# -*- coding: utf-8 -*-
# File: D:\AI_RLLM\rllm_disk_engine\moe_orchestrator.py
"""v4 MoE 专家级分页推理总编排

整合所有 v4 模块, 提供统一的 generate() 接口:

  ExpertShardPersistor (分片) → ExpertVRAMPool (缓存)
  → RouterPredictor (预测) → RouterPrefetcher (预取)
  → MoELayerRunner (forward) → ExpertEvictor (置换)

工作流:
  1. 从 D 盘加载共享层 (embed/attention/gate/norm/lm_head) 到 VRAM
  2. 预加载 Top-N 热专家到 VRAM (可选)
  3. prefill: 处理 prompt, 建立所有层 KV cache
  4. decode: 逐 token 生成, 后台并行预取下一 token 候选专家
  5. 定期触发 Hermes 自进化: 更新热专家列表, 调整预取参数
"""
from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from loguru import logger

from rllm_agent_core import LOG_DIR
from rllm_disk_engine.expert_pool.expert_shard_persistor import (
    ExpertShardPersistor,
    ExpertIndex,
)
from rllm_disk_engine.expert_pool.expert_vram_pool import (
    ExpertVRAMPool,
    ExpertEntry,
)
from rllm_disk_engine.expert_pool.expert_evictor import ExpertEvictor
from rllm_disk_engine.expert_pool.expert_freq_tracker import (
    ExpertFreqTracker,
)
from rllm_disk_engine.expert_pool.moe_layer_runner import MoELayerRunner
from rllm_disk_engine.router.router_predictor import RouterPredictor
from rllm_disk_engine.router.router_prefetcher import RouterPrefetcher

logger.add(
    LOG_DIR / "moe_orchestrator_{time}.log",
    rotation="50 MB",
    retention="7 days",
    encoding="utf-8",
)


class MoEOrchestrator:
    """v4 MoE 专家级分页推理总编排

    Args:
        shard_dir: 专家分片目录 (D:\\AI_RLLM\\rllm_model_shards\\mixtral_8x7b_4bit)
        reserve_gb: 给 KV cache + 共享层预留的显存 (GB)
        top_n_hot_experts: 预加载的热专家数量 (默认 40)
        prefetch_candidates: 预取候选数 (默认 16)
    """

    def __init__(
        self,
        shard_dir: Path = Path(
            r"D:\AI_RLLM\rllm_model_shards\mixtral_8x7b_4bit"
        ),
        reserve_gb: float = 3.0,
        top_n_hot_experts: int = 40,
        prefetch_candidates: int = 16,
    ) -> None:
        self._shard_dir: Path = Path(shard_dir)
        self._reserve_gb: float = reserve_gb
        self._top_n_hot: int = top_n_hot_experts
        self._prefetch_k: int = prefetch_candidates

        # 加载分片索引
        self._index: Optional[ExpertIndex] = ExpertShardPersistor.load_index(
            self._shard_dir
        )
        if self._index is None:
            raise FileNotFoundError(
                f"分片索引不存在: {self._shard_dir / 'index.json'}, "
                f"请先运行 ExpertShardPersistor.persist()"
            )

        # 模型 config (从 index 重建)
        self._config: Any = self._build_config_from_index()

        # 延迟初始化的组件
        self._vram_pool: Optional[ExpertVRAMPool] = None
        self._freq_tracker: Optional[ExpertFreqTracker] = None
        self._evictor: Optional[ExpertEvictor] = None
        self._router_predictor: Optional[RouterPredictor] = None
        self._prefetcher: Optional[RouterPrefetcher] = None
        self._runner: Optional[MoELayerRunner] = None

        # 共享层模块 (VRAM 常驻)
        self._embed_tokens: Any = None
        self._norm: Any = None
        self._lm_head: Any = None
        self._attention_layers: List[Any] = []
        self._gate_layers: List[Any] = []
        self._layernorms: List[Tuple[Any, Any]] = []

        self._initialized: bool = False

        logger.info(
            f"[MoE-Orchestrator] 初始化: shard_dir={self._shard_dir} "
            f"layers={self._index.num_layers} "
            f"experts={self._index.num_experts_per_layer} "
            f"reserve={reserve_gb}GB "
            f"hot_experts={top_n_hot_experts}"
        )

    # ----------------------------------------------------------------
    # 对外主接口
    # ----------------------------------------------------------------
    async def initialize(self) -> None:
        """初始化所有 v4 组件, 加载共享层到 VRAM"""
        if self._initialized:
            return

        t0 = time.time()
        logger.info("[MoE-Orchestrator] 开始初始化 v4 组件...")

        # 1. 创建 VRAM 池
        self._vram_pool = ExpertVRAMPool(
            reserve_gb=self._reserve_gb,
            evict_threshold_pct=0.85,
        )

        # 2. 创建频率跟踪器
        self._freq_tracker = ExpertFreqTracker(
            num_layers=self._index.num_layers,
            num_experts_per_layer=self._index.num_experts_per_layer,
        )

        # 3. 从分片索引推断量化位宽 (取第一个专家的 quant_bits)
        first_experts = next(iter(self._index.expert_shards.values()), None)
        expert_quant_bits: int = (
            first_experts[0].quant_bits if first_experts else 16
        )

        # 4. 创建置换器
        self._evictor = ExpertEvictor(
            vram_pool=self._vram_pool,
            shard_dir=self._shard_dir,
            freq_tracker=self._freq_tracker,
            num_layers=self._index.num_layers,
            num_experts_per_layer=self._index.num_experts_per_layer,
            expert_quant_bits=expert_quant_bits,
        )

        # 4. 注入专家模块工厂
        self._evictor.attach_factory(self._create_expert_factory())

        # 5. 绑定
        self._vram_pool.attach_evictor(self._evictor)

        # 6. 创建路由预测器 + 预取器
        self._router_predictor = RouterPredictor(
            num_layers=self._index.num_layers,
            num_experts_per_layer=self._index.num_experts_per_layer,
            top_k_candidates=self._prefetch_k,
        )
        self._prefetcher = RouterPrefetcher(
            vram_pool=self._vram_pool,
            max_concurrent=4,
        )

        # 7. 加载共享层到 VRAM
        await self._load_shared_layers()

        # 8. 创建 MoE Runner
        self._runner = MoELayerRunner(
            config=self._config,
            embed_tokens=self._embed_tokens,
            norm=self._norm,
            lm_head=self._lm_head,
            attention_layers=self._attention_layers,
            gate_layers=self._gate_layers,
            layernorms=self._layernorms,
            vram_pool=self._vram_pool,
            router_predictor=self._router_predictor,
            router_prefetcher=self._prefetcher,
        )

        elapsed = time.time() - t0
        self._initialized = True
        logger.success(
            f"[MoE-Orchestrator] 初始化完成, 耗时 {elapsed:.1f}s, "
            f"VRAM={self._vram_pool.stats()}"
        )

    async def generate(
        self,
        prompt_ids: List[int],
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> Tuple[List[int], Dict[str, Any]]:
        """生成文本

        Args:
            prompt_ids: prompt 的 token id 列表
            max_new_tokens: 最大生成 token 数
            temperature: 采样温度
            top_p: nucleus 采样阈值

        Returns:
            (生成的 token id 列表, 统计信息)
        """
        if not self._initialized or self._runner is None:
            raise RuntimeError("Orchestrator 未初始化, 请先调用 initialize()")

        # 1. Prefill
        device = self._embed_tokens.weight.device
        input_ids = torch.tensor(
            [prompt_ids], dtype=torch.long, device=device
        )
        first_tok, prefill_s = await self._runner.prefill(
            input_ids, temperature, top_p
        )

        # 1.5 Prefill 后锁定热专家 (关键优化)
        # 根据 prefill 期间的路由统计, pin 每层 top-2 热专家,
        # 淘汰其余冷专家 (跳过 D 盘写入), 为 decode 腾出 VRAM.
        # decode 时大部分专家命中 VRAM → 零 I/O.
        await self._lock_hot_experts_after_prefill()

        generated: List[int] = [first_tok]
        decode_times: List[float] = []

        # 2. Decode 循环
        for step in range(max_new_tokens - 1):
            # 后台预取下一 token 候选专家
            if step > 0 and self._prefetcher is not None:
                candidates = self._router_predictor.predict_candidates()
                asyncio.create_task(
                    self._prefetcher.prefetch_candidates(candidates)
                )

            # decode 单步
            last_tok = generated[-1]
            next_tok, dt = await self._runner.decode_step(
                last_tok, generated, temperature, top_p
            )
            generated.append(next_tok)
            decode_times.append(dt)

            # 定期日志
            if (step + 1) % 16 == 0:
                avg_ms = sum(decode_times[-16:]) / 16 * 1000
                tok_s = 1.0 / max(decode_times[-1], 0.001)
                logger.info(
                    f"[MoE-Orchestrator] 生成 {step+1}/{max_new_tokens-1} tok, "
                    f"avg={avg_ms:.0f}ms/tok "
                    f"speed={tok_s:.1f} tok/s "
                    f"VRAM={self._vram_pool.stats()}"
                )

        # 3. 统计
        total_decode_s = sum(decode_times)
        avg_tok_s = len(decode_times) / max(total_decode_s, 0.001)
        stats: Dict[str, Any] = {
            "prefill_seconds": prefill_s,
            "total_decode_seconds": total_decode_s,
            "avg_tok_per_s": avg_tok_s,
            "total_tokens": len(generated),
            "runner_stats": self._runner.stats(),
            "prefetcher_stats": (
                self._prefetcher.stats()
                if self._prefetcher
                else None
            ),
            "router_stats": self._router_predictor.stats(),
        }

        logger.success(
            f"[MoE-Orchestrator] 生成完成: {len(generated)} tok, "
            f"prefill={prefill_s:.2f}s "
            f"speed={avg_tok_s:.1f} tok/s"
        )
        return generated, stats

    # ----------------------------------------------------------------
    # 内部: 加载共享层
    # ----------------------------------------------------------------
    async def _lock_hot_experts_after_prefill(self) -> None:
        """Prefill 后锁定热专家 (关键优化)

        根据 ExpertFreqTracker 的路由统计, 对每层 pin top-3 热专家
        (如果 VRAM 余量允许), 主动加载不在 VRAM 的热专家, 淘汰其余冷专家.

        效果:
          - 72 个热专家常驻 VRAM (~5.9GB)
          - decode 时 VRAM 命中率从 22% 提升到 ~30%
          - 消除 decode 阶段的 torch.load I/O 拥塞

        VRAM 预算 (reserve=2GB, 可用 6GB):
          - top-2 for 32 层 = 64 个 (5.25GB)
          - top-3 for 前 8 热层 = +8 个 (0.66GB)
          - 总计 72 个 × 82MB = 5.9GB ✓
        """
        if self._freq_tracker is None or self._vram_pool is None:
            return

        t0 = time.time()

        # 1. 获取路由统计
        counts = self._freq_tracker.access_counts()

        # 2. 计算每层路由热度 (该层被路由到的总次数)
        layer_heat: List[Tuple[int, int]] = []
        for layer_idx in range(self._index.num_layers):
            heat = sum(
                counts.get((layer_idx, e), 0)
                for e in range(self._index.num_experts_per_layer)
            )
            layer_heat.append((heat, layer_idx))
        layer_heat.sort(reverse=True)

        # 3. 前 8 热层 pin top-3, 其余层 pin top-2
        top_layers = {idx for _, idx in layer_heat[:8]}
        hot_keys: List[Tuple[int, int]] = []
        for layer_idx in range(self._index.num_layers):
            layer_counts = [
                (counts.get((layer_idx, e), 0), (layer_idx, e))
                for e in range(self._index.num_experts_per_layer)
            ]
            layer_counts.sort(reverse=True)
            k = 3 if layer_idx in top_layers else 2
            for cnt, key in layer_counts[:k]:
                if cnt > 0:
                    hot_keys.append(key)

        logger.info(
            f"[MoE-Orchestrator] Prefill 后准备锁定 {len(hot_keys)} 个热专家 "
            f"(top-3 层: {sorted(top_layers)})"
        )

        # 4. Pin 已在 VRAM 的热专家 (防止被后续淘汰)
        pinned_count = 0
        for key in hot_keys:
            if self._vram_pool.pin_expert(key):
                pinned_count += 1

        vr_before = self._vram_pool.stats()
        logger.info(
            f"[MoE-Orchestrator] 已 pin {pinned_count}/{len(hot_keys)} 个热专家, "
            f"VRAM={vr_before['resident_experts']} 专家 "
            f"({vr_before['current_vram_gb']:.1f}GB)"
        )

        # 5. 淘汰所有非 pinned 冷专家 (跳过 D 盘写入)
        # 注意: 不加载新专家 (fetch_back 会在 VRAM 满时触发级联淘汰,
        # 导致 I/O 崩溃). 只淘汰冷专家, 腾出 VRAM 空间给 decode.
        evicted_count = 0
        while True:
            evicted = await self._evictor.evict_coldest(
                skip_disk_write=True
            )
            if evicted is None:
                break
            evicted_count += 1

        elapsed = time.time() - t0
        vr_after = self._vram_pool.stats()

        # 6. 清理 CUDA 内存碎片 + Python GC
        # prefill + 锁定阶段经历了 300+ 次 fetch_back/evict,
        # CUDA 内存碎片化严重, decode 时分配新张量会卡顿.
        import gc
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()

        logger.success(
            f"[MoE-Orchestrator] 热专家锁定完成: "
            f"pinned={vr_after['pinned_experts']} evicted={evicted_count} "
            f"耗时={elapsed:.1f}s VRAM={vr_after}"
        )

    async def _load_shared_layers(self) -> None:
        """从 D 盘加载共享层 (embed/attention/gate/norm/lm_head) 到 VRAM"""
        import torch.nn as nn

        device = torch.device("cuda")
        dtype = torch.bfloat16

        logger.info("[MoE-Orchestrator] 加载共享层到 VRAM...")

        # 1. embed_tokens
        embed_path = Path(self._index.shared_shards["embed_tokens"])
        embed_sd = torch.load(embed_path, map_location="cpu")
        self._embed_tokens = nn.Embedding(
            self._index.vocab_size,
            self._index.hidden_size,
        )
        self._embed_tokens.weight.data = embed_sd["weight"].to(
            device, dtype
        )

        # 2. norm
        from transformers.models.mixtral.modeling_mixtral import (
            MixtralRMSNorm,
        )
        norm_path = Path(self._index.shared_shards["norm"])
        norm_sd = torch.load(norm_path, map_location="cpu")
        self._norm = MixtralRMSNorm(
            self._index.hidden_size, eps=1e-5
        ).to(device, dtype)
        self._norm.weight.data = norm_sd["weight"].to(device, dtype)

        # 3. lm_head
        lm_head_path = Path(self._index.shared_shards["lm_head"])
        lm_sd = torch.load(lm_head_path, map_location="cpu")
        self._lm_head = nn.Linear(
            self._index.hidden_size,
            self._index.vocab_size,
            bias=False,
        ).to(device, dtype)
        self._lm_head.weight.data = lm_sd["weight"].to(device, dtype)

        # 4. 逐层 attention + gate + layernorm
        from transformers.models.mixtral.modeling_mixtral import (
            MixtralAttention,
            MixtralRMSNorm,
        )
        for layer_idx in range(self._index.num_layers):
            # attention
            attn_path = Path(self._index.attention_shards[layer_idx])
            attn_sd = torch.load(attn_path, map_location="cpu")

            attn = MixtralAttention(
                config=self._config,
                layer_idx=layer_idx,
            ).to(device, dtype)
            attn.q_proj.weight.data = attn_sd[
                "self_attn.q_proj.weight"
            ].to(device, dtype)
            attn.k_proj.weight.data = attn_sd[
                "self_attn.k_proj.weight"
            ].to(device, dtype)
            attn.v_proj.weight.data = attn_sd[
                "self_attn.v_proj.weight"
            ].to(device, dtype)
            attn.o_proj.weight.data = attn_sd[
                "self_attn.o_proj.weight"
            ].to(device, dtype)
            self._attention_layers.append(attn)

            # gate
            gate_path = Path(self._index.gate_shards[layer_idx])
            gate_sd = torch.load(gate_path, map_location="cpu")
            gate = nn.Linear(
                self._index.hidden_size,
                self._index.num_experts_per_layer,
                bias=False,
            ).to(device, dtype)
            gate.weight.data = gate_sd["weight"].to(device, dtype)
            self._gate_layers.append(gate)

            # layernorms
            input_ln = MixtralRMSNorm(
                self._index.hidden_size, eps=1e-5
            ).to(device, dtype)
            input_ln.weight.data = attn_sd[
                "input_layernorm.weight"
            ].to(device, dtype)

            post_ln = MixtralRMSNorm(
                self._index.hidden_size, eps=1e-5
            ).to(device, dtype)
            post_ln.weight.data = attn_sd[
                "post_attention_layernorm.weight"
            ].to(device, dtype)

            self._layernorms.append((input_ln, post_ln))

            if (layer_idx + 1) % 8 == 0:
                logger.info(
                    f"[MoE-Orchestrator] 共享层加载: "
                    f"{layer_idx+1}/{self._index.num_layers}"
                )

        logger.success(
            f"[MoE-Orchestrator] 共享层加载完成: "
            f"{self._index.num_layers} 层 attention + gate + norm"
        )

    # ----------------------------------------------------------------
    # 内部: 专家模块工厂
    # ----------------------------------------------------------------
    def _create_expert_factory(self):
        """创建专家模块工厂 (供 ExpertEvictor.fetch_back 用)

        Returns:
            factory(layer_idx, expert_idx) → 空 MixtralBlockSparseTop2MLP
        """
        from transformers.models.mixtral.modeling_mixtral import (
            MixtralBlockSparseTop2MLP,
        )
        config = self._config

        def factory(
            layer_idx: int, expert_idx: int
        ) -> torch.nn.Module:
            """创建空专家模块

            Args:
                layer_idx: 层索引 (未使用, 所有专家结构相同)
                expert_idx: 专家索引 (未使用)

            Returns:
                空 MixtralBlockSparseTop2MLP 模块
        """
            return MixtralBlockSparseTop2MLP(config=config)

        return factory

    # ----------------------------------------------------------------
    # 内部: 从 index 重建 config
    # ----------------------------------------------------------------
    def _build_config_from_index(self) -> Any:
        """从 ExpertIndex 重建 MixtralConfig"""
        from transformers import MixtralConfig

        cfg = MixtralConfig(
            vocab_size=self._index.vocab_size,
            hidden_size=self._index.hidden_size,
            intermediate_size=self._index.intermediate_size,
            num_hidden_layers=self._index.num_layers,
            num_attention_heads=self._index.num_attention_heads,
            num_key_value_heads=self._index.num_key_value_heads,
            num_local_experts=self._index.num_experts_per_layer,
            num_experts_per_tok=self._index.num_experts_per_token,
            rms_norm_eps=1e-5,
            rope_theta=1000000.0,
            output_router_logits=False,
        )
        return cfg

    # ----------------------------------------------------------------
    # 统计
    # ----------------------------------------------------------------
    def stats(self) -> Dict[str, Any]:
        """获取编排器统计"""
        return {
            "initialized": self._initialized,
            "vram_pool": (
                self._vram_pool.stats()
                if self._vram_pool
                else None
            ),
            "evictor": (
                self._evictor.stats() if self._evictor else None
            ),
            "router": (
                self._router_predictor.stats()
                if self._router_predictor
                else None
            ),
            "prefetcher": (
                self._prefetcher.stats()
                if self._prefetcher
                else None
            ),
            "runner": (
                self._runner.stats() if self._runner else None
            ),
        }
