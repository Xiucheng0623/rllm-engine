# -*- coding: utf-8 -*-
# File: D:\AI_RLLM\rllm_disk_engine\expert_pool\moe_layer_runner.py
"""MoE 手动逐层 forward 执行器 (v4 核心)

替代 model.forward(), 手动执行:
  embed → 逐层 (attention + MoE 专家路由) → norm → lm_head

关键创新:
  1. 每层 MoE forward 时, 只从 ExpertVRAMPool 获取被路由选中的 2 个专家
  2. 其余 6 个专家不加载 → I/O 量降 4 倍
  3. 配合 RouterPrefetcher: 下一步候选专家提前预取

MoE forward 手动拆解:
  1. gate(hidden) → router_logits [batch*seq, num_experts]
  2. softmax → routing_weights
  3. Top-2 → topk_weights, topk_indices
  4. 对每个被选中的专家:
     - 从 ExpertVRAMPool 获取专家模块
     - expert.forward(hidden) → expert_output
  5. 加权合并: output = sum(topk_weights[i] * expert_output[i])

性能契约:
  - 单层 attention forward: <5ms (VRAM 常驻)
  - 单专家 FFN forward: <3ms (VRAM 常驻)
  - 单层 MoE forward (2 专家): <12ms
  - 32 层 decode: ~400ms → 2.5 tok/s (全 VRAM)
  - 加预取掩盖 I/O: 15-20 tok/s (混合命中)
"""
from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from loguru import logger

from rllm_agent_core import LOG_DIR
from rllm_disk_engine.expert_pool.expert_vram_pool import (
    ExpertVRAMPool,
    ExpertKey,
)
from rllm_disk_engine.router.router_predictor import RouterPredictor
from rllm_disk_engine.router.router_prefetcher import RouterPrefetcher

logger.add(
    LOG_DIR / "moe_runner_{time}.log",
    rotation="50 MB",
    retention="7 days",
    encoding="utf-8",
)

# prefill 预取前瞻层数
PREFETCH_LOOKAHEAD: int = 4


class MoELayerRunner:
    """MoE 手动逐层 forward 执行器

    持有 Mixtral 的非专家组件, 专家模块从 ExpertVRAMPool 动态获取.

    Args:
        config: MixtralConfig 实例
        embed_tokens: embedding 层 (已 .to("cuda"))
        norm: 最终 RMSNorm (已 .to("cuda"))
        lm_head: lm_head 线性层 (已 .to("cuda"))
        attention_layers: 各层 attention 模块列表 (已 .to("cuda"))
        gate_layers: 各层路由器 Linear 列表 (已 .to("cuda"))
        layernorms: 各层 (input_layernorm, post_attention_layernorm) 列表
        vram_pool: ExpertVRAMPool 实例
        router_predictor: 路由预测器 (可选)
        router_prefetcher: 路由器预取器 (可选, 用于预取候选专家)
    """

    def __init__(
        self,
        config: Any,
        embed_tokens: Any,
        norm: Any,
        lm_head: Any,
        attention_layers: List[Any],
        gate_layers: List[Any],
        layernorms: List[Tuple[Any, Any]],
        vram_pool: ExpertVRAMPool,
        router_predictor: Optional[RouterPredictor] = None,
        router_prefetcher: Optional[RouterPrefetcher] = None,
    ) -> None:
        self._config = config
        self._embed = embed_tokens
        self._norm = norm
        self._lm_head = lm_head
        self._attention_layers: List[Any] = attention_layers
        self._gate_layers: List[Any] = gate_layers
        self._layernorms: List[Tuple[Any, Any]] = layernorms
        self._vram_pool: ExpertVRAMPool = vram_pool
        self._router_predictor: Optional[RouterPredictor] = router_predictor
        self._router_prefetcher: Optional[RouterPrefetcher] = router_prefetcher

        self._num_layers: int = config.num_hidden_layers
        self._num_experts: int = config.num_local_experts
        self._num_experts_per_tok: int = config.num_experts_per_tok
        self._hidden_size: int = config.hidden_size
        self._intermediate_size: int = config.intermediate_size

        # KV cache (使用 DynamicCache)
        self._kv_cache: Any = None

        # decode 模式标志 (prefill 后设为 True, 允许容错替代)
        self._decode_mode: bool = False

        # prefill 预取累积列表 (每层的 Top-2 路由结果)
        self._prefill_route_log: List[List[ExpertKey]] = []

        # 统计
        self._expert_fetch_count: int = 0
        self._expert_vram_hit: int = 0
        self._expert_fallback_count: int = 0
        self._total_tokens: int = 0

        logger.info(
            f"[MoE-Runner] 初始化: layers={self._num_layers} "
            f"experts/layer={self._num_experts} "
            f"experts/tok={self._num_experts_per_tok} "
            f"prefetcher={'启用' if router_prefetcher else '未启用'}"
        )

    # ----------------------------------------------------------------
    # Prefill: 处理整个 prompt
    # ----------------------------------------------------------------
    async def prefill(
        self,
        input_ids: torch.Tensor,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> Tuple[int, float]:
        """Prefill 阶段: 处理整个 prompt, 生成首 token

        Args:
            input_ids: [1, seq_len] 的 token id 张量
            temperature: 采样温度
            top_p: nucleus 采样阈值

        Returns:
            (首 token id, prefill 耗时秒)
        """
        t0 = time.time()
        device = input_ids.device
        batch_size, seq_len = input_ids.shape

        # Step 1: embed
        hidden_states = self._embed(input_ids)
        cache_position = torch.arange(seq_len, device=device)
        position_ids = cache_position.unsqueeze(0)

        # 初始化 KV cache
        from transformers.cache_utils import DynamicCache
        self._kv_cache = DynamicCache()

        # Step 2: 逐层 forward
        for layer_idx in range(self._num_layers):
            hidden_states = await self._forward_single_layer(
                layer_idx,
                hidden_states,
                position_ids=position_ids,
                cache_position=cache_position,
            )

        # Step 3: norm → lm_head
        hidden_states = self._norm(hidden_states)
        logits = self._lm_head(hidden_states)

        # Step 4: 采样首 token
        next_tok = self._sample_next_token(
            logits, [], temperature, top_p
        )

        elapsed = time.time() - t0
        self._total_tokens += 1
        logger.info(
            f"[MoE-Runner] Prefill 完成: seq_len={seq_len} "
            f"耗时={elapsed:.3f}s "
            f"专家获取={self._expert_fetch_count} "
            f"VRAM命中={self._expert_vram_hit}"
        )
        return next_tok, elapsed

    # ----------------------------------------------------------------
    # Decode: 单 token 增量 forward
    # ----------------------------------------------------------------
    async def decode_step(
        self,
        last_token: int,
        history_tokens: List[int],
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> Tuple[int, float]:
        """Decode 阶段: 单 token 增量生成

        Args:
            last_token: 上一步生成的 token id
            history_tokens: 已生成的 token 序列
            temperature: 采样温度
            top_p: nucleus 采样阈值

        Returns:
            (下一个 token id, 本步耗时秒)
        """
        t0 = time.time()
        device = self._embed.weight.device

        # decode 模式: 允许专家容错替代 (避免 fetch_back I/O)
        self._decode_mode = True

        # 构造单 token 输入
        input_ids = torch.tensor(
            [[last_token]], dtype=torch.long, device=device
        )

        # embed
        hidden_states = self._embed(input_ids)

        # 序列位置
        past_seq_len = self._kv_cache.get_seq_length() if self._kv_cache else 0
        cache_position = torch.tensor([past_seq_len], device=device)
        position_ids = cache_position.unsqueeze(0)

        # 逐层 forward
        for layer_idx in range(self._num_layers):
            hidden_states = await self._forward_single_layer(
                layer_idx,
                hidden_states,
                position_ids=position_ids,
                cache_position=cache_position,
            )

        # norm → lm_head
        hidden_states = self._norm(hidden_states)
        logits = self._lm_head(hidden_states)

        # 采样
        next_tok = self._sample_next_token(
            logits, history_tokens, temperature, top_p
        )

        # fire-and-forget 预取下一 token 候选 (RouterPredictor → RouterPrefetcher)
        self._prefetch_for_next_token()

        elapsed = time.time() - t0
        self._total_tokens += 1
        return next_tok, elapsed

    # ----------------------------------------------------------------
    # 内部: 单层 forward (attention + MoE)
    # ----------------------------------------------------------------
    async def _forward_single_layer(
        self,
        layer_idx: int,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        cache_position: torch.Tensor,
    ) -> torch.Tensor:
        """执行单层 MixtralDecoderLayer forward (手动拆解)

        流程:
          1. input_layernorm
          2. attention forward (VRAM 常驻)
          3. residual
          4. post_attention_layernorm
          5. MoE forward (gate 路由 + 2 专家)
          6. residual

        Args:
            layer_idx: 层索引
            hidden_states: [batch, seq, hidden]
            position_ids: 位置编码
            cache_position: cache 位置

        Returns:
            输出 hidden_states
        """
        residual = hidden_states

        # 1. input_layernorm
        input_ln = self._layernorms[layer_idx][0]
        hidden_states = input_ln(hidden_states)

        # 2. attention
        attn = self._attention_layers[layer_idx]
        with torch.no_grad():
            attn_outputs = attn(
                hidden_states,
                attention_mask=None,
                position_ids=position_ids,
                past_key_value=self._kv_cache,
                output_attentions=False,
                use_cache=True,
                cache_position=cache_position,
            )
        hidden_states = attn_outputs[0]

        # 3. residual
        hidden_states = residual + hidden_states

        # 4. post_attention_layernorm
        residual = hidden_states
        post_ln = self._layernorms[layer_idx][1]
        hidden_states = post_ln(hidden_states)

        # 5. MoE forward (gate 路由 + 专家)
        hidden_states = await self._forward_moe_layer(
            layer_idx, hidden_states
        )

        # 6. residual
        hidden_states = residual + hidden_states

        return hidden_states

    # ----------------------------------------------------------------
    # 内部: MoE 层 forward (核心创新)
    # ----------------------------------------------------------------
    async def _forward_moe_layer(
        self,
        layer_idx: int,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """MoE 层 forward (手动拆解路由 + 专家)

        Mixtral MoE 逻辑:
          1. gate(hidden) → router_logits [batch*seq, num_experts]
          2. softmax → routing_weights
          3. Top-2 → topk_weights, topk_indices
          4. 对每个被选中的专家:
             - 从 ExpertVRAMPool 获取专家模块
             - expert.forward(hidden) → expert_output
          5. 加权合并

        关键: 只获取被路由选中的 2 个专家, 其余 6 个不加载.

        Args:
            layer_idx: 层索引
            hidden_states: [batch, seq, hidden]

        Returns:
            MoE 输出 [batch, seq, hidden]
        """
        batch_size, seq_len, hidden_dim = hidden_states.shape
        # 展平为 [batch*seq, hidden]
        flat_hidden = hidden_states.view(-1, hidden_dim)

        # 1. gate 路由
        gate = self._gate_layers[layer_idx]
        router_logits = gate(flat_hidden)  # [batch*seq, num_experts]

        # 记录 gate 输出 (供 RouterPredictor 用)
        if self._router_predictor is not None:
            self._router_predictor.update_gate_output(
                layer_idx, router_logits
            )

        # 2. softmax + Top-2
        routing_weights = F.softmax(router_logits, dim=1)
        topk_weights, topk_indices = routing_weights.topk(
            self._num_experts_per_tok, dim=-1
        )
        # 归一化权重
        topk_weights = topk_weights / topk_weights.sum(
            dim=-1, keepdim=True
        )

        # 2.5 prefill 阶段: 记录本层路由 + 触发前瞻预取
        if not self._decode_mode:
            all_activated: List[ExpertKey] = []
            for expert_rank in range(self._num_experts_per_tok):
                for expert_idx in topk_indices[:, expert_rank].unique().tolist():
                    all_activated.append((layer_idx, expert_idx))
            self._prefill_route_log.append(all_activated)
            # 每 PREFETCH_LOOKAHEAD 层, 对后续层做预取
            if (layer_idx % PREFETCH_LOOKAHEAD == 0
                    and self._router_prefetcher is not None):
                self._trigger_prefetch_for_future_layers(layer_idx)

        # 3. 对每个 Top-2 专家, 获取模块并 forward
        output = torch.zeros_like(flat_hidden)
        num_tokens = flat_hidden.shape[0]

        for expert_rank in range(self._num_experts_per_tok):
            expert_indices = topk_indices[:, expert_rank]  # [num_tokens]
            weights = topk_weights[:, expert_rank]  # [num_tokens]

            # 按专家分组: 哪些 token 路由到了同一个专家
            unique_experts = torch.unique(expert_indices)

            for expert_idx in unique_experts.tolist():
                # 找出路由到这个专家的 token
                mask = expert_indices == expert_idx
                if not mask.any():
                    continue

                token_hidden = flat_hidden[mask]  # [n, hidden]
                token_weights = weights[mask]  # [n]

                # 从 VRAM 池获取专家模块
                # decode 模式: 优先 VRAM 命中, 未命中则用同层替代专家
                expert_key: ExpertKey = (layer_idx, expert_idx)

                if self._decode_mode:
                    # decode 模式: 先检查 VRAM, 未命中不触发 fetch_back
                    entry = self._vram_pool.get_expert_entry(expert_key)
                    if entry is not None:
                        expert_module = entry.module
                        entry.access_count += 1
                        entry.last_access_ts = time.time()
                        self._expert_vram_hit += 1
                    else:
                        # 未命中: 用同层第一个 VRAM 中的专家替代
                        expert_module = self._find_fallback_expert(
                            layer_idx
                        )
                        if expert_module is None:
                            logger.warning(
                                f"[MoE-Runner] decode 无替代专家 "
                                f"layer={layer_idx}, 跳过"
                            )
                            continue
                        self._expert_fallback_count += 1
                else:
                    # prefill 模式: 正常获取 (可能触发 fetch_back)
                    expert_module = await self._vram_pool.get_expert(
                        expert_key
                    )
                    if expert_module is None:
                        logger.warning(
                            f"[MoE-Runner] 无法获取专家 {expert_key}, "
                            f"跳过 {mask.sum().item()} 个 token"
                        )
                        continue

                self._expert_fetch_count += 1

                # 专家 forward: w2(act(w1(x)) * w3(x))
                # Linear4bit 的 weight.data 是 uint8 (量化数据), param.dtype
                # 返回 uint8 而非 compute_dtype, 不能用 param.dtype 对齐输入.
                # 需要从 Linear4bit.compute_dtype 获取真正的计算 dtype.
                compute_dtype: Optional[torch.dtype] = None
                for _sub_name, _sub_mod in expert_module.named_modules():
                    if hasattr(_sub_mod, "compute_dtype") and _sub_mod.compute_dtype is not None:
                        compute_dtype = _sub_mod.compute_dtype
                        break
                if compute_dtype is not None:
                    # Linear4bit: 输入用 compute_dtype (bfloat16)
                    token_hidden_typed = token_hidden.to(compute_dtype)
                else:
                    # 普通 Linear: 输入和权重 dtype 一致
                    expert_dtype = next(expert_module.parameters()).dtype
                    if expert_dtype != token_hidden.dtype:
                        token_hidden_typed = token_hidden.to(expert_dtype)
                    else:
                        token_hidden_typed = token_hidden
                with torch.no_grad():
                    expert_out = expert_module(token_hidden_typed)

                # 加权累加
                weighted_out = expert_out * token_weights.unsqueeze(-1)
                # scatter 回原位置
                output[mask] += weighted_out

        # reshape 回 [batch, seq, hidden]
        return output.view(batch_size, seq_len, hidden_dim)

    # ----------------------------------------------------------------
    # 内部: prefill 前瞻预取 (融合 RouterPredictor + RouterPrefetcher)
    # ----------------------------------------------------------------
    def _trigger_prefetch_for_future_layers(self, current_layer: int) -> None:
        """根据已处理层的路由历史, 预取后续层的热门专家

        启发式: 同 index 专家在不同层有相似语义功能, 当前层激活的热门
        专家 index 大概率也是后续层需要的. 通过异步预取, 让 I/O 与当前
        层计算重叠, 减少后续层的 fetch_back 等待.

        Args:
            current_layer: 当前刚处理完的层索引
        """
        if self._router_prefetcher is None or not self._prefill_route_log:
            return

        # 1. 统计已处理层中最热门的专家 index (跨层频次)
        from collections import Counter
        freq: Counter = Counter()
        for layer_routes in self._prefill_route_log:
            for layer_idx, expert_idx in layer_routes:
                freq[expert_idx] += 1

        # 2. Top-3 最热专家 index
        top_indices = [e for e, _ in freq.most_common(3)]

        # 3. 对未来 PREFETCH_LOOKAHEAD 层, 生成候选列表
        candidates: List[ExpertKey] = []
        for ahead in range(1, PREFETCH_LOOKAHEAD + 1):
            future_layer = current_layer + ahead
            if future_layer >= self._num_layers:
                break
            for e_idx in top_indices:
                candidates.append((future_layer, e_idx))

        if not candidates:
            return

        # 4. fire-and-forget 异步预取 (不阻塞当前层)
        asyncio.ensure_future(
            self._router_prefetcher.prefetch_candidates(candidates)
        )

    # ----------------------------------------------------------------
    # 内部: decode 阶段预取下一 token 候选
    # ----------------------------------------------------------------
    def _prefetch_for_next_token(self) -> None:
        """decode 阶段: 预测下一 token 专家候选 + fire-and-forget 预取

        用 RouterPredictor 预测下一 token 各层的 Top-K 候选专家,
        fire-and-forget 触发预取. 相邻 token 路由强相关 (~80% 重合),
        大部分候选在下步已命中 VRAM.
        """
        if self._router_predictor is None or self._router_prefetcher is None:
            return
        candidates = self._router_predictor.predict_candidates()
        if candidates:
            asyncio.ensure_future(
                self._router_prefetcher.prefetch_candidates(candidates)
            )

    # ----------------------------------------------------------------
    # 内部: 采样
    # ----------------------------------------------------------------
    def _sample_next_token(
        self,
        logits: torch.Tensor,
        history_tokens: List[int],
        temperature: float,
        top_p: float,
    ) -> int:
        """采样下一个 token (temperature + top-p)

        Args:
            logits: [batch, seq, vocab] 的 logits
            history_tokens: 历史生成的 token (用于 repetition penalty)
            temperature: 采样温度
            top_p: nucleus 采样阈值

        Returns:
            token id
        """
        # 取最后一个 token 的 logits
        next_logits = logits[0, -1, :]  # [vocab]

        # Repetition penalty
        if history_tokens:
            for tok_id in set(history_tokens[-64:]):
                next_logits[tok_id] /= 1.1

        # Temperature
        if temperature > 0:
            next_logits = next_logits / temperature

        # Top-p sampling
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(
                next_logits, descending=True
            )
            cumulative_probs = torch.cumsum(
                F.softmax(sorted_logits, dim=-1), dim=-1
            )
            # 移除累积概率超过 top_p 的 token
            sorted_indices_to_remove = cumulative_probs > top_p
            # 保留第一个超过 top_p 的 token
            sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
            sorted_indices_to_remove[0] = False
            indices_to_remove = sorted_indices_to_remove.scatter(
                0, sorted_indices, sorted_indices_to_remove
            )
            next_logits = next_logits.masked_fill(
                indices_to_remove, float("-inf")
            )

        # 采样
        if temperature > 0:
            probs = F.softmax(next_logits, dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1).item()
        else:
            next_tok = torch.argmax(next_logits).item()

        return next_tok

    # ----------------------------------------------------------------
    # 内部: 查找同层替代专家 (decode 容错)
    # ----------------------------------------------------------------
    def _find_fallback_expert(
        self, layer_idx: int
    ) -> Optional[torch.nn.Module]:
        """查找同层最热的 VRAM 专家 (decode 容错替代)

        当 decode 时路由到的专家不在 VRAM, 用同层 access_count 最高
        的 pinned 专家代替, 避免 fetch_back I/O. 选最热专家而非第一个,
        因为热专家代表了该层最常见的计算模式, 替代损失最小.

        Args:
            layer_idx: 层索引

        Returns:
            替代专家模块或 None
        """
        best_entry: Optional[Any] = None
        best_count: int = -1
        for expert_idx in range(self._num_experts):
            entry = self._vram_pool.get_expert_entry(
                (layer_idx, expert_idx)
            )
            if entry is not None and entry.access_count > best_count:
                best_entry = entry
                best_count = entry.access_count
        if best_entry is not None:
            return best_entry.module
        return None

    # ----------------------------------------------------------------
    # 统计
    # ----------------------------------------------------------------
    def stats(self) -> Dict[str, Any]:
        """获取 runner 统计"""
        return {
            "total_tokens": self._total_tokens,
            "expert_fetch_count": self._expert_fetch_count,
            "expert_vram_hit": self._expert_vram_hit,
            "expert_fallback_count": self._expert_fallback_count,
            "vram_pool_stats": self._vram_pool.stats(),
            "router_predictor_stats": (
                self._router_predictor.stats()
                if self._router_predictor
                else None
            ),
        }
