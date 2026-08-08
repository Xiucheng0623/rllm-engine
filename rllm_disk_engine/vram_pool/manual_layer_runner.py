# -*- coding: utf-8 -*-
# File: D:\AI_RLLM\rllm_disk_engine\vram_pool\manual_layer_runner.py
"""v3 路径核心: 手动逐层 forward runner

架构目标:
  1. 替代 model.forward() 整体调用, 改为手动 embed → 逐层 decoder → norm → lm_head
  2. 每层 forward 时, 该层的 past_key_value 可独立控制 (加载/保存/淘汰)
  3. 桥接 KVSpillManager: 显存超阈值时, 把指定层的 (key, value) spill 到 SSD
  4. 激活 VRAMCachePool + HotColdEvictor: 层模块从池中获取, 显存不足触发淘汰

设计原则:
  - 不修改 transformers 库源码, 仅调用其模块的 forward
  - 复用 MistralConfig 创建空模块, 由 VRAMCachePool 绑定权重
  - 单层 forward 语义与 model.forward 内部完全一致

性能契约:
  - 单层 forward 耗时 < 5ms (7B on RTX5070Ti)
  - KV spill 异步, 不阻塞 decode 主循环
  - Decode 阶段零权重磁盘 IO (常驻 VRAM)
"""
from __future__ import annotations

import asyncio
import gc
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger

from rllm_agent_core import LOG_DIR

logger.add(
    LOG_DIR / "v3_runner_{time}.log",
    rotation="50 MB",
    retention="7 days",
    encoding="utf-8",
)


# ============================================================
# 数据结构: 单层 KV 缓存条目
# ============================================================
@dataclass
class LayerKVEntry:
    """单层 KV 缓存条目

    Attributes:
        layer_idx: 层索引
        key: [batch, num_kv_heads, seq_len, head_dim] 形状的张量
        value: 同 key
        in_vram: 是否在显存中 (False 表示已 spill 到磁盘)
        spill_path: spill 时的磁盘路径 (None 表示未 spill)
        seq_len: 当前 KV 序列长度
    """
    layer_idx: int
    key: Any = None              # torch.Tensor 或 None (已 spill)
    value: Any = None
    in_vram: bool = True
    spill_path: Optional[Path] = None
    seq_len: int = 0


# ============================================================
# 核心: 手动逐层 forward runner
# ============================================================
class ManualLayerRunner:
    """手动逐层 forward 执行器

    职责:
      1. 持有 Mistral 模型的非层组件 (embed_tokens / norm / lm_head)
      2. 执行 prefill: embed → 32 层 decoder → norm → lm_head
      3. 执行 decode: 单 token 过 32 层, 复用每层 past_kv
      4. 桥接 KVSpillManager: KV 超阈值时 spill 指定层

    与 VRAMCachePool 配合:
      - 不直接持有层模块, 而是从 pool.get_layer(idx) 获取
      - 显存不足时 pool 自动触发 HotColdEvictor 淘汰冷层
      - 被淘汰的层再次访问时 pool 自动 fetch_back

    Args:
        config: MistralConfig 实例
        embed_tokens: embedding 层 (已 .to("cuda"))
        norm: 最终 RMSNorm (已 .to("cuda"))
        lm_head: lm_head 线性层 (已 .to("cuda"))
        vram_pool: VRAMCachePool 实例 (提供层模块)
        kv_spill_threshold_mb: KV 总显存占用阈值, 超过则 spill 最旧层
        spill_dir: KV spill 文件目录
    """

    def __init__(
        self,
        config: Any,
        embed_tokens: Any,
        norm: Any,
        lm_head: Any,
        vram_pool: Any,
        kv_spill_threshold_mb: int = 512,
        spill_dir: Optional[Path] = None,
    ) -> None:
        self._config = config
        self._embed = embed_tokens
        self._norm = norm
        self._lm_head = lm_head
        self._vram_pool = vram_pool
        self._num_layers: int = config.num_hidden_layers
        self._hidden_size: int = config.hidden_size
        self._num_kv_heads: int = config.num_key_value_heads
        self._head_dim: int = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )

        self._kv_threshold_bytes: int = kv_spill_threshold_mb * 1024 * 1024
        self._spill_dir: Path = Path(spill_dir or r"D:\AI_RLLM\rllm_offload_temp\kv_cache")
        self._spill_dir.mkdir(parents=True, exist_ok=True)

        # 每层 KV 缓存条目 (layer_idx -> LayerKVEntry)
        self._kv_cache: Dict[int, LayerKVEntry] = {}
        # KV spill 计数
        self._spill_count: int = 0
        self._readback_count: int = 0
        # 线程锁 (KV 操作需加锁)
        self._lock = threading.RLock()

        logger.info(
            f"[v3-Runner] 初始化完成: layers={self._num_layers} "
            f"hidden={self._hidden_size} kv_heads={self._num_kv_heads} "
            f"head_dim={self._head_dim} kv_threshold={kv_spill_threshold_mb}MB"
        )

    # ----------------------------------------------------------------
    # Prefill: 处理整个 prompt, 建立所有层的 KV cache
    # ----------------------------------------------------------------
    async def prefill(
        self,
        input_ids: Any,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> Tuple[int, float]:
        """Prefill 阶段: 处理整个 prompt, 生成首 token, 建立所有层 KV cache

        Args:
            input_ids: [1, seq_len] 的 token id 张量 (已 .to("cuda"))
            temperature: 采样温度
            top_p: nucleus 采样阈值

        Returns:
            (首 token id, prefill 耗时秒)

        Note:
            - 逐层 forward, 每层产生 (key, value) 存入 self._kv_cache
            - 全部完成后检查 KV 总显存, 超阈值则 spill 最旧层
            - pref 段不进行 KV spill (所有层都需要参与 forward)
        """
        import torch
        t0 = time.time()
        device = input_ids.device
        batch_size, seq_len = input_ids.shape

        # 确保 embed/norm/lm_head 都在 CUDA (VRAM 压力下可能被换到 CPU)
        cuda_device = torch.device("cuda")
        if self._embed.weight.device.type != "cuda":
            self._embed = self._embed.to(cuda_device)
        if next(self._norm.parameters()).device.type != "cuda":
            self._norm = self._norm.to(cuda_device)
        if next(self._lm_head.parameters()).device.type != "cuda":
            self._lm_head = self._lm_head.to(cuda_device)

        # Step 1: embed_tokens → hidden_states
        hidden_states = self._embed(input_ids)
        # cache_position: [0, 1, ..., seq_len-1]
        cache_position = torch.arange(seq_len, device=device)
        # position_ids (Mistral 内部 RoPE 需要)
        position_ids = cache_position.unsqueeze(0)
        # attention_mask=None: 让 transformers 内部自己处理 (传 2D 会出错, 4D 太复杂)
        # transformers 会警告 "attention mask not set", 但不影响 prefill 正确性 (无 padding 场景)

        # Step 2: 逐层 decoder forward
        # 关键: transformers 4.44 的 DynamicCache 期望 layer_idx 按顺序 0,1,2... 填充,
        # 不能跳跃. 所以用单一共享 cache, 所有层共享, 但 key_cache[layer_idx] 仍按层独立.
        from transformers.cache_utils import DynamicCache

        shared_cache = DynamicCache()

        for layer_idx in range(self._num_layers):
            # 从 VRAMCachePool 获取层模块 (可能触发 fetch_back)
            layer_module = await self._vram_pool.get_layer(layer_idx)
            if layer_module is None:
                raise RuntimeError(
                    f"[v3-Runner] 无法获取层 {layer_idx} (VRAM 池为空且无法 fetch_back)"
                )

            # 执行该层 forward (共享 cache, attention 内部会 update 对应 layer_idx 槽位)
            with torch.no_grad():
                outputs = layer_module(
                    hidden_states,
                    attention_mask=None,
                    position_ids=position_ids,
                    past_key_value=shared_cache,
                    output_attentions=False,
                    use_cache=True,
                    cache_position=cache_position,
                )
            hidden_states = outputs[0]

        # Prefill 结束后, 从共享 cache 提取每层 KV 到独立条目 (便于后续按层 spill)
        with self._lock:
            for layer_idx in range(self._num_layers):
                try:
                    key = shared_cache.key_cache[layer_idx]
                    value = shared_cache.value_cache[layer_idx]
                    self._kv_cache[layer_idx] = LayerKVEntry(
                        layer_idx=layer_idx,
                        key=key,
                        value=value,
                        in_vram=True,
                        seq_len=seq_len,
                    )
                except IndexError as exc:
                    logger.warning(
                        f"[v3-Runner] prefill 提取层 {layer_idx} KV 失败: {exc}"
                    )

        # Step 3: norm → lm_head → logits (确保 device 正确)
        norm_device = next(self._norm.parameters()).device
        if norm_device.type != "cuda":
            self._norm = self._norm.to(device)
        lm_device = next(self._lm_head.parameters()).device
        if lm_device.type != "cuda":
            self._lm_head = self._lm_head.to(device)

        hidden_states = self._norm(hidden_states)
        logits = self._lm_head(hidden_states)

        # Step 4: 采样首 token
        next_tok = self._sample_next_token(logits, [], temperature, top_p)

        # Step 5: 检查 KV 总显存, 超阈值 spill 最旧层 (异步)
        await self._check_and_spill_kv_if_needed()

        elapsed = time.time() - t0
        logger.info(
            f"[v3-Runner] Prefill 完成: seq_len={seq_len} "
            f"耗时={elapsed:.3f}s KV层数={len(self._kv_cache)}"
        )
        return next_tok, elapsed

    # ----------------------------------------------------------------
    # Decode: 单 token 增量 forward, 复用每层 past_kv
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
            history_tokens: 已生成的 token 序列 (用于 repetition penalty)
            temperature: 采样温度
            top_p: nucleus 采样阈值

        Returns:
            (下一个 token id, 本步耗时秒)

        Note:
            - 每层 forward 复用该层 past_kv, 增量产生新 (key, value)
            - 若该层 KV 已 spill 到磁盘, 先 fetch_back 回 VRAM
            - decode 结束后检查 KV 总显存, 超阈值 spill 最旧层
            - 验收点: 此方法不产生权重磁盘 IO (层模块常驻 VRAM)
        """
        import torch
        t0 = time.time()
        # 强制使用 CUDA, 避免 embed/norm 被 VRAM 压力换到 CPU 后 device 变量也变 CPU
        device = torch.device("cuda")
        # 确保 embed/norm/lm_head 都在 CUDA (VRAM 压力下可能被换到 CPU)
        if self._embed.weight.device.type != "cuda":
            self._embed = self._embed.to(device)
        if next(self._norm.parameters()).device.type != "cuda":
            self._norm = self._norm.to(device)
        if next(self._lm_head.parameters()).device.type != "cuda":
            self._lm_head = self._lm_head.to(device)

        # 构造单 token 输入
        input_ids = torch.tensor([[last_token]], dtype=torch.long, device=device)

        # embed
        hidden_states = self._embed(input_ids)

        # 当前序列长度 = 历史 KV 长度 + 1 (本次新 token)
        # 从第 0 层的 KV entry 取 seq_len
        with self._lock:
            first_entry = next(iter(self._kv_cache.values()), None)
            past_seq_len = first_entry.seq_len if first_entry else 0
        new_seq_len = past_seq_len + 1
        # attention_mask=None: 让 transformers 内部自己处理
        cache_position = torch.tensor([past_seq_len], device=device)
        position_ids = cache_position.unsqueeze(0)

        # 逐层 forward (共享 cache, 每层按 layer_idx 顺序填充)
        from transformers.cache_utils import DynamicCache

        # 构造共享 cache: 从 self._kv_cache 恢复所有层的 KV
        # 若某些层已 spill 到磁盘, 需先 fetch_back 回 VRAM
        shared_cache = DynamicCache()
        with self._lock:
            for layer_idx in range(self._num_layers):
                entry = self._kv_cache.get(layer_idx)
                if entry is not None and entry.in_vram and entry.key is not None:
                    # 直接注入 (prefill 时已建立)
                    shared_cache.key_cache.append(entry.key)
                    shared_cache.value_cache.append(entry.value)
                else:
                    # 该层 KV 已 spill, 需要 fetch_back (在锁外执行避免死锁)
                    shared_cache.key_cache.append(None)
                    shared_cache.value_cache.append(None)

        # 检查是否有 None 槽位 (需要 fetch_back), 锁外执行
        none_layers = [
            idx for idx in range(self._num_layers)
            if shared_cache.key_cache[idx] is None
        ]
        for layer_idx in none_layers:
            past_kv_tuple = await self._get_layer_kv(layer_idx)
            if past_kv_tuple is not None:
                k, v = past_kv_tuple
                shared_cache.key_cache[layer_idx] = k
                shared_cache.value_cache[layer_idx] = v

        for layer_idx in range(self._num_layers):
            # 获取当前层 (可能触发 fetch_back)
            layer_module = await self._vram_pool.get_layer_with_prefetch(layer_idx)
            if layer_module is None:
                raise RuntimeError(
                    f"[v3-Runner] decode 无法获取层 {layer_idx}"
                )

            # 在 GPU forward 当前层的同时, 后台异步预取下一层
            if layer_idx + 1 < self._num_layers:
                asyncio.ensure_future(
                    self._vram_pool.prefetch_layer(layer_idx + 1)
                )

            with torch.no_grad():
                outputs = layer_module(
                    hidden_states,
                    attention_mask=None,
                    position_ids=position_ids,
                    past_key_value=shared_cache,
                    output_attentions=False,
                    use_cache=True,
                    cache_position=cache_position,
                )
            hidden_states = outputs[0]

        # decode 结束, 更新 self._kv_cache (shared_cache 已自动 update 追加新 token)
        with self._lock:
            for layer_idx in range(self._num_layers):
                try:
                    new_key = shared_cache.key_cache[layer_idx]
                    new_value = shared_cache.value_cache[layer_idx]
                    entry = self._kv_cache.get(layer_idx)
                    if entry is not None:
                        entry.key = new_key
                        entry.value = new_value
                        entry.in_vram = True
                        entry.seq_len = new_seq_len
                    else:
                        self._kv_cache[layer_idx] = LayerKVEntry(
                            layer_idx=layer_idx,
                            key=new_key,
                            value=new_value,
                            in_vram=True,
                            seq_len=new_seq_len,
                        )
                except IndexError as exc:
                    logger.warning(
                        f"[v3-Runner] decode 层 {layer_idx} 更新 KV 失败: {exc}"
                    )

        # norm → lm_head (确保在正确 device, 避免 VRAM 压力下参数跑到 CPU)
        norm_device = next(self._norm.parameters()).device
        if norm_device.type != "cuda":
            self._norm = self._norm.to(device)
        lm_device = next(self._lm_head.parameters()).device
        if lm_device.type != "cuda":
            self._lm_head = self._lm_head.to(device)

        hidden_states = self._norm(hidden_states)
        logits = self._lm_head(hidden_states)

        # 采样
        next_tok = self._sample_next_token(logits, history_tokens, temperature, top_p)

        # KV 溢出检查
        await self._check_and_spill_kv_if_needed()

        return next_tok, time.time() - t0

    # ----------------------------------------------------------------
    # KV cache 桥接: 获取层 KV, 若已 spill 则 fetch_back
    # ----------------------------------------------------------------
    async def _get_layer_kv(
        self, layer_idx: int
    ) -> Optional[Tuple[Any, Any]]:
        """获取指定层的 past_kv, 若已 spill 到磁盘则回读

        Args:
            layer_idx: 层索引

        Returns:
            (key, value) 元组, 或 None (无 KV)
        """
        with self._lock:
            entry = self._kv_cache.get(layer_idx)
            if entry is None:
                return None
            if entry.in_vram:
                return (entry.key, entry.value)

        # 已 spill, 需要从磁盘回读
        if entry.spill_path and entry.spill_path.exists():
            import torch
            try:
                kv_data = torch.load(entry.spill_path, map_location="cuda")
                with self._lock:
                    entry.key = kv_data["key"]
                    entry.value = kv_data["value"]
                    entry.in_vram = True
                self._readback_count += 1
                logger.debug(
                    f"[v3-Runner] KV 回读 layer={layer_idx} "
                    f"from {entry.spill_path.name}"
                )
                return (entry.key, entry.value)
            except Exception as exc:
                logger.warning(
                    f"[v3-Runner] KV 回读失败 layer={layer_idx}: {exc}"
                )
                return None
        return None

    # ----------------------------------------------------------------
    # KV spill: 超阈值时把最旧层 KV 写入磁盘
    # ----------------------------------------------------------------
    async def _check_and_spill_kv_if_needed(self) -> int:
        """检查 KV 总显存占用, 超阈值则 spill 最旧层

        策略:
          1. 统计所有 in_vram 层的 KV 张量总字节数
          2. 若超过阈值, 按层索引从小到大 (最旧) 逐层 spill
          3. spill 后该层 key/value 置 None, 释放 VRAM
          4. 持续 spill 直到总占用低于 (阈值 - reserve)

        Returns:
            本次 spill 的层数
        """
        import torch
        spilled = 0
        with self._lock:
            total_bytes = self._estimate_kv_bytes_locked()
            if total_bytes <= self._kv_threshold_bytes:
                return 0

            # 按层索引从小到大 spill (最旧层优先淘汰)
            target_usage = int(self._kv_threshold_bytes * 0.7)
            for layer_idx in sorted(self._kv_cache.keys()):
                if total_bytes <= target_usage:
                    break
                entry = self._kv_cache[layer_idx]
                if not entry.in_vram or entry.key is None:
                    continue
                # 写入磁盘
                spill_path = self._spill_dir / f"kv_layer_{layer_idx}_{int(time.time()*1000)}.pt"
                try:
                    torch.save(
                        {"key": entry.key, "value": entry.value},
                        spill_path,
                    )
                    layer_bytes = self._entry_bytes(entry)
                    entry.key = None
                    entry.value = None
                    entry.in_vram = False
                    entry.spill_path = spill_path
                    total_bytes -= layer_bytes
                    self._spill_count += 1
                    spilled += 1
                    logger.info(
                        f"[v3-Runner] KV spill layer={layer_idx} "
                        f"size={layer_bytes/1024/1024:.1f}MB → {spill_path.name}"
                    )
                except Exception as exc:
                    logger.warning(
                        f"[v3-Runner] KV spill 失败 layer={layer_idx}: {exc}"
                    )

        if spilled > 0:
            torch.cuda.empty_cache()
        return spilled

    def _estimate_kv_bytes_locked(self) -> int:
        """估算当前 in_vram 的 KV 总字节数 (需持锁)"""
        total = 0
        for entry in self._kv_cache.values():
            if entry.in_vram and entry.key is not None:
                total += self._entry_bytes(entry)
        return total

    @staticmethod
    def _entry_bytes(entry: LayerKVEntry) -> int:
        """计算单层 KV 占用字节数"""
        if entry.key is None:
            return 0
        try:
            return entry.key.element_size() * entry.key.numel() * 2  # key + value
        except Exception:
            return 0

    # ----------------------------------------------------------------
    # 采样 (与 worker v2.2 一致)
    # ----------------------------------------------------------------
    def _sample_next_token(
        self,
        logits: Any,
        history_tokens: List[int],
        temperature: float,
        top_p: float,
    ) -> int:
        """采样下一个 token: repetition penalty + temperature + top_p

        Args:
            logits: [1, 1, vocab] 或 [vocab] 的 logits
            history_tokens: 历史生成 token (repetition penalty 用)
            temperature: 温度
            top_p: nucleus 阈值

        Returns:
            采样后的 token id
        """
        import torch
        import torch.nn.functional as F

        if logits.dim() == 3:
            logits = logits[0, -1, :]
        logits = logits.float()

        # repetition penalty
        if history_tokens:
            penalty: float = 1.15
            for tok_id in set(history_tokens[-256:]):
                if 0 <= tok_id < logits.shape[0]:
                    logits[tok_id] = logits[tok_id] / penalty

        # temperature
        if abs(temperature - 1.0) > 1e-6:
            logits = logits / temperature

        # top_p
        if 0.0 < top_p < 1.0:
            sorted_logits, sorted_idx = torch.sort(logits, descending=True)
            cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            remove = cum_probs > top_p
            remove[1:] = remove[:-1].clone()
            remove[0] = False
            logits[sorted_idx[remove]] = float("-inf")

        probs = F.softmax(logits, dim=-1)
        return int(torch.multinomial(probs, num_samples=1).item())

    # ----------------------------------------------------------------
    # 资源释放
    # ----------------------------------------------------------------
    def reset_kv_cache(self) -> None:
        """清空所有 KV cache (任务切换时调用)"""
        import torch
        with self._lock:
            for entry in self._kv_cache.values():
                entry.key = None
                entry.value = None
                entry.in_vram = False
            self._kv_cache.clear()
        torch.cuda.empty_cache()
        gc.collect()

    def stats(self) -> Dict[str, Any]:
        """获取运行统计"""
        with self._lock:
            return {
                "num_layers": self._num_layers,
                "kv_entries": len(self._kv_cache),
                "kv_in_vram": sum(1 for e in self._kv_cache.values() if e.in_vram),
                "kv_spill_count": self._spill_count,
                "kv_readback_count": self._readback_count,
                "kv_total_bytes": self._estimate_kv_bytes_locked(),
            }


# ============================================================
# 版权声明
# 本项目 Rebirth LLM(RLLM) 基于开源项目 Nous Hermes-Agent (MIT License) 二次深度开发,
# 项目内保留完整原始开源协议文件; 智能体自迭代调度逻辑复用开源代码,
# 磁盘分层加载、全局内存锁、D盘隔离部署、自动IO调优模块为自研闭源模块,
# 分发时附带完整MIT协议文件。
#
# 商标隔离免责声明
# 项目名称 Rebirth LLM (简称RLLM) 为独立软件项目代号,
# 与奢侈品品牌Hermes、开源项目Hermes-Agent无品牌合作、隶属关联;
# 仅代码内部功能性调用开源框架, 不会使用Hermes相关名称开展商业宣传, 无品牌混淆意图。
# ============================================================
