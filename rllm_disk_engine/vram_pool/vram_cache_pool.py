# -*- coding: utf-8 -*-
# File: D:\AI_RLLM\rllm_disk_engine\vram_pool\vram_cache_pool.py
"""全局 GPU 显存持久层缓存池

核心契约:
  1. Prefill 阶段一次性装入所有 Transformer 层 → 常驻 VRAM
  2. Decode 阶段零磁盘 IO, 纯 GPU forward
  3. 显存不足时由 HotColdEvictor 自动淘汰冷层回写 D 盘
  4. 支持 4bit NF4 / 8bit / bf16 自适应量化策略

性能目标:
  - 7B 模型 4bit 量化后 ~3.5GB, RTX 5070Ti 12GB 显存绰绰有余
  - Decode 阶段 forward 瓶颈转移至 GPU 浮点算力 (≥1 tok/s)
"""
from __future__ import annotations

import asyncio
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from loguru import logger

from rllm_agent_core import LOG_DIR

logger.add(
    LOG_DIR / "vram_pool_{time}.log",
    rotation="50 MB",
    retention="7 days",
    encoding="utf-8",
)


@dataclass
class VRAMLayerEntry:
    """单层 VRAM 缓存条目

    Attributes:
        layer_idx: 层索引
        module: 已 .to("cuda") 的 DecoderLayer 模块
        size_bytes: 实际占用 VRAM 字节数
        load_ts: 装入时间戳
        access_count: 累计访问次数 (热力计数)
        last_access_ts: 最近访问时间戳
        pinned: 是否锁定 (高频层, 不可淘汰)
        quant_bits: 实际量化位宽 (4/8/16)
    """
    layer_idx: int
    module: torch.nn.Module
    size_bytes: int
    load_ts: float = field(default_factory=time.time)
    access_count: int = 0
    last_access_ts: float = field(default_factory=time.time)
    pinned: bool = False
    quant_bits: int = 16


class VRAMCachePool:
    """全局 GPU 显存缓存池 (单例)

    设计要点:
      - 单例模式确保整个进程共享一个 VRAM 池
      - OrderedDict 维护 LRU 顺序
      - 显存水位超 usable_bytes 时触发冷热淘汰
    """

    _singleton: Optional["VRAMCachePool"] = None
    _lock_singleton: threading.Lock = threading.Lock()

    def __new__(cls, *args: Any, **kwargs: Any) -> "VRAMCachePool":
        """单例构造 (接受任意参数, 仅首次创建时使用)

        Args:
            *args: 位置参数 (传递给 __init__)
            **kwargs: 关键字参数 (传递给 __init__)
        """
        if cls._singleton is None:
            with cls._lock_singleton:
                if cls._singleton is None:
                    cls._singleton = super().__new__(cls)
        return cls._singleton

    def __init__(
        self,
        reserve_gb: float = 2.0,
        evict_threshold_pct: float = 0.85,
    ) -> None:
        """初始化显存池

        Args:
            reserve_gb: 给 KV cache + 临时张量预留的显存 (GB)
            evict_threshold_pct: 触发淘汰的水位百分比 (0-1)
        """
        if getattr(self, "_initialized", False):
            return
        self._initialized: bool = True

        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA 不可用, VRAMCachePool 需要 GPU 支持. "
                "请检查 PyTorch CUDA 版本与显卡驱动."
            )

        self._device: torch.device = torch.device("cuda")
        self._capacity_bytes: int = self._probe_vram_capacity()
        self._reserve_bytes: int = int(reserve_gb * 1024**3)
        self._usable_bytes: int = self._capacity_bytes - self._reserve_bytes
        self._evict_threshold_pct: float = evict_threshold_pct

        self._layers: OrderedDict[int, VRAMLayerEntry] = OrderedDict()
        self._current_bytes: int = 0
        self._lock: threading.RLock = threading.RLock()

        # 绑定冷热置换器
        from rllm_disk_engine.vram_pool.hot_cold_evictor import HotColdEvictor
        self._evictor: Optional[HotColdEvictor] = None  # 延迟绑定

        # 显存事件计数
        self._evict_count: int = 0
        self._fetch_back_count: int = 0
        self._prefetch_hit_count: int = 0
        self._prefetch_miss_count: int = 0

        # 异步预取任务追踪: {layer_idx: asyncio.Task}
        self._prefetch_tasks: Dict[int, asyncio.Task] = {}
        self._prefetch_lock: asyncio.Lock = asyncio.Lock()

        logger.success(
            f"[RLLM-VRAMPool] 初始化: device={torch.cuda.get_device_name(0)} "
            f"capacity={self._capacity_bytes/1024**3:.1f}GB "
            f"usable={self._usable_bytes/1024**3:.1f}GB "
            f"reserve={reserve_gb:.1f}GB "
            f"evict_threshold={evict_threshold_pct:.0%}"
        )

    def _probe_vram_capacity(self) -> int:
        """探测 GPU 显存容量

        Returns:
            显存总字节数
        """
        props = torch.cuda.get_device_properties(0)
        return int(props.total_memory)

    def attach_evictor(self, evictor) -> None:
        """绑定冷热置换器 (避免循环依赖, 延迟注入)"""
        self._evictor = evictor

    # ----------------------------------------------------------------
    # 对外核心接口
    # ----------------------------------------------------------------
    async def prefill_load_all(
        self,
        layer_loader,
        layer_module_factory: Callable[[int], torch.nn.Module],
        num_layers: int,
        quant_bits: int = 4,
    ) -> Tuple[int, float]:
        """Prefill 阶段: 一次性把所有层装入 VRAM 常驻

        Args:
            layer_loader: ZeroCopyShardLoader 实例
            layer_module_factory: 创建空 DecoderLayer 的工厂函数
            num_layers: 总层数 (Mistral-7B=32)
            quant_bits: 量化位宽 (4=NF4, 8=Int8, 16=bf16)

        Returns:
            (成功装入层数, 总耗时秒)
        """
        t0 = time.time()
        loaded: int = 0

        logger.info(
            f"[RLLM-VRAMPool] 开始 Prefill 装入: "
            f"layers={num_layers} quant={quant_bits}bit"
        )

        for layer_idx in range(num_layers):
            try:
                # 1. 零拷贝加载权重 (mmap 视图)
                state_dict, load_ms = await layer_loader.load_layer_zero_copy(
                    layer_idx
                )

                # 2. 创建空模块并搬到 VRAM
                module = layer_module_factory(layer_idx)
                module = module.to(self._device)

                # 3. 直接 in-place 绑定权重 + 量化
                size_bytes = self._bind_weights_inplace(
                    module, state_dict, quant_bits
                )

                # 4. 显存水位检查, 不足则淘汰冷层
                while not self._can_fit(size_bytes):
                    if self._evictor is None:
                        logger.warning(
                            f"[RLLM-VRAMPool] 层{layer_idx}无法装入, "
                            f"显存不足且未绑定 evictor, 跳过"
                        )
                        break
                    evicted = await self._evictor.evict_coldest()
                    if evicted is None:
                        logger.warning(
                            f"[RLLM-VRAMPool] 层{layer_idx}无法装入, "
                            f"显存不足且无冷层可淘汰, 跳过"
                        )
                        break

                # 5. 入池
                with self._lock:
                    entry = VRAMLayerEntry(
                        layer_idx=layer_idx,
                        module=module,
                        size_bytes=size_bytes,
                        quant_bits=quant_bits,
                    )
                    self._layers[layer_idx] = entry
                    self._current_bytes += size_bytes
                loaded += 1

                if loaded % 8 == 0:
                    logger.info(
                        f"[RLLM-VRAMPool] Prefill 进度: "
                        f"{loaded}/{num_layers} "
                        f"VRAM={self._current_bytes/1024**3:.1f}GB "
                        f"load_ms={load_ms:.0f}"
                    )

                # 每层结束强制释放 mmap 视图 + GC
                state_dict.clear()
                import gc as _gc
                _gc.collect()

            except Exception as exc:
                logger.exception(
                    f"[RLLM-VRAMPool] 层 {layer_idx} 装入失败: {exc}"
                )

        elapsed = time.time() - t0
        logger.success(
            f"[RLLM-VRAMPool] Prefill 完成: "
            f"{loaded}/{num_layers} 层装入, "
            f"VRAM={self._current_bytes/1024**3:.1f}GB, "
            f"耗时={elapsed:.1f}s"
        )
        return loaded, elapsed

    async def load_from_quantized_model(
        self,
        decoder_layers: List[torch.nn.Module],
        quant_bits: int = 4,
    ) -> int:
        """从已量化的模型中加载 decoder layers 到 VRAM 池

        与 prefill_load_all 不同, 此方法接收已经通过 from_pretrained
        + BitsAndBytesConfig 量化的层模块 (含正确的 Linear4bit).
        跳过 _bind_weights_inplace, 直接入池.

        Args:
            decoder_layers: 已量化的 DecoderLayer 模块列表 (已在 CUDA)
            quant_bits: 量化位宽 (用于统计)

        Returns:
            成功装入的层数
        """
        loaded: int = 0

        for layer_idx, layer_module in enumerate(decoder_layers):
            # 估算层占用 VRAM (遍历所有参数, 累加 bytes)
            size_bytes: int = 0
            for param in layer_module.parameters():
                size_bytes += param.numel() * param.element_size()

            # 入池
            with self._lock:
                entry = VRAMLayerEntry(
                    layer_idx=layer_idx,
                    module=layer_module,
                    size_bytes=size_bytes,
                    quant_bits=quant_bits,
                )
                self._layers[layer_idx] = entry
                self._current_bytes += size_bytes
            loaded += 1

            if loaded % 8 == 0:
                logger.info(
                    f"[RLLM-VRAMPool] 量化层装入进度: "
                    f"{loaded}/{len(decoder_layers)} "
                    f"VRAM={self._current_bytes/1024**3:.1f}GB"
                )

        logger.success(
            f"[RLLM-VRAMPool] 量化层装入完成: "
            f"{loaded}/{len(decoder_layers)} 层, "
            f"VRAM={self._current_bytes/1024**3:.1f}GB, "
            f"quant={quant_bits}bit"
        )
        return loaded

    async def get_layer(self, layer_idx: int) -> Optional[torch.nn.Module]:
        """Decode 阶段: 直接返回 VRAM 中的层模块 (零磁盘 IO)

        若该层已被淘汰到磁盘, 由 HotColdEvictor 异步从 D 盘读回.
        若已有 prefetch 任务在跑, 等待其完成, 避免重复 fetch_back 竞态.

        Args:
            layer_idx: 层索引

        Returns:
            DecoderLayer 模块 (已在 VRAM 中) 或 None (读回失败)
        """
        with self._lock:
            entry = self._layers.get(layer_idx)
            if entry is not None:
                entry.access_count += 1
                entry.last_access_ts = time.time()
                # LRU 更新: 移到末尾
                self._layers.move_to_end(layer_idx)
                return entry.module
            # 检查是否已有 prefetch 任务在跑 (避免重复 fetch_back 竞态)
            pending_task = self._prefetch_tasks.get(layer_idx)

        # 有 prefetch 任务在跑, 等待它完成 (复用, 不重复 fetch_back)
        if pending_task is not None:
            logger.debug(
                f"[RLLM-VRAMPool] 层 {layer_idx} 等待已存在的 prefetch 任务"
            )
            module = await pending_task
            with self._lock:
                self._prefetch_tasks.pop(layer_idx, None)
            if module is not None:
                with self._lock:
                    entry = self._layers.get(layer_idx)
                    if entry is not None:
                        entry.access_count += 1
                        entry.last_access_ts = time.time()
                        self._layers.move_to_end(layer_idx)
                return module
            # prefetch 失败, 走常规 fetch_back

        # 缺失: 触发冷层读回
        logger.debug(
            f"[RLLM-VRAMPool] 层 {layer_idx} 未在 VRAM, 触发冷层读回"
        )
        if self._evictor is not None:
            with self._lock:
                self._fetch_back_count += 1
            return await self._evictor.fetch_back(layer_idx)
        return None

    async def prefetch_layer(self, layer_idx: int) -> None:
        """异步预取层到 VRAM (不阻塞调用方)

        若层已在 VRAM 中, 直接返回 (命中). 否则启动后台 fetch_back 任务.
        预取任务完成后, 层已在 VRAM, 后续 get_layer 调用零等待.

        Args:
            layer_idx: 要预取的层索引
        """
        # 快速检查: 已在 VRAM?
        with self._lock:
            entry = self._layers.get(layer_idx)
            if entry is not None:
                self._prefetch_hit_count += 1
                return
            # 已有预取任务在跑?
            if layer_idx in self._prefetch_tasks:
                return

        # 启动后台预取任务
        if self._evictor is None:
            return

        async with self._prefetch_lock:
            # 再次检查 (可能在等锁期间被其他协程预取了)
            with self._lock:
                if layer_idx in self._layers:
                    self._prefetch_hit_count += 1
                    return
                if layer_idx in self._prefetch_tasks:
                    return

            # 创建后台 fetch_back 任务
            task = asyncio.create_task(
                self._evictor.fetch_back(layer_idx),
                name=f"prefetch-L{layer_idx}",
            )
            with self._lock:
                self._prefetch_tasks[layer_idx] = task

    async def get_layer_with_prefetch(self, layer_idx: int) -> Optional[torch.nn.Module]:
        """获取层 (支持预取衔接): 若有预取任务则等待其完成

        Args:
            layer_idx: 层索引

        Returns:
            DecoderLayer 模块或 None
        """
        # 检查是否已有预取任务
        with self._lock:
            entry = self._layers.get(layer_idx)
            if entry is not None:
                entry.access_count += 1
                entry.last_access_ts = time.time()
                self._layers.move_to_end(layer_idx)
                return entry.module
            task = self._prefetch_tasks.get(layer_idx)

        # 有预取任务在跑, 等待它完成
        if task is not None:
            logger.debug(f"[RLLM-VRAMPool] 层 {layer_idx} 等待预取任务完成")
            module = await task
            # 清理预取任务
            with self._lock:
                self._prefetch_tasks.pop(layer_idx, None)
            if module is not None:
                with self._lock:
                    entry = self._layers.get(layer_idx)
                    if entry is not None:
                        entry.access_count += 1
                        entry.last_access_ts = time.time()
                        self._layers.move_to_end(layer_idx)
                return module
            # 预取失败, 走常规路径
            self._prefetch_miss_count += 1

        # 常规 get_layer 路径
        return await self.get_layer(layer_idx)

    def pin_layer(self, layer_idx: int) -> bool:
        """锁定高频层, 禁止淘汰

        Args:
            layer_idx: 层索引

        Returns:
            是否锁定成功
        """
        with self._lock:
            entry = self._layers.get(layer_idx)
            if entry is None:
                return False
            entry.pinned = True
            logger.info(f"[RLLM-VRAMPool] 层 {layer_idx} 已锁定 (pinned)")
            return True

    # ----------------------------------------------------------------
    # 内部: 权重绑定 + 量化
    # ----------------------------------------------------------------
    def _bind_weights_inplace(
        self,
        module: torch.nn.Module,
        state_dict: Dict[str, Any],
        quant_bits: int,
    ) -> int:
        """直接 in-place 绑定权重, 跳过 load_state_dict 全量复制

        Args:
            module: 目标模块 (已在 VRAM)
            state_dict: 权重字典 (mmap 视图)
            quant_bits: 量化位宽

        Returns:
            实际占用 VRAM 字节数
        """
        total_bytes: int = 0

        for name, param in module.named_parameters():
            # 名称映射: 模块参数名 -> state_dict key
            # MistralDecoderLayer 的参数名形如 "self_attn.q_proj.weight"
            # state_dict key 形如 "model.layers.N.self_attn.q_proj.weight"
            matched_key: Optional[str] = None
            for k in state_dict.keys():
                if k.endswith(f".{name}") or k == name:
                    matched_key = k
                    break
            if matched_key is None:
                continue

            src = state_dict[matched_key]
            # DMA 直传: mmap → pinned → cuda
            if quant_bits == 4 and src.dim() == 2 and src.shape[0] >= 256:
                # 大 Linear 走 4bit NF4 量化
                try:
                    import bitsandbytes as bnb
                    # 先把 src 搬到 cuda
                    src_cuda = src.to(self._device, non_blocking=True).to(
                        torch.bfloat16
                    )
                    new_param = bnb.nn.Params4bit(
                        src_cuda.data,
                        requires_grad=False,
                        quant_type="nf4",
                        blocksize=64,
                    )
                    param.data = new_param
                    total_bytes += param.numel() * param.element_size()
                    # 立即释放 mmap 视图 (Windows 上算 RSS)
                    if matched_key in state_dict:
                        del state_dict[matched_key]
                    del src, src_cuda
                except Exception as exc:
                    logger.warning(
                        f"[RLLM-VRAMPool] 4bit 量化失败 {name}: {exc}, "
                        f"回退 bf16"
                    )
                    param.data = src.to(
                        self._device, non_blocking=True
                    ).to(torch.bfloat16)
                    total_bytes += param.numel() * 2
                    if matched_key in state_dict:
                        del state_dict[matched_key]
                    del src
            elif quant_bits == 8 and src.dim() == 2 and src.shape[0] >= 256:
                try:
                    import bitsandbytes as bnb
                    src_cuda = src.to(self._device, non_blocking=True).to(
                        torch.bfloat16
                    )
                    new_param = bnb.nn.Int8Params(
                        src_cuda.data,
                        requires_grad=False,
                    )
                    param.data = new_param
                    total_bytes += param.numel() * param.element_size()
                    if matched_key in state_dict:
                        del state_dict[matched_key]
                    del src, src_cuda
                except Exception as exc:
                    logger.warning(
                        f"[RLLM-VRAMPool] 8bit 量化失败 {name}: {exc}, "
                        f"回退 bf16"
                    )
                    param.data = src.to(
                        self._device, non_blocking=True
                    ).to(torch.bfloat16)
                    total_bytes += param.numel() * 2
                    if matched_key in state_dict:
                        del state_dict[matched_key]
                    del src
            else:
                # 嵌入层 / norm / 小张量: FP16 用 float16, 量化用 bfloat16
                target_dtype = torch.float16 if quant_bits == 16 else torch.bfloat16
                param.data = src.to(
                    self._device, non_blocking=False
                ).to(target_dtype)
                total_bytes += param.numel() * 2
                if matched_key in state_dict:
                    del state_dict[matched_key]
                del src

        # 强制 GC 释放本层所有 mmap 视图
        import gc
        gc.collect()
        return total_bytes

    def _can_fit(self, size_bytes: int) -> bool:
        """检查显存是否还能容纳 size_bytes"""
        with self._lock:
            threshold = int(self._usable_bytes * self._evict_threshold_pct)
            return self._current_bytes + size_bytes <= threshold

    # ----------------------------------------------------------------
    # 淘汰/读回接口 (供 HotColdEvictor 调用)
    # ----------------------------------------------------------------
    def remove_layer(self, layer_idx: int) -> Optional[VRAMLayerEntry]:
        """从池中移除层 (供 evictor 调用)"""
        with self._lock:
            entry = self._layers.pop(layer_idx, None)
            if entry is not None:
                self._current_bytes = max(
                    0, self._current_bytes - entry.size_bytes
                )
            return entry

    def add_layer(self, entry: VRAMLayerEntry) -> None:
        """添加层到池中 (供 evictor fetch_back 后回填)"""
        with self._lock:
            self._layers[entry.layer_idx] = entry
            self._current_bytes += entry.size_bytes

    def release_layer_vram(self, module: torch.nn.Module) -> None:
        """释放模块占用的 VRAM"""
        try:
            for p in module.parameters():
                p.data = torch.empty(0, device=self._device)
            torch.cuda.empty_cache()
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"[RLLM-VRAMPool] VRAM 释放异常: {exc}")

    # ----------------------------------------------------------------
    # 统计
    # ----------------------------------------------------------------
    def stats(self) -> Dict[str, Any]:
        """获取显存池统计"""
        with self._lock:
            return {
                "device": torch.cuda.get_device_name(0),
                "vram_capacity_gb": round(
                    self._capacity_bytes / 1024**3, 2
                ),
                "vram_used_gb": round(self._current_bytes / 1024**3, 2),
                "vram_usable_gb": round(
                    self._usable_bytes / 1024**3, 2
                ),
                "vram_layers_resident": len(self._layers),
                "evict_count": self._evict_count,
                "fetch_back_count": self._fetch_back_count,
                "prefetch_hit": self._prefetch_hit_count,
                "prefetch_miss": self._prefetch_miss_count,
                "prefetch_pending": len(self._prefetch_tasks),
            }

    def increment_evict_count(self) -> None:
        with self._lock:
            self._evict_count += 1

    async def force_evict_to_limit(self) -> int:
        """强制淘汰直到 VRAM 使用量降到阈值以下

        在 prefill 完成后调用, 确保不在 VRAM 中保留过多层
        (避免 CUDA 统一内存隐式分页导致的性能下降).

        Returns:
            淘汰的层数
        """
        evicted = 0
        while True:
            with self._lock:
                if self._current_bytes <= self._evict_threshold:
                    break
            if self._evictor is None:
                break
            result = await self._evictor.evict_coldest()
            if result is None:
                break
            evicted += 1
        if evicted > 0:
            logger.info(
                f"[RLLM-VRAMPool] 强制淘汰完成: {evicted} 层, "
                f"当前 VRAM={self._current_bytes/1024**3:.1f}GB"
            )
        return evicted

    def list_resident_layers(self) -> List[int]:
        """获取当前在 VRAM 的层索引列表"""
        with self._lock:
            return list(self._layers.keys())

    def get_layer_entry(self, layer_idx: int) -> Optional[VRAMLayerEntry]:
        """获取层条目 (供 evictor 评分)"""
        with self._lock:
            return self._layers.get(layer_idx)


# ============================================================
# 单例获取函数
# ============================================================
def get_vram_cache_pool(
    reserve_gb: float = 2.0,
    evict_threshold_pct: float = 0.85,
) -> VRAMCachePool:
    """获取全局 VRAMCachePool 单例

    Args:
        reserve_gb: 预留显存 (GB)
        evict_threshold_pct: 淘汰水位百分比

    Returns:
        VRAMCachePool 实例
    """
    pool = VRAMCachePool(
        reserve_gb=reserve_gb,
        evict_threshold_pct=evict_threshold_pct,
    )
    return pool
