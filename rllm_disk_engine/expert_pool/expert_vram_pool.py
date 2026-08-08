# -*- coding: utf-8 -*-
# File: D:\AI_RLLM\rllm_disk_engine\expert_pool\expert_vram_pool.py
"""专家级 VRAM 缓存池

与 v3 的 VRAMCachePool 区别:
  - 缓存粒度: 从"层" (layer_idx) → "专家" (layer_idx, expert_idx)
  - 缓存对象: 从 DecoderLayer → MixtralBLockSparseTop2MLP (单个专家)
  - 容量更大: 4bit 量化后每专家仅 ~130MB, RTX5070Ti 8GB 可放 ~60 个专家
  - 支持热专家常驻: Top-N 高频专家 pin 在 VRAM, 零 I/O

设计:
  - OrderedDict 维护 LRU 顺序
  - 显存水位超 usable_bytes 时由 ExpertEvictor 淘汰冷专家
  - 异步预取: 配合 RouterPrefetcher 提前装载候选专家
"""
from __future__ import annotations

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
    LOG_DIR / "expert_vram_pool_{time}.log",
    rotation="50 MB",
    retention="7 days",
    encoding="utf-8",
)


# 专家全局 ID: (layer_idx, expert_idx)
ExpertKey = Tuple[int, int]


@dataclass
class ExpertEntry:
    """单个专家 VRAM 缓存条目

    Attributes:
        key: (layer_idx, expert_idx)
        module: 已 .to("cuda") 的 MixtralBLockSparseTop2MLP 模块
        size_bytes: 实际占用 VRAM 字节数
        load_ts: 装入时间戳
        access_count: 累计访问次数
        last_access_ts: 最近访问时间戳
        pinned: 是否锁定 (高频热专家, 不可淘汰)
        quant_bits: 量化位宽 (4/8/16)
    """
    key: ExpertKey
    module: torch.nn.Module
    size_bytes: int
    load_ts: float = field(default_factory=time.time)
    access_count: int = 0
    last_access_ts: float = field(default_factory=time.time)
    pinned: bool = False
    quant_bits: int = 4


class ExpertVRAMPool:
    """专家级 VRAM 缓存池 (单例)

    与 v3 VRAMCachePool 并行存在, 不互相干扰.

    Args:
        reserve_gb: 给 KV cache + 共享层 + 临时张量预留的显存 (GB)
        evict_threshold_pct: 触发淘汰的水位百分比 (0-1)
    """

    _singleton: Optional["ExpertVRAMPool"] = None
    _lock_singleton: threading.Lock = threading.Lock()

    def __new__(cls, *args: Any, **kwargs: Any) -> "ExpertVRAMPool":
        if cls._singleton is None:
            with cls._lock_singleton:
                if cls._singleton is None:
                    cls._singleton = super().__new__(cls)
        return cls._singleton

    def __init__(
        self,
        reserve_gb: float = 3.0,
        evict_threshold_pct: float = 0.85,
    ) -> None:
        """初始化专家 VRAM 池

        Args:
            reserve_gb: 给 KV cache + 共享层预留的显存 (GB)
            evict_threshold_pct: 触发淘汰的水位百分比
        """
        if getattr(self, "_initialized", False):
            return
        self._initialized: bool = True

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA 不可用, ExpertVRAMPool 需要 GPU 支持")

        self._device: torch.device = torch.device("cuda")
        self._capacity_bytes: int = self._probe_vram_capacity()
        self._reserve_bytes: int = int(reserve_gb * 1024**3)
        self._usable_bytes: int = self._capacity_bytes - self._reserve_bytes
        self._evict_threshold_pct: float = evict_threshold_pct

        # 专家缓存: {(layer, expert): ExpertEntry}
        self._experts: OrderedDict[ExpertKey, ExpertEntry] = OrderedDict()
        self._current_bytes: int = 0
        self._lock: threading.RLock = threading.RLock()

        # 绑定冷热置换器 (延迟注入)
        self._evictor: Optional[Any] = None

        # 统计计数
        self._evict_count: int = 0
        self._fetch_back_count: int = 0
        self._prefetch_hit_count: int = 0
        self._prefetch_miss_count: int = 0

        # 异步预取任务追踪
        self._prefetch_tasks: Dict[ExpertKey, Any] = {}
        self._prefetch_lock: Any = None  # asyncio.Lock, 延迟创建

        logger.success(
            f"[ExpertVRAMPool] 初始化: device={torch.cuda.get_device_name(0)} "
            f"capacity={self._capacity_bytes/1024**3:.1f}GB "
            f"usable={self._usable_bytes/1024**3:.1f}GB "
            f"reserve={reserve_gb:.1f}GB"
        )

    def _probe_vram_capacity(self) -> int:
        """探测 GPU 显存容量"""
        props = torch.cuda.get_device_properties(0)
        return int(props.total_memory)

    def attach_evictor(self, evictor: Any) -> None:
        """绑定专家级冷热置换器"""
        self._evictor = evictor

    # ----------------------------------------------------------------
    # 对外核心接口
    # ----------------------------------------------------------------
    def add_expert(self, entry: ExpertEntry) -> None:
        """添加专家到 VRAM 池

        Args:
            entry: 专家缓存条目
        """
        with self._lock:
            self._experts[entry.key] = entry
            self._current_bytes += entry.size_bytes

    def remove_expert(self, key: ExpertKey) -> Optional[ExpertEntry]:
        """从池中移除专家 (不释放 VRAM, 由调用方处理)

        Args:
            key: (layer_idx, expert_idx)

        Returns:
            被移除的 ExpertEntry 或 None
        """
        with self._lock:
            entry = self._experts.pop(key, None)
            if entry is not None:
                self._current_bytes -= entry.size_bytes
            return entry

    def get_expert_entry(self, key: ExpertKey) -> Optional[ExpertEntry]:
        """获取专家条目 (不增加访问计数, 供 evictor 用)"""
        with self._lock:
            return self._experts.get(key)

    async def get_expert(self, key: ExpertKey) -> Optional[torch.nn.Module]:
        """获取专家模块 (Decode 阶段主路径)

        若专家在 VRAM → 直接返回 (零 I/O)
        若不在 → 触发 ExpertEvictor.fetch_back

        Args:
            key: (layer_idx, expert_idx)

        Returns:
            MixtralBLockSparseTop2MLP 模块或 None
        """
        with self._lock:
            entry = self._experts.get(key)
            if entry is not None:
                entry.access_count += 1
                entry.last_access_ts = time.time()
                self._experts.move_to_end(key)
                return entry.module

        # 缺失: 触发冷专家读回
        if self._evictor is not None:
            with self._lock:
                self._fetch_back_count += 1
            return await self._evictor.fetch_back(key)
        return None

    async def get_expert_with_prefetch(
        self, key: ExpertKey
    ) -> Optional[torch.nn.Module]:
        """获取专家 (支持预取衔接): 若有预取任务则等待完成"""
        with self._lock:
            entry = self._experts.get(key)
            if entry is not None:
                entry.access_count += 1
                entry.last_access_ts = time.time()
                self._experts.move_to_end(key)
                return entry.module
            task = self._prefetch_tasks.get(key)

        if task is not None:
            logger.debug(
                f"[ExpertVRAMPool] 专家 {key} 等待预取任务完成"
            )
            module = await task
            with self._lock:
                self._prefetch_tasks.pop(key, None)
            if module is not None:
                with self._lock:
                    entry = self._experts.get(key)
                    if entry is not None:
                        entry.access_count += 1
                        entry.last_access_ts = time.time()
                return module
            self._prefetch_miss_count += 1

        return await self.get_expert(key)

    def pin_expert(self, key: ExpertKey) -> bool:
        """锁定热专家, 禁止淘汰

        Args:
            key: (layer_idx, expert_idx)

        Returns:
            是否锁定成功
        """
        with self._lock:
            entry = self._experts.get(key)
            if entry is None:
                return False
            entry.pinned = True
            logger.info(f"[ExpertVRAMPool] 锁定热专家 {key}")
            return True

    def unpin_expert(self, key: ExpertKey) -> bool:
        """解锁专家"""
        with self._lock:
            entry = self._experts.get(key)
            if entry is None:
                return False
            entry.pinned = False
            return True

    def list_resident_experts(self) -> List[ExpertKey]:
        """列出所有常驻 VRAM 的专家 key"""
        with self._lock:
            return list(self._experts.keys())

    def _can_fit(self, size_bytes: int) -> bool:
        """检查 VRAM 是否还能容纳指定大小的专家"""
        with self._lock:
            threshold = int(
                self._usable_bytes * self._evict_threshold_pct
            )
            return self._current_bytes + size_bytes <= threshold

    def increment_evict_count(self) -> None:
        """递增淘汰计数"""
        with self._lock:
            self._evict_count += 1

    # ----------------------------------------------------------------
    # 预取接口 (供 RouterPrefetcher 调用)
    # ----------------------------------------------------------------
    async def prefetch_expert(self, key: ExpertKey) -> None:
        """异步预取专家到 VRAM (不阻塞调用方)

        Args:
            key: (layer_idx, expert_idx)
        """
        import asyncio

        if self._prefetch_lock is None:
            self._prefetch_lock = asyncio.Lock()

        # 快速检查: 已在 VRAM?
        with self._lock:
            if key in self._experts:
                self._prefetch_hit_count += 1
                return
            if key in self._prefetch_tasks:
                return

        if self._evictor is None:
            return

        async with self._prefetch_lock:
            with self._lock:
                if key in self._experts:
                    self._prefetch_hit_count += 1
                    return
                if key in self._prefetch_tasks:
                    return

            task = asyncio.create_task(
                self._evictor.fetch_back(key),
                name=f"prefetch-expert-L{key[0]}E{key[1]}",
            )
            with self._lock:
                self._prefetch_tasks[key] = task

    # ----------------------------------------------------------------
    # 统计与诊断
    # ----------------------------------------------------------------
    def stats(self) -> Dict[str, Any]:
        """获取缓存池统计"""
        with self._lock:
            pinned_count = sum(
                1 for e in self._experts.values() if e.pinned
            )
            return {
                "resident_experts": len(self._experts),
                "pinned_experts": pinned_count,
                "current_vram_bytes": self._current_bytes,
                "current_vram_gb": self._current_bytes / 1024**3,
                "usable_vram_gb": self._usable_bytes / 1024**3,
                "evict_count": self._evict_count,
                "fetch_back_count": self._fetch_back_count,
                "prefetch_hit": self._prefetch_hit_count,
                "prefetch_miss": self._prefetch_miss_count,
            }

    def get_vram_usage_bytes(self) -> int:
        """获取当前 VRAM 占用字节数"""
        with self._lock:
            return self._current_bytes
