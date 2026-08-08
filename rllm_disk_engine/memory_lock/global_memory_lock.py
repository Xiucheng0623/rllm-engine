# File: D:\AI_RLLM\rllm_disk_engine\memory_lock\global_memory_lock.py
"""全局CPU内存硬锁

核心：
  - 2GB CPU推理缓冲区封顶（构造函数强制校验）
  - 实时psutil监控进程RSS，采样间隔可配置
  - 超限触发：
      1. 释放三层记忆热层（下沉冷层）
      2. 释放调度器缓冲区层
      3. gc.collect()
      4. 仍超限则 raise MemoryError 阻断更多加载
"""
from __future__ import annotations

import asyncio
import gc
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional

from loguru import logger

from rllm_agent_core import LOG_DIR

try:
    import psutil
except ImportError:  # pragma: no cover
    psutil = None  # type: ignore[assignment]

logger.add(
    LOG_DIR / "memlock_{time}.log",
    rotation="50 MB",
    retention="7 days",
    encoding="utf-8",
)


@dataclass
class MemorySnapshot:
    """内存快照"""
    ts: float = field(default_factory=time.time)
    process_rss_mb: float = 0.0
    process_vms_mb: float = 0.0
    system_used_percent: float = 0.0
    managed_buffer_mb: float = 0.0
    breach_count: int = 0


class GlobalMemoryLock:
    """全局CPU内存硬锁（单进程保护）

    Args:
        limit_gb: CPU缓冲硬限(GB)，强制<=2.0
        monitor_interval_ms: 采样间隔(ms)
        force_swap: 是否超限自动触发swap动作
    """

    def __init__(
        self,
        limit_gb: float = 2.0,
        monitor_interval_ms: int = 100,
        force_swap: bool = True,
    ) -> None:
        if limit_gb > 2.0:
            raise ValueError(
                f"[RLLM-MemoryLock] CPU缓冲硬限2GB，禁止设置 {limit_gb}GB"
            )
        self._limit_bytes = int(limit_gb * 1024 ** 3)
        self._limit_gb = limit_gb
        self._monitor_interval_ms = monitor_interval_ms
        self._force_swap = force_swap
        self._breach_count: int = 0
        self._swap_callbacks: List[Callable[[int], None]] = []
        self._lock = threading.RLock()
        self._snapshots: List[MemorySnapshot] = []
        self._active_holders: int = 0  # 嵌套acquire计数
        self._stop_evt = threading.Event()

        # 获取进程对象
        self._process: Optional[object] = None
        if psutil is not None:
            try:
                self._process = psutil.Process(os.getpid())
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"[MemLock] psutil进程绑定失败: {exc}")

        # 启动后台监控线程
        self._monitor_thread: Optional[threading.Thread] = None
        self._start_monitor()

        logger.info(
            f"[RLLM-MemoryLock] 初始化完成: 硬限={limit_gb}GB "
            f"({self._limit_bytes} bytes), interval={monitor_interval_ms}ms, "
            f"force_swap={force_swap}"
        )

    # ----------------------------------------------------------------
    # 上下文管理器风格 acquire/release
    # ----------------------------------------------------------------
    async def acquire(self) -> None:
        """获取保护区（嵌套计数器），获取前后强制检查"""
        with self._lock:
            self._active_holders += 1
        await self._enforce_limit_once(caller="acquire")

    def release(self) -> None:
        with self._lock:
            self._active_holders = max(0, self._active_holders - 1)

    async def __aenter__(self) -> "GlobalMemoryLock":
        await self.acquire()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:  # noqa: D401
        self.release()

    # ----------------------------------------------------------------
    # 对外查询
    # ----------------------------------------------------------------
    @property
    def current_usage_mb(self) -> float:
        """当前进程RSS (MB)"""
        snap = self._take_snapshot()
        return snap.process_rss_mb

    @property
    def limit_mb(self) -> float:
        return self._limit_bytes / (1024 * 1024)

    @property
    def breach_count(self) -> int:
        with self._lock:
            return self._breach_count

    def add_breach_callback(self, cb: Callable[[int], None]) -> None:
        """注册超限回调"""
        self._swap_callbacks.append(cb)

    def latest_snapshot(self) -> Optional[MemorySnapshot]:
        with self._lock:
            return self._snapshots[-1] if self._snapshots else None

    # ----------------------------------------------------------------
    # 监控
    # ----------------------------------------------------------------
    def _start_monitor(self) -> None:
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            name="memlock_monitor",
            daemon=True,
        )
        self._monitor_thread.start()

    def _monitor_loop(self) -> None:
        while not self._stop_evt.is_set():
            try:
                asyncio.run(self._enforce_limit_once(caller="monitor"))
            except MemoryError:
                # 不抛给监控线程
                pass
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"[MemLock] 监控异常: {exc}")
            self._stop_evt.wait(self._monitor_interval_ms / 1000.0)

    def stop_monitor(self) -> None:
        self._stop_evt.set()

    # ----------------------------------------------------------------
    # 核心：超限执行
    # ----------------------------------------------------------------
    async def _enforce_limit_once(self, caller: str = "") -> None:
        snap = self._take_snapshot()
        with self._lock:
            self._snapshots.append(snap)
            if len(self._snapshots) > 1000:
                self._snapshots = self._snapshots[-500:]

        current_bytes = int(snap.process_rss_mb * 1024 * 1024)
        if current_bytes <= self._limit_bytes:
            return

        # 超限
        with self._lock:
            self._breach_count += 1
            bc = self._breach_count

        logger.warning(
            f"[RLLM-MemoryLock] 触发硬限 #{bc}: "
            f"RSS={snap.process_rss_mb:.1f}MB / 限制={self._limit_gb*1024:.0f}MB "
            f"(caller={caller})"
        )

        for cb in self._swap_callbacks:
            try:
                cb(bc)
            except Exception:  # noqa: BLE001
                pass

        if self._force_swap:
            await self._perform_force_swap(current_bytes)

        # 二次检查：仍超限 -> 阻断
        snap2 = self._take_snapshot()
        current_bytes2 = int(snap2.process_rss_mb * 1024 * 1024)
        if current_bytes2 > self._limit_bytes:
            msg = (
                f"[RLLM-MemoryLock] 强制swap后仍超硬限 "
                f"{snap2.process_rss_mb:.1f}MB>{self._limit_gb*1024:.0f}MB，"
                f"已累计超限{self._breach_count}次，阻断加载！"
            )
            logger.error(msg)
            raise MemoryError(msg)

    async def _perform_force_swap(self, current_bytes: int) -> None:
        """执行强制swap（释放各层内存）"""
        # 1) GC 先行
        gc.collect()
        released_mb = 0.0

        # 2) 调度器缓冲区清空
        try:
            from rllm_disk_engine.scheduler.async_page_scheduler import get_page_scheduler
            sched = get_page_scheduler()
            before = sched.current_buffer_bytes()
            # 强制卸载所有已加载层
            stats = sched.stats()
            loaded_layers = list(range(stats["loaded_layers"]))
            for idx in loaded_layers:
                await sched.unload_layer(idx)
            after = sched.current_buffer_bytes()
            released_mb += (before - after) / (1024 * 1024)
        except Exception as exc:  # noqa: BLE001
            logger.debug(f"[MemLock] 调度器释放失败: {exc}")

        # 3) 三层记忆热层下沉
        try:
            from rllm_agent_core.memory.three_layer_memory import get_memory_manager
            mm = get_memory_manager()
            # 热层驱逐下沉到暖层/冷层
            evicted = mm.hot.evict_all()
            for item in evicted:
                await mm.warm.put(item)
                if item.persistent_path is None:
                    await mm.cold.put(item)
            released_mb += sum(it.size_bytes for it in evicted) / (1024 * 1024)
        except Exception as exc:  # noqa: BLE001
            logger.debug(f"[MemLock] 记忆释放失败: {exc}")

        # 4) 再GC
        gc.collect()
        logger.info(f"[RLLM-MemoryLock] 强制swap释放约 {released_mb:.1f}MB")

    # ----------------------------------------------------------------
    def _take_snapshot(self) -> MemorySnapshot:
        snap = MemorySnapshot(managed_buffer_mb=0.0)
        if self._process is not None:
            try:
                mi = self._process.memory_info()  # type: ignore[union-attr]
                snap.process_rss_mb = mi.rss / (1024 * 1024)
                snap.process_vms_mb = mi.vms / (1024 * 1024)
                vm = psutil.virtual_memory()  # type: ignore[union-attr]
                snap.system_used_percent = float(vm.percent)
            except Exception:  # noqa: BLE001
                pass
        with self._lock:
            snap.breach_count = self._breach_count
        return snap


# 单例
_lock_singleton: Optional[RLLM-MemoryLock] = None
_lock_lock = threading.Lock()


def get_memory_lock(
    limit_gb: float = 2.0,
    **kwargs,
) -> GlobalMemoryLock:
    global _lock_singleton
    if _lock_singleton is None:
        with _lock_lock:
            if _lock_singleton is None:
                _lock_singleton = GlobalMemoryLock(limit_gb=limit_gb, **kwargs)
    return _lock_singleton
