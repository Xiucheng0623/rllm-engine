# File: D:\AI_RLLM\rllm_disk_engine\mmap_io\mmap_wrapper.py
"""mmap 磁盘映射封装（Windows适配）

目标：
  1. 优化大分片文件连续读（减少read syscall）
  2. 降低随机IO延迟（Windows页缓存利用）
  3. 自动handle mmap窗口，避免全量文件占用虚拟内存
  4. 分片文件自动关闭、引用计数管理
"""
from __future__ import annotations

import mmap
import os
import struct
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger

from rllm_agent_core import LOG_DIR

logger.add(
    LOG_DIR / "mmap_{time}.log",
    rotation="50 MB",
    retention="7 days",
    encoding="utf-8",
)


# ============================================================
# mmap窗口句柄
# ============================================================
@dataclass
class MmapFileHandle:
    """单个mmap窗口句柄（管理引用计数与自动关闭）"""
    path: Path
    offset: int
    length: int
    access_mode: int = mmap.ACCESS_READ
    ref_count: int = 0
    opened_at: float = field(default_factory=time.time)
    _mm: Optional[mmap.mmap] = None
    _f: Any = None  # 文件句柄

    def is_open(self) -> bool:
        return self._mm is not None

    def open(self) -> None:
        if self.is_open():
            return
        self._f = open(self.path, "rb")
        # Windows mmap要求offset必须是分配粒度(64KB)对齐
        align = 65536
        base_offset = (self.offset // align) * align
        delta = self.offset - base_offset
        map_len = self.length + delta
        try:
            self._mm = mmap.mmap(
                self._f.fileno(),
                length=map_len,
                access=self.access_mode,
                offset=base_offset,
            )
            # 为了保持对外接口 offset/length 不变，保存delta
            self._delta = delta  # type: ignore[attr-defined]
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"[MmapWrapper] mmap失败 {self.path.name}: {exc}, 回退标准IO")
            if self._f is not None:
                try:
                    self._f.close()
                except Exception:  # noqa: BLE001
                    pass
                self._f = None
            raise

    def close(self) -> None:
        if self._mm is not None:
            try:
                self._mm.close()
            except Exception:  # noqa: BLE001
                pass
            self._mm = None
        if self._f is not None:
            try:
                self._f.close()
            except Exception:  # noqa: BLE001
                pass
            self._f = None

    def read_bytes(self, offset: int, length: int) -> bytes:
        """读取offset..offset+length的字节"""
        if self._mm is None:
            self.open()
        assert self._mm is not None
        delta = getattr(self, "_delta", 0)
        start = offset - self.offset + delta
        end = start + length
        return bytes(self._mm[start:end])

    def read_int32(self, offset: int) -> int:
        b = self.read_bytes(offset, 4)
        return struct.unpack("<i", b)[0]

    def read_uint64(self, offset: int) -> int:
        b = self.read_bytes(offset, 8)
        return struct.unpack("<Q", b)[0]


# ============================================================
# mmap管理器（LRU窗口池）
# ============================================================
class MmapManager:
    """mmap窗口池管理器（LRU淘汰）

    约束：
      - 最大同时打开窗口数：max_open_handles (默认32)
      - 单窗口默认大小：128MB
    """

    def __init__(
        self,
        max_open_handles: int = 32,
        default_window_mb: int = 128,
    ) -> None:
        self._max_open = max_open_handles
        self._window_bytes = default_window_mb * 1024 * 1024
        self._handles: Dict[Tuple[str, int, int], MmapFileHandle] = {}
        self._lru: List[Tuple[str, int, int]] = []
        self._lock = threading.RLock()
        self._hit_count: int = 0
        self._miss_count: int = 0
        logger.info(
            f"[RLLM-MmapIO] 初始化: max_handles={max_open_handles}, "
            f"win_size={default_window_mb}MB"
        )

    # ----------------------------------------------------------------
    # 对外读接口
    # ----------------------------------------------------------------
    def read_shard(self, path: Path, offset: int, length: int) -> bytes:
        """读取分片文件 [offset, offset+length)，优先mmap，失败回退read"""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(path)

        # 若长度很大，直接普通IO更高效（不创建mmap）
        if length > self._window_bytes * 2:
            return self._fallback_read(path, offset, length)

        try:
            h = self._acquire_handle(path, offset, length)
            try:
                data = h.read_bytes(offset, length)
                self._hit_count += 1
                return data
            finally:
                self._release_handle(h)
        except Exception as exc:  # noqa: BLE001
            logger.debug(f"[RLLM-MmapIO] mmap读失败，回退: {exc}")
            self._miss_count += 1
            return self._fallback_read(path, offset, length)

    def stats(self) -> Dict[str, int]:
        with self._lock:
            return {
                "open_handles": len(self._handles),
                "hit_count": self._hit_count,
                "miss_count": self._miss_count,
            }

    def close_all(self) -> None:
        with self._lock:
            for h in self._handles.values():
                try:
                    h.close()
                except Exception:  # noqa: BLE001
                    pass
            self._handles.clear()
            self._lru.clear()

    # ----------------------------------------------------------------
    # 内部：LRU窗口池管理
    # ----------------------------------------------------------------
    def _acquire_handle(
        self, path: Path, offset: int, length: int
    ) -> MmapFileHandle:
        """根据offset/length对齐窗口，复用或创建handle"""
        win_start = (offset // self._window_bytes) * self._window_bytes
        win_end = win_start + self._window_bytes
        # 保证覆盖length
        if offset + length > win_end:
            win_end = offset + length
        win_len = win_end - win_start

        key = (str(path), win_start, win_len)
        with self._lock:
            handle = self._handles.get(key)
            if handle is not None:
                # LRU更新
                try:
                    self._lru.remove(key)
                except ValueError:
                    pass
                self._lru.append(key)
                handle.ref_count += 1
                return handle

            # 超限 -> 淘汰引用为0最旧
            while len(self._handles) >= self._max_open:
                victim = None
                for k in self._lru:
                    v = self._handles.get(k)
                    if v is not None and v.ref_count == 0:
                        victim = k
                        break
                if victim is None:
                    break
                vh = self._handles.pop(victim)
                self._lru.remove(victim)
                vh.close()

            handle = MmapFileHandle(
                path=path,
                offset=win_start,
                length=win_len,
            )
            handle.open()
            handle.ref_count = 1
            self._handles[key] = handle
            self._lru.append(key)
            return handle

    def _release_handle(self, handle: MmapFileHandle) -> None:
        with self._lock:
            handle.ref_count = max(0, handle.ref_count - 1)

    # ----------------------------------------------------------------
    @staticmethod
    def _fallback_read(path: Path, offset: int, length: int) -> bytes:
        """回退普通IO"""
        with open(path, "rb") as fp:
            fp.seek(offset, os.SEEK_SET)
            return fp.read(length)


# 单例
_mmap_singleton: Optional[RLLM-MmapIO] = None
_mmap_lock = threading.Lock()


def get_mmap_manager(**kwargs) -> MmapManager:
    global _mmap_singleton
    if _mmap_singleton is None:
        with _mmap_lock:
            if _mmap_singleton is None:
                _mmap_singleton = MmapManager(**kwargs)
    return _mmap_singleton
