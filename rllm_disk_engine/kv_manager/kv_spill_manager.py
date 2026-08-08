# File: D:\AI_RLLM\rllm_disk_engine\kv_manager\kv_spill_manager.py
"""KV缓存磁盘溢出管理器

职责：
  1. 跟踪当前KV缓存/图像特征张量的内存大小
  2. 超过阈值（默认512MB）时，自动将最旧KV块写入D盘offload_temp/kv_cache
  3. 读取时按需异步读回，无需常驻内存
  4. 配合三层记忆管理器冷层，实现大上下文不OOM
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import pickle
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple

from loguru import logger

from rllm_agent_core import LOG_DIR

logger.add(
    LOG_DIR / "kv_{time}.log",
    rotation="50 MB",
    retention="7 days",
    encoding="utf-8",
)


# ============================================================
# KV条目
# ============================================================
@dataclass
class KVCacheEntry:
    """单条KV缓存条目"""
    cache_id: str              # 唯一ID
    task_id: str
    layer_idx: int
    size_bytes: int
    created_ts: float = field(default_factory=time.time)
    last_access_ts: float = field(default_factory=time.time)
    in_memory: bool = True
    spill_file: Optional[Path] = None
    data: Any = None  # 内存驻留时的数据（不写盘时保留）

    def estimated_bytes(self) -> int:
        if self.size_bytes > 0:
            return self.size_bytes
        try:
            return len(pickle.dumps(self.data))
        except Exception:  # noqa: BLE001
            return 1024


# ============================================================
# 溢出管理器
# ============================================================
class KVSpillManager:
    """KV缓存磁盘溢出管理器"""

    def __init__(
        self,
        temp_dir: Path,
        spill_threshold_mb: int = 512,
        reserve_ratio: float = 0.2,
    ) -> None:
        self._temp_dir = Path(temp_dir)
        self._temp_dir.mkdir(parents=True, exist_ok=True)
        self._threshold_bytes = spill_threshold_mb * 1024 * 1024
        self._reserve_bytes = int(self._threshold_bytes * reserve_ratio)

        # 内存中KV条目
        self._in_mem: Dict[str, KVCacheEntry] = {}
        # LRU 双端队列（cache_id）
        self._lru: Deque[str] = Deque()
        # 当前内存占用估算
        self._current_bytes: int = 0
        # 溢出计数
        self._spill_count: int = 0
        self._read_back_count: int = 0

        # 索引
        self._index_path = self._temp_dir / "_kv_index.json"
        self._lock = threading.RLock()
        self._load_index()

        logger.info(
            f"[RLLM-KVSpill] 初始化完成: threshold={spill_threshold_mb}MB "
            f"dir={self._temp_dir}"
        )

    # ----------------------------------------------------------------
    # 对外接口
    # ----------------------------------------------------------------
    async def put(
        self,
        cache_id: str,
        data: Any,
        task_id: str = "",
        layer_idx: int = -1,
        size_hint_bytes: int = 0,
    ) -> bool:
        """写入KV缓存（内存优先，超限自动spill）"""
        entry = KVCacheEntry(
            cache_id=cache_id,
            task_id=task_id,
            layer_idx=layer_idx,
            size_bytes=size_hint_bytes,
            data=data,
        )
        if size_hint_bytes == 0:
            entry.size_bytes = entry.estimated_bytes()

        with self._lock:
            self._in_mem[cache_id] = entry
            self._lru.append(cache_id)
            self._current_bytes += entry.size_bytes
            await self._spill_if_needed_locked()
        return True

    async def get(self, cache_id: str) -> Optional[Any]:
        """读取KV缓存（按需回读）"""
        with self._lock:
            entry = self._in_mem.get(cache_id)
            if entry is not None and entry.in_memory:
                self._touch_lru(cache_id)
                return entry.data
            # 尝试从磁盘回读
            if entry is None:
                # 查索引
                meta = self._index.get(cache_id)
                if meta is None:
                    return None
                entry = KVCacheEntry(
                    cache_id=cache_id,
                    task_id=meta.get("task_id", ""),
                    layer_idx=meta.get("layer_idx", -1),
                    size_bytes=meta.get("size_bytes", 0),
                    in_memory=False,
                    spill_file=Path(meta["spill_file"]),
                )
            if entry.spill_file and entry.spill_file.exists():
                try:
                    with open(entry.spill_file, "rb") as fp:
                        data = pickle.load(fp)
                    self._read_back_count += 1
                    entry.data = data
                    entry.in_memory = True
                    self._in_mem[cache_id] = entry
                    self._lru.append(cache_id)
                    self._current_bytes += entry.size_bytes
                    self._touch_lru(cache_id)
                    await self._spill_if_needed_locked()
                    return data
                except Exception as exc:  # noqa: BLE001
                    logger.warning(f"[RLLM-KVSpill] 读回失败 {cache_id}: {exc}")
                    return None
            return None

    async def check_and_spill_if_needed(self) -> int:
        """Worker每步检查用：超阈值主动spill，返回本次spill条目数"""
        with self._lock:
            return await self._spill_if_needed_locked()

    async def get_spill_count(self) -> int:
        """返回累计spill次数"""
        with self._lock:
            return self._spill_count

    def stats(self) -> Dict[str, int]:
        with self._lock:
            return {
                "in_mem_count": len(self._in_mem),
                "in_mem_mb": self._current_bytes // (1024 * 1024),
                "spill_count": self._spill_count,
                "read_back_count": self._read_back_count,
                "spill_files": sum(1 for p in self._temp_dir.glob("*.kv.bin") if p.is_file()),
            }

    # ----------------------------------------------------------------
    # 内部
    # ----------------------------------------------------------------
    def _touch_lru(self, cache_id: str) -> None:
        """最近访问移到末尾"""
        try:
            self._lru.remove(cache_id)
        except ValueError:
            pass
        self._lru.append(cache_id)
        entry = self._in_mem.get(cache_id)
        if entry is not None:
            entry.last_access_ts = time.time()

    async def _spill_if_needed_locked(self) -> int:
        """超限则按LRU逐个spill到磁盘，返回spill数量"""
        spilled = 0
        target_usage = self._threshold_bytes - self._reserve_bytes
        while self._current_bytes > target_usage and self._lru:
            victim_id = self._lru.popleft()
            entry = self._in_mem.get(victim_id)
            if entry is None or not entry.in_memory or entry.data is None:
                continue
            if await self._spill_entry(entry):
                spilled += 1
                entry.in_memory = False
                entry.data = None
                self._current_bytes = max(0, self._current_bytes - entry.size_bytes)
                self._spill_count += 1
        if spilled > 0:
            self._save_index()
            logger.debug(
                f"[RLLM-KVSpill] 触发spill: {spilled}条, "
                f"当前内存={self._current_bytes/1024/1024:.1f}MB, 阈值={self._threshold_bytes/1024/1024:.0f}MB"
            )
        return spilled

    async def _spill_entry(self, entry: KVCacheEntry) -> bool:
        """单条spill落D盘"""
        try:
            safe = hashlib.md5(entry.cache_id.encode("utf-8")).hexdigest()
            spill_file = self._temp_dir / f"{safe}_{int(time.time())}.kv.bin"
            with open(spill_file, "wb") as fp:
                pickle.dump(entry.data, fp, protocol=pickle.HIGHEST_PROTOCOL)
            entry.spill_file = spill_file
            # 更新索引
            self._index[entry.cache_id] = {
                "task_id": entry.task_id,
                "layer_idx": entry.layer_idx,
                "size_bytes": entry.size_bytes,
                "spill_file": str(spill_file),
                "spill_ts": time.time(),
            }
            return True
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"[RLLM-KVSpill] spill失败 {entry.cache_id}: {exc}")
            return False

    # ----------------------------------------------------------------
    # 索引持久化
    # ----------------------------------------------------------------
    def _load_index(self) -> None:
        self._index: Dict[str, Dict[str, Any]] = {}
        if self._index_path.exists():
            try:
                with open(self._index_path, "r", encoding="utf-8") as fp:
                    self._index = json.load(fp)
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"[RLLM-KVSpill] 索引加载失败: {exc}")

    def _save_index(self) -> None:
        try:
            with open(self._index_path, "w", encoding="utf-8") as fp:
                json.dump(self._index, fp, ensure_ascii=False, indent=2)
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"[RLLM-KVSpill] 索引保存失败: {exc}")


# 单例
_kv_singleton: Optional[KVSpillManager] = None
_kv_lock = threading.Lock()


def get_kv_manager(
    temp_dir: Optional[Path] = None,
    spill_threshold_mb: int = 512,
) -> KVSpillManager:
    global _kv_singleton
    if _kv_singleton is None:
        with _kv_lock:
            if _kv_singleton is None:
                d = temp_dir or Path(r"D:\AI_RLLM\rllm_offload_temp\kv_cache")
                _kv_singleton = KVSpillManager(d, spill_threshold_mb)
    return _kv_singleton
