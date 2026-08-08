# File: D:\AI_RLLM\rllm_agent_core\memory\three_layer_memory.py
"""RLLM 三层记忆架构(底层复用Hermes)（重构版）

改造原生Hermes记忆，新增冷层磁盘持久化：
  L1 热层 (Hot):   内存中，仅最近N轮对话，小容量（<=128MB）
  L2 暖层 (Warm):  磁盘cache，中容量，diskcache管理
  L3 冷层 (Cold):  D盘持久化文件，大容量，所有权重/KV溢出均落此层
内存溢出时强制张量swap冷层磁盘，不截断推理任务。
"""
from __future__ import annotations

import abc
import asyncio
import json
import pickle
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from diskcache import Cache
from loguru import logger

from rllm_agent_core import HERMES_ROOT, LOG_DIR

logger.add(
    LOG_DIR / "memory_{time}.log",
    rotation="50 MB",
    retention="7 days",
    encoding="utf-8",
)

# 冷层磁盘路径（张量swap/KV溢出）
COLD_DISK_DIR: Path = HERMES_ROOT / "offload_temp" / "tensor_swap"
COLD_DISK_DIR.mkdir(parents=True, exist_ok=True)
WARM_CACHE_DIR: Path = HERMES_ROOT / "offload_temp" / "warm_cache"
WARM_CACHE_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 数据结构
# ============================================================
@dataclass
class MemoryItem:
    """统一记忆条目（强类型）"""
    key: str
    value: Any
    size_bytes: int = 0
    created_ts: float = field(default_factory=time.time)
    last_access_ts: float = field(default_factory=time.time)
    layer: str = "unknown"  # hot / warm / cold
    persistent_path: Optional[Path] = None  # 冷层落地路径

    def touch(self) -> None:
        """更新访问时间"""
        self.last_access_ts = time.time()


# ============================================================
# 记忆层抽象
# ============================================================
class MemoryLayer(abc.ABC):
    """记忆层抽象基类"""
    name: str = "base"
    capacity_bytes: int = 0

    @abc.abstractmethod
    async def get(self, key: str) -> Optional[MemoryItem]:
        raise NotImplementedError

    @abc.abstractmethod
    async def put(self, item: MemoryItem) -> bool:
        raise NotImplementedError

    @abc.abstractmethod
    async def delete(self, key: str) -> bool:
        raise NotImplementedError

    @abc.abstractmethod
    def current_usage_bytes(self) -> int:
        raise NotImplementedError


# ============================================================
# L1 热层：内存LRU
# ============================================================
class HotMemoryLayer(MemoryLayer):
    """L1 热层：内存LRU，容量限制，超限自动下移暖层

    Attributes:
        capacity_bytes: 热层容量上限（默认64MB）
    """
    name = "hot"

    def __init__(self, capacity_bytes: int = 64 * 1024 * 1024) -> None:
        self.capacity_bytes = capacity_bytes
        self._store: Dict[str, MemoryItem] = {}
        self._lock = threading.RLock()

    async def get(self, key: str) -> Optional[MemoryItem]:
        with self._lock:
            item = self._store.get(key)
            if item is not None:
                item.touch()
                return item
        return None

    async def put(self, item: MemoryItem) -> bool:
        """写入热层，超限返回False表示需要下沉暖层"""
        with self._lock:
            if item.size_bytes > self.capacity_bytes:
                return False
            while self.current_usage_bytes() + item.size_bytes > self.capacity_bytes:
                # LRU驱逐最早访问的
                if not self._store:
                    return False
                victim = min(self._store.values(), key=lambda x: x.last_access_ts)
                del self._store[victim.key]
            item.layer = "hot"
            self._store[item.key] = item
            return True

    async def delete(self, key: str) -> bool:
        with self._lock:
            return self._store.pop(key, None) is not None

    def current_usage_bytes(self) -> int:
        with self._lock:
            return sum(it.size_bytes for it in self._store.values())

    def evict_all(self) -> List[MemoryItem]:
        """清空热层，返回所有条目供下沉"""
        with self._lock:
            items = list(self._store.values())
            self._store.clear()
            return items


# ============================================================
# L2 暖层：diskcache
# ============================================================
class WarmMemoryLayer(MemoryLayer):
    """L2 暖层：diskcache 磁盘KV存储"""
    name = "warm"

    def __init__(self, cache_dir: Path = WARM_CACHE_DIR, size_limit_bytes: int = 2 * 1024 * 1024 * 1024) -> None:
        self._cache_dir = cache_dir
        self.capacity_bytes = size_limit_bytes
        self._cache = Cache(str(cache_dir), size_limit=size_limit_bytes)

    async def get(self, key: str) -> Optional[MemoryItem]:
        try:
            raw: Optional[bytes] = self._cache.get(key, default=None)
            if raw is None:
                return None
            item: MemoryItem = pickle.loads(raw)
            item.layer = "warm"
            item.touch()
            return item
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"[WarmLayer] 读取失败: {exc}")
            return None

    async def put(self, item: MemoryItem) -> bool:
        try:
            item.layer = "warm"
            self._cache.set(item.key, pickle.dumps(item))
            return True
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"[WarmLayer] 写入失败: {exc}")
            return False

    async def delete(self, key: str) -> bool:
        self._cache.delete(key)
        return True

    def current_usage_bytes(self) -> int:
        try:
            return int(self._cache.volume())
        except Exception:  # noqa: BLE001
            return 0

    def close(self) -> None:
        self._cache.close()


# ============================================================
# L3 冷层：D盘文件持久化
# ============================================================
class ColdDiskMemoryLayer(MemoryLayer):
    """L3 冷层：D盘文件系统持久化

    用途：
      - 溢出张量swap（推理中间状态）
      - KV缓存超量spill
      - 长期技能与策略存档
    """
    name = "cold"

    def __init__(self, root_dir: Path = COLD_DISK_DIR) -> None:
        self._root = root_dir
        self._root.mkdir(parents=True, exist_ok=True)
        self._index_path = root_dir / "_cold_index.json"
        self._index: Dict[str, Dict[str, Any]] = self._load_index()
        self.capacity_bytes = 1024 ** 4  # 1TB 逻辑上限
        self._lock = threading.RLock()

    def _load_index(self) -> Dict[str, Dict[str, Any]]:
        if self._index_path.exists():
            try:
                with open(self._index_path, "r", encoding="utf-8") as fp:
                    return json.load(fp)
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"[ColdLayer] 索引加载失败，重建: {exc}")
        return {}

    def _save_index(self) -> None:
        with open(self._index_path, "w", encoding="utf-8") as fp:
            json.dump(self._index, fp, ensure_ascii=False, indent=2)

    def _path_for(self, key: str) -> Path:
        safe_key = key.replace("/", "_").replace("\\", "_").replace(":", "_")
        return self._root / f"{safe_key}.cold.bin"

    async def get(self, key: str) -> Optional[MemoryItem]:
        with self._lock:
            meta = self._index.get(key)
            if meta is None:
                return None
            path = Path(meta["path"])
            if not path.exists():
                # 索引残留
                del self._index[key]
                self._save_index()
                return None
            try:
                with open(path, "rb") as fp:
                    raw = fp.read()
                value = pickle.loads(raw)
                item = MemoryItem(
                    key=key,
                    value=value,
                    size_bytes=meta["size_bytes"],
                    created_ts=meta["created_ts"],
                    last_access_ts=time.time(),
                    layer="cold",
                    persistent_path=path,
                )
                return item
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"[ColdLayer] 读取冷层失败 {key}: {exc}")
                return None

    async def put(self, item: MemoryItem) -> bool:
        with self._lock:
            path = self._path_for(item.key)
            try:
                with open(path, "wb") as fp:
                    fp.write(pickle.dumps(item.value))
                item.layer = "cold"
                item.persistent_path = path
                size_bytes = path.stat().st_size
                item.size_bytes = size_bytes
                self._index[item.key] = {
                    "path": str(path),
                    "size_bytes": size_bytes,
                    "created_ts": item.created_ts,
                }
                self._save_index()
                logger.debug(f"[ColdLayer] 持久化 {item.key} ({size_bytes} bytes)")
                return True
            except Exception as exc:  # noqa: BLE001
                logger.error(f"[ColdLayer] 冷层写入失败 {item.key}: {exc}")
                return False

    async def delete(self, key: str) -> bool:
        with self._lock:
            meta = self._index.pop(key, None)
            if meta is None:
                return False
            try:
                Path(meta["path"]).unlink(missing_ok=True)
            except Exception:  # noqa: BLE001
                pass
            self._save_index()
            return True

    def current_usage_bytes(self) -> int:
        with self._lock:
            return sum(v["size_bytes"] for v in self._index.values())


# ============================================================
# 三层记忆管理器
# ============================================================
class ThreeTierMemoryManager:
    """Hermes三层记忆统一管理器

    策略：
      1. 写入优先L1热层，L1满自动下沉L2暖层
      2. L2暖层满或张量过大，直接落L3冷层D盘
      3. 读取按 L1->L2->L3 顺序，命中自动回写更热层
      4. 提供强制swap_tensors_to_disk()，内存超限立即触发
    """

    def __init__(
        self,
        hot_cap_bytes: int = 128 * 1024 * 1024,
        warm_cap_bytes: int = 2 * 1024 * 1024 * 1024,
    ) -> None:
        self.hot = HotMemoryLayer(hot_cap_bytes)
        self.warm = WarmMemoryLayer(size_limit_bytes=warm_cap_bytes)
        self.cold = ColdDiskMemoryLayer()
        self._lock = threading.RLock()
        self._swap_count: int = 0
        logger.info(
            f"[RLLM-3TierMem] 初始化完成 "
            f"热层={hot_cap_bytes//1024//1024}MB "
            f"暖层={warm_cap_bytes//1024//1024//1024}GB "
            f"冷层={COLD_DISK_DIR}"
        )

    # ----------------------------------------------------------------
    async def store(self, key: str, value: Any, size_hint_bytes: int = 0) -> bool:
        """写入记忆（自动选层）"""
        size = size_hint_bytes
        if size == 0:
            try:
                size = len(pickle.dumps(value))
            except Exception:  # noqa: BLE001
                size = 1024
        item = MemoryItem(key=key, value=value, size_bytes=size)
        with self._lock:
            if await self.hot.put(item):
                return True
            if await self.warm.put(item):
                return True
            return await self.cold.put(item)

    async def fetch(self, key: str) -> Optional[Any]:
        """读取记忆（L1→L2→L3）"""
        with self._lock:
            item = await self.hot.get(key)
            if item is not None:
                return item.value
            item = await self.warm.get(key)
            if item is not None:
                # 回写热层
                await self.hot.put(item)
                return item.value
            item = await self.cold.get(key)
            if item is not None:
                # 回写暖层
                await self.warm.put(item)
                return item.value
            return None

    async def delete(self, key: str) -> bool:
        ok1 = await self.hot.delete(key)
        ok2 = await self.warm.delete(key)
        ok3 = await self.cold.delete(key)
        return ok1 or ok2 or ok3

    # ----------------------------------------------------------------
    async def force_swap_tensors_to_disk(
        self,
        tensor_dict: Dict[str, Any],
    ) -> Dict[str, Path]:
        """强制将张量dict落冷层磁盘（内存超限保护）

        Args:
            tensor_dict: {name: tensor/任意可pickle对象}

        Returns:
            落盘路径映射 {name: Path on D盘}
        """
        self._swap_count += 1
        result: Dict[str, Path] = {}
        for name, tensor in tensor_dict.items():
            key = f"swap_{self._swap_count}_{name}"
            try:
                size = len(pickle.dumps(tensor))
                item = MemoryItem(key=key, value=tensor, size_bytes=size)
                if await self.cold.put(item) and item.persistent_path is not None:
                    result[name] = item.persistent_path
            except Exception as exc:  # noqa: BLE001
                logger.error(f"[RLLM-3TierMem] swap失败 {name}: {exc}")
        logger.warning(
            f"[RLLM-3TierMem] 触发强制swap，共 {len(tensor_dict)} 个张量，"
            f"成功 {len(result)} 个，累计swap次数={self._swap_count}"
        )
        return result

    def usage_summary(self) -> Dict[str, int]:
        """返回各层使用量(字节)"""
        return {
            "hot_bytes": self.hot.current_usage_bytes(),
            "warm_bytes": self.warm.current_usage_bytes(),
            "cold_bytes": self.cold.current_usage_bytes(),
            "swap_count": self._swap_count,
        }


# 单例接口
_instance: Optional[ThreeTierMemoryManager] = None
_instance_lock = threading.Lock()


def get_memory_manager() -> ThreeTierMemoryManager:
    """获取三层记忆管理器单例"""
    global _instance
    if _instance is None:
        with _instance_lock:
            if _instance is None:
                _instance = ThreeTierMemoryManager()
    return _instance
