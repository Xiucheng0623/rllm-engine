# File: D:\AI_RLLM\rllm_disk_engine\scheduler\async_page_scheduler.py
"""异步磁盘分页调度器

核心设计：
  1. 内存缓冲区硬锁2GB，最多同时加载1-2层权重（计算层+预取层）
  2. 每层计算完成后立即调用 unload_layer() 释放内存引用 + gc.collect()
  3. 异步线程池在计算当前层时，预取 N 层到就绪队列（prefetch_layers_ahead）
  4. 可选mmap封装加速大文件连续读
"""
from __future__ import annotations

import asyncio
import collections
import gc
import threading
import time
from concurrent.futures import ThreadPoolExecutor, Future
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple

from loguru import logger

from rllm_agent_core import LOG_DIR
from rllm_disk_engine.sharding.shard_persistor import (
    ModelShardPersistor,
    ShardMeta,
    QuantizationType,
    get_shard_persistor,
)

logger.add(
    LOG_DIR / "scheduler_{time}.log",
    rotation="50 MB",
    retention="7 days",
    encoding="utf-8",
)


# ============================================================
# 结果结构
# ============================================================
@dataclass
class LayerLoadResult:
    """单层加载结果"""
    layer_idx: int
    tensors: Dict[str, Any]
    metas: List[ShardMeta]
    load_time_ms: float  # 磁盘读取耗时ms
    in_memory_bytes: int  # 占用内存字节数


# ============================================================
# 调度器
# ============================================================
class AsyncPageScheduler:
    """异步磁盘分页调度器

    内存策略：
      buffer_capacity_gb = 2.0 (强制，构造函数会覆盖>2GB的值)
    """

    def __init__(
        self,
        shards_dir: Path,
        model_name: str = "default_model",
        prefetch_layers_ahead: int = 2,
        prefetch_threads: int = 4,
        enable_mmap: bool = True,
        buffer_capacity_gb: float = 2.0,
        raw_model_dir: Path = None,
        weight_map: dict = None,
    ) -> None:
        if buffer_capacity_gb > 2.0:
            raise ValueError(
                f"CPU缓冲区硬限2GB，禁止设置 buffer_capacity_gb={buffer_capacity_gb}"
            )
        self._model_name = model_name
        self._prefetch_ahead = prefetch_layers_ahead
        self._enable_mmap = enable_mmap
        self._buffer_capacity_bytes = int(buffer_capacity_gb * 1024 ** 3)
        self._shards_dir = Path(shards_dir)
        self._persistor: ModelShardPersistor = get_shard_persistor()
        # 真实模型 safetensors 路径和权重索引
        self._raw_model_dir = Path(raw_model_dir) if raw_model_dir else None
        self._weight_map: dict = weight_map or {}

        # 当前已加载层 {layer_idx: LayerLoadResult}
        self._loaded: Dict[int, LayerLoadResult] = {}
        # 预取就绪队列: {layer_idx: Future}
        self._prefetch_futures: Dict[int, Future] = {}
        # 预取完成缓存 (若用户load顺序一致可直接命中)
        self._prefetch_cache: Dict[int, Tuple[Dict[str, Any], List[ShardMeta], float]] = {}

        self._lock = threading.RLock()
        self._pool = ThreadPoolExecutor(max_workers=max(1, prefetch_threads), thread_name_prefix="disk_prefetch")
        self._current_buffer_bytes: int = 0
        self._total_load_count: int = 0
        self._cache_hit_count: int = 0
        logger.info(
            f"[RLLM-PageScheduler] 初始化完成: prefetch_ahead={prefetch_layers_ahead} "
            f"threads={prefetch_threads} mmap={enable_mmap} "
            f"buffer_cap={buffer_capacity_gb}GB"
        )

    # ----------------------------------------------------------------
    # 对外接口
    # ----------------------------------------------------------------
    async def load_layer(self, layer_idx: int) -> Tuple[Any, float]:
        """加载单层权重（核心入口）

        优先级：
          1. 命中已加载内存 -> 直接返回
          2. 命中预取就绪缓存 -> 取出并入内存
          3. 同步等待预取future完成
          4. 兜底同步读取

        Returns:
            (layer_object_or_dict, read_latency_ms)
        """
        self._total_load_count += 1
        t0 = time.time()

        # Step1: 已加载命中
        with self._lock:
            if layer_idx in self._loaded:
                self._cache_hit_count += 1
                return self._loaded[layer_idx].tensors, self._loaded[layer_idx].load_time_ms

        # Step2: 启动本层及之后预取
        await self._schedule_prefetch(start_layer=layer_idx)

        # Step3: 等待本层预取结果或同步加载
        with self._lock:
            fut = self._prefetch_futures.get(layer_idx)
            cached = self._prefetch_cache.pop(layer_idx, None)
        if cached is not None:
            tensors, metas, ms = cached
            load_ms = ms
        elif fut is not None:
            try:
                tensors, metas, load_ms = await asyncio.wrap_future(fut)
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"[RLLM-PageScheduler] 预取失败，回退同步: {exc}")
                tensors, metas, load_ms = await self._sync_load(layer_idx)
        else:
            tensors, metas, load_ms = await self._sync_load(layer_idx)

        # Step4: 放入已加载集合，强制执行缓冲区裁剪
        mem_bytes = sum(m.size_bytes for m in metas)
        result = LayerLoadResult(
            layer_idx=layer_idx,
            tensors=tensors,
            metas=metas,
            load_time_ms=load_ms,
            in_memory_bytes=mem_bytes,
        )
        with self._lock:
            self._loaded[layer_idx] = result
            self._current_buffer_bytes += mem_bytes
        await self._evict_buffer_if_needed()

        total_ms = (time.time() - t0) * 1000.0
        return tensors, total_ms

    async def unload_layer(self, layer_idx: int) -> bool:
        """卸载单层，立即释放内存 + gc"""
        with self._lock:
            loaded = self._loaded.pop(layer_idx, None)
            if loaded is None:
                return False
            self._current_buffer_bytes -= loaded.in_memory_bytes
            if self._current_buffer_bytes < 0:
                self._current_buffer_bytes = 0
        # 删除引用
        try:
            del loaded
        except Exception:  # noqa: BLE001
            pass
        gc.collect()
        return True

    def current_buffer_bytes(self) -> int:
        with self._lock:
            return self._current_buffer_bytes

    def stats(self) -> Dict[str, int]:
        with self._lock:
            return {
                "loaded_layers": len(self._loaded),
                "buffer_bytes": self._current_buffer_bytes,
                "total_loads": self._total_load_count,
                "cache_hits": self._cache_hit_count,
                "prefetch_inflight": len(self._prefetch_futures),
            }

    # ----------------------------------------------------------------
    # 内部：预取调度
    # ----------------------------------------------------------------
    async def _schedule_prefetch(self, start_layer: int) -> None:
        """异步调度：预取 start_layer ~ start_layer+ahead 层"""
        total_layers = 32
        with self._lock:
            for i in range(self._prefetch_ahead + 1):
                lidx = start_layer + i
                if lidx >= total_layers:
                    continue
                if lidx in self._prefetch_futures or lidx in self._prefetch_cache or lidx in self._loaded:
                    continue
                # 提交预取任务
                future = self._pool.submit(self._load_sync_worker, lidx)
                self._prefetch_futures[lidx] = future
                # 挂载完成回调写入缓存
                future.add_done_callback(
                    lambda f, idx=lidx: self._on_prefetch_done(idx, f)
                )

    def _on_prefetch_done(self, layer_idx: int, fut: Future) -> None:
        """预取完成回调，写入缓存"""
        try:
            result = fut.result()
            with self._lock:
                self._prefetch_futures.pop(layer_idx, None)
                self._prefetch_cache[layer_idx] = result
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"[RLLM-PageScheduler] 预取层{layer_idx}失败: {exc}")
            with self._lock:
                self._prefetch_futures.pop(layer_idx, None)

    # ----------------------------------------------------------------
    # 内部：同步加载（在线程池内执行）
    # ----------------------------------------------------------------
    def _load_sync_worker(
        self, layer_idx: int
    ) -> Tuple[Dict[str, Any], List[ShardMeta], float]:
        """从原始 safetensors 按层读真实 bf16 权重（不走 .shard 占位文件）"""
        t0 = time.time()
        all_tensors: Dict[str, Any] = {}
        metas: List[ShardMeta] = []

        if self._raw_model_dir and self._weight_map:
            # === 真实模式：从 safetensors 按层读权重 ===
            from safetensors import safe_open
            layer_prefix = f"model.layers.{layer_idx}."
            # 按 safetensors 文件分组，减少 safe_open 次数
            file_to_keys: Dict[str, List[str]] = {}
            for key, st_file in self._weight_map.items():
                if key.startswith(layer_prefix):
                    file_to_keys.setdefault(st_file, []).append(key)
            for st_file, keys in file_to_keys.items():
                fp = self._raw_model_dir / st_file
                try:
                    with safe_open(str(fp), framework="pt", device="cpu") as f:
                        for key in keys:
                            if key in f.keys():
                                all_tensors[key] = f.get_tensor(key)
                except Exception as exc:
                    logger.warning(f"[RLLM-PageScheduler] safetensors读取失败 {st_file}: {exc}")
            mem_bytes = sum(t.numel() * t.element_size() for t in all_tensors.values())
            from rllm_disk_engine.sharding.shard_persistor import ShardMeta
            meta = ShardMeta(
                shard_id=f"layer_{layer_idx:03d}_real",
                layer_idx=layer_idx,
                tensor_keys=list(all_tensors.keys()),
                file_path=str(self._raw_model_dir),
                file_offset_bytes=0,
                size_bytes=mem_bytes,
                stored_bytes=mem_bytes,
                quantization="bf16",
                sha1="",
                shape_info={},
            )
            metas.append(meta)
            logger.debug(f"[RLLM-PageScheduler] 层{layer_idx} 真实权重加载: {len(all_tensors)} tensors, {mem_bytes/1024/1024:.1f}MB")
        else:
            # === 兼容模式：走 .shard 分片（旧逻辑）===
            shard_ids = self._persistor.list_layer_shard_ids(self._model_name, layer_idx)
            if not shard_ids:
                raw_dir = self._shards_dir / "_raw"
                try:
                    self._persistor.slice_and_persist(raw_dir, self._model_name)
                    shard_ids = self._persistor.list_layer_shard_ids(self._model_name, layer_idx)
                except Exception as exc:
                    logger.warning(f"[RLLM-PageScheduler] 骨架生成失败: {exc}")
            for sid in shard_ids:
                try:
                    tensors, meta = self._persistor.load_shard(
                        self._model_name, sid, QuantizationType.INT8
                    )
                    all_tensors.update({f"{sid}_{k}": v for k, v in tensors.items()})
                    metas.append(meta)
                except Exception as exc:
                    logger.warning(f"[RLLM-PageScheduler] 加载分片{sid}失败: {exc}")

        load_ms = (time.time() - t0) * 1000.0
        return all_tensors, metas, load_ms

    async def _sync_load(
        self, layer_idx: int
    ) -> Tuple[Dict[str, Any], List[ShardMeta], float]:
        """兜底同步加载（直接线程池跑）"""
        fut = self._pool.submit(self._load_sync_worker, layer_idx)
        return await asyncio.wrap_future(fut)

    # ----------------------------------------------------------------
    # 内部：缓冲区裁剪
    # ----------------------------------------------------------------
    async def _evict_buffer_if_needed(self) -> None:
        """超2GB硬限时，强制卸载最早加载的层"""
        evicted = 0
        with self._lock:
            while self._current_buffer_bytes > self._buffer_capacity_bytes and self._loaded:
                # 卸载最小编号（最早加载）的层
                victim_idx = min(self._loaded.keys())
                victim = self._loaded.pop(victim_idx)
                self._current_buffer_bytes -= victim.in_memory_bytes
                evicted += 1
        if evicted > 0:
            if self._current_buffer_bytes < 0:
                self._current_buffer_bytes = 0
            logger.warning(
                f"[RLLM-PageScheduler] 缓冲区超限，强制卸载 {evicted} 层，"
                f"当前 {self._current_buffer_bytes/1024/1024:.1f}MB"
            )
            gc.collect()


# 单例
_scheduler_singleton: Optional[AsyncPageScheduler] = None
_sched_lock = threading.Lock()


def get_page_scheduler(
    shards_dir: Optional[Path] = None,
    **kwargs: Any,
) -> AsyncPageScheduler:
    global _scheduler_singleton
    if _scheduler_singleton is None:
        with _sched_lock:
            if _scheduler_singleton is None:
                root = shards_dir or Path(r"D:\AI_RLLM\rllm_model_shards")
                _scheduler_singleton = AsyncPageScheduler(root, **kwargs)
    return _scheduler_singleton
