# -*- coding: utf-8 -*-
# File: D:\AI_RLLM\rllm_disk_engine\zero_copy_loader\zero_copy_shard_loader.py
"""零拷贝分片加载器

核心契约:
  1. 进程启动时打开所有 safetensors 文件句柄一次, 永不关闭 (消除重复 open 开销)
  2. mmap 模式读取, 数据驻留 page cache, 不进 Python 堆
  3. 通过 pinned memory + .to("cuda", non_blocking=True) DMA 直传 VRAM
  4. 绕过 load_state_dict 的张量全量复制

设计目标:
  - 单层加载耗时 < 100ms (相比原 safe_open+load_state_dict 的 250ms)
  - 消除每 token 重复打开文件的 syscall 开销
"""
from __future__ import annotations

import asyncio
import threading
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger

from rllm_agent_core import LOG_DIR

logger.add(
    LOG_DIR / "zero_copy_{time}.log",
    rotation="50 MB",
    retention="7 days",
    encoding="utf-8",
)


@dataclass
class LayerIndexEntry:
    """层索引条目"""
    layer_idx: int
    tensor_keys: List[str]
    st_files: List[str]  # 涉及的 safetensors 文件


class ZeroCopyShardLoader:
    """零拷贝分片加载器 (全局单例)

    Attributes:
        _handles: 持久 safetensors 文件句柄池 (st_file -> safe_open句柄)
        _layer_index: 层索引 (layer_idx -> LayerIndexEntry)
        _lock: 线程锁
    """

    _singleton: Optional["ZeroCopyShardLoader"] = None
    _lock_singleton: threading.Lock = threading.Lock()

    def __new__(cls) -> "ZeroCopyShardLoader":
        """单例构造"""
        if cls._singleton is None:
            with cls._lock_singleton:
                if cls._singleton is None:
                    cls._singleton = super().__new__(cls)
        return cls._singleton

    def __init__(self) -> None:
        """初始化加载器

        Note:
            实际句柄打开在 initialize() 中完成, 构造函数仅设置标志位
        """
        if getattr(self, "_initialized", False):
            return
        self._initialized: bool = True
        self._raw_model_dir: Optional[Path] = None
        self._weight_map: Dict[str, str] = {}
        self._handles: Dict[str, Any] = {}
        self._layer_index: Dict[int, LayerIndexEntry] = {}
        self._lock: threading.RLock = threading.RLock()
        self._open_ts: float = 0.0
        self._total_loads: int = 0
        self._total_load_ms: float = 0.0
        logger.info("[RLLM-ZeroCopy] 加载器实例化完成 (待 initialize)")

    # ----------------------------------------------------------------
    # 初始化: 进程启动时调用一次
    # ----------------------------------------------------------------
    def initialize(
        self,
        raw_model_dir: Path,
        weight_map: Dict[str, str],
        num_layers: int,
    ) -> None:
        """打开所有 safetensors 文件句柄 + 建立 layer_idx 索引

        Args:
            raw_model_dir: 模型原始目录 (含 model.safetensors.index.json)
            weight_map: {tensor_key: st_file} 映射
            num_layers: 模型总层数

        Raises:
            FileNotFoundError: 模型目录不存在
            RuntimeError: safetensors 文件打开失败
        """
        from safetensors import safe_open

        self._raw_model_dir = Path(raw_model_dir)
        if not self._raw_model_dir.exists():
            raise FileNotFoundError(f"模型目录不存在: {self._raw_model_dir}")

        self._weight_map = dict(weight_map)
        t0 = time.time()

        # 1. 收集所有涉及的 safetensors 文件
        st_files = sorted(set(weight_map.values()))
        logger.info(
            f"[RLLM-ZeroCopy] 开始打开 {len(st_files)} 个 safetensors 文件句柄"
        )

        # 2. 持久打开每个文件
        for st_file in st_files:
            fp = self._raw_model_dir / st_file
            if not fp.exists():
                raise FileNotFoundError(f"safetensors 文件缺失: {fp}")
            try:
                # framework="pt" 启用 mmap, 数据驻留 page cache
                self._handles[st_file] = safe_open(
                    str(fp), framework="pt", device="cpu"
                )
            except Exception as exc:
                raise RuntimeError(
                    f"safetensors 文件打开失败 {fp}: {exc}"
                ) from exc

        # 3. 建立 layer_idx -> [tensor_keys + st_files] 索引
        for layer_idx in range(num_layers):
            prefix = f"model.layers.{layer_idx}."
            keys = [k for k in weight_map.keys() if k.startswith(prefix)]
            files = sorted(set(weight_map[k] for k in keys))
            self._layer_index[layer_idx] = LayerIndexEntry(
                layer_idx=layer_idx,
                tensor_keys=keys,
                st_files=files,
            )

        self._open_ts = time.time() - t0
        logger.success(
            f"[RLLM-ZeroCopy] 初始化完成: "
            f"handles={len(self._handles)} layers={num_layers} "
            f"耗时={self._open_ts:.2f}s"
        )

    # ----------------------------------------------------------------
    # 对外核心接口
    # ----------------------------------------------------------------
    async def load_layer_zero_copy(
        self, layer_idx: int
    ) -> Tuple[Dict[str, Any], float]:
        """零拷贝加载单层权重 (mmap 视图, 不复制)

        Args:
            layer_idx: 层索引

        Returns:
            (state_dict, 加载耗时ms)
            state_dict 的 value 是 mmap 视图, 调用方负责 .to("cuda") 拷贝

        Raises:
            KeyError: layer_idx 未在索引中
            RuntimeError: 文件句柄已关闭
        """
        if layer_idx not in self._layer_index:
            raise KeyError(f"层 {layer_idx} 未在索引中, 请先 initialize()")

        t0 = time.time()
        entry: LayerIndexEntry = self._layer_index[layer_idx]
        state_dict: Dict[str, Any] = {}

        # 按 safetensors 文件分组, 复用同一句柄 (减少 syscall)
        file_to_keys: Dict[str, List[str]] = defaultdict(list)
        for k in entry.tensor_keys:
            file_to_keys[self._weight_map[k]].append(k)

        for st_file, ks in file_to_keys.items():
            handle = self._handles.get(st_file)
            if handle is None:
                raise RuntimeError(f"文件句柄已关闭: {st_file}")
            for k in ks:
                # safe_open 已开启 mmap, get_tensor 返回 mmap 视图
                state_dict[k] = handle.get_tensor(k)

        load_ms = (time.time() - t0) * 1000.0
        with self._lock:
            self._total_loads += 1
            self._total_load_ms += load_ms
        return state_dict, load_ms

    def get_aux_tensors(
        self, keys: List[str]
    ) -> Dict[str, Any]:
        """加载非层权重 (embed_tokens / norm / lm_head)

        Args:
            keys: 张量名列表

        Returns:
            {tensor_key: mmap_tensor}
        """
        out: Dict[str, Any] = {}
        for k in keys:
            st_file = self._weight_map.get(k)
            if st_file is None:
                continue
            handle = self._handles.get(st_file)
            if handle is None:
                continue
            out[k] = handle.get_tensor(k)
        return out

    # ----------------------------------------------------------------
    # 生命周期
    # ----------------------------------------------------------------
    def release(self) -> None:
        """进程退出时关闭所有句柄"""
        with self._lock:
            for st_file, h in list(self._handles.items()):
                try:
                    del h
                except Exception as exc:  # noqa: BLE001
                    logger.warning(f"[RLLM-ZeroCopy] 句柄关闭失败 {st_file}: {exc}")
            self._handles.clear()
            self._layer_index.clear()
        logger.info("[RLLM-ZeroCopy] 所有句柄已释放")

    def stats(self) -> Dict[str, float]:
        """获取加载器统计"""
        with self._lock:
            avg_ms = (
                self._total_load_ms / self._total_loads
                if self._total_loads > 0
                else 0.0
            )
            return {
                "handles": float(len(self._handles)),
                "indexed_layers": float(len(self._layer_index)),
                "total_loads": float(self._total_loads),
                "avg_load_ms": avg_ms,
                "init_open_sec": self._open_ts,
            }


# ============================================================
# 单例获取函数
# ============================================================
def get_zero_copy_loader() -> ZeroCopyShardLoader:
    """获取全局 ZeroCopyShardLoader 单例

    Returns:
        ZeroCopyShardLoader 实例
    """
    return ZeroCopyShardLoader()
