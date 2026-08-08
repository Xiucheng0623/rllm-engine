# File: D:\AI_RLLM\rllm_disk_engine\__init__.py
"""磁盘分页调度引擎包

自研磁盘分页低内存推理核心：
  sharding:  模型分片持久化器（按Transformer层切片+4/8bit量化+索引元数据）
  scheduler: 异步磁盘分页调度器（单层载入、异步预取、计算完销毁）
  kv_manager: KV缓存磁盘溢出管理器
  memory_lock: 全局内存硬锁（2GB封顶，超限强制swap）
  mmap_io: mmap磁盘映射封装
"""
from __future__ import annotations

from pathlib import Path
from typing import Final

DISK_ENGINE_ROOT: Final[Path] = Path(r"D:\AI_RLLM\rllm_disk_engine")

__all__ = ["DISK_ENGINE_ROOT"]
