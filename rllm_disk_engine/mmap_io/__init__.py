# File: D:\AI_RLLM\rllm_disk_engine\mmap_io\__init__.py
"""mmap 磁盘映射封装包"""
from .mmap_wrapper import (
    MmapFileHandle,
    MmapManager,
    get_mmap_manager,
)

__all__ = ["MmapFileHandle", "MmapManager", "get_mmap_manager"]
