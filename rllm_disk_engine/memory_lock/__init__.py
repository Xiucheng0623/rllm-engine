# File: D:\AI_RLLM\rllm_disk_engine\memory_lock\__init__.py
"""全局内存硬锁包"""
from .global_memory_lock import GlobalMemoryLock, get_memory_lock

__all__ = ["GlobalMemoryLock", "get_memory_lock"]
