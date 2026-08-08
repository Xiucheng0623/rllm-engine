# -*- coding: utf-8 -*-
"""GPU 显存持久层缓存池包

提供 Prefill 一次性装入 + Decode 零磁盘 IO 的能力。
"""
from rllm_disk_engine.vram_pool.vram_cache_pool import (
    VRAMCachePool,
    VRAMLayerEntry,
    get_vram_cache_pool,
)
from rllm_disk_engine.vram_pool.hot_cold_evictor import (
    HotColdEvictor,
    LayerFreqTracker,
)

__all__ = [
    "VRAMCachePool",
    "VRAMLayerEntry",
    "get_vram_cache_pool",
    "HotColdEvictor",
    "LayerFreqTracker",
]
