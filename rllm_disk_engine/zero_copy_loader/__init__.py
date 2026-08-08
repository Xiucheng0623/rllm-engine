# -*- coding: utf-8 -*-
"""零拷贝分片加载器包

提供持久 safetensors 句柄 + mmap + DMA 直传 VRAM 的零拷贝加载能力。
"""
from rllm_disk_engine.zero_copy_loader.zero_copy_shard_loader import (
    ZeroCopyShardLoader,
    get_zero_copy_loader,
)

__all__ = ["ZeroCopyShardLoader", "get_zero_copy_loader"]
