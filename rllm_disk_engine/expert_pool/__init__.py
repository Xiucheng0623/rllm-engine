# -*- coding: utf-8 -*-
# File: D:\AI_RLLM\rllm_disk_engine\expert_pool\__init__.py
"""v4 专家级分页架构 (Expert-Level Paging)

将分页粒度从"Transformer 层"细化到"MoE 专家", 大幅降低每 token 的 I/O 量.

核心模块:
  - ExpertShardPersistor: Mixtral 模型按专家分片写 D 盘 + 生成 index.json
  - ExpertVRAMPool: 专家级 VRAM 缓存池
  - ExpertEvictor: 专家级冷热置换 (直写 D 盘)
  - ExpertFreqTracker: 专家访问频率统计 (Hermes 自进化用)
  - MoELayerRunner: MoE 手动逐层 forward (替代 model.forward)
"""
from rllm_disk_engine.expert_pool.expert_shard_persistor import (
    ExpertShardPersistor,
    ExpertIndex,
    ExpertShardMeta,
)
from rllm_disk_engine.expert_pool.expert_vram_pool import (
    ExpertVRAMPool,
    ExpertEntry,
)
from rllm_disk_engine.expert_pool.expert_evictor import ExpertEvictor
from rllm_disk_engine.expert_pool.expert_freq_tracker import (
    ExpertFreqTracker,
)
from rllm_disk_engine.expert_pool.moe_layer_runner import MoELayerRunner

__all__ = [
    "ExpertShardPersistor",
    "ExpertIndex",
    "ExpertShardMeta",
    "ExpertVRAMPool",
    "ExpertEntry",
    "ExpertEvictor",
    "ExpertFreqTracker",
    "MoELayerRunner",
]
