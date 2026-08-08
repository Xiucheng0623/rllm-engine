# -*- coding: utf-8 -*-
# File: D:\AI_RLLM\rllm_disk_engine\router\__init__.py
"""路由器预取模块

基于路由预测, 提前从 D 盘预取候选专家到 VRAM, 掩盖 I/O 延迟.

核心模块:
  - RouterPredictor: 轻量路由预测 (Top-16 候选专家)
  - RouterPrefetcher: 并行预取候选专家到 VRAM
"""
from rllm_disk_engine.router.router_predictor import RouterPredictor
from rllm_disk_engine.router.router_prefetcher import RouterPrefetcher

__all__ = ["RouterPredictor", "RouterPrefetcher"]
