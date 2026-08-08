# -*- coding: utf-8 -*-
# File: D:\AI_RLLM\rllm_engine\__init__.py
"""RLLM Engine - 本地大模型推理引擎

让大模型在消费级 GPU 上跑起来.

核心创新:
  - v3 层级分页: 7B/13B dense 模型, 已验证 23.6 tok/s + 正常输出
  - v4 专家级分页: 47B MoE 模型, 256 专家独立分片 (实验性)

用法:
    from rllm_engine import RLLMEngine

    # 7B dense 模型 (推荐)
    engine = RLLMEngine("Nous-Hermes-2-Mistral-7B-DPO")
    engine.load()
    print(engine.generate("你好"))

    # 47B MoE 模型 (实验性)
    engine = RLLMEngine("mixtral-8x7b")
    engine.load()

跨平台支持:
  - Windows: 自动检测 D 盘 → C 盘用户目录
  - Linux/macOS: ~/.rllm
  - 可通过环境变量 RLLM_HOME 覆盖

硬件要求:
  - NVIDIA GPU (8GB+ VRAM), 16GB+ RAM
  - NVMe SSD 推荐
  - 可选: D 盘用于模型分片存储
"""
from __future__ import annotations

from rllm_engine.engine import RLLMEngine, EngineConfig
from rllm_engine.platform_paths import get_rllm_home, ensure_dirs

__version__ = "1.0.0"
__all__ = ["RLLMEngine", "EngineConfig", "get_rllm_home", "ensure_dirs"]
