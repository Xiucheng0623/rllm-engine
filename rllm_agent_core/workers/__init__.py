# File: D:\AI_RLLM\rllm_agent_core\workers\__init__.py
"""RLLM Worker 注册中心(底层复用Hermes)

新增 DiskLLMInferWorker 替代原生全内存推理Worker。
"""
from .worker_registry import (
    WorkerBase,
    WorkerRegistry,
    DiskLLMInferWorker,
    register_default_workers,
    get_worker,
)

__all__ = [
    "WorkerBase",
    "WorkerRegistry",
    "DiskLLMInferWorker",
    "register_default_workers",
    "get_worker",
]
