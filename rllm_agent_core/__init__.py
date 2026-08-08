# File: D:\AI_RLLM\rllm_agent_core\__init__.py
"""Rebirth LLM(RLLM) 核心包

改造自 Nous Hermes-Agent 开源框架，深度耦合磁盘分页推理引擎。
所有路径硬绑定 D:\\AI_RLLM，实现零C盘占用。
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Final

# ========== 全局唯一D盘根路径真值源 ==========
HERMES_ROOT: Final[Path] = Path(r"D:\AI_RLLM")
SKILL_STORAGE_DIR: Final[Path] = HERMES_ROOT / "skill_storage"
SKILL_ARCHIVE_DIR: Final[Path] = SKILL_STORAGE_DIR / "archive"
OUTPUT_DATASET_DIR: Final[Path] = HERMES_ROOT / "output_dataset"
INPUT_DATA_DIR: Final[Path] = HERMES_ROOT / "input_data"
LOG_DIR: Final[Path] = HERMES_ROOT / "logs"

# ========== 强制离线模式（禁止联网下载模型） ==========
os.environ.setdefault("HF_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

# ========== 将Hermes模块路径加入sys.path ==========
for _sub in ("rllm_agent_core", "rllm_disk_engine", "rllm_auto_evo", "rllm_pipeline"):
    _p = str(HERMES_ROOT / _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ========== 确保日志/技能目录存在 ==========
for _d in (SKILL_STORAGE_DIR, SKILL_ARCHIVE_DIR, OUTPUT_DATASET_DIR, LOG_DIR):
    _d.mkdir(parents=True, exist_ok=True)

__version__ = "1.0.0-DiskOffload"
__all__ = [
    "HERMES_ROOT",
    "SKILL_STORAGE_DIR",
    "SKILL_ARCHIVE_DIR",
    "OUTPUT_DATASET_DIR",
    "INPUT_DATA_DIR",
    "LOG_DIR",
]
