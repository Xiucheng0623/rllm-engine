# -*- coding: utf-8 -*-
# File: D:\AI_RLLM\rllm_engine\platform_paths.py
r"""跨平台路径管理

自动检测最佳存储位置:
  - Windows: 优先 D 盘 → C 盘 %USERPROFILE%\.rllm
  - Linux/macOS: ~/.rllm
  - 可通过环境变量 RLLM_HOME 覆盖

用法:
    from rllm_engine.platform_paths import get_rllm_home

    home = get_rllm_home()
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional


def _detect_rllm_home() -> Path:
    """自动检测 RLLM 根目录

    优先级:
      1. 环境变量 RLLM_HOME
      2. Windows: D:\AI_RLLM (如果 D 盘存在)
      3. Windows: %USERPROFILE%\.rllm
      4. Linux/macOS: ~/.rllm

    Returns:
        RLLM 根目录路径
    """
    # 1. 环境变量覆盖
    env_home = os.environ.get("RLLM_HOME", "")
    if env_home:
        return Path(env_home)

    # 2. Windows D 盘优先 (NVMe 性能更好)
    if sys.platform == "win32":
        d_drive = Path("D:\\AI_RLLM")
        if Path("D:\\").exists():
            return d_drive
        # 无 D 盘 → C 盘用户目录
        user_home = Path(os.environ.get("USERPROFILE", str(Path.home())))
        return user_home / ".rllm"

    # 3. Linux/macOS
    return Path.home() / ".rllm"


# 单例: 全局 RLLM 根目录
_RLLM_HOME: Optional[Path] = None


def get_rllm_home() -> Path:
    """获取 RLLM 根目录 (首次调用时自动检测)

    Returns:
        RLLM 根目录路径
    """
    global _RLLM_HOME
    if _RLLM_HOME is None:
        _RLLM_HOME = _detect_rllm_home()
    return _RLLM_HOME


def get_models_dir() -> Path:
    """获取模型存储目录"""
    return get_rllm_home() / "models"


def get_cache_dir() -> Path:
    """获取 HF 缓存目录"""
    return get_rllm_home() / "cache"


def get_logs_dir() -> Path:
    """获取日志目录"""
    return get_rllm_home() / "logs"


def ensure_dirs() -> None:
    """确保所有必需目录存在"""
    home = get_rllm_home()
    dirs = [
        home,
        get_models_dir(),
        get_cache_dir(),
        get_logs_dir(),
        home / "offload_temp",
        home / "output",
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
