# File: D:\AI_RLLM\env_config.py
"""
Rebirth LLM(RLLM) 全局环境变量硬注入配置模块（代码级，不依赖activate.bat）
===================================================================
**设计意图**：
  用户可能用 `python xxx.py` 裸跑，也可能 IDE 直接调试，此时未必走 `.venv\Scripts\activate.bat`，
  导致 HF_HOME/TRANSFORMERS_CACHE/HF_OFFLINE/TORCH_HOME 全部丢失，
  进而触发 validate_env_RLLM 面板的「严重问题」警告，或意外触发 HuggingFace 联网下载。

**使用方式（所有入口文件顶部第一行）**：
  >>> import env_config   # 本模块一加载就立刻设置环境变量，之后 import torch/transformers 才会生效

**硬隔离承诺**：
  本模块设置的所有磁盘路径 100% 位于 D:\\AI_RLLM\\，绝不写 C 盘。
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# =========================================================
# 1. 硬编码项目根（唯一真理）
# =========================================================
PROJECT_ROOT: Path = Path(r"D:\AI_RLLM").resolve()

# =========================================================
# 2. 必须存在的 40 个目录（与 init_RLLM_env.bat 第 2 步保持一致）
# =========================================================
REQUIRED_DIRS: List[Path] = [
    PROJECT_ROOT / ".venv",
    PROJECT_ROOT / "hf_cache" / "hub",
    PROJECT_ROOT / "hf_cache" / "datasets",
    PROJECT_ROOT / "hf_cache" / "torch",
    PROJECT_ROOT / "input_data",
    PROJECT_ROOT / "logs",
    PROJECT_ROOT / "rllm_agent_core" / "config",
    PROJECT_ROOT / "rllm_agent_core" / "workers",
    PROJECT_ROOT / "rllm_agent_core" / "memory",
    PROJECT_ROOT / "rllm_agent_core" / "skills",
    PROJECT_ROOT / "rllm_agent_core" / "review",
    PROJECT_ROOT / "rllm_disk_engine" / "sharding",
    PROJECT_ROOT / "rllm_disk_engine" / "scheduler",
    PROJECT_ROOT / "rllm_disk_engine" / "kv_manager",
    PROJECT_ROOT / "rllm_disk_engine" / "memory_lock",
    PROJECT_ROOT / "rllm_disk_engine" / "mmap_io",
    PROJECT_ROOT / "rllm_auto_evo" / "metrics",
    PROJECT_ROOT / "rllm_auto_evo" / "strategy",
    PROJECT_ROOT / "rllm_auto_evo" / "tuner",
    PROJECT_ROOT / "rllm_pipeline" / "batch_reader",
    PROJECT_ROOT / "rllm_pipeline" / "writer",
    PROJECT_ROOT / "rllm_pipeline" / "checkpoint",
    PROJECT_ROOT / "rllm_model_shards" / "indexes",
    PROJECT_ROOT / "rllm_model_shards" / "tokenizer",
    PROJECT_ROOT / "rllm_model_shards" / "_raw",
    PROJECT_ROOT / "rllm_offload_temp" / "kv_cache",
    PROJECT_ROOT / "rllm_offload_temp" / "tensor_swap",
    PROJECT_ROOT / "rllm_offload_temp" / "warm_cache",
    PROJECT_ROOT / "rllm_output_dataset",
    PROJECT_ROOT / "rllm_skill_storage" / "evo_reports",
    PROJECT_ROOT / "rllm_skill_storage" / "archive",
    PROJECT_ROOT / "rllm_tests",
]

# =========================================================
# 3. 需要强制注入的环境变量（key -> value）
# =========================================================
REQUIRED_ENV_VARS: List[Tuple[str, str]] = [
    ("PROJECT_ROOT",          str(PROJECT_ROOT)),
    ("HF_HOME",               str(PROJECT_ROOT / "hf_cache")),
    ("TRANSFORMERS_CACHE",    str(PROJECT_ROOT / "hf_cache" / "hub")),
    ("HUGGINGFACE_HUB_CACHE", str(PROJECT_ROOT / "hf_cache" / "hub")),
    ("TORCH_HOME",            str(PROJECT_ROOT / "hf_cache" / "torch")),
    ("HF_DATASETS_CACHE",     str(PROJECT_ROOT / "hf_cache" / "datasets")),
    # ★ 强制离线模式：禁止 transformers 任何自动联网下载
    ("HF_HUB_OFFLINE",        "1"),
    ("TRANSFORMERS_OFFLINE",  "1"),
    ("HF_DATASETS_OFFLINE",   "1"),
    ("HF_OFFLINE",            "1"),  # 兼容 validate_env_RLLM 的旧字段
    # 全局 CPU buffer 硬限 2GB 提示
    ("RLLM_CPU_BUFFER_GB",    "2.0"),
]


def ensure_dirs() -> Dict[str, bool]:
    """确保所有 REQUIRED_DIRS 目录存在（缺少则自动 mkdir -p）。

    Returns:
        Dict[str, bool]: 每个目录的「是否刚刚新建」标志。
    """
    result: Dict[str, bool] = {}
    for d in REQUIRED_DIRS:
        was_missing: bool = not d.exists()
        try:
            d.mkdir(parents=True, exist_ok=True)
        except (OSError, PermissionError) as e:  # 防御式：磁盘只读/权限不足
            print(f"[env_config] ⚠ 创建目录失败 {d}: {e}")
        result[str(d)] = was_missing
    return result



def apply_env(verbose: bool = False) -> Dict[str, Tuple[str, str]]:
    """把 REQUIRED_ENV_VARS 全部写入 os.environ（覆盖已有值，确保正确性）。
    Windows 特别处理：os.add_dll_directory + PATH 双保险，解决 torch fbgemm.dll / cublas64_*.dll
    之类的「WinError 126 找不到指定模块」经典坑。

    Returns:
        Dict[str, Tuple[str, str]]: key -> (old_value, new_value) 的变更记录。
    """
    # --- Windows torch DLL 搜索双保险（必加，否则fbgemm.dll依赖找不到）---
    _torch_lib = PROJECT_ROOT / ".venv" / "Lib" / "site-packages" / "torch" / "lib"
    if _torch_lib.exists() and sys.platform.startswith("win") and hasattr(os, "add_dll_directory"):
        try:
            os.add_dll_directory(str(_torch_lib))
        except (OSError, FileNotFoundError) as _e:
            print(f"[env_config] add_dll_directory失败(不致命): {_e}")
        os.environ["PATH"] = str(_torch_lib) + ";" + os.environ.get("PATH", "")
    # --- END Windows torch DLL 双保险 ---

    changes: Dict[str, Tuple[str, str]] = {}
    for k, v in REQUIRED_ENV_VARS:
        old: str = os.environ.get(k, "")
        if old != v:
            os.environ[k] = v
            changes[k] = (old, v)
    # 把项目根 + 两个核心包目录加到 sys.path 首（最高优先级）
    for p in [str(PROJECT_ROOT),
              str(PROJECT_ROOT / "rllm_agent_core"),
              str(PROJECT_ROOT / "rllm_disk_engine"),
              str(PROJECT_ROOT / "rllm_auto_evo"),
              str(PROJECT_ROOT / "rllm_pipeline")]:
        if p not in sys.path:
            sys.path.insert(0, p)
    if verbose and changes:
        print(f"[env_config] 已注入 {len(changes)} 条环境变量，所有路径约束于 {PROJECT_ROOT}")
    return changes

# =========================================================
# 4. 模块一加载即执行（核心：先导入本模块的脚本都会自动拿到正确的环境）
# =========================================================
ensure_dirs()
apply_env(verbose=False)


__all__ = [
    "PROJECT_ROOT",
    "REQUIRED_DIRS",
    "REQUIRED_ENV_VARS",
    "ensure_dirs",
    "apply_env",
]
