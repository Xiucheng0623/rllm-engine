# -*- coding: utf-8 -*-
# File: D:\AI_RLLM\rllm_manager.py
"""RLLM Engine 模型管理器

管理模型生命周期: 下载 → 4bit量化分片 → 验证.

用法:
    python rllm_manager.py download mixtral-8x7b     # 下载原始模型
    python rllm_manager.py shard mixtral-8x7b         # 4bit 量化分片
    python rllm_manager.py verify mixtral-8x7b        # 验证分片完整性
    python rllm_manager.py info mixtral-8x7b          # 查看模型信息
    python rllm_manager.py list                       # 列出已安装模型
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

_RLLM_ROOT = Path(r"D:\AI_RLLM")
if str(_RLLM_ROOT) not in sys.path:
    sys.path.insert(0, str(_RLLM_ROOT))

# 已知模型注册表
KNOWN_MODELS: Dict[str, Dict[str, Any]] = {
    "mixtral-8x7b": {
        "hf_repo": "NousResearch/Mixtral-8x7B-Instruct-v0.1",
        "desc": "Mixtral-8x7B-Instruct (47B MoE, Apache 2.0)",
        "raw_dir": str(_RLLM_ROOT / "rllm_model_shards" / "mixtral_8x7b_raw"),
        "shard_dir": str(
            _RLLM_ROOT / "rllm_model_shards" / "mixtral_8x7b_v4_shards"
        ),
        "min_disk_gb": 100,
        "min_vram_gb": 8,
    },
    "mixtral-8x7b-instruct": {
        "hf_repo": "NousResearch/Mixtral-8x7B-Instruct-v0.1",
        "desc": "Mixtral-8x7B-Instruct-v0.1 (47B MoE, Apache 2.0)",
        "raw_dir": str(_RLLM_ROOT / "rllm_model_shards" / "mixtral_8x7b_raw"),
        "shard_dir": str(
            _RLLM_ROOT / "rllm_model_shards" / "mixtral_8x7b_v4_shards"
        ),
        "min_disk_gb": 100,
        "min_vram_gb": 8,
    },
}

SHARD_SCRIPT = str(_RLLM_ROOT / "tests" / "shard_mixtral_4bit.py")


def parse_args() -> argparse.Namespace:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="RLLM Engine 模型管理器",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="子命令")

    # download
    dl = subparsers.add_parser("download", help="下载原始模型 (HF)")
    dl.add_argument("model", help="模型名称")
    dl.add_argument(
        "--mirror",
        action="store_true",
        help="使用 HF Mirror 加速 (国内推荐)",
    )

    # shard
    sh = subparsers.add_parser("shard", help="4bit 量化分片")
    sh.add_argument("model", help="模型名称")

    # verify
    vf = subparsers.add_parser("verify", help="验证分片完整性")
    vf.add_argument("model", help="模型名称")

    # info
    info = subparsers.add_parser("info", help="查看模型信息")
    info.add_argument("model", help="模型名称")

    # list
    subparsers.add_parser("list", help="列出已安装模型")

    return parser.parse_args()


def cmd_download(model_name: str, use_mirror: bool = False) -> int:
    """下载原始模型

    Args:
        model_name: 模型名称
        use_mirror: 是否使用 HF Mirror

    Returns:
        退出码
    """
    if model_name not in KNOWN_MODELS:
        print(f"未知模型: {model_name}")
        print(f"可用模型: {', '.join(KNOWN_MODELS)}")
        return 1

    info = KNOWN_MODELS[model_name]
    raw_dir = Path(info["raw_dir"])

    # 检查是否已下载
    if (raw_dir / "config.json").exists():
        files = list(raw_dir.glob("*.safetensors"))
        if files:
            total_size = sum(f.stat().st_size for f in files) / 1024**3
            print(f"模型已下载: {raw_dir} ({total_size:.1f}GB)")
            return 0

    print(f"\n{'='*60}")
    print(f"  下载模型: {info['hf_repo']}")
    print(f"  目标: {raw_dir}")
    print(f"  预计大小: ~93GB")
    print(f"{'='*60}\n")

    raw_dir.mkdir(parents=True, exist_ok=True)

    # 构建 huggingface-cli download 命令
    env_prefix = ""
    if use_mirror:
        env_prefix = "set HF_ENDPOINT=https://hf-mirror.com && "

    cmd = (
        f'{env_prefix}'
        f'huggingface-cli download {info["hf_repo"]} '
        f'--local-dir "{raw_dir}" '
        f'--local-dir-use-symlinks False '
        f'--resume-download '
        f'--include "*.json" "*.safetensors" "tokenizer.*" '
    )

    print(f"执行: {cmd[:120]}...")
    import subprocess

    result = subprocess.run(
        cmd,
        shell=True,
        cwd=str(_RLLM_ROOT),
    )

    if result.returncode != 0:
        print(f"\n下载失败 (exit={result.returncode})")
        print("提示: 国内网络推荐使用 --mirror 参数")
        return result.returncode

    print(f"\n下载完成: {raw_dir}")
    return 0


def cmd_shard(model_name: str) -> int:
    """4bit 量化分片

    从 safetensors 按需加载 → 4bit NF4 量化 → 写入 D 盘。
    内存峰值 ~1.5GB, 不爆 32GB RAM。

    Args:
        model_name: 模型名称

    Returns:
        退出码
    """
    if model_name not in KNOWN_MODELS:
        print(f"未知模型: {model_name}")
        return 1

    info = KNOWN_MODELS[model_name]
    raw_dir = Path(info["raw_dir"])

    if not (raw_dir / "config.json").exists():
        print(f"原始模型未下载: {raw_dir}")
        print(f"请先运行: python rllm_manager.py download {model_name}")
        return 1

    if not Path(SHARD_SCRIPT).exists():
        print(f"分片脚本缺失: {SHARD_SCRIPT}")
        return 1

    print(f"\n{'='*60}")
    print(f"  Mixtral-8x7B 4bit 量化分片")
    print(f"  预计: ~25GB ~3 分钟")
    print(f"{'='*60}\n")

    t0 = time.time()
    import subprocess

    env = {"HF_HOME": str(_RLLM_ROOT / "hf_cache")}

    result = subprocess.run(
        [sys.executable, SHARD_SCRIPT],
        env={**__import__("os").environ, **env},
        cwd=str(_RLLM_ROOT),
    )

    elapsed = time.time() - t0

    if result.returncode == 0:
        print(f"\n分片完成! 耗时 {elapsed:.1f}s")
        return 0
    else:
        print(f"\n分片失败 (exit={result.returncode})")
        return result.returncode


def cmd_verify(model_name: str) -> int:
    """验证分片完整性

    Args:
        model_name: 模型名称

    Returns:
        退出码 (0=完整, 1=不完整)
    """
    if model_name not in KNOWN_MODELS:
        print(f"未知模型: {model_name}")
        return 1

    info = KNOWN_MODELS[model_name]
    shard_dir = Path(info["shard_dir"])

    if not shard_dir.exists():
        print(f"分片目录不存在: {shard_dir}")
        return 1

    # 检查 index.json
    index_path = shard_dir / "index.json"
    if not index_path.exists():
        print(f"索引缺失: {index_path}")
        return 1

    with open(index_path, "r", encoding="utf-8") as f:
        index = json.load(f)

    num_layers = index.get("num_layers", 32)
    num_experts = index.get("num_experts_per_layer", 8)

    print(f"\n{'='*60}")
    print(f"  验证分片: {model_name}")
    print(f"  层数: {num_layers}, 专家/层: {num_experts}")
    print(f"{'='*60}\n")

    missing: List[str] = []
    total_files = 0

    # 共享层
    shared_dir = shard_dir / "shared"
    for fname in ["embed_tokens.pt", "norm.pt", "lm_head.pt"]:
        if not (shared_dir / fname).exists():
            missing.append(f"shared/{fname}")
        else:
            total_files += 1

    # 每层
    for layer_idx in range(num_layers):
        layer_dir = shard_dir / f"layer_{layer_idx:02d}"

        # attention + gate
        for fname in ["attention.pt", "gate.pt"]:
            if not (layer_dir / fname).exists():
                missing.append(f"layer_{layer_idx:02d}/{fname}")
            else:
                total_files += 1

        # 专家
        experts_dir = layer_dir / "experts"
        for expert_idx in range(num_experts):
            expert_file = experts_dir / f"expert_{expert_idx}.pt"
            if not expert_file.exists():
                missing.append(
                    f"layer_{layer_idx:02d}/experts/expert_{expert_idx}.pt"
                )
            else:
                total_files += 1

    if missing:
        print(f"[FAIL] 缺失 {len(missing)} 个文件:")
        for m in missing[:10]:
            print(f"  - {m}")
        if len(missing) > 10:
            print(f"  ... 及其他 {len(missing) - 10} 个")
        return 1

    # 统计大小
    total_size = sum(
        f.stat().st_size
        for f in shard_dir.rglob("*.pt")
        if f.is_file()
    )
    print(f"[OK] {total_files} 个文件完整, {total_size/1024**3:.1f}GB")
    return 0


def cmd_info(model_name: str) -> int:
    """查看模型信息"""
    if model_name not in KNOWN_MODELS:
        print(f"未知模型: {model_name}")
        return 1

    info = KNOWN_MODELS[model_name]
    raw_dir = Path(info["raw_dir"])
    shard_dir = Path(info["shard_dir"])

    print(f"\n模型: {model_name}")
    print(f"  说明: {info['desc']}")
    print(f"  HF: {info['hf_repo']}")
    print(f"  原始目录: {raw_dir}")
    print(f"  分片目录: {shard_dir}")

    if (raw_dir / "config.json").exists():
        sf = list(raw_dir.glob("*.safetensors"))
        sz = sum(f.stat().st_size for f in sf) / 1024**3
        print(f"  原始模型: 已下载 ({sz:.1f}GB)")
    else:
        print(f"  原始模型: 未下载")

    idx = shard_dir / "index.json"
    if idx.exists():
        with open(idx, "r", encoding="utf-8") as f:
            index = json.load(f)
        print(f"  层数: {index.get('num_layers', '?')}")
        print(f"  专家/层: {index.get('num_experts_per_layer', '?')}")
        total_sz = sum(
            f.stat().st_size for f in shard_dir.rglob("*.pt") if f.is_file()
        ) / 1024**3
        print(f"  分片大小: {total_sz:.1f}GB")
    else:
        print(f"  分片: 未分片")

    return 0


def cmd_list() -> int:
    """列出已安装模型"""
    print(f"\n已注册模型:")
    for name, info in KNOWN_MODELS.items():
        raw_ok = (Path(info["raw_dir"]) / "config.json").exists()
        shard_ok = (Path(info["shard_dir"]) / "index.json").exists()
        status = []
        if raw_ok:
            status.append("原始 ✓")
        if shard_ok:
            status.append("分片 ✓")
        if not status:
            status.append("未安装")
        print(f"  {name}: {' | '.join(status)} - {info['desc']}")
    return 0


def main() -> int:
    """主入口"""
    args = parse_args()

    if args.command == "download":
        return cmd_download(args.model, args.mirror)
    elif args.command == "shard":
        return cmd_shard(args.model)
    elif args.command == "verify":
        return cmd_verify(args.model)
    elif args.command == "info":
        return cmd_info(args.model)
    elif args.command == "list":
        return cmd_list()
    else:
        print("请指定子命令: download | shard | verify | info | list")
        print("例如: python rllm_manager.py info mixtral-8x7b")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
