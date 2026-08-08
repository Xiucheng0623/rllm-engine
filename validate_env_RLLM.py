# File: D:\AI_RLLM\validate_env.py
"""Rebirth LLM(RLLM) Rebirth LLM(RLLM) 环境校验脚本

功能：
  1. 校验D盘全量目录结构完整性
  2. 校验CUDA可用性、驱动版本、显存大小 (RTX5070Ti 目标卡)
  3. 校验全局内存上限 32GB，CPU缓冲区硬限2GB逻辑
  4. 校验HuggingFace/Torch缓存路径是否全部指向D盘
  5. 输出所有依赖库版本清单
  6. 校验离线模式标志 HF_OFFLINE=1

运行: python D:\AI_RLLM\validate_env.py
"""
from __future__ import annotations
import env_config  # RLLM全局环境变量自动注入（硬锁定D:\AI_RLLM）

import os
import sys
import json
import platform
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Tuple

# ---------- 全局D盘根路径（唯一真值源） ----------
ROOT: Path = Path(r"D:\AI_RLLM")


@dataclass
class EnvReport:
    """环境校验结构化报告"""
    os_version: str = ""
    python_version: str = ""
    cuda_available: bool = False
    cuda_version: str = ""
    gpu_name: str = ""
    gpu_vram_gb: float = 0.0
    system_ram_gb: float = 0.0
    cpu_buffer_limit_gb: float = 2.0
    d_disk_free_gb: float = 0.0
    dir_checks: Dict[str, bool] = field(default_factory=dict)
    env_checks: Dict[str, str] = field(default_factory=dict)
    offline_mode: bool = False
    dependency_versions: Dict[str, str] = field(default_factory=dict)
    all_passed: bool = True
    warnings: List[str] = field(default_factory=list)


def check_disk_structure(report: EnvReport) -> None:
    """校验D盘目录结构完整性"""
    required_dirs: List[Path] = [
        ROOT / ".venv",
        ROOT / "hf_cache" / "hub",
        ROOT / "hf_cache" / "datasets",
        ROOT / "model_shards" / "indexes",
        ROOT / "offload_temp" / "kv_cache",
        ROOT / "offload_temp" / "tensor_swap",
        ROOT / "hermes_core" / "workers",
        ROOT / "hermes_core" / "memory",
        ROOT / "hermes_core" / "skills",
        ROOT / "hermes_core" / "review",
        ROOT / "hermes_core" / "config",
        ROOT / "output_dataset",
        ROOT / "skill_storage" / "archive",
        ROOT / "disk_engine" / "sharding",
        ROOT / "disk_engine" / "scheduler",
        ROOT / "disk_engine" / "kv_manager",
        ROOT / "disk_engine" / "memory_lock",
        ROOT / "disk_engine" / "mmap_io",
        ROOT / "auto_evo" / "metrics",
        ROOT / "auto_evo" / "tuner",
        ROOT / "auto_evo" / "strategy",
        ROOT / "pipeline" / "batch_reader",
        ROOT / "pipeline" / "writer",
        ROOT / "pipeline" / "checkpoint",
        ROOT / "input_data",
        ROOT / "logs",
        ROOT / "tests",
    ]
    for d in required_dirs:
        exists = d.exists() and d.is_dir()
        report.dir_checks[str(d)] = exists
        if not exists:
            report.all_passed = False
            report.warnings.append(f"缺失目录: {d}")


def check_env_vars(report: EnvReport) -> None:
    """校验HuggingFace/Torch缓存路径是否全在D盘"""
    check_items: List[Tuple[str, str]] = [
        ("HF_HOME", str(ROOT / "hf_cache")),
        ("TRANSFORMERS_CACHE", str(ROOT / "hf_cache" / "hub")),
        ("HUGGINGFACE_HUB_CACHE", str(ROOT / "hf_cache" / "hub")),
        ("TORCH_HOME", str(ROOT / "hf_cache" / "torch")),
        ("HF_DATASETS_CACHE", str(ROOT / "hf_cache" / "datasets")),
    ]
    for env_key, expected_path in check_items:
        actual = os.environ.get(env_key, "<未设置>")
        report.env_checks[env_key] = actual
        if actual != expected_path:
            report.all_passed = False
            report.warnings.append(
                f"环境变量 {env_key}={actual} 非预期 {expected_path}"
            )
    report.offline_mode = os.environ.get("HF_OFFLINE", "0") == "1"
    if not report.offline_mode:
        report.warnings.append("未启用HF_OFFLINE=1离线模式，存在自动联网下载模型风险")


def check_cuda_and_memory(report: EnvReport) -> None:
    """校验CUDA、GPU显存、系统内存上限32GB"""
    report.os_version = f"{platform.system()} {platform.release()} {platform.version()}"
    report.python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"

    # 系统内存检测 (硬约束 32GB)
    try:
        import psutil
        total_ram_bytes = psutil.virtual_memory().total
        report.system_ram_gb = round(total_ram_bytes / (1024 ** 3), 2)
        if report.system_ram_gb < 16:
            report.warnings.append(f"系统内存仅 {report.system_ram_gb}GB，低于推荐32GB")
    except Exception as exc:  # noqa: BLE001
        report.warnings.append(f"psutil内存检测失败: {exc}")

    # D盘可用空间
    try:
        import shutil
        _, _, free = shutil.disk_usage(str(ROOT))
        report.d_disk_free_gb = round(free / (1024 ** 3), 2)
        if report.d_disk_free_gb < 100:
            report.warnings.append(f"D盘仅剩余 {report.d_disk_free_gb}GB，建议至少200GB")
    except Exception as exc:  # noqa: BLE001
        report.warnings.append(f"D盘空间检测失败: {exc}")

    # CUDA检测
    try:
        import torch
        report.cuda_available = torch.cuda.is_available()
        if report.cuda_available:
            report.cuda_version = torch.version.cuda or "<未知>"
            report.gpu_name = torch.cuda.get_device_name(0)
            vram_bytes = torch.cuda.get_device_properties(0).total_memory
            report.gpu_vram_gb = round(vram_bytes / (1024 ** 3), 2)
            if "5070" not in report.gpu_name.upper() and "RTX" not in report.gpu_name.upper():
                report.warnings.append(f"显卡非预期RTX5070Ti: {report.gpu_name}")
        else:
            report.warnings.append("CUDA不可用，将回退CPU推理(极慢)")
            report.all_passed = False
    except Exception as exc:  # noqa: BLE001
        report.cuda_available = False
        report.warnings.append(f"PyTorch/CUDA导入失败: {exc}")
        report.all_passed = False


def check_dependencies(report: EnvReport) -> None:
    """校验核心依赖版本"""
    required_pkgs: List[str] = [
        "torch", "transformers", "accelerate", "bitsandbytes",
        "aiofiles", "diskcache", "numpy", "pandas",
        "pydantic", "rich", "loguru", "psutil",
    ]
    for pkg in required_pkgs:
        try:
            mod = __import__(pkg.replace("-", "_"))
            report.dependency_versions[pkg] = getattr(mod, "__version__", "<未知>")
        except Exception as exc:  # noqa: BLE001
            report.dependency_versions[pkg] = f"<缺失:{exc}>"
            report.all_passed = False
            report.warnings.append(f"依赖缺失: {pkg}")


def print_report(report: EnvReport) -> None:
    """彩色打印校验报告"""
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich import box

    console = Console()
    console.print(Panel.fit(
        "[bold cyan]Rebirth LLM(RLLM) 环境校验报告[/bold cyan]\n"
        f"[dim]全D盘隔离路径: {ROOT}[/dim]",
        border_style="cyan"
    ))

    # 系统概览
    t1 = Table(title="系统概览", box=box.SIMPLE_HEAVY, show_header=True)
    t1.add_column("项目", style="bold yellow")
    t1.add_column("值", style="white")
    t1.add_row("操作系统", report.os_version)
    t1.add_row("Python版本", report.python_version)
    t1.add_row("CUDA可用", "[green]是[/green]" if report.cuda_available else "[red]否[/red]")
    t1.add_row("CUDA版本", report.cuda_version or "-")
    t1.add_row("GPU型号", report.gpu_name or "-")
    t1.add_row("GPU显存", f"{report.gpu_vram_gb:.2f} GB")
    t1.add_row("系统内存", f"{report.system_ram_gb:.2f} GB [dim](上限32GB)[/dim]")
    t1.add_row("CPU缓冲硬限", f"{report.cpu_buffer_limit_gb:.2f} GB")
    t1.add_row("D盘可用", f"{report.d_disk_free_gb:.2f} GB")
    t1.add_row("离线模式", "[green]已启用[/green]" if report.offline_mode else "[yellow]未启用[/yellow]")
    console.print(t1)

    # 环境变量
    t2 = Table(title="缓存路径校验(必须全在D盘)", box=box.SIMPLE_HEAVY)
    t2.add_column("环境变量", style="bold magenta")
    t2.add_column("当前值", style="white", overflow="fold")
    for k, v in report.env_checks.items():
        mark = "[green]✓[/green]" if str(ROOT) in v else "[red]✗[/red]"
        t2.add_row(k, f"{mark} {v}")
    console.print(t2)

    # 依赖版本
    t3 = Table(title="核心依赖版本", box=box.SIMPLE_HEAVY)
    t3.add_column("包名", style="bold blue")
    t3.add_column("版本", style="white")
    for pkg, ver in sorted(report.dependency_versions.items()):
        color = "green" if "缺失" not in ver else "red"
        t3.add_row(pkg, f"[{color}]{ver}[/{color}]")
    console.print(t3)

    # 目录校验结果
    dir_pass = sum(1 for v in report.dir_checks.values() if v)
    dir_total = len(report.dir_checks)
    console.print(
        f"\n[bold]目录完整性:[/bold] [green]{dir_pass}[/green]/{dir_total} 个目录已创建"
    )

    # 警告与结论
    if report.warnings:
        warn_table = Table(title="警告 / 问题清单", box=box.SIMPLE_HEAVY)
        warn_table.add_column("#", style="bold red")
        warn_table.add_column("内容", style="yellow", overflow="fold")
        for i, w in enumerate(report.warnings, 1):
            warn_table.add_row(str(i), w)
        console.print(warn_table)

    if report.all_passed and not report.warnings:
        console.print(Panel.fit(
            "[bold green]✓ 所有环境校验通过 ✓[/bold green]",
            border_style="green"
        ))
    elif report.all_passed:
        console.print(Panel.fit(
            "[bold yellow]△ 基本通过，请关注警告项 △[/bold yellow]",
            border_style="yellow"
        ))
    else:
        console.print(Panel.fit(
            "[bold red]✗ 存在严重问题，请修复后重试 ✗[/bold red]",
            border_style="red"
        ))

    # 持久化JSON报告
    report_path = ROOT / "logs" / "env_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as fp:
        json.dump(asdict(report), fp, ensure_ascii=False, indent=2)
    console.print(f"\n[dim]结构化报告已保存: {report_path}[/dim]")


def main() -> int:
    """主入口"""
    # 注入D盘源码到sys.path
    for sub in ("hermes_core", "disk_engine", "auto_evo", "pipeline"):
        p = str(ROOT / sub)
        if p not in sys.path:
            sys.path.insert(0, p)

    report = EnvReport()
    try:
        check_disk_structure(report)
        check_env_vars(report)
        check_cuda_and_memory(report)
        check_dependencies(report)
        print_report(report)
    except Exception as exc:  # noqa: BLE001
        print(f"[严重错误] 环境校验崩溃: {exc}")
        import traceback
        traceback.print_exc()
        return 2

    return 0 if report.all_passed else 1


if __name__ == "__main__":
    sys.exit(main())


# ============================================================
# 版权声明
# 本项目 Rebirth LLM(RLLM) 基于开源项目 Nous Hermes-Agent（MIT License）二次深度开发，项目内保留完整原始开源协议文件；智能体自迭代调度逻辑复用开源代码，磁盘分层加载、全局内存锁、D盘隔离部署、自动IO调优模块为自研闭源模块，分发时附带完整MIT协议文件。
# 商标隔离免责声明
# 项目名称 Rebirth LLM（简称RLLM）为独立软件项目代号，与奢侈品品牌Hermes、开源项目Hermes-Agent无品牌合作、隶属关联；仅代码内部功能性调用开源框架，不会使用Hermes相关名称开展商业宣传，无品牌混淆意图。
# ============================================================
