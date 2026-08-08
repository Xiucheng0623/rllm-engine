# -*- coding: utf-8 -*-
# File: D:\AI_RLLM\rllm_engine\demo.py
"""RLLM Engine 快速启动 Demo

安装后一键验证效果, 无需任何配置.

用法:
    rllm-demo                     # 自动检测模型
    rllm-demo --model "Nous-Hermes-2-Mistral-7B-DPO"
    rllm-demo --prompt "写一首诗"
    rllm-demo --interactive        # 交互模式
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

# ANSI 颜色 (跨平台)
C_BOLD = "\033[1m"
C_GREEN = "\033[92m"
C_YELLOW = "\033[93m"
C_CYAN = "\033[96m"
C_RED = "\033[91m"
C_GRAY = "\033[90m"
C_RESET = "\033[0m"

# ASCII Banner
BANNER = r"""
  ██████╗ ██╗     ██╗     ███╗   ███╗
  ██╔══██╗██║     ██║     ████╗ ████║
  ██████╔╝██║     ██║     ██╔████╔██║
  ██╔══██╗██║     ██║     ██║╚██╔╝██║
  ██║  ██║███████╗███████╗██║ ╚═╝ ██║
  ╚═╝  ╚═╝╚══════╝╚══════╝╚═╝     ╚═╝

  Local LLM Inference Engine
  消费级 GPU 运行大模型
"""

# 默认 Demo Prompts
DEMO_PROMPTS = [
    ("自我介绍", "请用一句话介绍自己。"),
    ("简单问答", "What is the capital of France?"),
    ("创作", "请用中文写一首关于秋天的短诗，四行。"),
    ("编程", "用 Python 写一个快速排序函数，只输出代码。"),
    ("翻译", "将以下英文翻译成中文: Machine learning is transforming every industry."),
]


def _detect_model() -> Optional[str]:
    """自动检测可用的模型"""
    candidates: List[Tuple[str, Path]] = []

    # 从 rllm_engine.platform_paths 获取根目录
    from rllm_engine.platform_paths import get_rllm_home

    home = get_rllm_home()
    model_dirs = [
        home / "models",
        home / "rllm_model_shards",
        home / "rllm_model_shards" / "_raw",
        home / "rllm_model_shards" / "_quantized",
    ]

    for base in model_dirs:
        if not base.exists():
            continue
        for sub in base.iterdir():
            if not sub.is_dir():
                continue
            if (sub / "config.json").exists():
                candidates.append((sub.name, sub))
            # 递归子目录
            for deep in sub.iterdir():
                if deep.is_dir() and (deep / "config.json").exists():
                    candidates.append((f"{sub.name}/{deep.name}", deep))

    if not candidates:
        return None
    return candidates[0][0]


def _get_version() -> str:
    try:
        from rllm_engine import __version__
        return __version__
    except ImportError:
        return "1.0.0"


def print_header(title: str) -> None:
    """打印带颜色的标题"""
    print(f"\n{C_BOLD}{C_CYAN}  📋 {title}{C_RESET}")
    print(f"  {'─' * 50}")


def print_stat(label: str, value: str, good: bool = True) -> None:
    """打印统计项"""
    color = C_GREEN if good else C_YELLOW
    print(f"  {label:<18} {color}{value}{C_RESET}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="RLLM Engine 快速 Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  rllm-demo                                      # 自动检测模型并运行 Demo
  rllm-demo --model Nous-Hermes-2-Mistral-7B-DPO # 指定模型
  rllm-demo --prompt "写一首诗"                   # 自定义 prompt
  rllm-demo --interactive                        # 交互模式
        """,
    )
    parser.add_argument(
        "--model", "-m", default="", help="模型名称或路径"
    )
    parser.add_argument(
        "--prompt", "-p", default="", help="自定义 prompt"
    )
    parser.add_argument(
        "--max-tokens", type=int, default=128, help="最大生成 token 数"
    )
    parser.add_argument(
        "--interactive", "-i", action="store_true", help="交互模式"
    )
    parser.add_argument(
        "--list-models", action="store_true", help="列出可用模型"
    )
    return parser.parse_args()


def run_demo(engine, prompt: str, label: str) -> Tuple[float, int]:
    """运行单个 Demo prompt"""
    print(f"\n  {C_BOLD}>>> {label}{C_RESET}")
    print(f"  {C_GRAY}{prompt}{C_RESET}")

    t0 = time.time()
    try:
        result = engine.generate(prompt)
    except Exception as e:
        print(f"  {C_RED}[错误] {e}{C_RESET}")
        return 0.0, 0
    elapsed = time.time() - t0

    tok_count = len(result)
    tps = tok_count / max(elapsed, 0.001)

    # 截断过长输出
    display = result[:300]
    if len(result) > 300:
        display += "..."

    print(f"  {C_GREEN}{display}{C_RESET}")
    print(f"  {C_GRAY}({tps:.1f} tok/s, {elapsed:.1f}s){C_RESET}")

    return tps, tok_count


def main() -> int:
    args = parse_args()

    # Banner
    print(f"{C_BOLD}{C_CYAN}{BANNER}{C_RESET}")
    print(f"  v{_get_version()}  |  消费级 GPU 运行大模型")
    print(f"  {'=' * 50}")

    # 列出模型
    if args.list_models:
        models = _list_all_models()
        if not models:
            print(f"\n  {C_YELLOW}未检测到本地模型.{C_RESET}")
            print(f"  请下载模型到 {get_rllm_home()}/models/")
        else:
            print(f"\n  {C_BOLD}可用模型:{C_RESET}")
            for name, path in models:
                sz = sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1024**3
                print(f"    • {C_GREEN}{name}{C_RESET} ({sz:.1f}GB)")
        return 0

    # 确定模型
    model_name = args.model or _detect_model()
    if not model_name:
        print(f"\n  {C_YELLOW}未检测到本地模型.{C_RESET}")
        print(f"  {C_GRAY}请先下载模型或直接指定路径.{C_RESET}")
        print(f"  {C_GRAY}例如: rllm-demo --model /path/to/model{C_RESET}")
        return 1

    # 系统信息
    print_header("系统信息")
    from rllm_engine.platform_paths import get_rllm_home

    print_stat("模型", model_name)
    print_stat("根目录", str(get_rllm_home()))
    print_stat("平台", sys.platform)

    try:
        import torch
        if torch.cuda.is_available():
            gpu = torch.cuda.get_device_name(0)
            vram = getattr(torch.cuda.get_device_properties(0), "total_memory", 0) / 1024**3
            print_stat("GPU", f"{gpu} ({vram:.1f}GB)")
        else:
            print_stat("GPU", f"{C_RED}未检测到 (需要 NVIDIA GPU){C_RESET}", False)
            return 1
    except ImportError:
        print_stat("GPU", f"{C_RED}PyTorch 未安装{C_RESET}", False)
        return 1

    # 加载模型
    print_header("加载模型")
    from rllm_engine import RLLMEngine

    engine = RLLMEngine(
        model_name_or_path=model_name,
        max_new_tokens=args.max_tokens,
    )

    t_load = time.time()
    try:
        engine.load()
    except FileNotFoundError as e:
        print(f"\n  {C_RED}[错误] 模型未找到: {e}{C_RESET}")
        return 1
    except Exception as e:
        print(f"\n  {C_RED}[错误] {e}{C_RESET}")
        return 1
    load_s = time.time() - t_load

    vram_idle = 0.0
    try:
        import torch
        vram_idle = torch.cuda.memory_allocated() / 1024**3
    except Exception:
        pass

    print(f"\n  {C_GREEN}模型已就绪!{C_RESET}")
    print_stat("加载时间", f"{load_s:.1f}s")
    print_stat("VRAM 占用", f"{vram_idle:.1f}GB")
    print_stat("引擎类型", engine.model_type)

    # 自定义 prompt 模式
    if args.prompt:
        print_header("自定义 Prompt")
        run_demo(engine, args.prompt, "Prompt")
        engine.unload()
        return 0

    # 交互模式
    if args.interactive:
        engine.chat()
        engine.unload()
        return 0

    # Demo 模式
    print_header(f"Demo 演示 ({min(3, len(DEMO_PROMPTS))} 个场景)")

    speeds: List[float] = []
    total_tokens = 0
    total_time = 0.0

    for label, prompt in DEMO_PROMPTS[:3]:
        tps, tok_count = run_demo(engine, prompt, label)
        if tps > 0:
            speeds.append(tps)
            total_tokens += tok_count

    # 汇总
    print_header("性能汇总")

    avg_speed = sum(speeds) / max(len(speeds), 1)
    peak_speed = max(speeds) if speeds else 0

    print_stat("平均速度", f"{avg_speed:.1f} tok/s")
    print_stat("峰值速度", f"{peak_speed:.1f} tok/s")
    print_stat("VRAM", f"{vram_idle:.1f}GB ({'4bit' if '4' in str(engine.config.quant_bits) else str(engine.config.quant_bits)+'bit'})")

    # 对比
    print(f"\n  {C_BOLD}与传统方案对比:{C_RESET}")
    print(f"  {'─' * 45}")
    print(f"  {'指标':<16} {'传统 GPU':<15} {'RLLM Engine':<15}")
    print(f"  {'─' * 45}")
    print(f"  {'VRAM':<16} {'14GB (FP16)':<15} {C_GREEN}{vram_idle:.1f}GB{C_RESET}")
    print(f"  {'速度':<16} {'~20 tok/s':<15} {C_GREEN}{avg_speed:.1f} tok/s{C_RESET}")
    print(f"  {'GPU 门槛':<16} {'16GB+':<15} {C_GREEN}8GB+{C_RESET}")
    print(f"  {'─' * 45}")

    # 下一步
    print(f"\n  {C_BOLD}🎉 Demo 完成!{C_RESET}")
    print(f"  {C_GRAY}下一步:{C_RESET}")
    print(f"  {C_GRAY}  • python -c \"from rllm_engine import RLLMEngine; RLLMEngine('{model_name}').load().chat()\"{C_RESET}")
    print(f"  {C_GRAY}  • rllm-demo --interactive  # 交互对话{C_RESET}")
    print(f"  {C_GRAY}  • rllm-demo --prompt \"你的问题\"  # 自定义提问{C_RESET}")

    engine.unload()
    return 0


def _list_all_models():
    """列出所有可用模型"""
    from rllm_engine.platform_paths import get_rllm_home

    home = get_rllm_home()
    models = []

    for base in [
        home / "models",
        home / "rllm_model_shards",
    ]:
        if not base.exists():
            continue
        for sub in base.rglob("config.json"):
            parent = sub.parent
            models.append((parent.name, parent))

    return models


if __name__ == "__main__":
    raise SystemExit(main())
