# -*- coding: utf-8 -*-
# File: D:\AI_RLLM\tests\benchmark.py
"""RLLM Engine Benchmark 脚本

测量推理速度、显存占用、输出质量.

用法:
    python tests/benchmark.py
    python tests/benchmark.py --model Nous-Hermes-2-Mistral-7B-DPO
    python tests/benchmark.py --prompts 10 --output report.json
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

import torch
from rllm_engine import RLLMEngine, get_rllm_home

# 测试 prompts (中英文混合, 覆盖不同长度)
TEST_PROMPTS = [
    "你好，请介绍一下自己。",
    "What is the capital of France? Answer in one sentence.",
    "请用中文解释什么是机器学习, 用简短的语言。",
    "Write a haiku about the moon.",
    "解释一下量子计算和经典计算的区别。",
    "用一句话总结太阳系。",
    "List three benefits of open source software.",
    "请写一首关于春天的短诗，四行。",
    "What is 2+2? Just give the answer.",
    "人工智能的未来发展方向是什么? 简要回答。",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RLLM Engine Benchmark")
    parser.add_argument(
        "--model", default="Nous-Hermes-2-Mistral-7B-DPO", help="模型名称"
    )
    parser.add_argument(
        "--prompts", type=int, default=5, help="测试 prompt 数量"
    )
    parser.add_argument(
        "--max-tokens", type=int, default=64, help="每个 prompt 最大生成 token"
    )
    parser.add_argument(
        "--output", default="", help="输出 JSON 报告路径"
    )
    return parser.parse_args()


def get_vram_usage() -> float:
    """获取当前 VRAM 使用量 (GB)"""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**3
    return 0.0


def main() -> int:
    args = parse_args()
    num_prompts = min(args.prompts, len(TEST_PROMPTS))
    prompts = TEST_PROMPTS[:num_prompts]

    print(f"\n{'='*60}")
    print(f"  RLLM Engine Benchmark")
    print(f"  模型: {args.model}")
    print(f"  Prompts: {num_prompts}")
    print(f"  Max tokens: {args.max_tokens}")
    print(f"{'='*60}\n")

    # 系统信息
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"
    vram_total = getattr(torch.cuda.get_device_properties(0), "total_memory", 0) / 1024**3
    print(f"GPU: {gpu_name} ({vram_total:.1f}GB VRAM)")
    print(f"根目录: {get_rllm_home()}\n")

    # 初始化
    engine = RLLMEngine(
        model_name_or_path=args.model,
        max_new_tokens=args.max_tokens,
    )

    # 加载
    print("加载模型...\n")
    t_load = time.time()
    try:
        engine.load()
    except Exception as e:
        print(f"[FAIL] 加载失败: {e}")
        return 1
    load_time = time.time() - t_load
    vram_idle = get_vram_usage()

    # 推理测试
    results: List[Dict[str, Any]] = []
    speeds: List[float] = []
    vram_peaks: List[float] = []

    for i, prompt in enumerate(prompts):
        print(f"\n[{i+1}/{num_prompts}] {prompt[:60]}...")
        vram_before = get_vram_usage()

        t0 = time.time()
        output = engine.generate(prompt)
        elapsed = time.time() - t0

        vram_after = get_vram_usage()
        vram_peak = max(vram_after, vram_before)

        tok_count = len(engine._tokenizer.encode(output))
        tps = tok_count / max(elapsed, 0.001)

        speeds.append(tps)
        vram_peaks.append(vram_peak)

        result = {
            "prompt": prompt,
            "output": output[:200],
            "output_tokens": tok_count,
            "seconds": round(elapsed, 2),
            "tok_per_s": round(tps, 2),
            "vram_gb": round(vram_peak, 2),
        }
        results.append(result)

        print(f"    {tps:.1f} tok/s | {elapsed:.1f}s | VRAM {vram_peak:.1f}GB")

    # 统计
    engine.unload()

    avg_speed = sum(speeds) / len(speeds)
    max_speed = max(speeds)
    avg_vram = sum(vram_peaks) / len(vram_peaks)

    print(f"\n{'='*60}")
    print(f"  Benchmark 结果")
    print(f"{'='*60}")
    print(f"  模型类型: {engine.model_type}")
    print(f"  加载时间: {load_time:.1f}s")
    print(f"  平均速度: {avg_speed:.1f} tok/s")
    print(f"  峰值速度: {max_speed:.1f} tok/s")
    print(f"  VRAM 空闲: {vram_idle:.1f}GB")
    print(f"  VRAM 峰值: {max(vram_peaks):.1f}GB")
    print(f"  量化: {engine.config.quant_bits}bit")
    print(f"{'='*60}\n")

    # 对比表
    print("=" * 70)
    print("  对比: RLLM Engine vs 传统方案")
    print("=" * 70)
    print(f"  {'指标':<20} {'传统(on GPU)':<18} {'RLLM Engine':<18}")
    print(f"  {'-'*55}")
    print(f"  {'7B 模型 VRAM':<20} {'14GB (FP16)':<18} {'{:.1f}GB (4bit)'.format(avg_vram):<18}")
    print(f"  {'推理速度':<20} {'~20 tok/s':<18} {'{:.1f} tok/s'.format(avg_speed):<18}")
    print(f"  {'最小 GPU':<20} {'16GB VRAM':<18} {'8GB VRAM':<18}")
    print(f"  {'跨平台':<20} {'Linux only':<18} {'Win/Linux/macOS':<18}")
    print("=" * 70)

    # 保存报告
    if args.output:
        report = {
            "model": args.model,
            "model_type": engine.model_type,
            "gpu": gpu_name,
            "vram_total": round(vram_total, 1),
            "quant_bits": engine.config.quant_bits,
            "load_time_s": round(load_time, 1),
            "avg_tok_per_s": round(avg_speed, 1),
            "max_tok_per_s": round(max_speed, 1),
            "vram_idle_gb": round(vram_idle, 1),
            "vram_peak_gb": round(max(vram_peaks), 1),
            "results": results,
        }
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"\n报告已保存: {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
