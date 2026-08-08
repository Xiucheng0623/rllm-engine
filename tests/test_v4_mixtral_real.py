# -*- coding: utf-8 -*-
# File: D:\AI_RLLM\tests\test_v4_mixtral_real.py
"""v4 真实 Mixtral-8x7B 推理测试

测试内容:
  1. 从 v4 4bit 分片目录加载
  2. 中文 prompt 推理
  3. 测速 (prefill + decode)
  4. 对比 v3 (23.6 tok/s)

使用:
  D:\\AI_RLLM\\.venv\\Scripts\\python.exe D:\\AI_RLLM\\tests\\test_v4_mixtral_real.py
"""
from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

RLLM_ROOT = Path(r"D:\AI_RLLM")
if str(RLLM_ROOT) not in sys.path:
    sys.path.insert(0, str(RLLM_ROOT))

import torch
from loguru import logger

from rllm_agent_core import LOG_DIR
from rllm_disk_engine.moe_orchestrator import MoEOrchestrator

logger.add(
    LOG_DIR / "test_v4_real_{time}.log",
    rotation="50 MB",
    retention="7 days",
    encoding="utf-8",
)

SHARD_DIR = Path(r"D:\AI_RLLM\rllm_model_shards\mixtral_8x7b_v4_shards")
RAW_DIR = Path(r"D:\AI_RLLM\rllm_model_shards\mixtral_8x7b_raw")

# 测试 prompt
TEST_PROMPTS = [
    "你好，请用中文介绍一下你自己。",
    "什么是混合专家模型（MoE）？请简要解释。",
    "写一首关于秋天的小诗。",
]


async def run_test() -> int:
    """运行真实 Mixtral 推理测试"""
    print("=" * 60)
    print("  v4 真实 Mixtral-8x7B 推理测试")
    print(f"  分片目录: {SHARD_DIR}")
    print("=" * 60)

    # 检查分片
    if not (SHARD_DIR / "index.json").exists():
        print("[FAIL] 分片不存在, 请先运行 shard_mixtral_4bit.py")
        return 1

    # 加载 tokenizer
    print("\n[1/4] 加载 tokenizer...")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(RAW_DIR))
    print(f"  vocab_size: {tokenizer.vocab_size}")

    # 初始化 MoEOrchestrator
    print("\n[2/4] 初始化 MoEOrchestrator...")
    orchestrator = MoEOrchestrator(
        shard_dir=SHARD_DIR,
        reserve_gb=2.0,
        top_n_hot_experts=32,
        prefetch_candidates=8,
    )
    await orchestrator.initialize()
    print("  [OK] 初始化完成")

    # 逐个测试 prompt
    print("\n[3/4] 开始推理测试...")
    results = []

    for i, prompt in enumerate(TEST_PROMPTS):
        print(f"\n  --- 测试 {i+1}/{len(TEST_PROMPTS)} ---")
        print(f"  Prompt: {prompt}")

        # 编码 (Mistral instruct format)
        messages = [{"role": "user", "content": prompt}]
        try:
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            text = f"[INST] {prompt} [/INST]"

        input_ids = tokenizer.encode(text, return_tensors="pt")[0].tolist()
        print(f"  Input tokens: {len(input_ids)}")

        # 生成
        t0 = time.time()
        generated_ids, stats = await orchestrator.generate(
            prompt_ids=input_ids,
            max_new_tokens=64,
            temperature=0.7,
            top_p=0.9,
        )
        total_s = time.time() - t0

        # 解码
        output_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        print(f"  输出: {output_text[:200]}")
        print(f"  生成 {len(generated_ids)} tokens")
        print(f"  prefill: {stats['prefill_seconds']:.2f}s")
        print(f"  decode: {stats['total_decode_seconds']:.2f}s")
        print(f"  速度: {stats['avg_tok_per_s']:.2f} tok/s")

        results.append({
            "prompt": prompt,
            "output": output_text,
            "tokens": len(generated_ids),
            "tok_per_s": stats["avg_tok_per_s"],
            "prefill_s": stats["prefill_seconds"],
            "decode_s": stats["total_decode_seconds"],
        })

    # 总结
    print(f"\n{'=' * 60}")
    print("  [4/4] 测试总结")
    print("=" * 60)
    print(f"  {'Prompt':<30} | {'速度':>10} | {'prefill':>8}")
    print(f"  {'-'*30}-+-{'-'*10}-+-{'-'*8}")
    for r in results:
        short = r["prompt"][:28]
        print(f"  {short:<30} | {r['tok_per_s']:>8.1f} t/s | {r['prefill_s']:>6.2f}s")

    avg_speed = sum(r["tok_per_s"] for r in results) / len(results)
    print(f"\n  平均速度: {avg_speed:.2f} tok/s")
    print(f"  v3 基准: 23.6 tok/s (7B 4bit)")
    print(f"  v4 ({'快' if avg_speed > 23.6 else '慢'} {abs(avg_speed - 23.6):.1f} tok/s)")

    # v4 架构统计
    print(f"\n  --- v4 架构统计 ---")
    print(f"  模型: Mixtral-8x7B (47B 参数, 4bit NF4)")
    print(f"  专家: 32 层 × 8 专家 = 256 个")
    print(f"  每 token 激活: 2 专家 + 共享层 (~23% 参数)")

    return 0


def main() -> int:
    if not torch.cuda.is_available():
        print("[FAIL] CUDA 不可用")
        return 1
    print(f"CUDA: {torch.cuda.get_device_name(0)}")
    return asyncio.run(run_test())


if __name__ == "__main__":
    sys.exit(main())
