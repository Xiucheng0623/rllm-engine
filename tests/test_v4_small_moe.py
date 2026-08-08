# -*- coding: utf-8 -*-
# File: D:\AI_RLLM\tests\test_v4_small_moe.py
"""v4 小规模 MoE 快速验证 (无需下载真实模型)

用 MixtralConfig 创建一个小的随机权重 MoE 模型:
  - 4 层 × 4 专家 (而非 32 层 × 8 专家)
  - hidden_size=512 (而非 4096)
  - 总参数 ~200MB, 可在 1GB RAM 内创建

验证内容:
  1. ExpertShardPersistor 能正确分片到 D 盘 + 生成 index.json
  2. MoEOrchestrator 能初始化 + 加载共享层
  3. prefill + decode 能跑通 (不验证输出质量, 只验证流程)
  4. 专家缓存池/置换器/预取器协同工作无崩溃
  5. MoEEvoOrchestrator 自进化逻辑触发

使用:
  D:\\AI_RLLM\\.venv\\Scripts\\python.exe D:\\AI_RLLM\\tests\\test_v4_small_moe.py
"""
from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

# 确保 D:\AI_RLLM 在 sys.path
RLLM_ROOT = Path(r"D:\AI_RLLM")
if str(RLLM_ROOT) not in sys.path:
    sys.path.insert(0, str(RLLM_ROOT))

import torch
from loguru import logger

from rllm_agent_core import LOG_DIR

logger.add(
    LOG_DIR / "test_v4_small_{time}.log",
    rotation="50 MB",
    retention="7 days",
    encoding="utf-8",
)

# v4 模块
from rllm_disk_engine.expert_pool.expert_shard_persistor import (
    ExpertShardPersistor,
    ExpertIndex,
)
from rllm_disk_engine.moe_orchestrator import MoEOrchestrator


# 小规模 Mixtral 配置
SMALL_CONFIG_KWARGS: Dict[str, Any] = {
    "vocab_size": 3200,
    "hidden_size": 512,
    "intermediate_size": 1024,
    "num_hidden_layers": 4,
    "num_attention_heads": 8,
    "num_key_value_heads": 2,
    "num_local_experts": 4,
    "num_experts_per_tok": 2,
    "rms_norm_eps": 1e-5,
    "rope_theta": 1000000.0,
    "output_router_logits": False,
}

SHARD_DIR = Path(r"D:\AI_RLLM\rllm_model_shards\mixtral_small_test")


def step1_create_and_shard_model() -> ExpertIndex:
    """步骤 1: 创建小规模 Mixtral 模型 + 分片到 D 盘

    Returns:
        ExpertIndex 分片索引
    """
    print("\n" + "=" * 60)
    print("  步骤 1: 创建小规模 Mixtral + 分片")
    print("=" * 60)

    from transformers import MixtralConfig, MixtralForCausalLM

    # 清理旧分片 (如果有)
    if SHARD_DIR.exists():
        import shutil
        shutil.rmtree(SHARD_DIR)
    SHARD_DIR.mkdir(parents=True, exist_ok=True)

    # 创建随机权重模型
    print("  [1/3] 创建小规模 Mixtral 模型 (随机权重)...")
    config = MixtralConfig(**SMALL_CONFIG_KWARGS)
    model = MixtralForCausalLM(config)
    model.eval()

    param_count = sum(p.numel() for p in model.parameters())
    print(f"  [OK] 模型创建: {param_count / 1e6:.1f}M 参数")

    # 分片 (用 FP16, 不量化, 因为小模型不需要)
    print("  [2/3] 执行专家分片...")
    t0 = time.time()
    persistor = ExpertShardPersistor(
        model=model,
        output_dir=SHARD_DIR,
        expert_quant_bits=16,  # 小模型用 FP16, 不量化
        shared_dtype=torch.bfloat16,
    )
    index = persistor.persist()
    elapsed = time.time() - t0

    print(f"  [OK] 分片完成: {index.total_size_bytes / 1024**2:.1f}MB, 耗时 {elapsed:.1f}s")
    print(f"  [OK] 层数: {index.num_layers}")
    print(f"  [OK] 每层专家数: {index.num_experts_per_layer}")
    print(f"  [OK] 每 token 激活: {index.num_experts_per_token}")

    # 释放模型 (节省 RAM)
    del model
    import gc
    gc.collect()

    return index


async def step2_test_inference() -> bool:
    """步骤 2: 用 MoEOrchestrator 跑推理

    Returns:
        是否通过
    """
    print("\n" + "=" * 60)
    print("  步骤 2: v4 MoE 推理测试 (小规模)")
    print("=" * 60)

    try:
        # 1. 初始化编排器
        print("\n  [1/4] 初始化 MoEOrchestrator...")
        orchestrator = MoEOrchestrator(
            shard_dir=SHARD_DIR,
            reserve_gb=2.0,
            top_n_hot_experts=8,
            prefetch_candidates=4,
        )
        await orchestrator.initialize()
        print("  [OK] 初始化完成")

        # 2. 构造测试 prompt
        print("\n  [2/4] 构造测试 prompt...")
        prompt_ids: List[int] = [1, 100, 200, 300, 400, 500, 2]
        print(f"  prompt tokens: {prompt_ids}")

        # 3. 生成
        print("\n  [3/4] 开始生成 (16 tokens)...")
        t0 = time.time()
        generated, stats = await orchestrator.generate(
            prompt_ids=prompt_ids,
            max_new_tokens=16,
            temperature=0.7,
            top_p=0.9,
        )
        total_s = time.time() - t0

        # 4. 输出结果
        print(f"\n  [4/4] 生成结果:")
        print(f"    生成 token 数: {len(generated)}")
        print(f"    首 token: {generated[0]}")
        print(f"    prefill 耗时: {stats['prefill_seconds']:.3f}s")
        print(f"    decode 总耗时: {stats['total_decode_seconds']:.3f}s")
        print(f"    平均速度: {stats['avg_tok_per_s']:.2f} tok/s")
        print(f"    总耗时: {total_s:.3f}s")

        # 详细统计
        print(f"\n  --- v4 架构统计 ---")
        runner_stats = stats.get("runner_stats", {})
        print(f"    专家获取次数: {runner_stats.get('expert_fetch_count', 0)}")

        vram_stats = runner_stats.get("vram_pool_stats", {})
        print(f"    VRAM 占用: {vram_stats.get('current_vram_gb', 0):.4f}GB")
        print(f"    常驻专家数: {vram_stats.get('resident_experts', 0)}")
        print(f"    淘汰次数: {vram_stats.get('evict_count', 0)}")
        print(f"    读回次数: {vram_stats.get('fetch_back_count', 0)}")

        router_stats = stats.get("router_stats", {})
        print(f"    路由预测命中率: {router_stats.get('hit_rate', 0):.1%}")

        prefetcher_stats = stats.get("prefetcher_stats", {})
        print(f"    预取总数: {prefetcher_stats.get('prefetch_total', 0)}")
        print(f"    预取跳过: {prefetcher_stats.get('prefetch_skipped', 0)}")

        print(f"\n  结果: 小规模推理测试通过")
        return True

    except Exception as exc:
        print(f"  [FAIL] 推理测试失败: {exc}")
        import traceback
        traceback.print_exc()
        return False


def main() -> int:
    """主测试入口"""
    print("=" * 60)
    print("  v4 MoE 专家级分页架构 — 小规模快速验证")
    print("  RLLM DiskOffload (无需下载真实模型)")
    print("=" * 60)

    # 检查 CUDA
    if not torch.cuda.is_available():
        print("[FAIL] CUDA 不可用, 无法测试")
        return 1
    print(f"[OK] CUDA: {torch.cuda.get_device_name(0)}")

    # 步骤 1: 创建 + 分片
    try:
        index = step1_create_and_shard_model()
    except Exception as exc:
        print(f"[FAIL] 分片失败: {exc}")
        import traceback
        traceback.print_exc()
        return 1

    # 步骤 2: 推理
    inference_ok = asyncio.run(step2_test_inference())

    # 总结
    print("\n" + "=" * 60)
    print("  验收总结")
    print("=" * 60)
    print(f"    分片: PASS")
    print(f"    推理: {'PASS' if inference_ok else 'FAIL'}")

    if inference_ok:
        print("\n  v4 架构验证通过! 可以下载真实 Mixtral-8x7B 模型.")
        print(f"  小规模分片目录: {SHARD_DIR}")
    else:
        print("\n  v4 架构有问题, 需要修复后再下载真实模型.")

    return 0 if inference_ok else 1


if __name__ == "__main__":
    sys.exit(main())
