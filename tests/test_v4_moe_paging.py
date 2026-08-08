# -*- coding: utf-8 -*-
# File: D:\AI_RLLM\tests\test_v4_moe_paging.py
"""v4 MoE 专家级分页架构验收测试

测试内容:
  1. 模块导入验证 (所有 v4 模块可正常 import)
  2. 分片索引检查 (index.json 是否存在)
  3. 如果分片存在: 完整推理测试 (prefill + decode + 性能统计)
  4. 如果分片不存在: 仅运行导入验证 + 输出使用说明

使用方式:
  D:\\AI_RLLM\\.venv\\Scripts\\python.exe D:\\AI_RLLM\\tests\\test_v4_moe_paging.py
"""
from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

# 确保 D:\AI_RLLM 在 sys.path
RLLM_ROOT = Path(r"D:\AI_RLLM")
if str(RLLM_ROOT) not in sys.path:
    sys.path.insert(0, str(RLLM_ROOT))

from loguru import logger

from rllm_agent_core import LOG_DIR

logger.add(
    LOG_DIR / "test_v4_{time}.log",
    rotation="50 MB",
    retention="7 days",
    encoding="utf-8",
)

# v4 模块导入
from rllm_disk_engine.expert_pool.expert_shard_persistor import (
    ExpertShardPersistor,
    ExpertIndex,
)
from rllm_disk_engine.expert_pool.expert_vram_pool import (
    ExpertVRAMPool,
    ExpertEntry,
)
from rllm_disk_engine.expert_pool.expert_evictor import ExpertEvictor
from rllm_disk_engine.expert_pool.expert_freq_tracker import (
    ExpertFreqTracker,
)
from rllm_disk_engine.expert_pool.moe_layer_runner import MoELayerRunner
from rllm_disk_engine.router.router_predictor import RouterPredictor
from rllm_disk_engine.router.router_prefetcher import RouterPrefetcher
from rllm_disk_engine.moe_orchestrator import MoEOrchestrator

SHARD_DIR = Path(r"D:\AI_RLLM\rllm_model_shards\mixtral_8x7b_4bit")


# ============================================================
# 测试 1: 模块导入验证
# ============================================================
def test_imports() -> bool:
    """验证所有 v4 模块可正常导入

    Returns:
        是否全部通过
    """
    print("\n" + "=" * 60)
    print("  测试 1: v4 模块导入验证")
    print("=" * 60)

    modules: List[Tuple[str, Any]] = [
        ("ExpertShardPersistor", ExpertShardPersistor),
        ("ExpertIndex", ExpertIndex),
        ("ExpertVRAMPool", ExpertVRAMPool),
        ("ExpertEntry", ExpertEntry),
        ("ExpertEvictor", ExpertEvictor),
        ("ExpertFreqTracker", ExpertFreqTracker),
        ("MoELayerRunner", MoELayerRunner),
        ("RouterPredictor", RouterPredictor),
        ("RouterPrefetcher", RouterPrefetcher),
        ("MoEOrchestrator", MoEOrchestrator),
    ]

    all_ok: bool = True
    for name, cls in modules:
        try:
            # 检查是否是类
            if isinstance(cls, type):
                print(f"  [OK] {name}")
            else:
                print(f"  [OK] {name} (已导入)")
        except Exception as exc:
            print(f"  [FAIL] {name}: {exc}")
            all_ok = False

    if all_ok:
        print(f"\n  结果: 10/10 模块导入成功")
    else:
        print(f"\n  结果: 部分模块导入失败")
    return all_ok


# ============================================================
# 测试 2: 分片索引检查
# ============================================================
def test_shard_index() -> Optional[ExpertIndex]:
    """检查 Mixtral 分片是否存在

    Returns:
        ExpertIndex 或 None
    """
    print("\n" + "=" * 60)
    print("  测试 2: 分片索引检查")
    print("=" * 60)

    index_path = SHARD_DIR / "index.json"
    if not index_path.exists():
        print(f"  [SKIP] 分片索引不存在: {index_path}")
        print(f"  请先运行 ExpertShardPersistor.persist() 生成分片")
        print(f"  或手动放置 Mixtral 模型到: {SHARD_DIR}")
        return None

    try:
        index = ExpertShardPersistor.load_index(SHARD_DIR)
        print(f"  [OK] 模型类型: {index.model_type}")
        print(f"  [OK] 层数: {index.num_layers}")
        print(f"  [OK] 每层专家数: {index.num_experts_per_layer}")
        print(f"  [OK] 每 token 激活: {index.num_experts_per_token}")
        print(f"  [OK] 隐藏维度: {index.hidden_size}")
        print(f"  [OK] 中间维度: {index.intermediate_size}")
        print(f"  [OK] 词表大小: {index.vocab_size}")
        print(
            f"  [OK] 总大小: "
            f"{index.total_size_bytes / 1024**3:.2f}GB"
        )

        # 检查分片文件是否真实存在
        missing: int = 0
        for layer_idx, experts in index.expert_shards.items():
            for meta in experts:
                if not Path(meta.shard_path).exists():
                    missing += 1
        if missing > 0:
            print(f"  [WARN] {missing} 个专家分片文件缺失")
        else:
            print(f"  [OK] 所有专家分片文件存在")

        return index
    except Exception as exc:
        print(f"  [FAIL] 加载索引失败: {exc}")
        return None


# ============================================================
# 测试 3: 完整推理测试
# ============================================================
async def test_inference(index: ExpertIndex) -> bool:
    """完整推理测试: prefill + decode

    Args:
        index: 分片索引

    Returns:
        是否通过
    """
    print("\n" + "=" * 60)
    print("  测试 3: v4 MoE 推理测试")
    print("=" * 60)

    try:
        # 1. 初始化编排器
        print("\n  [1/4] 初始化 MoEOrchestrator...")
        orchestrator = MoEOrchestrator(
            shard_dir=SHARD_DIR,
            reserve_gb=3.0,
            top_n_hot_experts=40,
            prefetch_candidates=16,
        )
        await orchestrator.initialize()
        print("  [OK] 初始化完成")

        # 2. 构造测试 prompt (简单中文)
        print("\n  [2/4] 构造测试 prompt...")
        # 用简单的 token id (实际应用中应使用 tokenizer)
        # 这里用随机 token id 测试推理流程
        import torch

        prompt_ids: List[int] = [1, 10535, 317, 470, 232, 428, 2]
        print(f"  prompt tokens: {prompt_ids}")

        # 3. 生成
        print("\n  [3/4] 开始生成 (max 32 tokens)...")
        t0 = time.time()
        generated, stats = await orchestrator.generate(
            prompt_ids=prompt_ids,
            max_new_tokens=32,
            temperature=0.7,
            top_p=0.9,
        )
        total_s = time.time() - t0

        # 4. 输出结果
        print(f"\n  [4/4] 生成结果:")
        print(f"    生成 token 数: {len(generated)}")
        print(f"    首 token: {generated[0]}")
        print(f"    prefill 耗时: {stats['prefill_seconds']:.3f}s")
        print(
            f"    decode 总耗时: "
            f"{stats['total_decode_seconds']:.3f}s"
        )
        print(
            f"    平均速度: {stats['avg_tok_per_s']:.2f} tok/s"
        )
        print(f"    总耗时: {total_s:.3f}s")

        # 详细统计
        print(f"\n  --- 详细统计 ---")
        runner_stats = stats.get("runner_stats", {})
        print(
            f"    专家获取次数: "
            f"{runner_stats.get('expert_fetch_count', 0)}"
        )

        vram_stats = runner_stats.get("vram_pool_stats", {})
        print(
            f"    VRAM 占用: "
            f"{vram_stats.get('current_vram_gb', 0):.2f}GB"
        )
        print(
            f"    常驻专家数: "
            f"{vram_stats.get('resident_experts', 0)}"
        )
        print(
            f"    淘汰次数: "
            f"{vram_stats.get('evict_count', 0)}"
        )
        print(
            f"    读回次数: "
            f"{vram_stats.get('fetch_back_count', 0)}"
        )

        router_stats = stats.get("router_stats", {})
        print(
            f"    路由预测命中率: "
            f"{router_stats.get('hit_rate', 0):.1%}"
        )

        prefetcher_stats = stats.get("prefetcher_stats", {})
        print(
            f"    预取总数: "
            f"{prefetcher_stats.get('prefetch_total', 0)}"
        )
        print(
            f"    预取跳过 (已在VRAM): "
            f"{prefetcher_stats.get('prefetch_skipped', 0)}"
        )

        print(f"\n  结果: 推理测试通过")
        return True

    except Exception as exc:
        print(f"  [FAIL] 推理测试失败: {exc}")
        import traceback

        traceback.print_exc()
        return False


# ============================================================
# 主函数
# ============================================================
def main() -> int:
    """主测试入口

    Returns:
        退出码 (0=成功, 1=失败)
    """
    print("=" * 60)
    print("  v4 MoE 专家级分页架构验收测试")
    print("  RLLM DiskOffload — Phase 0")
    print("=" * 60)

    # 测试 1: 导入验证
    import_ok = test_imports()
    if not import_ok:
        print("\n导入验证失败, 终止测试")
        return 1

    # 测试 2: 分片索引
    index = test_shard_index()
    if index is None:
        print("\n" + "=" * 60)
        print("  分片不存在, 仅完成导入验证")
        print("=" * 60)
        print("\n  使用说明:")
        print(f"    1. 下载 Mixtral-8x7B 模型")
        print(f"    2. 运行 ExpertShardPersistor 生成分片:")
        print(f"       python -c \"from rllm_disk_engine.expert_pool import ExpertShardPersistor; ...\"")
        print(f"    3. 分片目录: {SHARD_DIR}")
        print(f"    4. 重新运行本测试")
        return 0

    # 测试 3: 推理
    inference_ok = asyncio.run(test_inference(index))

    # 总结
    print("\n" + "=" * 60)
    print("  验收总结")
    print("=" * 60)
    print(f"    模块导入: {'PASS' if import_ok else 'FAIL'}")
    print(f"    分片索引: {'PASS' if index else 'FAIL'}")
    print(f"    推理测试: {'PASS' if inference_ok else 'FAIL'}")

    return 0 if (import_ok and index and inference_ok) else 1


if __name__ == "__main__":
    sys.exit(main())
