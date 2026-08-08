# -*- coding: utf-8 -*-
# File: D:\AI_RLLM\tests\test_v3_4bit.py
"""v3 路径 + 4-bit NF4 量化集成测试

验证 ManualLayerRunner + VRAMCachePool + QuantizedModelLoader 全链路:
  1. QuantizedModelLoader 用 from_pretrained 加载 4-bit 模型
  2. 提取 decoder layers 注入 VRAMCachePool
  3. ManualLayerRunner 逐层 forward + KV cache 管理
  4. 5 条中文 prompt 验证生成质量 + 速度
"""
from __future__ import annotations

import asyncio
import gc
import os
import sys
import time
from pathlib import Path

import torch
from loguru import logger

ROOT = Path(r"D:\AI_RLLM")
os.environ["HF_HOME"] = str(ROOT / "hf_cache")
os.environ["TRANSFORMERS_CACHE"] = str(ROOT / "hf_cache" / "hub")
os.environ["HUGGINGFACE_HUB_CACHE"] = str(ROOT / "hf_cache" / "hub")
os.environ["TORCH_HOME"] = str(ROOT / "hf_cache" / "torch")
os.environ["HF_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["BITSANDBYTES_NOWELCOME"] = "1"

sys.path.insert(0, str(ROOT))
for sub in ("rllm_agent_core", "rllm_disk_engine", "rllm_auto_evo", "rllm_pipeline"):
    sys.path.insert(0, str(ROOT / sub))

MODEL_DIR = ROOT / "rllm_model_shards" / "_raw" / "Nous-Hermes-2-Mistral-7B-DPO"


async def main() -> None:
    """主函数: v3 + 4-bit 集成测试."""
    from rllm_disk_engine.quantized_model_loader import QuantizedModelLoader
    from rllm_disk_engine.vram_pool.vram_cache_pool import VRAMCachePool
    from rllm_disk_engine.vram_pool.manual_layer_runner import ManualLayerRunner

    logger.info("=" * 60)
    logger.info("v3 路径 + 4-bit NF4 量化集成测试")
    logger.info("=" * 60)

    # 清理显存
    gc.collect()
    torch.cuda.empty_cache()

    free, total = torch.cuda.mem_get_info()
    logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    logger.info(f"VRAM: {free / 1024**3:.2f} GB free / {total / 1024**3:.2f} GB total")

    # ============================================================
    # Step 1: QuantizedModelLoader 加载 4-bit 模型
    # ============================================================
    logger.info("\n--- Step 1: 加载 4-bit 量化模型 ---")
    loader = QuantizedModelLoader(
        model_dir=MODEL_DIR,
        quant_bits=4,
    )
    model, tokenizer, config = loader.load_model()

    # ============================================================
    # Step 2: 提取 decoder layers + 辅助模块
    # ============================================================
    logger.info("\n--- Step 2: 提取 decoder layers ---")
    components = loader.extract_layers()

    decoder_layers = components["decoder_layers"]
    embed_tokens = components["embed_tokens"]
    norm_module = components["norm"]
    lm_head = components["lm_head"]
    num_layers = len(decoder_layers)
    quant_bits = components["quant_bits"]

    logger.info(f"提取完成: {num_layers} 层, quant={quant_bits}bit")

    # 释放模型壳 (层引用已提取, 壳可以释放)
    loader.cleanup_model_shell()
    gc.collect()

    # ============================================================
    # Step 3: 注入 VRAMCachePool
    # ============================================================
    logger.info("\n--- Step 3: 注入 VRAMCachePool ---")

    # 重置单例
    VRAMCachePool._singleton = None

    # 4-bit 7B ~3.3GB, 8GB VRAM, reserve 2GB 给 KV cache
    vram_pool = VRAMCachePool(reserve_gb=2.0)
    logger.info(
        f"VRAMCachePool: capacity={vram_pool._capacity_bytes/1024**3:.1f}GB "
        f"usable={vram_pool._usable_bytes/1024**3:.1f}GB"
    )

    # 直接注入已量化的层 (无需 prefill_load_all)
    loaded = await vram_pool.load_from_quantized_model(
        decoder_layers=decoder_layers,
        quant_bits=quant_bits,
    )

    pool_stats = vram_pool.stats()
    logger.info(
        f"VRAM 池: resident={pool_stats['vram_layers_resident']} "
        f"VRAM={pool_stats['vram_used_gb']:.1f}GB"
    )

    # ============================================================
    # Step 4: 创建 ManualLayerRunner
    # ============================================================
    logger.info("\n--- Step 4: 创建 ManualLayerRunner ---")
    kv_dir = ROOT / "rllm_offload_temp" / "kv_cache_v3_4bit"

    runner = ManualLayerRunner(
        config=config,
        embed_tokens=embed_tokens,
        norm=norm_module,
        lm_head=lm_head,
        vram_pool=vram_pool,
        kv_spill_threshold_mb=512,  # 4-bit 下 KV 空间充足
        spill_dir=kv_dir,
    )
    logger.info("ManualLayerRunner 创建完成")

    # ============================================================
    # Step 5: 推理测试
    # ============================================================
    prompts = [
        "请用中文介绍一下中国的四大发明。",
        "请解释一下什么是量子计算。",
        "用简短的中文描述一下春天的景色。",
        "写一首关于月亮的短诗。",
        "人工智能的未来发展方向是什么？",
    ]

    results = []

    for pi, prompt in enumerate(prompts):
        logger.info(f"\n--- Prompt {pi+1}/{len(prompts)} ---")
        logger.info(f"输入: {prompt}")

        tokens = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")
        input_length = tokens.shape[1]
        logger.info(f"输入 token 数: {input_length}")

        # Prefill
        t0 = time.time()
        first_token, _ = await runner.prefill(
            input_ids=tokens, temperature=0.7, top_p=0.9,
        )
        prefill_time = time.time() - t0
        generated_ids = [first_token]
        logger.info(
            f"Prefill: {input_length} tok → 1 tok, {prefill_time:.2f}s"
        )

        # Decode
        max_new_tokens = 128
        t_decode = time.time()
        for step in range(max_new_tokens):
            next_id, _ = await runner.decode_step(
                last_token=generated_ids[-1],
                history_tokens=generated_ids,
                temperature=0.7,
                top_p=0.9,
            )
            generated_ids.append(next_id)
            if next_id == tokenizer.eos_token_id:
                break
        decode_time = time.time() - t_decode
        decode_count = len(generated_ids) - 1
        tps = decode_count / max(decode_time, 0.001)

        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

        logger.info(f"生成: {generated_text[:300]}")
        logger.info(
            f"首token: {prefill_time:.2f}s | "
            f"decode: {decode_count} tok / {decode_time:.2f}s = {tps:.1f} tok/s"
        )

        # 调度统计
        stats = runner.stats()
        pool_stats = vram_pool.stats()
        logger.info(
            f"KV: entries={stats.get('kv_entries', '?')} "
            f"spill={stats.get('kv_spill_count', '?')} "
            f"readback={stats.get('kv_readback_count', '?')} | "
            f"VRAM: resident={pool_stats.get('vram_layers_resident', '?')} "
            f"evict={pool_stats.get('evict_count', '?')} "
            f"fetch_back={pool_stats.get('fetch_back_count', '?')}"
        )

        results.append({
            "prompt": prompt,
            "prefill_time": prefill_time,
            "decode_count": decode_count,
            "decode_time": decode_time,
            "tps": tps,
            "text": generated_text[:200],
        })

        # 重置 runner 的 KV cache (每条 prompt 独立)
        runner._kv_cache.clear()
        runner._spill_count = 0
        runner._readback_count = 0

    # ============================================================
    # Step 6: 汇总
    # ============================================================
    logger.info("\n" + "=" * 60)
    logger.info("v3 + 4-bit 集成测试汇总")
    logger.info("=" * 60)

    avg_prefill = sum(r["prefill_time"] for r in results) / len(results)
    avg_tps = sum(r["tps"] for r in results) / len(results)

    for i, r in enumerate(results):
        logger.info(
            f"  Prompt {i+1}: prefill={r['prefill_time']:.2f}s, "
            f"{r['decode_count']} tok / {r['decode_time']:.2f}s = {r['tps']:.1f} tok/s"
        )

    logger.info(f"\n  平均 prefill: {avg_prefill:.2f}s")
    logger.info(f"  平均 decode 速度: {avg_tps:.1f} tok/s")
    logger.info(f"  目标速度: 15.0 tok/s")
    logger.info(f"  {'✅ 达标' if avg_tps >= 15.0 else '⚠️ 未达标'}")

    # 最终 VRAM 统计
    free_end, total_end = torch.cuda.mem_get_info()
    logger.info(
        f"\n  最终 VRAM: {free_end/1024**3:.2f} GB free / "
        f"{total_end/1024**3:.2f} GB total"
    )


if __name__ == "__main__":
    asyncio.run(main())
