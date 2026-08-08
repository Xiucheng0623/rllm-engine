# -*- coding: utf-8 -*-
# File: D:\AI_RLLM\tests\test_speculative_decoding.py
"""Speculative Decoding 集成测试

4-bit 草稿模型 (3.3GB VRAM) + FP16 验证模型 (CPU RAM 缓存)
验证全量 FP16 模型在 8GB VRAM 上通过 Speculative Decoding 达到 5+ tok/s
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
EVICTED_DIR = ROOT / "rllm_offload_temp" / "evicted_layers_spec"
KV_DIR = ROOT / "rllm_offload_temp" / "kv_cache_spec"


async def main() -> None:
    """主函数: Speculative Decoding 测试."""
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    from rllm_disk_engine.vram_pool.vram_cache_pool import VRAMCachePool
    from rllm_disk_engine.vram_pool.hot_cold_evictor import HotColdEvictor
    from rllm_disk_engine.vram_pool.manual_layer_runner import ManualLayerRunner
    from rllm_disk_engine.zero_copy_loader.zero_copy_shard_loader import ZeroCopyShardLoader
    from rllm_disk_engine.speculative_decoder import SpeculativeDecoder

    logger.info("=" * 60)
    logger.info("Speculative Decoding 测试 — 4-bit 草稿 + FP16 验证")
    logger.info("=" * 60)

    gc.collect()
    torch.cuda.empty_cache()

    free, total = torch.cuda.mem_get_info()
    logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    logger.info(f"VRAM: {free / 1024**3:.2f} GB free / {total / 1024**3:.2f} GB total")

    # ============================================================
    # Step 1: 加载 4-bit 草稿模型 (全装 VRAM)
    # ============================================================
    logger.info("\n--- Step 1: 加载 4-bit 草稿模型 ---")
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        str(MODEL_DIR), local_files_only=True
    )
    if tokenizer.eos_token_id is None:
        tokenizer.eos_token_id = 2
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    t0 = time.time()
    draft_model = AutoModelForCausalLM.from_pretrained(
        str(MODEL_DIR),
        quantization_config=quant_config,
        device_map={"": 0},
        torch_dtype=torch.float16,
        local_files_only=True,
        low_cpu_mem_usage=True,
    )
    draft_model.eval()
    draft_load_time = time.time() - t0

    free_after_draft, _ = torch.cuda.mem_get_info()
    draft_vram = (free - free_after_draft) / 1024**3
    logger.info(
        f"草稿模型加载: {draft_load_time:.1f}s, "
        f"VRAM 占用 ~{draft_vram:.2f}GB"
    )

    # ============================================================
    # Step 2: 加载 FP16 验证模型 (CPU RAM 缓存)
    # ============================================================
    logger.info("\n--- Step 2: 加载 FP16 验证模型层 ---")
    from transformers.models.mistral.configuration_mistral import MistralConfig
    from transformers.models.mistral.modeling_mistral import (
        MistralDecoderLayer,
        MistralRMSNorm,
    )
    import json

    config = MistralConfig.from_pretrained(str(MODEL_DIR / "config.json"))
    num_layers = config.num_hidden_layers

    # 构建 weight_map
    index_path = MODEL_DIR / "model.safetensors.index.json"
    index_data = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map = index_data.get("weight_map", {})

    # ZeroCopyShardLoader
    loader = ZeroCopyShardLoader()
    loader.initialize(
        raw_model_dir=MODEL_DIR,
        weight_map=weight_map,
        num_layers=num_layers,
    )

    # VRAMCachePool — 4-bit 草稿 ~3.3GB + embed/norm/lm_head ~1GB = 4.3GB 固定
    # 剩余 ~3.7GB: 1GB 给 KV cache, 2.7GB 给 FP16 层池 (能装 ~6 层)
    VRAMCachePool._singleton = None
    vram_pool = VRAMCachePool(reserve_gb=5.0)
    vram_pool._usable_bytes = 3 * 1024**3  # 3GB 给 FP16 层 (能装 7 层)
    vram_pool._evict_threshold = int(vram_pool._usable_bytes * 0.8)
    logger.info(
        f"VRAMCachePool: usable={vram_pool._usable_bytes/1024**3:.1f}GB "
        f"evict_threshold={vram_pool._evict_threshold/1024**3:.1f}GB"
    )

    # HotColdEvictor
    def _layer_factory(layer_idx: int):
        return MistralDecoderLayer(config, layer_idx=layer_idx).half()

    evictor = HotColdEvictor(
        vram_pool=vram_pool,
        evict_dir=EVICTED_DIR,
        num_layers=num_layers,
    )
    evictor.attach_factory(_layer_factory, quant_bits=16)
    vram_pool.attach_evictor(evictor)

    # 加载 FP16 层
    logger.info("预加载 FP16 层到 VRAM 池...")
    loaded, _ = await vram_pool.prefill_load_all(
        layer_loader=loader,
        layer_module_factory=lambda idx: MistralDecoderLayer(config, layer_idx=idx).half(),
        num_layers=num_layers,
        quant_bits=16,
    )
    logger.info(f"FP16 层预加载完成: {loaded}/{num_layers}")

    # 强制淘汰到 VRAM 阈值以下
    evicted = await vram_pool.force_evict_to_limit()
    pool_stats = vram_pool.stats()
    logger.info(
        f"淘汰后: resident={pool_stats['vram_layers_resident']} "
        f"VRAM={pool_stats['vram_used_gb']:.1f}GB"
    )

    # 提取辅助权重
    aux_weights = loader.get_aux_tensors([
        "model.embed_tokens.weight",
        "model.norm.weight",
        "lm_head.weight",
    ])
    embed_module = torch.nn.Embedding(
        config.vocab_size, config.hidden_size, dtype=torch.float16
    )
    embed_module.weight.data = aux_weights["model.embed_tokens.weight"].clone()
    embed_module = embed_module.to("cuda")

    norm_module = MistralRMSNorm(
        config.hidden_size, eps=config.rms_norm_eps
    ).half()
    norm_module.weight.data = aux_weights["model.norm.weight"].clone()
    norm_module = norm_module.to("cuda")

    lm_head_module = torch.nn.Linear(
        config.hidden_size, config.vocab_size, bias=False
    ).half()
    lm_head_module.weight.data = aux_weights["lm_head.weight"].clone()
    lm_head_module = lm_head_module.to("cuda")

    # ManualLayerRunner (FP16 验证模型)
    verify_runner = ManualLayerRunner(
        config=config,
        embed_tokens=embed_module,
        norm=norm_module,
        lm_head=lm_head_module,
        vram_pool=vram_pool,
        kv_spill_threshold_mb=256,
        spill_dir=KV_DIR,
    )
    logger.info("FP16 验证模型 ManualLayerRunner 创建完成")

    # ============================================================
    # Step 3: 创建 SpeculativeDecoder
    # ============================================================
    logger.info("\n--- Step 3: 创建 SpeculativeDecoder ---")
    spec_decoder = SpeculativeDecoder(
        draft_model=draft_model,
        draft_tokenizer=tokenizer,
        verify_runner=verify_runner,
        draft_size=8,
    )

    # ============================================================
    # Step 4: 推理测试
    # ============================================================
    prompts = [
        "请用中文介绍一下中国的四大发明。",
        "请解释一下什么是量子计算。",
        "用简短的中文描述一下春天的景色。",
    ]

    results = []
    for pi, prompt in enumerate(prompts):
        logger.info(f"\n--- Prompt {pi+1}/{len(prompts)} ---")
        logger.info(f"输入: {prompt}")

        # 重置验证模型和草稿模型的 KV cache
        verify_runner._kv_cache.clear()
        verify_runner._spill_count = 0
        verify_runner._readback_count = 0
        spec_decoder.reset_draft_cache()

        # 重置统计
        spec_decoder.accept_count = 0
        spec_decoder.reject_count = 0
        spec_decoder.total_rounds = 0
        spec_decoder.total_verify_time = 0.0
        spec_decoder.total_draft_time = 0.0

        t0 = time.time()
        text, stats = await spec_decoder.generate(
            prompt=prompt,
            max_new_tokens=128,
            temperature=0.1,  # 低温提升草稿接受率
            top_p=0.95,
        )
        total_time = time.time() - t0

        logger.info(f"生成: {text[:300]}")
        logger.info(
            f"Speculative: {stats['total_tokens']} tok / {stats['total_time']:.1f}s "
            f"= {stats['tps']:.1f} tok/s | "
            f"accept_rate={stats['accept_rate']:.0%} | "
            f"rounds={stats['rounds']} | "
            f"avg_verify={stats['avg_verify_time']:.2f}s | "
            f"avg_draft={stats['avg_draft_time']:.2f}s"
        )

        # 调度统计
        pool_stats = vram_pool.stats()
        logger.info(
            f"VRAM: resident={pool_stats['vram_layers_resident']} "
            f"evict={pool_stats['evict_count']} "
            f"fetch_back={pool_stats['fetch_back_count']}"
        )

        results.append(stats)

    # ============================================================
    # Step 5: 汇总
    # ============================================================
    logger.info("\n" + "=" * 60)
    logger.info("Speculative Decoding 汇总")
    logger.info("=" * 60)

    avg_tps = sum(r["tps"] for r in results) / len(results)
    avg_accept = sum(r["accept_rate"] for r in results) / len(results)

    for i, r in enumerate(results):
        logger.info(
            f"  Prompt {i+1}: {r['total_tokens']} tok / {r['total_time']:.1f}s "
            f"= {r['tps']:.1f} tok/s | accept={r['accept_rate']:.0%}"
        )

    logger.info(f"\n  平均速度: {avg_tps:.1f} tok/s")
    logger.info(f"  平均接受率: {avg_accept:.0%}")
    logger.info(f"  普通 FP16 基线: 0.9 tok/s")
    logger.info(f"  提升倍数: {avg_tps / 0.9:.1f}x")
    logger.info(f"  目标速度: 5.0 tok/s")
    logger.info(f"  {'✅ 达标' if avg_tps >= 5.0 else '⚠️ 未达标, 需调优'}")


if __name__ == "__main__":
    asyncio.run(main())
