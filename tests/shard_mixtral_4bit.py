# -*- coding: utf-8 -*-
# File: D:\AI_RLLM\tests\shard_mixtral_4bit.py
"""Mixtral-8x7B 4bit 量化分片脚本 (从 safetensors 按需加载)

核心设计:
  93GB FP16 模型无法在 32GB RAM 中完整加载.
  使用 safetensors 的 lazy loading, 每次只读取单个张量,
  4bit 量化后写入 D 盘, 然后释放.

  内存峰值: ~1.5GB (单个专家 + 量化开销)

  输出格式完全兼容 ExpertShardPersistor, MoEOrchestrator 可直接加载.

使用:
  D:\\AI_RLLM\\.venv\\Scripts\\python.exe D:\\AI_RLLM\\tests\\shard_mixtral_4bit.py
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from loguru import logger

RLLM_ROOT = Path(r"D:\AI_RLLM")
if str(RLLM_ROOT) not in sys.path:
    sys.path.insert(0, str(RLLM_ROOT))

from rllm_agent_core import LOG_DIR
from rllm_disk_engine.expert_pool.expert_shard_persistor import (
    ExpertShardMeta,
    ExpertIndex,
)

logger.add(
    LOG_DIR / "shard_mixtral_{time}.log",
    rotation="50 MB",
    retention="7 days",
    encoding="utf-8",
)

RAW_DIR = Path(r"D:\AI_RLLM\rllm_model_shards\mixtral_8x7b_raw")
SHARD_DIR = Path(r"D:\AI_RLLM\rllm_model_shards\mixtral_8x7b_v4_shards")


def load_weight_map() -> Dict[str, str]:
    """加载 safetensors 权重名 → shard 文件映射"""
    index_path = RAW_DIR / "model.safetensors.index.json"
    with open(index_path, "r", encoding="utf-8") as f:
        index = json.load(f)
    return index.get("weight_map", {})


def load_config() -> Dict[str, Any]:
    """加载 config.json"""
    with open(RAW_DIR / "config.json", "r", encoding="utf-8") as f:
        return json.load(f)


def extract_tensor(
    weight_name: str,
    weight_map: Dict[str, str],
) -> torch.Tensor:
    """从 safetensors 文件提取单个张量 (lazy, 不加载整个文件)"""
    from safetensors import safe_open

    shard_name = weight_map[weight_name]
    shard_path = RAW_DIR / shard_name

    with safe_open(str(shard_path), framework="pt", device="cpu") as f:
        return f.get_tensor(weight_name)


def move_quant_state_to_cpu(qs: Any) -> Any:
    """用 QuantState 构造函数重建 CPU 版本 (递归处理 state2).

    Params4bit.cpu() 会丢失 quant_state (退化为普通 Parameter), 必须手动
    搬移 quant_state 的所有 tensor 到 CPU. 用构造函数重建比 dir() 遍历更
    稳健, 避免漏掉属性或拷贝 properties.

    Args:
        qs: bitsandbytes.functional.QuantState 对象 (CUDA 或 CPU)

    Returns:
        新的 QuantState 对象 (所有 tensor 在 CPU 上)
    """
    if qs is None:
        return None
    from bitsandbytes.functional import QuantState

    # 递归处理嵌套的 state2 (double quantization)
    new_state2 = None
    if getattr(qs, "state2", None) is not None:
        s2 = qs.state2
        # offset 是 1-element tensor, 需要 .cpu()
        s2_offset_cpu = None
        if getattr(s2, "offset", None) is not None and isinstance(s2.offset, torch.Tensor):
            s2_offset_cpu = s2.offset.cpu()
        else:
            s2_offset_cpu = s2.offset
        new_state2 = QuantState(
            absmax=s2.absmax.cpu() if s2.absmax is not None else None,
            shape=s2.shape,
            code=s2.code.cpu() if s2.code is not None else None,
            blocksize=s2.blocksize,
            quant_type=s2.quant_type,
            dtype=s2.dtype,
            offset=s2_offset_cpu,
            state2=None,
        )

    # offset 是 1-element tensor (nested quantization 时), 需要 .cpu()
    qs_offset_cpu = None
    if getattr(qs, "offset", None) is not None and isinstance(qs.offset, torch.Tensor):
        qs_offset_cpu = qs.offset.cpu()
    else:
        qs_offset_cpu = qs.offset

    return QuantState(
        absmax=qs.absmax.cpu() if qs.absmax is not None else None,
        shape=qs.shape,
        code=qs.code.cpu() if qs.code is not None else None,
        blocksize=qs.blocksize,
        quant_type=qs.quant_type,
        dtype=qs.dtype,
        offset=qs_offset_cpu,
        state2=new_state2,
    )


def quantize_nf4(weight: torch.Tensor) -> Any:
    """4bit NF4 量化 (在 CUDA 上量化, 然后搬到 CPU 保存)

    关键修复: Params4bit(w_cuda, ...) 不会自动量化 (bnb_quantized=False).
    正确流程:
      1. 在 CPU 创建 bf16 权重, 包装成 Params4bit (未量化)
      2. 调用 param.cuda() 触发 _quantize (因为 bnb_quantized=False, to() 调 _quantize)
      3. 用构造函数重建 CPU 版本 (不用 .cpu(): 会丢失 quant_state 退化为普通 Parameter)

    Args:
        weight: 原始 FP16/BF16 权重张量

    Returns:
        Params4bit 对象 (CPU, 含 quant_state, bnb_quantized=True)
    """
    from bitsandbytes.nn import Params4bit

    w_bf16 = weight.to(torch.bfloat16)
    if torch.cuda.is_available():
        # 1. 在 CPU 创建 Params4bit (未量化)
        param = Params4bit(
            w_bf16,
            requires_grad=False,
            quant_type="nf4",
            compress_statistics=True,
        )
        # 2. .cuda() 触发 _quantize (量化 data → uint8 packed, 设置 quant_state)
        param = param.cuda()

        # 3. 用构造函数重建 CPU 版本 (保留 quant_state, 不用 .cpu())
        data_cpu = param.data.cpu()
        qs_cpu = move_quant_state_to_cpu(param.quant_state)
        param_cpu = Params4bit(
            data_cpu,
            requires_grad=False,
            quant_state=qs_cpu,
            quant_type="nf4",
            compress_statistics=True,
            bnb_quantized=True,
        )
        del param
    else:
        # 无 CUDA 时直接存 bf16 (回退)
        param_cpu = w_bf16.cpu()

    # 显式释放
    del w_bf16
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return param_cpu


def persist_shared(
    config: Dict[str, Any],
    weight_map: Dict[str, str],
    index: ExpertIndex,
    shared_dir: Path,
) -> int:
    """保存共享层 (embed_tokens, norm, lm_head)"""
    total_bytes: int = 0
    shared_dir.mkdir(parents=True, exist_ok=True)

    # embed_tokens
    embed_path = shared_dir / "embed_tokens.pt"
    if not embed_path.exists():
        t = extract_tensor("model.embed_tokens.weight", weight_map)
        torch.save({"weight": t.to(torch.bfloat16).cpu()}, embed_path)
        del t
    index.shared_shards["embed_tokens"] = str(embed_path)
    total_bytes += embed_path.stat().st_size
    print(f"    embed_tokens: {embed_path.stat().st_size / 1024**2:.0f}MB")

    # final norm
    norm_path = shared_dir / "norm.pt"
    if not norm_path.exists():
        t = extract_tensor("model.norm.weight", weight_map)
        torch.save({"weight": t.to(torch.bfloat16).cpu()}, norm_path)
        del t
    index.shared_shards["norm"] = str(norm_path)
    total_bytes += norm_path.stat().st_size

    # lm_head
    lm_head_path = shared_dir / "lm_head.pt"
    if not lm_head_path.exists():
        t = extract_tensor("lm_head.weight", weight_map)
        torch.save({"weight": t.to(torch.bfloat16).cpu()}, lm_head_path)
        del t
    index.shared_shards["lm_head"] = str(lm_head_path)
    total_bytes += lm_head_path.stat().st_size
    print(f"    lm_head: {lm_head_path.stat().st_size / 1024**2:.0f}MB")

    return total_bytes


def persist_layer_shared(
    layer_idx: int,
    weight_map: Dict[str, str],
    index: ExpertIndex,
    layer_dir: Path,
) -> int:
    """保存单层 attention + layernorm + gate"""
    total_bytes: int = 0

    # attention.pt (q/k/v/o proj + 2 layernorm)
    attn_path = layer_dir / "attention.pt"
    if not attn_path.exists():
        sd: Dict[str, torch.Tensor] = {}
        parts = [
            "self_attn.q_proj.weight",
            "self_attn.k_proj.weight",
            "self_attn.v_proj.weight",
            "self_attn.o_proj.weight",
            "input_layernorm.weight",
            "post_attention_layernorm.weight",
        ]
        for part in parts:
            wname = f"model.layers.{layer_idx}.{part}"
            if wname in weight_map:
                t = extract_tensor(wname, weight_map)
                sd[part] = t.to(torch.bfloat16).cpu()
                del t
        torch.save(sd, attn_path)
    index.attention_shards[layer_idx] = str(attn_path)
    total_bytes += attn_path.stat().st_size

    # gate.pt (路由器)
    gate_path = layer_dir / "gate.pt"
    if not gate_path.exists():
        wname = f"model.layers.{layer_idx}.block_sparse_moe.gate.weight"
        t = extract_tensor(wname, weight_map)
        torch.save({"weight": t.to(torch.bfloat16).cpu()}, gate_path)
        del t
    index.gate_shards[layer_idx] = str(gate_path)
    total_bytes += gate_path.stat().st_size

    return total_bytes


def persist_expert(
    layer_idx: int,
    expert_idx: int,
    weight_map: Dict[str, str],
    experts_dir: Path,
) -> Optional[ExpertShardMeta]:
    """4bit 量化 + 保存单个专家"""
    expert_path = experts_dir / f"expert_{expert_idx}.pt"

    # 断点续传
    if expert_path.exists():
        sd = torch.load(expert_path, map_location="cpu")
        return ExpertShardMeta(
            layer_idx=layer_idx,
            expert_idx=expert_idx,
            shard_path=str(expert_path),
            size_bytes=expert_path.stat().st_size,
            quant_bits=4,
            w1_shape=list(sd.get("w1_shape", [0, 0])),
            w2_shape=list(sd.get("w2_shape", [0, 0])),
            w3_shape=list(sd.get("w3_shape", [0, 0])),
        )

    prefix = f"model.layers.{layer_idx}.block_sparse_moe.experts.{expert_idx}"

    try:
        w1 = extract_tensor(f"{prefix}.w1.weight", weight_map)
        w2 = extract_tensor(f"{prefix}.w2.weight", weight_map)
        w3 = extract_tensor(f"{prefix}.w3.weight", weight_map)

        w1_shape: List[int] = list(w1.shape)
        w2_shape: List[int] = list(w2.shape)
        w3_shape: List[int] = list(w3.shape)

        # 4bit NF4 量化
        sd = {
            "w1": quantize_nf4(w1),
            "w2": quantize_nf4(w2),
            "w3": quantize_nf4(w3),
            "w1_shape": w1_shape,
            "w2_shape": w2_shape,
            "w3_shape": w3_shape,
            "quant_bits": 4,
            "quant_type": "nf4",
        }
        torch.save(sd, expert_path)

        del w1, w2, w3, sd

        return ExpertShardMeta(
            layer_idx=layer_idx,
            expert_idx=expert_idx,
            shard_path=str(expert_path),
            size_bytes=expert_path.stat().st_size,
            quant_bits=4,
            w1_shape=w1_shape,
            w2_shape=w2_shape,
            w3_shape=w3_shape,
        )
    except Exception as exc:
        logger.exception(f"专家 L{layer_idx}E{expert_idx} 失败: {exc}")
        return None


def main() -> int:
    """主入口"""
    print("=" * 60)
    print("  Mixtral-8x7B 4bit 量化分片")
    print(f"  源: {RAW_DIR}")
    print(f"  目标: {SHARD_DIR}")
    print("=" * 60)

    if not (RAW_DIR / "config.json").exists():
        print("[FAIL] 模型未下载, 请先运行 _download_mixtral.py")
        return 1

    # 不清空目录: 保留 shared/attention/gate (断点续传跳过),
    # 只重新生成被删除的 expert_*.pt (避免重复提取共享层)
    SHARD_DIR.mkdir(parents=True, exist_ok=True)

    config = load_config()
    weight_map = load_weight_map()
    num_layers = config["num_hidden_layers"]
    num_experts = config["num_local_experts"]

    print(f"\n  层数: {num_layers}")
    print(f"  专家/层: {num_experts}")
    print(f"  激活/token: {config['num_experts_per_tok']}")
    print(f"  权重总数: {len(weight_map)}")

    # 构建 ExpertIndex
    index = ExpertIndex(
        model_type="mixtral",
        num_layers=num_layers,
        num_experts_per_layer=num_experts,
        num_experts_per_token=config["num_experts_per_tok"],
        hidden_size=config["hidden_size"],
        intermediate_size=config["intermediate_size"],
        num_attention_heads=config["num_attention_heads"],
        num_key_value_heads=config["num_key_value_heads"],
        vocab_size=config["vocab_size"],
        created_at=time.strftime("%Y-%m-%d %H:%M:%S"),
    )

    total_bytes: int = 0
    t0 = time.time()

    # 1. 共享层
    print("\n[1/3] 保存共享层...")
    shared_dir = SHARD_DIR / "shared"
    total_bytes += persist_shared(config, weight_map, index, shared_dir)
    print(f"  共享层: {total_bytes / 1024**2:.0f}MB")

    # 2. 逐层
    print(f"\n[2/3] 保存 {num_layers} 层 (attention + gate + {num_experts} 专家/层)...")
    for layer_idx in range(num_layers):
        layer_dir = SHARD_DIR / f"layer_{layer_idx:02d}"
        experts_dir = layer_dir / "experts"
        layer_dir.mkdir(parents=True, exist_ok=True)
        experts_dir.mkdir(parents=True, exist_ok=True)

        # attention + gate
        total_bytes += persist_layer_shared(
            layer_idx, weight_map, index, layer_dir
        )

        # 专家
        for expert_idx in range(num_experts):
            meta = persist_expert(
                layer_idx, expert_idx, weight_map, experts_dir
            )
            if meta is not None:
                if layer_idx not in index.expert_shards:
                    index.expert_shards[layer_idx] = []
                index.expert_shards[layer_idx].append(meta)
                total_bytes += meta.size_bytes

        if (layer_idx + 1) % 4 == 0 or layer_idx == 0:
            elapsed = time.time() - t0
            done = (layer_idx + 1) * num_experts
            total = num_layers * num_experts
            speed = done / max(elapsed, 0.001)
            remaining = (total - done) / max(speed, 0.001)
            print(
                f"  [{layer_idx + 1}/{num_layers}] "
                f"{total_bytes / 1024**3:.2f}GB, "
                f"剩余 ~{remaining / 60:.0f}min"
            )

    index.total_size_bytes = total_bytes
    elapsed = time.time() - t0

    # 3. 写 index.json
    print("\n[3/3] 写入索引...")
    index_path = SHARD_DIR / "index.json"
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(index.to_dict(), f, indent=2, ensure_ascii=False)

    print(f"\n{'=' * 60}")
    print(f"  分片完成!")
    print(f"  总大小: {total_bytes / 1024**3:.2f} GB")
    print(f"  耗时: {elapsed / 60:.1f} 分钟")
    print(f"  路径: {SHARD_DIR}")
    print(f"  索引: {index_path}")
    print(f"{'=' * 60}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
