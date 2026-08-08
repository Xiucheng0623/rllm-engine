# -*- coding: utf-8 -*-
# File: D:\AI_RLLM\rllm_disk_engine\expert_pool\expert_shard_persistor.py
"""MoE 专家级分片持久化器

职责:
  1. 接收 MixtralForCausalLM 模型, 按"专家"粒度拆分权重
  2. 专家 FFN 权重 4bit NF4 量化, 共享层保持 bfloat16
  3. 写入 D 盘分片目录, 生成 index.json 路由表
  4. 支持断点续传 (已写分片跳过)

分片布局:
  D:\\AI_RLLM\\rllm_model_shards\\mixtral_8x7b_4bit\\
  ├── index.json                # 路由表 + 偏移量 + 量化信息
  ├── shared\\
  │   ├── embed_tokens.pt       # bfloat16
  │   ├── norm.pt
  │   └── lm_head.pt
  ├── layer_00\\
  │   ├── attention.pt          # q/k/v/o proj + 2 个 RMSNorm (bfloat16)
  │   ├── gate.pt               # 路由器 Linear (bfloat16, 很小)
  │   └── experts\\
  │       ├── expert_0.pt       # w1/w2/w3 4bit NF4 量化
  │       └── ...
  └── ...

量化策略:
  - 共享层 (embed/attention/norm/lm_head): bfloat16 (保证质量, 占 ~3GB)
  - 专家 FFN (w1/w2/w3): 4bit NF4 (压缩 4x, 每 expert ~130MB)
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from loguru import logger

from rllm_agent_core import LOG_DIR

logger.add(
    LOG_DIR / "expert_persistor_{time}.log",
    rotation="50 MB",
    retention="7 days",
    encoding="utf-8",
)


# ============================================================
# 数据结构: 分片元信息
# ============================================================
@dataclass
class ExpertShardMeta:
    """单个专家分片元信息

    Attributes:
        layer_idx: 所属 Transformer 层索引
        expert_idx: 专家索引 (在该层内)
        shard_path: 分片文件绝对路径
        size_bytes: 文件大小 (字节)
        quant_bits: 量化位宽 (4=NF4, 16=bf16)
        w1_shape: gate_proj 权重形状
        w2_shape: down_proj 权重形状
        w3_shape: up_proj 权重形状
    """
    layer_idx: int
    expert_idx: int
    shard_path: str
    size_bytes: int
    quant_bits: int
    w1_shape: List[int]
    w2_shape: List[int]
    w3_shape: List[int]


@dataclass
class ExpertIndex:
    """专家分片索引表 (写入 index.json)

    Attributes:
        model_type: 模型类型 (mixtral)
        num_layers: Transformer 层数
        num_experts_per_layer: 每层专家数
        num_experts_per_token: 每 token 激活专家数
        hidden_size: 隐藏维度
        intermediate_size: FFN 中间维度
        num_attention_heads: attention head 数
        num_key_value_heads: KV head 数
        vocab_size: 词表大小
        shared_shards: 共享层分片路径
        attention_shards: 各层 attention 分片路径
        gate_shards: 各层路由器分片路径
        expert_shards: 所有专家分片元信息 [layer][expert]
        total_size_bytes: 总分片大小
        created_at: 创建时间戳
    """
    model_type: str = "mixtral"
    num_layers: int = 32
    num_experts_per_layer: int = 8
    num_experts_per_token: int = 2
    hidden_size: int = 4096
    intermediate_size: int = 14336
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    vocab_size: int = 32000
    shared_shards: Dict[str, str] = field(default_factory=dict)
    attention_shards: Dict[int, str] = field(default_factory=dict)
    gate_shards: Dict[int, str] = field(default_factory=dict)
    expert_shards: Dict[int, List[ExpertShardMeta]] = field(default_factory=dict)
    total_size_bytes: int = 0
    created_at: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """序列化为可 JSON 化的字典"""
        d = asdict(self)
        # attention_shards / gate_shards 的 key 是 int, JSON 需要 str
        d["attention_shards"] = {str(k): v for k, v in self.attention_shards.items()}
        d["gate_shards"] = {str(k): v for k, v in self.gate_shards.items()}
        d["expert_shards"] = {
            str(k): [asdict(e) for e in v]
            for k, v in self.expert_shards.items()
        }
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ExpertIndex":
        """从字典反序列化"""
        expert_shards: Dict[int, List[ExpertShardMeta]] = {}
        for k, v in d.get("expert_shards", {}).items():
            expert_shards[int(k)] = [ExpertShardMeta(**e) for e in v]
        return cls(
            model_type=d.get("model_type", "mixtral"),
            num_layers=d.get("num_layers", 32),
            num_experts_per_layer=d.get("num_experts_per_layer", 8),
            num_experts_per_token=d.get("num_experts_per_token", 2),
            hidden_size=d.get("hidden_size", 4096),
            intermediate_size=d.get("intermediate_size", 14336),
            num_attention_heads=d.get("num_attention_heads", 32),
            num_key_value_heads=d.get("num_key_value_heads", 8),
            vocab_size=d.get("vocab_size", 32000),
            shared_shards=d.get("shared_shards", {}),
            attention_shards={int(k): v for k, v in d.get("attention_shards", {}).items()},
            gate_shards={int(k): v for k, v in d.get("gate_shards", {}).items()},
            expert_shards=expert_shards,
            total_size_bytes=d.get("total_size_bytes", 0),
            created_at=d.get("created_at", ""),
        )


# ============================================================
# 核心持久化器
# ============================================================
class ExpertShardPersistor:
    """MoE 专家级分片持久化器

    将 Mixtral 模型按专家粒度拆分, 量化后写入 D 盘.

    Args:
        model: MixtralForCausalLM 实例
        output_dir: 分片输出目录 (默认 D:\\AI_RLLM\\rllm_model_shards\\mixtral_8x7b_4bit)
        expert_quant_bits: 专家量化位宽 (4=NF4, 8=Int8, 16=bf16)
        shared_dtype: 共享层数据类型 (默认 bfloat16)
    """

    def __init__(
        self,
        model: Any,
        output_dir: Path = Path(
            r"D:\AI_RLLM\rllm_model_shards\mixtral_8x7b_4bit"
        ),
        expert_quant_bits: int = 4,
        shared_dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        self._model = model
        self._output_dir: Path = Path(output_dir)
        self._expert_quant_bits: int = expert_quant_bits
        self._shared_dtype: torch.dtype = shared_dtype

        # 从模型 config 提取结构信息
        cfg = model.config
        self._num_layers: int = cfg.num_hidden_layers
        self._num_experts: int = cfg.num_local_experts
        self._num_experts_per_tok: int = cfg.num_experts_per_tok
        self._hidden_size: int = cfg.hidden_size
        self._intermediate_size: int = cfg.intermediate_size
        self._num_heads: int = cfg.num_attention_heads
        self._num_kv_heads: int = cfg.num_key_value_heads
        self._vocab_size: int = cfg.vocab_size

        # 创建目录结构
        self._shared_dir: Path = self._output_dir / "shared"
        self._shared_dir.mkdir(parents=True, exist_ok=True)
        for i in range(self._num_layers):
            (self._output_dir / f"layer_{i:02d}" / "experts").mkdir(
                parents=True, exist_ok=True
            )

        logger.info(
            f"[ExpertPersistor] 初始化: layers={self._num_layers} "
            f"experts/layer={self._num_experts} "
            f"experts/tok={self._num_experts_per_tok} "
            f"quant={expert_quant_bits}bit "
            f"output={self._output_dir}"
        )

    # ----------------------------------------------------------------
    # 对外主入口
    # ----------------------------------------------------------------
    def persist(self) -> ExpertIndex:
        """执行分片持久化 (支持断点续传)

        Returns:
            ExpertIndex 索引表
        """
        t0 = time.time()
        index = ExpertIndex(
            model_type="mixtral",
            num_layers=self._num_layers,
            num_experts_per_layer=self._num_experts,
            num_experts_per_token=self._num_experts_per_tok,
            hidden_size=self._hidden_size,
            intermediate_size=self._intermediate_size,
            num_attention_heads=self._num_heads,
            num_key_value_heads=self._num_kv_heads,
            vocab_size=self._vocab_size,
            created_at=time.strftime("%Y-%m-%d %H:%M:%S"),
        )

        total_bytes: int = 0

        # 1. 持久化共享层 (embed_tokens, norm, lm_head)
        total_bytes += self._persist_shared(index)

        # 2. 逐层持久化 attention + gate + experts
        for layer_idx in range(self._num_layers):
            layer_module = self._model.model.layers[layer_idx]

            # 2a. attention + layernorm
            total_bytes += self._persist_attention(layer_idx, layer_module, index)

            # 2b. gate (路由器)
            total_bytes += self._persist_gate(layer_idx, layer_module, index)

            # 2c. 8 个专家
            for expert_idx in range(self._num_experts):
                meta = self._persist_expert(
                    layer_idx, expert_idx, layer_module
                )
                if meta is not None:
                    if layer_idx not in index.expert_shards:
                        index.expert_shards[layer_idx] = []
                    index.expert_shards[layer_idx].append(meta)
                    total_bytes += meta.size_bytes

            if (layer_idx + 1) % 4 == 0 or layer_idx == self._num_layers - 1:
                logger.info(
                    f"[ExpertPersistor] 进度: {layer_idx + 1}/{self._num_layers} 层, "
                    f"累计 {total_bytes / 1024**3:.2f}GB"
                )

        index.total_size_bytes = total_bytes

        # 3. 写 index.json
        index_path = self._output_dir / "index.json"
        with open(index_path, "w", encoding="utf-8") as f:
            json.dump(index.to_dict(), f, indent=2, ensure_ascii=False)

        elapsed = time.time() - t0
        logger.success(
            f"[ExpertPersistor] 完成: {self._num_layers} 层 × {self._num_experts} 专家, "
            f"总大小 {total_bytes / 1024**3:.2f}GB, "
            f"耗时 {elapsed:.1f}s, "
            f"index={index_path}"
        )
        return index

    # ----------------------------------------------------------------
    # 内部: 共享层持久化
    # ----------------------------------------------------------------
    def _persist_shared(self, index: ExpertIndex) -> int:
        """持久化 embed_tokens / norm / lm_head (bfloat16)"""
        total_bytes: int = 0

        # embed_tokens
        embed_path = self._shared_dir / "embed_tokens.pt"
        if not embed_path.exists():
            sd = {
                "weight": self._model.model.embed_tokens.weight.data.to(
                    self._shared_dtype
                ).cpu()
            }
            torch.save(sd, embed_path)
        index.shared_shards["embed_tokens"] = str(embed_path)
        total_bytes += embed_path.stat().st_size

        # final norm
        norm_path = self._shared_dir / "norm.pt"
        if not norm_path.exists():
            sd = {
                "weight": self._model.model.norm.weight.data.to(
                    self._shared_dtype
                ).cpu()
            }
            torch.save(sd, norm_path)
        index.shared_shards["norm"] = str(norm_path)
        total_bytes += norm_path.stat().st_size

        # lm_head
        lm_head_path = self._shared_dir / "lm_head.pt"
        if not lm_head_path.exists():
            sd = {
                "weight": self._model.lm_head.weight.data.to(
                    self._shared_dtype
                ).cpu()
            }
            torch.save(sd, lm_head_path)
        index.shared_shards["lm_head"] = str(lm_head_path)
        total_bytes += lm_head_path.stat().st_size

        logger.info(
            f"[ExpertPersistor] 共享层完成: {total_bytes / 1024**2:.0f}MB"
        )
        return total_bytes

    # ----------------------------------------------------------------
    # 内部: attention + layernorm 持久化
    # ----------------------------------------------------------------
    def _persist_attention(
        self,
        layer_idx: int,
        layer_module: Any,
        index: ExpertIndex,
    ) -> int:
        """持久化单层 attention + 2 个 RMSNorm (bfloat16)"""
        attn_path = (
            self._output_dir
            / f"layer_{layer_idx:02d}"
            / "attention.pt"
        )
        if not attn_path.exists():
            attn = layer_module.self_attn
            sd = {
                "q_proj.weight": attn.q_proj.weight.data.to(self._shared_dtype).cpu(),
                "k_proj.weight": attn.k_proj.weight.data.to(self._shared_dtype).cpu(),
                "v_proj.weight": attn.v_proj.weight.data.to(self._shared_dtype).cpu(),
                "o_proj.weight": attn.o_proj.weight.data.to(self._shared_dtype).cpu(),
            }
            # layernorm 也放这里 (避免分散文件)
            sd["input_layernorm.weight"] = (
                layer_module.input_layernorm.weight.data.to(self._shared_dtype).cpu()
            )
            sd["post_attention_layernorm.weight"] = (
                layer_module.post_attention_layernorm.weight.data.to(self._shared_dtype).cpu()
            )
            torch.save(sd, attn_path)

        index.attention_shards[layer_idx] = str(attn_path)
        return attn_path.stat().st_size

    # ----------------------------------------------------------------
    # 内部: gate (路由器) 持久化
    # ----------------------------------------------------------------
    def _persist_gate(
        self,
        layer_idx: int,
        layer_module: Any,
        index: ExpertIndex,
    ) -> int:
        """持久化路由器 gate Linear (bfloat16, 很小)"""
        gate_path = (
            self._output_dir
            / f"layer_{layer_idx:02d}"
            / "gate.pt"
        )
        if not gate_path.exists():
            gate = layer_module.block_sparse_moe.gate
            sd = {
                "weight": gate.weight.data.to(self._shared_dtype).cpu()
            }
            torch.save(sd, gate_path)

        index.gate_shards[layer_idx] = str(gate_path)
        return gate_path.stat().st_size

    # ----------------------------------------------------------------
    # 内部: 单个专家持久化 (4bit NF4 量化)
    # ----------------------------------------------------------------
    def _persist_expert(
        self,
        layer_idx: int,
        expert_idx: int,
        layer_module: Any,
    ) -> Optional[ExpertShardMeta]:
        """持久化单个专家 FFN (w1/w2/w3)

        量化策略:
          - 4bit: 用 bitsandbytes NF4 量化 (压缩 4x)
          - 16bit: 直接 bfloat16 (不量化)

        Args:
            layer_idx: 层索引
            expert_idx: 专家索引
            layer_module: MixtralDecoderLayer

        Returns:
            ExpertShardMeta 或 None (失败时)
        """
        expert_path = (
            self._output_dir
            / f"layer_{layer_idx:02d}"
            / "experts"
            / f"expert_{expert_idx}.pt"
        )

        # 断点续传: 已存在则跳过
        if expert_path.exists():
            sd = torch.load(
                expert_path, map_location="cpu", weights_only=False
            )
            return ExpertShardMeta(
                layer_idx=layer_idx,
                expert_idx=expert_idx,
                shard_path=str(expert_path),
                size_bytes=expert_path.stat().st_size,
                quant_bits=self._expert_quant_bits,
                w1_shape=list(sd.get("w1_shape", [0, 0])),
                w2_shape=list(sd.get("w2_shape", [0, 0])),
                w3_shape=list(sd.get("w3_shape", [0, 0])),
            )

        try:
            expert = layer_module.block_sparse_moe.experts[expert_idx]
            w1 = expert.w1.weight.data  # [intermediate, hidden]
            w2 = expert.w2.weight.data  # [hidden, intermediate]
            w3 = expert.w3.weight.data  # [intermediate, hidden]

            w1_shape: List[int] = list(w1.shape)
            w2_shape: List[int] = list(w2.shape)
            w3_shape: List[int] = list(w3.shape)

            if self._expert_quant_bits == 4:
                # 4bit NF4 量化
                sd = self._quantize_nf4(w1, w2, w3)
            elif self._expert_quant_bits == 8:
                # 8bit 量化
                sd = self._quantize_int8(w1, w2, w3)
            else:
                # 不量化, bfloat16
                sd = {
                    "w1": w1.to(torch.bfloat16).cpu(),
                    "w2": w2.to(torch.bfloat16).cpu(),
                    "w3": w3.to(torch.bfloat16).cpu(),
                }

            # 保存原始形状 (反量化时需要)
            sd["w1_shape"] = w1_shape
            sd["w2_shape"] = w2_shape
            sd["w3_shape"] = w3_shape
            sd["quant_bits"] = self._expert_quant_bits

            torch.save(sd, expert_path)

            return ExpertShardMeta(
                layer_idx=layer_idx,
                expert_idx=expert_idx,
                shard_path=str(expert_path),
                size_bytes=expert_path.stat().st_size,
                quant_bits=self._expert_quant_bits,
                w1_shape=w1_shape,
                w2_shape=w2_shape,
                w3_shape=w3_shape,
            )

        except Exception as exc:
            logger.exception(
                f"[ExpertPersistor] 专家 L{layer_idx}E{expert_idx} 持久化失败: {exc}"
            )
            return None

    def _quantize_nf4(
        self,
        w1: torch.Tensor,
        w2: torch.Tensor,
        w3: torch.Tensor,
    ) -> Dict[str, Any]:
        """4bit NF4 量化

        Args:
            w1: gate_proj 权重 [intermediate, hidden]
            w2: down_proj 权重 [hidden, intermediate]
            w3: up_proj 权重 [intermediate, hidden]

        Returns:
            包含量化参数的 state_dict
        """
        try:
            import bitsandbytes as bnb
            from bitsandbytes.nn import Params4bit

            def quantize_linear(weight: torch.Tensor) -> Params4bit:
                """量化单个 Linear 权重为 NF4"""
                # Params4bit 期望 [out_features, in_features]
                param = Params4bit(
                    weight.to(torch.bfloat16).cpu(),
                    requires_grad=False,
                    quant_type="nf4",
                    compress_statistics=True,
                )
                return param

            return {
                "w1": quantize_linear(w1),
                "w2": quantize_linear(w2),
                "w3": quantize_linear(w3),
                "quant_type": "nf4",
            }
        except ImportError:
            logger.warning(
                "[ExpertPersistor] bitsandbytes 未安装, 回退 bfloat16"
            )
            return {
                "w1": w1.to(torch.bfloat16).cpu(),
                "w2": w2.to(torch.bfloat16).cpu(),
                "w3": w3.to(torch.bfloat16).cpu(),
                "quant_type": "bf16_fallback",
            }

    def _quantize_int8(
        self,
        w1: torch.Tensor,
        w2: torch.Tensor,
        w3: torch.Tensor,
    ) -> Dict[str, Any]:
        """8bit 量化 (近无损)"""
        try:
            from bitsandbytes.nn import Int8Params

            def quantize_int8(weight: torch.Tensor) -> Any:
                param = Int8Params(
                    weight.to(torch.bfloat16).cpu(),
                    requires_grad=False,
                    has_fp16_weights=False,
                )
                return param

            return {
                "w1": quantize_int8(w1),
                "w2": quantize_int8(w2),
                "w3": quantize_int8(w3),
                "quant_type": "int8",
            }
        except ImportError:
            logger.warning(
                "[ExpertPersistor] bitsandbytes 未安装, 回退 bfloat16"
            )
            return {
                "w1": w1.to(torch.bfloat16).cpu(),
                "w2": w2.to(torch.bfloat16).cpu(),
                "w3": w3.to(torch.bfloat16).cpu(),
                "quant_type": "bf16_fallback",
            }

    # ----------------------------------------------------------------
    # 静态方法: 加载 index.json
    # ----------------------------------------------------------------
    @staticmethod
    def load_index(
        shard_dir: Path = Path(
            r"D:\AI_RLLM\rllm_model_shards\mixtral_8x7b_4bit"
        ),
    ) -> Optional[ExpertIndex]:
        """从 D 盘加载 index.json

        Args:
            shard_dir: 分片目录

        Returns:
            ExpertIndex 或 None (文件不存在时)
        """
        index_path = Path(shard_dir) / "index.json"
        if not index_path.exists():
            logger.warning(f"[ExpertPersistor] index.json 不存在: {index_path}")
            return None
        with open(index_path, "r", encoding="utf-8") as f:
            d = json.load(f)
        index = ExpertIndex.from_dict(d)
        logger.info(
            f"[ExpertPersistor] 加载索引: {index.num_layers} 层 × "
            f"{index.num_experts_per_layer} 专家, "
            f"总 {index.total_size_bytes / 1024**3:.2f}GB"
        )
        return index
