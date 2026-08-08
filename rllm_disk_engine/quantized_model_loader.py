# -*- coding: utf-8 -*-
# File: D:\AI_RLLM\rllm_disk_engine\quantized_model_loader.py
"""4-bit NF4 量化模型加载器

职责:
  1. 使用 BitsAndBytesConfig 加载完整 4-bit 量化模型 (from_pretrained)
  2. 从加载的模型中提取 decoder layers / embed_tokens / norm / lm_head
  3. 将提取的层注入 VRAMCachePool, 供 v3 ManualLayerRunner 使用

设计要点:
  - from_pretrained 是 bitsandbytes 量化的标准路径, 生成正确的 Linear4bit 模块
  - 提取后模型壳可释放, 只保留层模块, 节省内存
  - 4-bit 7B ~3.3GB, 32 层全部可常驻 8GB VRAM, 无需磁盘分页

Args:
    model_dir: 模型目录 (含 config.json + safetensors)
    quant_bits: 量化位宽 (4=Nf4, 8=Int8)
    device_map: 设备映射 (默认 {"": 0} 全装 GPU)
"""
from __future__ import annotations

import gc
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from loguru import logger


class QuantizedModelLoader:
    """4-bit/8-bit 量化模型加载器

    通过 from_pretrained + BitsAndBytesConfig 加载量化模型,
    然后提取 decoder layers 供 v3 路径使用.

    Attributes:
        model_dir: 模型目录路径
        quant_bits: 量化位宽 (4 或 8)
        config: 模型配置 (MistralConfig 等)
        tokenizer: 关联的 tokenizer
    """

    def __init__(
        self,
        model_dir: str | Path,
        quant_bits: int = 4,
    ) -> None:
        """初始化量化加载器

        Args:
            model_dir: 模型目录 (含 config.json + safetensors)
            quant_bits: 量化位宽, 4=NF4 (推荐), 8=INT8
        """
        self.model_dir: Path = Path(model_dir)
        self.quant_bits: int = quant_bits
        self.config: Optional[Any] = None
        self.tokenizer: Optional[Any] = None
        self._model: Optional[Any] = None

        if not self.model_dir.exists():
            raise FileNotFoundError(
                f"模型目录不存在: {self.model_dir}"
            )

        logger.info(
            f"[QuantLoader] 初始化: dir={self.model_dir} "
            f"quant={quant_bits}bit"
        )

    def _build_quant_config(self) -> "BitsAndBytesConfig":
        """构建 BitsAndBytesConfig

        Returns:
            BitsAndBytesConfig 实例
        """
        from transformers import BitsAndBytesConfig

        if self.quant_bits == 4:
            config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
        elif self.quant_bits == 8:
            config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_has_fp16_weight=True,
            )
        else:
            raise ValueError(
                f"不支持的量化位宽: {self.quant_bits}, 仅支持 4 或 8"
            )
        return config

    def load_model(self) -> Tuple[Any, Any, Any]:
        """加载完整量化模型

        使用 from_pretrained + BitsAndBytesConfig 标准路径加载.
        这是 bitsandbytes 量化的正确方式, 生成 Linear4bit/Linear8bit 模块.

        Returns:
            (model, tokenizer, config) 三元组

        Raises:
            RuntimeError: 加载失败
        """
        from transformers import (
            AutoTokenizer,
            AutoModelForCausalLM,
            AutoConfig,
        )

        t0 = time.time()

        # 1. 加载 config
        self.config = AutoConfig.from_pretrained(
            str(self.model_dir), local_files_only=True
        )
        num_layers = self.config.num_hidden_layers
        logger.info(
            f"[QuantLoader] Config: layers={num_layers} "
            f"hidden={self.config.hidden_size} "
            f"vocab={self.config.vocab_size}"
        )

        # 2. 加载 tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(self.model_dir), local_files_only=True
        )
        if self.tokenizer.eos_token_id is None:
            self.tokenizer.eos_token_id = 2
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # 3. 构建量化配置
        quant_config = self._build_quant_config()
        logger.info(f"[QuantLoader] BitsAndBytesConfig: {quant_config.to_dict()}")

        # 4. 加载模型 — 全部装进 GPU
        logger.info("[QuantLoader] 开始加载量化模型...")
        self._model = AutoModelForCausalLM.from_pretrained(
            str(self.model_dir),
            quantization_config=quant_config,
            device_map={"": 0},
            torch_dtype=torch.float16,
            local_files_only=True,
            low_cpu_mem_usage=True,
        )
        self._model.eval()

        load_time = time.time() - t0
        free, total = torch.cuda.mem_get_info()
        used_gb = (total - free) / 1024**3

        logger.success(
            f"[QuantLoader] 模型加载完成: {load_time:.1f}s, "
            f"VRAM 占用 ~{used_gb:.2f}GB / {total/1024**3:.2f}GB"
        )
        return self._model, self.tokenizer, self.config

    def extract_layers(self) -> Dict[str, Any]:
        """从已加载的模型中提取 v3 路径所需组件

        提取:
          - decoder_layers: List[nn.Module] (32 个 DecoderLayer, 已量化)
          - embed_tokens: embedding 层
          - norm: 最终 RMSNorm
          - lm_head: 语言模型头
          - config: 模型配置
          - tokenizer: tokenizer

        Returns:
            包含所有组件的字典

        Raises:
            RuntimeError: 模型未加载
        """
        if self._model is None:
            raise RuntimeError("模型未加载, 请先调用 load_model()")

        # 提取 decoder layers (适配多种模型结构)
        decoder_layers: List[torch.nn.Module] = []

        if hasattr(self._model, "model") and hasattr(self._model.model, "layers"):
            # MistralForCausalLM → model.layers
            decoder_layers = list(self._model.model.layers)
            embed_tokens = self._model.model.embed_tokens
            norm = self._model.model.norm
        elif hasattr(self._model, "transformer") and hasattr(self._model.transformer, "h"):
            # GPT 风格: transformer.h
            decoder_layers = list(self._model.transformer.h)
            embed_tokens = self._model.transformer.wte
            norm = self._model.transformer.ln_f
        else:
            raise RuntimeError(
                f"无法识别模型结构: {type(self._model)}, "
                f"请手动提取层"
            )

        # lm_head (可能和 embed_tokens 共享权重)
        if hasattr(self._model, "lm_head"):
            lm_head = self._model.lm_head
        else:
            # 共享权重场景
            lm_head = torch.nn.Linear(
                self.config.hidden_size,
                self.config.vocab_size,
                bias=False,
            )
            lm_head.weight = embed_tokens.weight
            lm_head = lm_head.to("cuda")

        num_layers = len(decoder_layers)
        logger.info(
            f"[QuantLoader] 提取完成: {num_layers} decoder layers, "
            f"embed={type(embed_tokens).__name__}, "
            f"norm={type(norm).__name__}, "
            f"lm_head={type(lm_head).__name__}"
        )

        # 统计量化情况
        quant_layer_count = 0
        for layer in decoder_layers:
            for name, module in layer.named_modules():
                if "Linear4bit" in type(module).__name__:
                    quant_layer_count += 1
                    break
        logger.info(
            f"[QuantLoader] 量化层统计: {quant_layer_count}/{num_layers} 层 "
            f"包含 Linear4bit 模块"
        )

        return {
            "decoder_layers": decoder_layers,
            "embed_tokens": embed_tokens,
            "norm": norm,
            "lm_head": lm_head,
            "config": self.config,
            "tokenizer": self.tokenizer,
            "quant_bits": self.quant_bits,
        }

    def cleanup_model_shell(self) -> None:
        """释放模型壳 (保留提取的层引用)

        提取层后调用此方法, 释放模型壳的 Python 对象引用.
        注意: 层模块的权重仍在 VRAM 中, 不会被释放.
        """
        if self._model is not None:
            # 断开模型壳对层的引用, 但不释放层本身
            if hasattr(self._model, "model") and hasattr(self._model.model, "layers"):
                self._model.model.layers = torch.nn.ModuleList()
            elif hasattr(self._model, "transformer") and hasattr(self._model.transformer, "h"):
                self._model.transformer.h = torch.nn.ModuleList()
            gc.collect()
            logger.info("[QuantLoader] 模型壳已释放")
