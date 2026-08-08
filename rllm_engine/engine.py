# -*- coding: utf-8 -*-
# File: D:\AI_RLLM\rllm_engine\engine.py
"""RLLM Engine 核心推理引擎

在消费级 GPU 上运行大模型的本地推理引擎.

支持双引擎:
  - v3 Dense: 层级分页, 7B/13B dense 模型 (已验证 23.6 tok/s + 正常输出)
  - v4 MoE:   专家级分页, 47B Mixtral MoE 模型 (实验性)

用法:
    from rllm_engine import RLLMEngine

    engine = RLLMEngine("Nous-Hermes-2-Mistral-7B-DPO")
    engine.load()

    # 生成
    print(engine.generate("你好"))

    # 流式
    for chunk in engine.generate_stream("写一首诗"):
        print(chunk, end="")
"""
from __future__ import annotations

import asyncio
import gc
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple

from loguru import logger

from rllm_engine.platform_paths import (
    ensure_dirs,
    get_cache_dir,
    get_logs_dir,
    get_models_dir,
    get_rllm_home,
)

# ------------------------------------------------------------------
# 配置
# ------------------------------------------------------------------


@dataclass
class EngineConfig:
    """RLLM Engine 配置

    Attributes:
        model_name_or_path: 模型名称或本地路径
        max_new_tokens: 默认最大生成 token 数
        temperature: 采样温度
        top_p: nucleus 采样阈值
        reserve_gb: VRAM 保留量
        quant_bits: 量化位宽 (4 或 8)
    """

    model_name_or_path: str = "Nous-Hermes-2-Mistral-7B-DPO"
    max_new_tokens: int = 128
    temperature: float = 0.7
    top_p: float = 0.9
    reserve_gb: float = 2.0
    quant_bits: int = 4

    @property
    def model_path(self) -> Path:
        """解析模型路径"""
        p = Path(self.model_name_or_path)
        if p.is_absolute() and p.exists():
            return p
        # 尝试 models 目录
        candidate = get_models_dir() / self.model_name_or_path
        if candidate.exists():
            return candidate
        # 尝试原始路径
        return candidate


# ------------------------------------------------------------------
# 引擎
# ------------------------------------------------------------------


class RLLMEngine:
    """RLLM 本地大模型推理引擎

    核心接口: load() → generate() / generate_stream() / chat()

    自动检测模型类型:
      - Mixtral/MoE → v4 专家级分页引擎
      - Dense (Mistral/Hermes/LLaMA) → v3 层级分页引擎

    Attributes:
        config: 引擎配置
        is_loaded: 模型是否已加载
        stats: 最近一次推理统计
        model_type: 模型类型 ("dense" 或 "moe")
    """

    def __init__(self, model_name_or_path: str = "Nous-Hermes-2-Mistral-7B-DPO",
                 **kwargs: Any) -> None:
        """初始化引擎

        Args:
            model_name_or_path: 模型名称或本地路径
                支持:
                - HuggingFace 本地路径: "path/to/model"
                - models 目录下的模型: "Nous-Hermes-2-Mistral-7B-DPO"
                - Mixtral MoE 模型: "mixtral-8x7b"
                - MoE 分片路径: "path/to/mixtral_8x7b_v4_shards"
            **kwargs: 覆盖 EngineConfig 配置项
        """
        self.config: EngineConfig = EngineConfig(
            model_name_or_path=model_name_or_path, **kwargs
        )
        self._is_loaded: bool = False
        self._last_stats: Dict[str, Any] = {}
        self._conversation_history: List[Dict[str, str]] = []
        self.model_type: str = "unknown"

        # v3 组件 (dense)
        self._loader: Any = None
        self._vram_pool: Any = None
        self._runner: Any = None
        self._tokenizer: Any = None

        # v4 组件 (MoE)
        self._orchestrator: Any = None

        # 确保目录
        ensure_dirs()
        self._setup_env()

    # ------------------------------------------------------------------
    # 公共 API
    # ------------------------------------------------------------------

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded

    @property
    def stats(self) -> Dict[str, Any]:
        return self._last_stats

    def load(self) -> None:
        """加载模型

        自动检测模型类型并加载对应的引擎:
          - MoE 模型 (含 "mixtral" / "moe" 关键词) → v4 引擎
          - Dense 模型 → v3 引擎

        Raises:
            FileNotFoundError: 模型路径不存在
            RuntimeError: CUDA 不可用
        """
        if self._is_loaded:
            return

        print(f"\n{'='*60}")
        print(f"  RLLM Engine v1.0")
        print(f"  模型: {self.config.model_name_or_path}")
        print(f"  平台: {sys.platform}")
        print(f"  根目录: {get_rllm_home()}")
        print(f"{'='*60}\n")

        # 检测模型类型
        model_path = self._resolve_model_path()
        self.model_type = self._detect_model_type(model_path)

        if self.model_type == "moe":
            self._load_v4_moe(model_path)
        else:
            self._load_v3_dense(model_path)

        self._is_loaded = True
        self._print_ready()

    def generate(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
    ) -> str:
        """生成文本

        Args:
            prompt: 输入文本
            max_new_tokens: 最大生成 token 数
            temperature: 采样温度
            top_p: nucleus 采样阈值

        Returns:
            生成的完整文本
        """
        self._ensure_loaded()
        return asyncio.run(
            self._generate_async(
                prompt,
                max_new_tokens or self.config.max_new_tokens,
                temperature or self.config.temperature,
                top_p or self.config.top_p,
            )
        )

    def generate_stream(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
    ) -> Generator[str, None, None]:
        """流式生成 (逐句输出)"""
        self._ensure_loaded()
        result = self.generate(prompt, max_new_tokens, temperature, top_p)
        for sentence in result.split("\n"):
            yield sentence + "\n"

    def chat(self) -> None:
        """交互式对话 (Ctrl+C 退出)"""
        self._ensure_loaded()

        print(f"\n{'='*60}")
        print(f"  RLLM Engine 对话模式 ({self.model_type})")
        print(f"  输入 'exit' 退出, 'clear' 清空历史")
        print(f"{'='*60}\n")

        self._conversation_history = []

        while True:
            try:
                user_input = input("\n>>> 用户: ").strip()
                if not user_input:
                    continue
                if user_input.lower() in ("exit", "quit", "q"):
                    break
                if user_input.lower() == "clear":
                    self._conversation_history = []
                    print("对话历史已清空。")
                    continue

                # 单轮（v3/V4 目前不保持 KV cache 跨 prompt）
                print("\n    助手: ", end="", flush=True)
                result = self.generate(user_input)
                print(result)
            except KeyboardInterrupt:
                print("\n\n对话结束。")
                break

    def unload(self) -> None:
        """卸载模型, 释放 VRAM"""
        if not self._is_loaded:
            return
        print("释放 VRAM...")
        self._loader = None
        self._vram_pool = None
        self._runner = None
        self._orchestrator = None
        self._tokenizer = None
        self._is_loaded = False
        gc.collect()
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("VRAM 已释放。")

    # ------------------------------------------------------------------
    # v3 Dense 加载
    # ------------------------------------------------------------------

    def _load_v3_dense(self, model_path: Path) -> None:
        """加载 v3 层积分页引擎 (dense 模型)

        流程: QuantizedModelLoader → VRAMCachePool → ManualLayerRunner
        """
        from rllm_disk_engine.quantized_model_loader import QuantizedModelLoader
        from rllm_disk_engine.vram_pool.vram_cache_pool import VRAMCachePool
        from rllm_disk_engine.vram_pool.manual_layer_runner import (
            ManualLayerRunner,
        )

        print("[1/5] 加载量化模型...")
        self._loader = QuantizedModelLoader(
            model_dir=model_path,
            quant_bits=self.config.quant_bits,
        )
        model, tokenizer, config = self._loader.load_model()
        self._tokenizer = tokenizer
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        components = self._loader.extract_layers()
        decoder_layers = components["decoder_layers"]
        embed_tokens = components["embed_tokens"]
        norm_module = components["norm"]
        lm_head = components["lm_head"]
        num_layers = len(decoder_layers)

        print(f"  层数: {num_layers}, {self.config.quant_bits}bit")
        self._loader.cleanup_model_shell()
        gc.collect()

        print("[2/5] 初始化 VRAM 缓存池...")
        VRAMCachePool._singleton = None
        self._vram_pool = VRAMCachePool(reserve_gb=self.config.reserve_gb)
        asyncio.run(
            self._vram_pool.load_from_quantized_model(
                decoder_layers=decoder_layers,
                quant_bits=self.config.quant_bits,
            )
        )

        print("[3/5] 创建推理 runner...")
        kv_dir = get_rllm_home() / "offload_temp" / "kv_cache"
        self._runner = ManualLayerRunner(
            config=config,
            embed_tokens=embed_tokens,
            norm=norm_module,
            lm_head=lm_head,
            vram_pool=self._vram_pool,
            kv_spill_threshold_mb=512,
            spill_dir=kv_dir,
        )

        print("[4/5] 验证 GPU...")
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA 不可用")
        gpu_name = torch.cuda.get_device_name(0)
        print(f"  GPU: {gpu_name}")

        # 重置单例确保后续 load 可复用
        VRAMCachePool._singleton = None

    # ------------------------------------------------------------------
    # v4 MoE 加载
    # ------------------------------------------------------------------

    def _load_v4_moe(self, model_path: Path) -> None:
        """加载 v4 专家级分页引擎 (MoE 模型)"""
        from rllm_disk_engine.moe_orchestrator import MoEOrchestrator

        print("[1/4] 加载 tokenizer...")
        from transformers import AutoTokenizer
        # MoE 需要 tokenizer 在分片目录下或原始模型目录
        raw_dir = model_path.parent / f"{model_path.name.replace('_v4_shards', '')}_raw"
        if not (raw_dir / "tokenizer.model").exists() and not (raw_dir / "tokenizer.json").exists():
            # 尝试同级目录
            for candidate in [model_path.parent, model_path]:
                for fname in ["tokenizer.model", "tokenizer.json"]:
                    if (candidate / fname).exists():
                        raw_dir = candidate
                        break
        self._tokenizer = AutoTokenizer.from_pretrained(str(raw_dir), use_fast=True)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        print("[2/4] 初始化专家分页引擎...")
        self._orchestrator = MoEOrchestrator(
            shard_dir=model_path,
            reserve_gb=self.config.reserve_gb,
        )

        async def _init():
            await self._orchestrator.initialize()
        asyncio.run(_init())

        print("[3/4] 验证 GPU...")
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA 不可用")
        gpu_name = torch.cuda.get_device_name(0)
        print(f"  GPU: {gpu_name}")

    # ------------------------------------------------------------------
    # 推理核心
    # ------------------------------------------------------------------

    async def _generate_async(
        self,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
    ) -> str:
        """异步推理"""
        prompt_s = prompt[:100].replace("\n", " ")
        print(f"\n[推理] {prompt_s}...")

        t0 = time.time()

        if self.model_type == "moe":
            output_text = await self._generate_v4(
                prompt, max_new_tokens, temperature, top_p
            )
        else:
            output_text = await self._generate_v3(
                prompt, max_new_tokens, temperature, top_p
            )

        total_s = time.time() - t0
        self._last_stats["total_seconds"] = total_s
        return output_text

    async def _generate_v3(
        self, prompt: str, max_new_tokens: int, temperature: float,
        top_p: float,
    ) -> str:
        """v3 Dense 推理"""
        tokens = self._tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")
        input_len = tokens.shape[1]

        # Prefill
        t0 = time.time()
        first_token, _ = await self._runner.prefill(
            input_ids=tokens, temperature=temperature, top_p=top_p,
        )
        prefill_s = time.time() - t0

        generated_ids = [first_token]

        # Decode
        t_dec = time.time()
        for step in range(max_new_tokens - 1):
            next_id, _ = await self._runner.decode_step(
                last_token=generated_ids[-1],
                history_tokens=generated_ids,
                temperature=temperature,
                top_p=top_p,
            )
            generated_ids.append(next_id)
            if next_id == self._tokenizer.eos_token_id:
                break
        decode_s = time.time() - t_dec
        decode_count = len(generated_ids) - 1

        output = self._tokenizer.decode(generated_ids, skip_special_tokens=True)
        tps = decode_count / max(decode_s, 0.001)

        self._last_stats = {
            "model_type": "dense",
            "input_tokens": input_len,
            "output_tokens": len(generated_ids),
            "prefill_seconds": prefill_s,
            "decode_seconds": decode_s,
            "avg_tok_per_s": tps,
        }

        print(f"  速度: {tps:.1f} tok/s | prefill={prefill_s:.1f}s decode={decode_s:.1f}s")
        return output

    async def _generate_v4(
        self, prompt: str, max_new_tokens: int, temperature: float,
        top_p: float,
    ) -> str:
        """v4 MoE 推理"""
        input_ids = self._tokenizer.encode(prompt, add_special_tokens=False)

        generated_ids, stats = await self._orchestrator.generate(
            prompt_ids=input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )

        output = self._tokenizer.decode(generated_ids, skip_special_tokens=True)

        self._last_stats = {
            "model_type": "moe",
            "input_tokens": len(input_ids),
            "output_tokens": len(generated_ids),
            **stats,
        }

        print(f"  速度: {stats['avg_tok_per_s']:.1f} tok/s")
        return output

    # ------------------------------------------------------------------
    # 辅助
    # ------------------------------------------------------------------

    def _setup_env(self) -> None:
        """设置环境变量"""
        cache_dir = str(get_cache_dir())
        os.environ["HF_HOME"] = cache_dir
        os.environ["TRANSFORMERS_CACHE"] = cache_dir
        os.environ["HUGGINGFACE_HUB_CACHE"] = cache_dir
        os.environ["TORCH_HOME"] = cache_dir
        os.environ["HF_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["BITSANDBYTES_NOWELCOME"] = "1"

        # 确保在 sys.path 中
        rllm_root = str(get_rllm_home())
        if rllm_root not in sys.path:
            sys.path.insert(0, rllm_root)

    def _resolve_model_path(self) -> Path:
        """解析模型路径

        尝试顺序:
          1. 绝对路径 (如 D:\AI_RLLM\models\model_name)
          2. models 目录下的子目录
          3. rllm_model_shards 递归搜索 (兼容旧路径)
        """
        p = Path(self.config.model_name_or_path)
        if p.is_absolute():
            return p

        # models 目录
        candidate = get_models_dir() / self.config.model_name_or_path
        if candidate.exists():
            return candidate

        # 递归搜索 rllm_model_shards
        old_path = get_rllm_home() / "rllm_model_shards"
        if old_path.exists():
            model_name = self.config.model_name_or_path
            for sub in old_path.rglob("*"):
                if sub.is_dir() and model_name in sub.name:
                    return sub

        return candidate

    def _detect_model_type(self, model_path: Path) -> str:
        """检测模型类型

        Returns:
            "moe" 或 "dense"
        """
        name_lower = str(model_path).lower()
        moe_keywords = ["mixtral", "moe", "mixture", "v4_shards"]
        if any(kw in name_lower for kw in moe_keywords):
            return "moe"

        # 检查 index.json (v4 MoE 分片的标志)
        if (model_path / "index.json").exists():
            import json
            with open(model_path / "index.json", "r", encoding="utf-8") as f:
                idx = json.load(f)
            if "num_experts_per_layer" in idx:
                return "moe"

        return "dense"

    def _ensure_loaded(self) -> None:
        if not self._is_loaded:
            raise RuntimeError("模型未加载, 请先调用 engine.load()")

    def _print_ready(self) -> None:
        print(f"\n{'='*60}")
        print(f"  RLLM Engine 就绪! ({self.model_type})")
        print(f"  输入 engine.generate('你好') 开始推理")
        print(f"{'='*60}\n")
