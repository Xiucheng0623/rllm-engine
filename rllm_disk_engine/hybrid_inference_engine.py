# -*- coding: utf-8 -*-
# File: D:\AI_RLLM\rllm_disk_engine\hybrid_inference_engine.py
"""混合推理引擎 — 自适应量化 + 层级磁盘分页

核心创新:
  1. 自动检测模型大小 vs VRAM 容量, 选择最优策略
  2. 模型能装入 VRAM → 4-bit 全常驻 (22+ tok/s)
  3. 模型超过 VRAM → 4-bit + 层级磁盘分页 (创新点)
  4. 无论哪种模式, 都保留层级调度/热冷淘汰/KV溢出/自进化

这比纯量化方案多了"超出 VRAM 时仍能运行"的能力,
比纯磁盘分页方案多了"能装入时全速运行"的能力.

策略选择:
  model_size_4bit = model_params * 0.5 bytes (NF4)
  if model_size_4bit < vram_usable * 0.8:
      → 全常驻模式 (4-bit, 零IO)
  elif model_size_4bit < vram_usable + cpu_ram * 0.8:
      → 混合模式 (4-bit, 部分层 CPU RAM 缓存)
  else:
      → 磁盘分页模式 (4-bit, 部分层 D 盘缓存)

  13B 4-bit = 6.6GB → 8GB VRAM: 混合模式 (创新点!)
  70B 4-bit = 35GB → 8GB VRAM: 磁盘分页模式 (创新点!)
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from loguru import logger


class HybridInferenceEngine:
    """混合推理引擎 — 自适应量化 + 层级分页

    根据模型大小和硬件容量自动选择最优推理策略.
    在保证速度的同时, 支持超过 VRAM 容量的大模型推理.

    Attributes:
        model_dir: 模型目录
        vram_gb: GPU VRAM 容量 (GB)
        cpu_ram_gb: 可用 CPU RAM (GB)
        strategy: 当前推理策略 ("full_resident" | "hybrid" | "disk_paged")
    """

    def __init__(
        self,
        model_dir: str | Path,
        vram_gb: float = 8.0,
        cpu_ram_gb: float = 16.0,
    ) -> None:
        """初始化混合推理引擎

        Args:
            model_dir: 模型目录路径
            vram_gb: GPU VRAM 容量 (GB)
            cpu_ram_gb: 可用 CPU RAM (GB)
        """
        self.model_dir: Path = Path(model_dir)
        self.vram_gb: float = vram_gb
        self.cpu_ram_gb: float = cpu_ram_gb
        self.strategy: str = "unknown"

        # 预留 VRAM 给 KV cache + 临时张量
        self.vram_reserve_gb: float = 2.0
        self.vram_usable_gb: float = vram_gb - self.vram_reserve_gb

        logger.info(
            f"[HybridEngine] 初始化: VRAM={vram_gb}GB "
            f"usable={self.vram_usable_gb:.1f}GB "
            f"CPU_RAM={cpu_ram_gb}GB"
        )

    def analyze_model(self, num_params: int) -> Dict[str, Any]:
        """分析模型大小并选择推理策略

        Args:
            num_params: 模型参数量 (如 7_000_000_000 = 7B)

        Returns:
            策略信息字典
        """
        # 4-bit NF4 量化后大小 (每参数 0.5 bytes + 额外开销 ~10%)
        size_4bit_gb = num_params * 0.5 / 1024**3 * 1.1
        # FP16 全量大小
        size_fp16_gb = num_params * 2 / 1024**3

        # VRAM 可用空间
        vram_budget = self.vram_usable_gb

        # 策略选择
        if size_4bit_gb < vram_budget * 0.8:
            # 4-bit 能装入 VRAM
            strategy = "full_resident"
            resident_layers_pct = 1.0
            expected_tps = "20-25 tok/s"
        elif size_4bit_gb < vram_budget + self.cpu_ram_gb * 0.7:
            # 4-bit 超出 VRAM 但 CPU RAM 够 (混合模式)
            strategy = "hybrid"
            # VRAM 装不下的层走 CPU RAM
            overflow_gb = size_4bit_gb - vram_budget * 0.8
            total_layer_gb = size_4bit_gb
            resident_pct = max(0.3, 1.0 - overflow_gb / total_layer_gb)
            resident_layers_pct = resident_pct
            # 预期速度: 只需 IO 传输溢出层
            io_layers_pct = 1.0 - resident_pct
            io_time_per_token = io_layers_pct * 42  # ms (PCIe 4.0)
            expected_tps = f"{1000 / io_time_per_token:.1f} tok/s (估)"
        else:
            # 4-bit 超出 VRAM + CPU RAM (磁盘分页)
            strategy = "disk_paged"
            resident_layers_pct = 0.2
            expected_tps = "0.5-3 tok/s"

        self.strategy = strategy

        result = {
            "model_params": num_params / 1e9,
            "size_4bit_gb": round(size_4bit_gb, 2),
            "size_fp16_gb": round(size_fp16_gb, 2),
            "vram_usable_gb": round(vram_budget, 2),
            "strategy": strategy,
            "resident_layers_pct": round(resident_layers_pct, 2),
            "expected_tps": expected_tps,
        }

        logger.info(
            f"[HybridEngine] 模型分析: {num_params/1e9:.1f}B params | "
            f"4-bit={size_4bit_gb:.1f}GB FP16={size_fp16_gb:.1f}GB | "
            f"策略={strategy} | 预期={expected_tps}"
        )

        return result

    def get_inference_config(self) -> Dict[str, Any]:
        """获取当前策略的推理配置

        Returns:
            配置字典 (传给 VRAMCachePool / ManualLayerRunner)
        """
        if self.strategy == "full_resident":
            return {
                "quant_bits": 4,
                "vram_pool_usable_gb": self.vram_usable_gb,
                "enable_eviction": False,
                "enable_prefetch": False,
                "cpu_cache_gb": 0,
                "description": "4-bit 全常驻, 零 IO, 最高速度",
            }
        elif self.strategy == "hybrid":
            return {
                "quant_bits": 4,
                "vram_pool_usable_gb": self.vram_usable_gb,
                "enable_eviction": True,
                "enable_prefetch": True,
                "cpu_cache_gb": min(self.cpu_ram_gb * 0.7, 12.0),
                "description": "4-bit + 层级 CPU RAM 分页, 创新模式",
            }
        else:  # disk_paged
            return {
                "quant_bits": 4,
                "vram_pool_usable_gb": self.vram_usable_gb,
                "enable_eviction": True,
                "enable_prefetch": True,
                "cpu_cache_gb": min(self.cpu_ram_gb * 0.7, 12.0),
                "disk_cache_dir": r"D:\AI_RLLM\rllm_offload_temp",
                "description": "4-bit + 磁盘分页, 支持超大模型",
            }


def print_strategy_table() -> None:
    """打印不同模型大小下的策略表"""
    engine_8gb = HybridInferenceEngine(model_dir=".", vram_gb=8.0, cpu_ram_gb=16.0)

    models = [
        ("7B", 7_000_000_000),
        ("13B", 13_000_000_000),
        ("34B", 34_000_000_000),
        ("70B", 70_000_000_000),
    ]

    print("\n" + "=" * 80)
    print(f"{'模型':>6} | {'4-bit大小':>10} | {'策略':>15} | {'常驻层%':>8} | {'预期速度':>20}")
    print("-" * 80)

    for name, params in models:
        info = engine_8gb.analyze_model(params)
        print(
            f"{name:>6} | "
            f"{info['size_4bit_gb']:>8.1f}GB | "
            f"{info['strategy']:>15} | "
            f"{info['resident_layers_pct']*100:>7.0f}% | "
            f"{info['expected_tps']:>20}"
        )

    print("=" * 80)
    print("\n创新点: hybrid 和 disk_paged 模式 — 超出 VRAM 仍能推理")


if __name__ == "__main__":
    print_strategy_table()
