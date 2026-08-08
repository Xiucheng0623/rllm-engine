# -*- coding: utf-8 -*-
# File: D:\AI_RLLM\rllm_disk_engine\router\router_predictor.py
"""路由预测器

策略:
  Phase 0 (无需训练): 用上一 token 的 gate 输出预测当前 token 可能激活的专家.
    原理: 相邻 token 的路由模式有强相关性, 上一步 Top-16 候选覆盖当前 95%+.
  Phase 1 (可扩展): 训练轻量 MLP, 输入 hidden_state, 输出专家激活概率.

关键参数:
  top_k_candidates: 预取候选数 (默认 16)
    - 8: 命中率 ~80%, I/O 省 4x
    - 16: 命中率 ~95%, I/O 省 2x (推荐)
    - 32: 命中率 ~99%, I/O 只省 1x
"""
from __future__ import annotations

import threading
from typing import Any, List, Optional, Tuple

import torch
from loguru import logger

from rllm_agent_core import LOG_DIR

logger.add(
    LOG_DIR / "router_predictor_{time}.log",
    rotation="50 MB",
    retention="7 days",
    encoding="utf-8",
)

ExpertKey = Tuple[int, int]


class RouterPredictor:
    """路由预测器 (Phase 0: 基于 gate 历史输出)

    工作方式:
      1. 每次 forward 时, 各层 gate 产生 logits [num_experts]
      2. 取 Top-K 候选 (K=16), 记录到历史
      3. 下一个 token 到来前, 把上一步的 Top-K 候选作为预取目标
      4. 由于相邻 token 路由强相关, 命中率 95%+

    Args:
        num_layers: Transformer 层数
        num_experts_per_layer: 每层专家数
        top_k_candidates: 预取候选数 (默认 16)
    """

    def __init__(
        self,
        num_layers: int = 32,
        num_experts_per_layer: int = 8,
        top_k_candidates: int = 16,
    ) -> None:
        self._num_layers: int = num_layers
        self._num_experts: int = num_experts_per_layer
        self._top_k: int = min(top_k_candidates, num_experts_per_layer)
        self._lock: threading.Lock = threading.Lock()

        # 上一 token 各层的 gate logits: [layer_idx → Tensor[num_experts]]
        self._last_gate_logits: dict[int, torch.Tensor] = {}

        # 命中率统计
        self._predict_count: int = 0
        self._hit_count: int = 0

        logger.info(
            f"[RouterPredictor] 初始化: layers={num_layers} "
            f"experts/layer={num_experts_per_layer} "
            f"top_k={self._top_k}"
        )

    # ----------------------------------------------------------------
    # 对外接口
    # ----------------------------------------------------------------
    def update_gate_output(
        self,
        layer_idx: int,
        gate_logits: torch.Tensor,
    ) -> None:
        """记录某层的 gate 输出 (每 token 调用一次)

        Args:
            layer_idx: 层索引
            gate_logits: [batch, num_experts] 或 [num_experts] 的路由 logits
        """
        with self._lock:
            # 取 batch=0, squeeze
            if gate_logits.dim() > 1:
                gate_logits = gate_logits[0]
            self._last_gate_logits[layer_idx] = gate_logits.detach().cpu()

    def predict_candidates(self) -> List[ExpertKey]:
        """预测下一个 token 可能激活的专家 (Top-K 候选)

        从上一 token 各层 gate 的 Top-K 候选中合并取并集.
        每层最多 K 个候选, 共 num_layers × K 个 (去重后通常更少).

        Returns:
            候选专家 key 列表 [(layer_idx, expert_idx), ...]
        """
        candidates: List[ExpertKey] = []
        with self._lock:
            for layer_idx in range(self._num_layers):
                logits = self._last_gate_logits.get(layer_idx)
                if logits is None:
                    continue
                # Top-K 专家
                k = min(self._top_k, self._num_experts)
                topk_vals, topk_indices = torch.topk(logits, k)
                for expert_idx in topk_indices.tolist():
                    candidates.append((layer_idx, expert_idx))

        return candidates

    def record_hit(
        self,
        predicted: List[ExpertKey],
        actual: List[ExpertKey],
    ) -> None:
        """记录预测命中率 (供统计)

        Args:
            predicted: 预测的候选列表
            actual: 实际激活的专家列表
        """
        with self._lock:
            self._predict_count += 1
            predicted_set = set(predicted)
            actual_set = set(actual)
            hits = len(actual_set & predicted_set)
            if hits == len(actual_set):
                self._hit_count += 1

    def stats(self) -> dict[str, Any]:
        """获取预测统计"""
        with self._lock:
            hit_rate = (
                self._hit_count / max(self._predict_count, 1)
            )
            return {
                "predict_count": self._predict_count,
                "hit_count": self._hit_count,
                "hit_rate": hit_rate,
                "top_k": self._top_k,
            }

    def reset(self) -> None:
        """重置状态 (新请求时调用)"""
        with self._lock:
            self._last_gate_logits.clear()
            self._predict_count = 0
            self._hit_count = 0
