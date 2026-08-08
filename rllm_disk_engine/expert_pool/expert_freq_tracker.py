# -*- coding: utf-8 -*-
# File: D:\AI_RLLM\rllm_disk_engine\expert_pool\expert_freq_tracker.py
"""专家访问频率跟踪器 (线程安全)

用途:
  1. 实时统计每个专家被路由激活的次数
  2. 混合评分 (LRU + 频次) 供 ExpertEvictor 决定淘汰哪个冷专家
  3. 定期输出 Top-N 热专家列表, 供 Hermes 自进化提升至 VRAM 常驻

混合评分公式:
    score = w * freq_norm + (1-w) * recency_norm
    其中 freq_norm = count / max_count
         recency_norm = 1 / (1 + minutes_since_last_access)
"""
from __future__ import annotations

import threading
import time
from typing import Dict, List, Tuple

from loguru import logger

from rllm_agent_core import LOG_DIR

logger.add(
    LOG_DIR / "expert_freq_{time}.log",
    rotation="50 MB",
    retention="7 days",
    encoding="utf-8",
)

# 专家全局 ID: (layer_idx, expert_idx)
ExpertKey = Tuple[int, int]


class ExpertFreqTracker:
    """专家访问频率跟踪器 (线程安全)

    Attributes:
        _counts: 每个专家的累计激活次数
        _last_ts: 每个专家的最近访问时间戳
        _lru_weight: LRU 权重 (0=纯频次, 1=纯LRU)
    """

    def __init__(
        self,
        num_layers: int = 32,
        num_experts_per_layer: int = 8,
        lru_weight_vs_freq: float = 0.5,
    ) -> None:
        """初始化频率跟踪器

        Args:
            num_layers: Transformer 层数
            num_experts_per_layer: 每层专家数
            lru_weight_vs_freq: LRU 权重 (默认 0.5)
        """
        self._num_layers: int = num_layers
        self._num_experts: int = num_experts_per_layer
        self._lru_weight: float = lru_weight_vs_freq
        self._lock: threading.Lock = threading.Lock()

        # key: (layer_idx, expert_idx) → count / last_ts
        self._counts: Dict[ExpertKey, int] = {}
        self._last_ts: Dict[ExpertKey, float] = {}
        for layer_idx in range(num_layers):
            for expert_idx in range(num_experts_per_layer):
                key = (layer_idx, expert_idx)
                self._counts[key] = 0
                self._last_ts[key] = 0.0

        logger.info(
            f"[ExpertFreq] 初始化: {num_layers}层 × {num_experts_per_layer}专家 "
            f"= {num_layers * num_experts_per_layer} 个专家, "
            f"lru_weight={lru_weight_vs_freq}"
        )

    def record_access(self, key: ExpertKey) -> None:
        """记录一次专家访问

        Args:
            key: (layer_idx, expert_idx)
        """
        with self._lock:
            self._counts[key] = self._counts.get(key, 0) + 1
            self._last_ts[key] = time.time()

    def frequency_score(self, key: ExpertKey) -> float:
        """计算专家频次评分 (0-1, 越高越热)

        Args:
            key: (layer_idx, expert_idx)

        Returns:
            频次评分 [0, 1]
        """
        with self._lock:
            max_cnt = max(self._counts.values()) if self._counts else 1
            max_cnt = max(max_cnt, 1)
            cnt = self._counts.get(key, 0)
            cnt_score: float = cnt / max_cnt

            now = time.time()
            last_ts = self._last_ts.get(key, 0.0)
            if last_ts > 0:
                minutes_diff = (now - last_ts) / 60.0
            else:
                minutes_diff = 9999.0
            recency_score: float = 1.0 / (1.0 + minutes_diff)

            return self._lru_weight * recency_score + (
                1 - self._lru_weight
            ) * cnt_score

    def snapshot(self) -> Dict[ExpertKey, float]:
        """获取所有专家评分快照"""
        with self._lock:
            return {k: self.frequency_score(k) for k in self._counts}

    def top_n_hot_experts(self, n: int = 40) -> List[Tuple[ExpertKey, float]]:
        """获取 Top-N 热门专家 (供 Hermes 自进化提升至 VRAM 常驻)

        Args:
            n: 返回前 N 个

        Returns:
            [(expert_key, score), ...] 按评分降序
        """
        with self._lock:
            scored = [
                (k, self.frequency_score(k)) for k in self._counts
            ]
            scored.sort(key=lambda x: x[1], reverse=True)
            return scored[:n]

    def access_counts(self) -> Dict[ExpertKey, int]:
        """获取所有专家的原始访问计数"""
        with self._lock:
            return dict(self._counts)
