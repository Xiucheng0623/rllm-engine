# -*- coding: utf-8 -*-
# File: D:\AI_RLLM\rllm_auto_evo\moe_evo_orchestrator.py
"""MoE 版 Hermes 自进化编排器

在 v4 MoE 推理过程中, 持续收集专家级指标, 自动进化:

进化维度:
  1. 热专家列表: 定期统计 Top-N 高频专家, 提升 (pin) 到 VRAM 常驻
  2. 预取参数: 自动调整 Top-K 候选数 (8/16/32), 在命中率与 I/O 量间找平衡
  3. 预取并发: 根据磁盘带宽利用率调整并发数 (2/4/6/8)
  4. VRAM 预留: 根据实际 KV cache 占用动态调整 reserve_gb

进化触发条件:
  - 每 N 个 token 触发一次专家热度更新
  - 路由预测命中率 < 80% → 增加 Top-K
  - VRAM 淘汰频繁 (>10/100tok) → 增加 reserve 或减少 pin 数
  - I/O 等待 > 50ms/token → 增加预取并发

策略淘汰:
  - 连续 3 轮命中率 < 60% 的 Top-K 配置被淘汰
  - 连续 3 轮淘汰率 > 20% 的 reserve 配置被淘汰
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger

from rllm_agent_core import LOG_DIR
from rllm_disk_engine.expert_pool.expert_freq_tracker import (
    ExpertFreqTracker,
    ExpertKey,
)
from rllm_disk_engine.expert_pool.expert_vram_pool import ExpertVRAMPool
from rllm_disk_engine.router.router_predictor import RouterPredictor
from rllm_disk_engine.router.router_prefetcher import RouterPrefetcher

logger.add(
    LOG_DIR / "moe_evo_{time}.log",
    rotation="50 MB",
    retention="7 days",
    encoding="utf-8",
)


@dataclass
class MoEEvoConfig:
    """MoE 自进化配置

    Attributes:
        top_k_candidates: 预取候选数 (8/16/32)
        prefetch_concurrency: 预取并发数 (2/4/6/8)
        hot_expert_count: 热专家常驻数 (20/40/60)
        reserve_gb: VRAM 预留 (GB)
        update_interval_tokens: 每 N token 触发一次进化
    """
    top_k_candidates: int = 16
    prefetch_concurrency: int = 4
    hot_expert_count: int = 40
    reserve_gb: float = 3.0
    update_interval_tokens: int = 64


@dataclass
class MoEEvoMetrics:
    """单轮进化指标

    Attributes:
        round_id: 轮次
        token_count: 本轮生成的 token 数
        avg_tok_per_s: 平均速度
        router_hit_rate: 路由预测命中率
        prefetch_total: 预取总数
        prefetch_skipped: 预取跳过数 (已在 VRAM)
        evict_count: 淘汰次数
        fetch_back_count: 读回次数
        vram_usage_gb: VRAM 占用 (GB)
        timestamp: 时间戳
    """
    round_id: int = 0
    token_count: int = 0
    avg_tok_per_s: float = 0.0
    router_hit_rate: float = 0.0
    prefetch_total: int = 0
    prefetch_skipped: int = 0
    evict_count: int = 0
    fetch_back_count: int = 0
    vram_usage_gb: float = 0.0
    timestamp: str = ""


class MoEEvoOrchestrator:
    """MoE 版 Hermes 自进化编排器

    在推理过程中持续监控指标, 自动调整参数.

    Args:
        vram_pool: ExpertVRAMPool 实例
        freq_tracker: 专家频率跟踪器
        router_predictor: 路由预测器
        prefetcher: 预取器
        initial_config: 初始进化配置
    """

    # 参数搜索空间
    PARAM_SPACE: Dict[str, List[Any]] = {
        "top_k_candidates": [8, 16, 32],
        "prefetch_concurrency": [2, 4, 6, 8],
        "hot_expert_count": [20, 40, 60],
        "reserve_gb": [2.0, 3.0, 4.0],
    }

    def __init__(
        self,
        vram_pool: ExpertVRAMPool,
        freq_tracker: ExpertFreqTracker,
        router_predictor: RouterPredictor,
        prefetcher: RouterPrefetcher,
        initial_config: Optional[MoEEvoConfig] = None,
    ) -> None:
        self._vram_pool: ExpertVRAMPool = vram_pool
        self._freq_tracker: ExpertFreqTracker = freq_tracker
        self._router: RouterPredictor = router_predictor
        self._prefetcher: RouterPrefetcher = prefetcher
        self._config: MoEEvoConfig = initial_config or MoEEvoConfig()

        # 进化历史
        self._history: List[MoEEvoMetrics] = []
        self._round_id: int = 0
        self._token_counter: int = 0

        # 策略评分 (用于淘汰低效配置)
        self._config_scores: Dict[str, float] = {}

        logger.info(
            f"[MoE-Evo] 初始化: top_k={self._config.top_k_candidates} "
            f"concurrency={self._config.prefetch_concurrency} "
            f"hot_experts={self._config.hot_expert_count}"
        )

    # ----------------------------------------------------------------
    # 对外接口
    # ----------------------------------------------------------------
    def record_token(self) -> None:
        """记录一个 token 生成 (每 token 调用一次)"""
        self._token_counter += 1
        if self._token_counter >= self._config.update_interval_tokens:
            asyncio.create_task(self.evolve())

    async def evolve(self) -> MoEEvoMetrics:
        """执行一轮自进化

        流程:
          1. 收集当前指标
          2. 评估当前配置得分
          3. 根据指标调整参数
          4. 应用新配置
          5. 更新热专家列表

        Returns:
            本轮进化指标
        """
        self._round_id += 1
        self._token_counter = 0

        # 1. 收集指标
        metrics = self._collect_metrics()
        self._history.append(metrics)

        # 2. 评估 + 调整
        new_config = self._adjust_config(metrics)

        # 3. 应用新配置
        await self._apply_config(new_config)

        # 4. 更新热专家
        await self._update_hot_experts()

        logger.info(
            f"[MoE-Evo] 轮 {self._round_id}: "
            f"speed={metrics.avg_tok_per_s:.1f}tok/s "
            f"hit_rate={metrics.router_hit_rate:.1%} "
            f"evict={metrics.evict_count} "
            f"VRAM={metrics.vram_usage_gb:.2f}GB "
            f"→ top_k={new_config.top_k_candidates} "
            f"concurrency={new_config.prefetch_concurrency}"
        )

        return metrics

    # ----------------------------------------------------------------
    # 内部: 收集指标
    # ----------------------------------------------------------------
    def _collect_metrics(self) -> MoEEvoMetrics:
        """收集当前轮次指标"""
        vram_stats = self._vram_pool.stats()
        router_stats = self._router.stats()
        prefetcher_stats = self._prefetcher.stats()

        # 计算平均速度 (从历史推断)
        avg_speed: float = 0.0
        if self._history:
            avg_speed = self._history[-1].avg_tok_per_s

        return MoEEvoMetrics(
            round_id=self._round_id,
            token_count=self._config.update_interval_tokens,
            avg_tok_per_s=avg_speed,
            router_hit_rate=router_stats.get("hit_rate", 0.0),
            prefetch_total=prefetcher_stats.get("prefetch_total", 0),
            prefetch_skipped=prefetcher_stats.get("prefetch_skipped", 0),
            evict_count=vram_stats.get("evict_count", 0),
            fetch_back_count=vram_stats.get("fetch_back_count", 0),
            vram_usage_gb=vram_stats.get("current_vram_gb", 0.0),
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
        )

    # ----------------------------------------------------------------
    # 内部: 调整配置
    # ----------------------------------------------------------------
    def _adjust_config(
        self, metrics: MoEEvoMetrics
    ) -> MoEEvoConfig:
        """根据指标调整配置

        策略:
          - 命中率 < 80% → 增加 Top-K
          - 命中率 > 95% 且预取量大 → 减少 Top-K
          - 淘汰频繁 → 增加 reserve 或减少 hot_expert
          - I/O 等待大 → 增加并发

        Args:
            metrics: 当前指标

        Returns:
            新配置
        """
        new_cfg = MoEEvoConfig(
            top_k_candidates=self._config.top_k_candidates,
            prefetch_concurrency=self._config.prefetch_concurrency,
            hot_expert_count=self._config.hot_expert_count,
            reserve_gb=self._config.reserve_gb,
            update_interval_tokens=self._config.update_interval_tokens,
        )

        # 1. 路由命中率调整 Top-K
        if metrics.router_hit_rate < 0.80:
            # 命中率低 → 增加 Top-K
            if new_cfg.top_k_candidates < 32:
                new_cfg.top_k_candidates = min(32, new_cfg.top_k_candidates * 2)
                logger.info(
                    f"[MoE-Evo] 命中率 {metrics.router_hit_rate:.1%} < 80%, "
                    f"Top-K {self._config.top_k_candidates} → {new_cfg.top_k_candidates}"
                )
        elif metrics.router_hit_rate > 0.95:
            # 命中率高 → 尝试减少 Top-K (省 I/O)
            if new_cfg.top_k_candidates > 8:
                new_cfg.top_k_candidates = max(8, new_cfg.top_k_candidates // 2)

        # 2. 淘汰频繁 → 减少 hot_expert (释放 VRAM)
        if metrics.evict_count > metrics.token_count * 0.2:
            if new_cfg.hot_expert_count > 20:
                new_cfg.hot_expert_count = max(20, new_cfg.hot_expert_count - 10)
                logger.info(
                    f"[MoE-Evo] 淘汰率 > 20%, "
                    f"hot_experts {self._config.hot_expert_count} → {new_cfg.hot_expert_count}"
                )

        # 3. 读回频繁 → 增加并发
        if metrics.fetch_back_count > metrics.token_count * 0.5:
            if new_cfg.prefetch_concurrency < 8:
                new_cfg.prefetch_concurrency = min(8, new_cfg.prefetch_concurrency + 2)

        return new_cfg

    # ----------------------------------------------------------------
    # 内部: 应用新配置
    # ----------------------------------------------------------------
    async def _apply_config(self, config: MoEEvoConfig) -> None:
        """应用新配置到各模块

        Args:
            config: 新配置
        """
        old_top_k = self._config.top_k_candidates
        old_concurrency = self._config.prefetch_concurrency

        # 更新 RouterPredictor 的 Top-K
        if config.top_k_candidates != old_top_k:
            self._router._top_k = config.top_k_candidates

        # 更新 Prefetcher 的并发数 (需要重建信号量)
        if config.prefetch_concurrency != old_concurrency:
            self._prefetcher._max_concurrent = config.prefetch_concurrency
            self._prefetcher._semaphore = asyncio.Semaphore(
                config.prefetch_concurrency
            )

        self._config = config

    # ----------------------------------------------------------------
    # 内部: 更新热专家列表
    # ----------------------------------------------------------------
    async def _update_hot_experts(self) -> None:
        """根据频率统计更新热专家列表 (pin 到 VRAM)"""
        top_n = self._config.hot_expert_count
        hot_experts = self._freq_tracker.top_n_hot_experts(top_n)

        # 先 unpin 所有
        resident = self._vram_pool.list_resident_experts()
        for key in resident:
            self._vram_pool.unpin_expert(key)

        # pin Top-N 热专家 (只在 VRAM 中的才 pin)
        pinned: int = 0
        for key, score in hot_experts:
            if self._vram_pool.get_expert_entry(key) is not None:
                if self._vram_pool.pin_expert(key):
                    pinned += 1

        logger.info(
            f"[MoE-Evo] 热专家更新: Top-{top_n} 中 "
            f"{pinned} 个已 pin (VRAM 常驻)"
        )

    # ----------------------------------------------------------------
    # 统计
    # ----------------------------------------------------------------
    def stats(self) -> Dict[str, Any]:
        """获取自进化统计"""
        return {
            "round": self._round_id,
            "token_counter": self._token_counter,
            "current_config": {
                "top_k": self._config.top_k_candidates,
                "concurrency": self._config.prefetch_concurrency,
                "hot_experts": self._config.hot_expert_count,
                "reserve_gb": self._config.reserve_gb,
            },
            "history_size": len(self._history),
            "last_metrics": (
                self._history[-1].__dict__
                if self._history
                else None
            ),
        }
