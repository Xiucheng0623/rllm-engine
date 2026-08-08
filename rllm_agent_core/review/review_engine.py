# File: D:\AI_RLLM\rllm_agent_core\review\review_engine.py
"""RLLM 复盘引擎(底层复用Hermes架构)（改造版）

改造原生 review_engine.py，接入：
  1. 磁盘IO指标采集（单层读取耗时、SSD缓存命中率、IO阻塞、碎片率）
  2. 内存/吞吐指标（内存峰值、批量吞吐、KV溢出次数）
  3. 自进化触发逻辑（延迟涨20%/内存超限/IO阻塞>30s/失败率>0.5%）
  4. 调用 auto_evo 模块产出新配置，写回 DiskOffloadInferSkill
"""
from __future__ import annotations

import asyncio
import json
import statistics
import threading
import time
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from loguru import logger

from rllm_agent_core import SKILL_STORAGE_DIR, LOG_DIR
from rllm_agent_core.skills.skill_loader import (
    SkillRegistry,
    SkillInvocation,
    DiskOffloadInferSkill,
    DiskOffloadSkillConfig,
    load_skill,
)

logger.add(
    LOG_DIR / "review_{time}.log",
    rotation="50 MB",
    retention="7 days",
    encoding="utf-8",
)

EVO_REPORT_DIR: Path = SKILL_STORAGE_DIR / "evo_reports"
EVO_REPORT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 进化触发枚举
# ============================================================
class EvolutionTrigger(str, Enum):
    """自进化触发条件枚举"""
    NONE = "none"
    LATENCY_INCREASE = "latency_increase_20pct"
    MEMORY_BREACH = "memory_threshold_breach"
    IO_BLOCK = "io_block_over_30s"
    FAILURE_RATE = "failure_rate_over_0_5pct"
    PERIODIC = "periodic_scheduled"


# ============================================================
# 复盘指标结构
# ============================================================
@dataclass
class ReviewMetrics:
    """复盘引擎聚合指标（强类型）"""
    # ---- 基础统计 ----
    window_rounds: int = 0
    success_count: int = 0
    failure_count: int = 0
    failure_rate: float = 0.0

    # ---- 时延 ----
    latency_list_ms: List[float] = field(default_factory=list)
    avg_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    latency_change_ratio: float = 0.0  # 较历史基线变化比例

    # ---- 内存 ----
    peak_memory_list_mb: List[float] = field(default_factory=list)
    avg_peak_memory_mb: float = 0.0
    max_peak_memory_mb: float = 0.0
    memory_breach_count: int = 0  # 超2GB次数

    # ---- 吞吐 ----
    throughput_list_tps: List[float] = field(default_factory=list)
    avg_throughput_tps: float = 0.0

    # ---- 磁盘IO ----
    avg_layer_read_ms: float = 0.0
    total_io_block_ms: float = 0.0
    max_io_block_ms: float = 0.0
    kv_spill_total: int = 0
    ssd_cache_hit_ratio: float = 0.0
    disk_fragmentation_score: float = 0.0

    # ---- 量化平衡值 (精度得分/耗时比) ----
    quant_accuracy_time_score: float = 0.0

    # ---- 结论 ----
    triggers: List[EvolutionTrigger] = field(default_factory=list)
    should_evolve: bool = False


# ============================================================
# 复盘引擎
# ============================================================
class ReviewEngine:
    """Hermes复盘引擎（改造接入自进化闭环）

    工作流程：
      1. 周期性从 SkillRegistry 拉取 invocation 日志
      2. 聚合成 ReviewMetrics，按触发条件判断
      3. 若触发进化，调用 auto_evo.tuner 生成候选配置
      4. 对比新旧性能，优胜劣汰，写回 Skill 持久化到D盘
    """

    def __init__(self, baseline_rounds: int = 50) -> None:
        self._baseline_rounds = baseline_rounds
        self._round_counter: int = 0
        self._baseline_metrics: Optional[ReviewMetrics] = None
        self._history_windows: List[ReviewMetrics] = []
        self._lock = threading.RLock()
        self._best_strategy_sig: Optional[str] = None
        self._best_score: float = float("-inf")
        self._best_config: Optional[DiskOffloadSkillConfig] = None
        self._evo_count: int = 0
        self._load_state()
        logger.info("[RLLM-ReviewEngine] 复盘引擎初始化完成")

    # ----------------------------------------------------------------
    # 状态持久化（D盘）
    # ----------------------------------------------------------------
    _STATE_PATH: Path = SKILL_STORAGE_DIR / "review_engine_state.json"

    def _load_state(self) -> None:
        if self._STATE_PATH.exists():
            try:
                with open(self._STATE_PATH, "r", encoding="utf-8") as fp:
                    raw = json.load(fp)
                self._best_strategy_sig = raw.get("best_strategy_sig")
                self._best_score = float(raw.get("best_score", float("-inf")))
                self._evo_count = int(raw.get("evo_count", 0))
                if raw.get("best_config"):
                    self._best_config = DiskOffloadSkillConfig(**raw["best_config"])
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"[RLLM-ReviewEngine] 状态加载失败: {exc}")

    def _save_state(self) -> None:
        payload: Dict[str, Any] = {
            "best_strategy_sig": self._best_strategy_sig,
            "best_score": self._best_score,
            "evo_count": self._evo_count,
            "best_config": asdict(self._best_config) if self._best_config else None,
            "updated_ts": time.time(),
        }
        with open(self._STATE_PATH, "w", encoding="utf-8") as fp:
            json.dump(payload, fp, ensure_ascii=False, indent=2)

    # ----------------------------------------------------------------
    # 1. 指标聚合
    # ----------------------------------------------------------------
    def collect_and_aggregate(
        self,
        invocations: List[SkillInvocation],
    ) -> ReviewMetrics:
        """聚合一批调用日志为结构化指标"""
        m = ReviewMetrics(window_rounds=len(invocations))
        if not invocations:
            return m

        latency_ms_list: List[float] = []
        mem_list: List[float] = []
        tps_list: List[float] = []
        layer_read_ms: List[float] = []
        io_block_ms_list: List[float] = []
        kv_spill_sum = 0
        io_block_total = 0.0
        io_block_max = 0.0

        for inv in invocations:
            if inv.success:
                m.success_count += 1
            else:
                m.failure_count += 1
            dur_ms = (inv.end_ts - inv.start_ts) * 1000 if inv.end_ts > 0 else 0.0
            latency_ms_list.append(dur_ms)

            met = inv.metrics
            mem_list.append(float(met.get("peak_memory_mb", 0.0)))
            tps_list.append(float(met.get("throughput_tps", 0.0)))
            layer_read_ms.append(float(met.get("avg_layer_read_ms", 0.0)))
            block_ms = float(met.get("io_block_ms", 0.0))
            io_block_ms_list.append(block_ms)
            io_block_total += block_ms
            io_block_max = max(io_block_max, block_ms)
            kv_spill_sum += int(met.get("kv_spill_count", 0))

            # 内存突破硬锁次数
            if float(met.get("peak_memory_mb", 0.0)) > 2048.0:
                m.memory_breach_count += 1

        total = len(invocations)
        m.failure_rate = m.failure_count / max(1, total)
        m.latency_list_ms = latency_ms_list
        m.avg_latency_ms = self._safe_mean(latency_ms_list)
        m.p95_latency_ms = self._percentile(latency_ms_list, 0.95)
        m.peak_memory_list_mb = mem_list
        m.avg_peak_memory_mb = self._safe_mean(mem_list)
        m.max_peak_memory_mb = max(mem_list) if mem_list else 0.0
        m.throughput_list_tps = tps_list
        m.avg_throughput_tps = self._safe_mean(tps_list)
        m.avg_layer_read_ms = self._safe_mean(layer_read_ms)
        m.total_io_block_ms = io_block_total
        m.max_io_block_ms = io_block_max
        m.kv_spill_total = kv_spill_sum
        # SSD缓存命中率：模拟 (1 - kv_spill_total/total 归一化)
        m.ssd_cache_hit_ratio = max(0.0, min(1.0, 1.0 - (kv_spill_sum / max(1, total * 10))))
        # 磁盘碎片率估算：IO阻塞时长 / 总时长
        total_dur = sum(latency_ms_list)
        m.disk_fragmentation_score = io_block_total / max(1.0, total_dur)
        # 量化精度-耗时平衡分 (tps * (1 - failure_rate))
        m.quant_accuracy_time_score = m.avg_throughput_tps * max(0.0, 1.0 - m.failure_rate)

        # 较基线延迟变化率
        if self._baseline_metrics and self._baseline_metrics.avg_latency_ms > 0:
            m.latency_change_ratio = (
                (m.avg_latency_ms - self._baseline_metrics.avg_latency_ms)
                / self._baseline_metrics.avg_latency_ms
            )
        return m

    # ----------------------------------------------------------------
    # 2. 触发判定
    # ----------------------------------------------------------------
    def evaluate_triggers(
        self,
        metrics: ReviewMetrics,
        config: Optional[DiskOffloadSkillConfig] = None,
    ) -> List[EvolutionTrigger]:
        """判定进化触发条件"""
        triggers: List[EvolutionTrigger] = []
        # (a) 延迟上涨20%
        if metrics.latency_change_ratio >= 0.20:
            triggers.append(EvolutionTrigger.LATENCY_INCREASE)
        # (b) 内存突破阈值（次数>0）
        if metrics.memory_breach_count > 0:
            triggers.append(EvolutionTrigger.MEMORY_BREACH)
        # (c) IO阻塞超30s (单次)
        if metrics.max_io_block_ms >= 30_000:
            triggers.append(EvolutionTrigger.IO_BLOCK)
        # (d) 失败率>0.5%
        if metrics.failure_rate > 0.005:
            triggers.append(EvolutionTrigger.FAILURE_RATE)
        return triggers

    # ----------------------------------------------------------------
    # 3. 执行进化闭环
    # ----------------------------------------------------------------
    async def run_review_cycle(self, min_samples: int = 10) -> Dict[str, Any]:
        """执行一次复盘-进化周期

        Returns:
            复盘结果字典（用于外部监控面板）
        """
        with self._lock:
            # 拉取调用记录
            invs = SkillRegistry().drain_invocations(limit=5000)
            if len(invs) < min_samples and self._baseline_metrics is not None:
                return {"skipped": True, "reason": "样本不足", "samples": len(invs)}

            metrics = self.collect_and_aggregate(invs)
            metrics.triggers = self.evaluate_triggers(metrics)
            metrics.should_evolve = len(metrics.triggers) > 0

            # 建立初始基线
            if self._baseline_metrics is None and metrics.window_rounds >= self._baseline_rounds:
                self._baseline_metrics = metrics
                logger.info(f"[RLLM-ReviewEngine] 建立初始基线: 平均延迟={metrics.avg_latency_ms:.1f}ms")

            # 分数（综合：吞吐-惩罚）
            score = self._score_metrics(metrics)

            skill: DiskOffloadInferSkill = load_skill("disk_offload_infer")  # type: ignore[assignment]
            current_cfg = skill.get_config()

            # 保存最优
            if score > self._best_score and metrics.failure_rate <= 0.01:
                self._best_score = score
                self._best_config = DiskOffloadSkillConfig(**asdict(current_cfg))
                self._best_strategy_sig = current_cfg.signature()
                logger.info(f"[RLLM-ReviewEngine] 新最优策略 sig={self._best_strategy_sig} score={score:.3f}")

            result: Dict[str, Any] = {
                "round": self._round_counter,
                "samples": metrics.window_rounds,
                "score": score,
                "best_score": self._best_score,
                "triggers": [t.value for t in metrics.triggers],
                "should_evolve": metrics.should_evolve,
                "failure_rate": metrics.failure_rate,
                "avg_latency_ms": metrics.avg_latency_ms,
                "avg_throughput_tps": metrics.avg_throughput_tps,
                "avg_peak_memory_mb": metrics.avg_peak_memory_mb,
                "max_io_block_ms": metrics.max_io_block_ms,
            }

            # 触发进化：调用 auto_evo 调优器
            if metrics.should_evolve or (self._round_counter % 100 == 0 and self._round_counter > 0):
                await self._do_evolve(current_cfg, metrics)

            self._round_counter += 1
            self._save_state()

            # 持久化复盘报告
            self._write_report(locals())
            return result

    # ----------------------------------------------------------------
    # 4. 进化动作（调用 auto_evo.tuner）
    # ----------------------------------------------------------------
    async def _do_evolve(
        self,
        current_cfg: DiskOffloadSkillConfig,
        metrics: ReviewMetrics,
    ) -> DiskOffloadSkillConfig:
        """调用自进化调优器产出新配置并应用"""
        self._evo_count += 1
        try:
            from rllm_auto_evo.tuner.auto_tuner import AutoTuner
            from rllm_auto_evo.strategy.strategy_pool import StrategyPool

            pool = StrategyPool()
            tuner = AutoTuner(strategy_pool=pool)
            new_cfg = tuner.suggest_next_config(current_cfg, asdict(metrics) if metrics else {})
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"[RLLM-ReviewEngine] 调优器不可用，使用启发式调优: {exc}")
            new_cfg = self._heuristic_tune(current_cfg, metrics)

        skill: DiskOffloadInferSkill = load_skill("disk_offload_infer")  # type: ignore[assignment]
        skill.apply_config(new_cfg)
        logger.info(
            f"[RLLM-ReviewEngine] 第{self._evo_count}次自进化完成，"
            f"配置更新 sig={new_cfg.signature()}，已持久化到D盘"
        )
        return new_cfg

    def _heuristic_tune(
        self,
        cfg: DiskOffloadSkillConfig,
        metrics: Dict[str, Any],
    ) -> DiskOffloadSkillConfig:
        """启发式调优（调优器不可用时兜底）"""
        new_cfg = DiskOffloadSkillConfig(**asdict(cfg))
        peak_mem = float(metrics.get("avg_peak_memory_mb", 0.0)) if isinstance(metrics, dict) else 0.0
        io_block_ms = float(metrics.get("max_io_block_ms", 0.0)) if isinstance(metrics, dict) else 0.0

        # 内存超限 -> 降位4bit、降预取、降KV阈值
        if peak_mem > 1800.0:
            new_cfg.quantization_bits = 4
            new_cfg.prefetch_layers_ahead = max(1, cfg.prefetch_layers_ahead - 1)
            new_cfg.kv_spill_threshold_mb = max(128, cfg.kv_spill_threshold_mb // 2)
        # IO阻塞严重 -> 提预取线程、开mmap、增shard
        if io_block_ms > 5000:
            new_cfg.prefetch_threads = min(16, cfg.prefetch_threads + 2)
            new_cfg.shard_size_mb = min(2048, cfg.shard_size_mb * 2)
        return new_cfg

    # ----------------------------------------------------------------
    # 辅助
    # ----------------------------------------------------------------
    @staticmethod
    def _safe_mean(values: List[float]) -> float:
        return float(statistics.mean(values)) if values else 0.0

    @staticmethod
    def _percentile(values: List[float], p: float) -> float:
        if not values:
            return 0.0
        s = sorted(values)
        k = (len(s) - 1) * p
        f = int(k)
        c = min(f + 1, len(s) - 1)
        if f == c:
            return float(s[f])
        return float(s[f] + (s[c] - s[f]) * (k - f))

    @staticmethod
    def _score_metrics(m: ReviewMetrics) -> float:
        """综合评分（越大越好）"""
        # 基础分 = 吞吐 * 100
        score = m.avg_throughput_tps * 100.0
        # 惩罚：延迟ms / 10
        score -= m.avg_latency_ms / 10.0
        # 惩罚：内存MB (超2GB重罚)
        score -= min(500.0, m.avg_peak_memory_mb) / 10.0
        if m.max_peak_memory_mb > 2048.0:
            score -= (m.max_peak_memory_mb - 2048.0) * 2
        # 惩罚：失败率
        score -= m.failure_rate * 10_000
        # 惩罚：IO阻塞s
        score -= m.total_io_block_ms / 1000.0
        return round(score, 4)

    def _write_report(self, data: Dict[str, Any]) -> None:
        """落D盘复盘报告"""
        path = EVO_REPORT_DIR / f"review_{int(time.time())}.json"
        try:
            with open(path, "w", encoding="utf-8") as fp:
                json.dump(data, fp, ensure_ascii=False, indent=2)
        except Exception:  # noqa: BLE001
            pass


# 单例
_instance_rev: Optional[RLLM-ReviewEngine] = None
_rev_lock = threading.Lock()


def get_review_engine() -> ReviewEngine:
    global _instance_rev
    if _instance_rev is None:
        with _rev_lock:
            if _instance_rev is None:
                _instance_rev = ReviewEngine()
    return _instance_rev
