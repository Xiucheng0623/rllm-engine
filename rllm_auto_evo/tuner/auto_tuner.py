# File: D:\AI_RLLM\rllm_auto_evo\tuner\auto_tuner.py
"""自进化自动调优器

触发条件任意满足即迭代：
  - 推理延迟上涨20%
  - 内存突破阈值(2GB)
  - IO阻塞超30s
  - 生成失败率>0.5%

自动调整参数空间：
  - shard_size_mb:    [256, 512, 1024]
  - prefetch_threads: [2, 4, 6, 8]
  - quantization_bits: [4, 8]
  - kv_spill_threshold_mb: [256, 512, 1024]
  - prefetch_layers_ahead: [1, 2, 3, 4]

策略：
  - Grid邻域搜索 + 历史最优回放
  - 每轮调整1-2个参数，避免大震荡
  - 与策略池联动，防止重复尝试已淘汰配置
"""
from __future__ import annotations

import copy
import itertools
import random
import threading
import time
from dataclasses import asdict
from typing import Any, Dict, Iterable, List, Optional

from loguru import logger

from rllm_agent_core import LOG_DIR
from rllm_agent_core.skills.skill_loader import DiskOffloadSkillConfig
from rllm_auto_evo.strategy.strategy_pool import (
    StrategyPool,
    StrategyPerformance,
    get_strategy_pool,
)

logger.add(
    LOG_DIR / "tuner_{time}.log",
    rotation="50 MB",
    retention="7 days",
    encoding="utf-8",
)


# 默认调参搜索空间
# quantization_bits: 4=NF4量化, 8=INT8量化, 16=FP16全量(磁盘分页)
DEFAULT_PARAM_SPACE: Dict[str, List[Any]] = {
    "prefetch_layers_ahead": [1, 2, 3, 4],
    "prefetch_threads": [2, 4, 6, 8],
    "quantization_bits": [4, 8, 16],
    "kv_spill_threshold_mb": [256, 512, 1024],
    "shard_size_mb": [256, 512, 1024],
}


class AutoTuner:
    """自动调优器"""

    def __init__(
        self,
        strategy_pool: Optional[RLLM-StrategyPool] = None,
        param_space: Optional[Dict[str, List[Any]]] = None,
        max_adj_per_round: int = 2,
        seed: int = 20260807,
    ) -> None:
        self._pool = strategy_pool or get_strategy_pool()
        self._param_space = param_space or copy.deepcopy(DEFAULT_PARAM_SPACE)
        self._max_adj = max_adj_per_round
        self._rng = random.Random(seed)
        self._evo_round: int = 0
        self._lock = threading.RLock()
        self._visited_sigs: set = set()
        logger.info(
            f"[RLLM-AutoTuner] 初始化完成: 搜索空间维度={len(self._param_space)}, "
            f"每轮最多调参数={max_adj_per_round}"
        )

    # ----------------------------------------------------------------
    # 主入口：给下一轮配置
    # ----------------------------------------------------------------
    def suggest_next_config(
        self,
        current_cfg: DiskOffloadSkillConfig,
        metrics_dict: Dict[str, Any],
        trigger: str = "heuristic",
    ) -> DiskOffloadSkillConfig:
        """基于当前配置和指标，建议下一配置

        优先顺序：
          1. 根据触发条件做定向强干预
          2. 邻域搜索（1~2个参数扰动）
          3. 回放到历史最优
          4. 随机新组合
        """
        with self._lock:
            self._evo_round += 1
            rid = self._evo_round

        # 1) 强干预（根据触发条件）
        forced = self._force_adjust(current_cfg, metrics_dict, trigger)
        if forced.signature() != current_cfg.signature():
            self._commit(forced, f"forced_{trigger}", rid)
            logger.info(f"[RLLM-AutoTuner] 触发强干预: {trigger} -> sig={forced.signature()}")
            return forced

        # 2) 邻域搜索，尝试未访问过的
        neighborhood = self._generate_neighbors(current_cfg)
        for cand in neighborhood:
            sig = cand.signature()
            if sig not in self._visited_sigs and not self._is_discard(sig):
                self._visited_sigs.add(sig)
                self._commit(cand, f"neighbor_{trigger}", rid)
                logger.info(f"[RLLM-AutoTuner] 邻域搜索 sig={sig}")
                return cand

        # 3) 回放到历史最优
        best = self._pool.get_best()
        if best is not None and not best.discard:
            best_cfg = self._pool.get_config(best.sig)
            if best_cfg is not None and best_cfg.signature() != current_cfg.signature():
                logger.info(f"[RLLM-AutoTuner] 回放到历史最优 sig={best.sig} score={best.performance.score:.2f}")
                self._commit(best_cfg, "best_replay", rid)
                return best_cfg

        # 4) 兜底：随机未访问组合
        cand = self._random_config(current_cfg)
        tries = 0
        while cand.signature() in self._visited_sigs and tries < 50:
            cand = self._random_config(current_cfg)
            tries += 1
        self._visited_sigs.add(cand.signature())
        self._commit(cand, f"random_{trigger}", rid)
        logger.info(f"[RLLM-AutoTuner] 随机配置 sig={cand.signature()}")
        return cand

    # ----------------------------------------------------------------
    # 触发条件 -> 强干预
    # ----------------------------------------------------------------
    def _force_adjust(
        self,
        cfg: DiskOffloadSkillConfig,
        metrics: Dict[str, Any],
        trigger: str,
    ) -> DiskOffloadSkillConfig:
        new_cfg = DiskOffloadSkillConfig(**asdict(cfg))
        peak_mem = float(metrics.get("max_peak_memory_mb", metrics.get("avg_peak_memory_mb", 0.0)))
        io_block = float(metrics.get("max_io_block_ms", 0.0))
        latency_chg = float(metrics.get("latency_change_ratio", 0.0))
        fail_rate = float(metrics.get("failure_rate", metrics.get("failure_flag", 0.0)))

        # 内存超限 -> 量化降位/降预取/降KV阈值
        if peak_mem > 1800:
            new_cfg.quantization_bits = 4
            new_cfg.prefetch_layers_ahead = max(1, cfg.prefetch_layers_ahead - 1)
            new_cfg.kv_spill_threshold_mb = max(256, cfg.kv_spill_threshold_mb // 2)
            return new_cfg

        # IO阻塞严重 -> 提预取线程 / 增大分片 / 增预取层
        if io_block > 5000:
            new_cfg.prefetch_threads = min(16, cfg.prefetch_threads + 2)
            new_cfg.shard_size_mb = min(2048, cfg.shard_size_mb * 2)
            new_cfg.prefetch_layers_ahead = min(4, cfg.prefetch_layers_ahead + 1)
            return new_cfg

        # 延迟大涨20%+ -> 试切量化/降KV阈值
        if latency_chg >= 0.20:
            if cfg.quantization_bits == 16:
                # FP16 太慢, 降级到 8-bit
                new_cfg.quantization_bits = 8
            elif cfg.quantization_bits == 8:
                new_cfg.quantization_bits = 4
            else:
                new_cfg.kv_spill_threshold_mb = max(256, cfg.kv_spill_threshold_mb // 2)
            return new_cfg

        # 失败率高 -> 降位量化/降预取压力
        if fail_rate > 0.005:
            if cfg.quantization_bits == 16:
                new_cfg.quantization_bits = 8
            else:
                new_cfg.quantization_bits = 4
            new_cfg.prefetch_threads = max(2, cfg.prefetch_threads - 2)
            return new_cfg

        return new_cfg

    # ----------------------------------------------------------------
    # 邻域搜索
    # ----------------------------------------------------------------
    def _generate_neighbors(
        self, cfg: DiskOffloadSkillConfig
    ) -> List[DiskOffloadSkillConfig]:
        cfg_dict = asdict(cfg)
        results: List[DiskOffloadSkillConfig] = []
        param_names = list(self._param_space.keys())
        # 1参数邻域
        for name in param_names:
            cur_val = cfg_dict[name]
            space = self._param_space[name]
            if cur_val not in space:
                continue
            idx = space.index(cur_val)
            for delta in (-1, 1):
                ni = idx + delta
                if 0 <= ni < len(space):
                    new_dict = dict(cfg_dict)
                    new_dict[name] = space[ni]
                    results.append(DiskOffloadSkillConfig(**new_dict))
        # 2参数邻域（部分）
        for n1, n2 in itertools.combinations(param_names, 2):
            for d1 in (-1, 1):
                for d2 in (-1, 1):
                    try:
                        i1 = self._param_space[n1].index(cfg_dict[n1]) + d1
                        i2 = self._param_space[n2].index(cfg_dict[n2]) + d2
                        if 0 <= i1 < len(self._param_space[n1]) and 0 <= i2 < len(self._param_space[n2]):
                            new_dict = dict(cfg_dict)
                            new_dict[n1] = self._param_space[n1][i1]
                            new_dict[n2] = self._param_space[n2][i2]
                            results.append(DiskOffloadSkillConfig(**new_dict))
                    except ValueError:
                        continue
                    if len(results) > 50:
                        return results
        return results

    # ----------------------------------------------------------------
    def _random_config(self, base: DiskOffloadSkillConfig) -> DiskOffloadSkillConfig:
        cfg_dict = asdict(base)
        chosen = self._rng.sample(list(self._param_space.keys()), k=min(self._max_adj, len(self._param_space)))
        for name in chosen:
            cfg_dict[name] = self._rng.choice(self._param_space[name])
        return DiskOffloadSkillConfig(**cfg_dict)

    # ----------------------------------------------------------------
    def _commit(
        self, cfg: DiskOffloadSkillConfig, trigger: str, rid: int
    ) -> None:
        try:
            self._pool.add_or_update(cfg, trigger=trigger, evo_round=rid)
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"[RLLM-AutoTuner] 策略池写入失败: {exc}")

    def _is_discard(self, sig: str) -> bool:
        rec = None
        for r in self._pool.list_all():
            if r.sig == sig:
                rec = r
                break
        return bool(rec and rec.discard)

    # ----------------------------------------------------------------
    # 与复盘引擎协同：写入评分
    # ----------------------------------------------------------------
    def record_score(
        self,
        sig: str,
        score: float,
        avg_latency_ms: float,
        avg_tps: float,
        avg_mem_mb: float,
        fail_rate: float,
    ) -> None:
        perf = StrategyPerformance(
            avg_latency_ms=avg_latency_ms,
            avg_throughput_tps=avg_tps,
            avg_peak_memory_mb=avg_mem_mb,
            failure_rate=fail_rate,
            score=score,
        )
        self._pool.update_performance(sig, perf)


# 单例
_tuner_singleton: Optional[RLLM-AutoTuner] = None
_tuner_lock = threading.Lock()


def get_auto_tuner(**kwargs) -> AutoTuner:
    global _tuner_singleton
    if _tuner_singleton is None:
        with _tuner_lock:
            if _tuner_singleton is None:
                _tuner_singleton = AutoTuner(**kwargs)
    return _tuner_singleton
