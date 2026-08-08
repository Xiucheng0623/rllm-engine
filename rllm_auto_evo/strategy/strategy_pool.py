# File: D:\AI_RLLM\rllm_auto_evo\strategy\strategy_pool.py
"""磁盘调度策略池（优胜劣汰）

机制：
  - 每次配置变化产生一个 StrategyRecord（签名+配置+性能+触发源）
  - 存档上限 100 条，超出自动淘汰最低分的
  - 最优策略标记为 active，复盘引擎优先推荐
  - 策略持久化到 D:\\AI_RLLM\\rllm_skill_storage\\strategy_pool.json
"""
from __future__ import annotations

import copy
import json
import threading
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger

from rllm_agent_core import SKILL_STORAGE_DIR, LOG_DIR
from rllm_agent_core.skills.skill_loader import DiskOffloadSkillConfig

logger.add(
    LOG_DIR / "strategy_{time}.log",
    rotation="50 MB",
    retention="7 days",
    encoding="utf-8",
)

POOL_PATH: Path = SKILL_STORAGE_DIR / "strategy_pool.json"


@dataclass
class StrategyPerformance:
    """某策略运行时性能统计"""
    samples: int = 0
    avg_latency_ms: float = 0.0
    avg_throughput_tps: float = 0.0
    avg_peak_memory_mb: float = 0.0
    avg_io_block_ms: float = 0.0
    failure_rate: float = 0.0
    score: float = 0.0  # 综合得分 = 吞吐*100 - 延迟 - 失败罚分


@dataclass
class StrategyRecord:
    """策略存档记录"""
    sig: str
    config: Dict[str, Any]  # DiskOffloadSkillConfig.asdict
    created_ts: float = field(default_factory=time.time)
    trigger: str = "manual"
    performance: StrategyPerformance = field(default_factory=StrategyPerformance)
    evo_round: int = 0
    is_best: bool = False
    discard: bool = False  # 淘汰标记


class StrategyPool:
    """策略池（最多保留max_records条）"""

    def __init__(
        self,
        pool_path: Path = POOL_PATH,
        max_records: int = 100,
    ) -> None:
        self._path = Path(pool_path)
        self._max = max_records
        self._records: Dict[str, StrategyRecord] = {}
        self._lock = threading.RLock()
        self._best_sig: Optional[str] = None
        self._load()
        logger.info(
            f"[RLLM-StrategyPool] 初始化完成: path={self._path}, "
            f"现有策略={len(self._records)}, 上限={max_records}"
        )

    # ----------------------------------------------------------------
    # 增 / 更新
    # ----------------------------------------------------------------
    def add_or_update(
        self,
        config: DiskOffloadSkillConfig,
        trigger: str = "manual",
        evo_round: int = 0,
    ) -> StrategyRecord:
        """新增或更新策略记录"""
        cfg_dict = asdict(config)
        sig = config.signature()
        with self._lock:
            rec = self._records.get(sig)
            if rec is None:
                rec = StrategyRecord(
                    sig=sig,
                    config=cfg_dict,
                    trigger=trigger,
                    evo_round=evo_round,
                )
                self._records[sig] = rec
                logger.info(f"[RLLM-StrategyPool] 新增策略 sig={sig} trigger={trigger}")
            else:
                rec.trigger = trigger
                rec.evo_round = evo_round
            self._maybe_evict_locked()
            self._save_locked()
            return rec

    def update_performance(
        self,
        sig: str,
        perf: StrategyPerformance,
    ) -> None:
        with self._lock:
            rec = self._records.get(sig)
            if rec is None:
                return
            rec.performance = perf
            self._recompute_best_locked()
            self._save_locked()

    def mark_discard(self, sig: str) -> None:
        with self._lock:
            rec = self._records.get(sig)
            if rec:
                rec.discard = True
                self._save_locked()

    # ----------------------------------------------------------------
    # 查询
    # ----------------------------------------------------------------
    def get_best(self) -> Optional[StrategyRecord]:
        with self._lock:
            if self._best_sig:
                return self._records.get(self._best_sig)
            return self._sorted_records_locked()[0] if self._records else None

    def get_config(self, sig: str) -> Optional[DiskOffloadSkillConfig]:
        with self._lock:
            rec = self._records.get(sig)
        if rec is None:
            return None
        return DiskOffloadSkillConfig(**rec.config)

    def list_all(self) -> List[StrategyRecord]:
        with self._lock:
            return list(self._records.values())

    def list_sorted(self, top_n: int = 20) -> List[StrategyRecord]:
        with self._lock:
            return self._sorted_records_locked()[:top_n]

    # ----------------------------------------------------------------
    # 内部
    # ----------------------------------------------------------------
    def _recompute_best_locked(self) -> None:
        ranked = self._sorted_records_locked()
        if ranked:
            new_best = ranked[0].sig
            if new_best != self._best_sig:
                for r in self._records.values():
                    r.is_best = False
                ranked[0].is_best = True
                self._best_sig = new_best
                logger.info(f"[RLLM-StrategyPool] 新最优策略 sig={new_best} score={ranked[0].performance.score:.2f}")

    def _sorted_records_locked(self) -> List[StrategyRecord]:
        def _key(r: StrategyRecord) -> float:
            if r.discard:
                return float("-inf")
            return r.performance.score
        return sorted(self._records.values(), key=_key, reverse=True)

    def _maybe_evict_locked(self) -> None:
        if len(self._records) <= self._max:
            return
        ranked = self._sorted_records_locked()
        # 淘汰末尾
        to_remove = [r.sig for r in ranked[self._max:]]
        for sig in to_remove:
            self._records.pop(sig, None)
        logger.info(f"[RLLM-StrategyPool] 淘汰 {len(to_remove)} 条历史低效策略")

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            with open(self._path, "r", encoding="utf-8") as fp:
                raw = json.load(fp)
            for item in raw.get("records", []):
                perf_dict = item.pop("performance", {})
                perf = StrategyPerformance(**perf_dict)
                rec = StrategyRecord(**item, performance=perf)
                self._records[rec.sig] = rec
                if rec.is_best:
                    self._best_sig = rec.sig
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"[RLLM-StrategyPool] 加载失败: {exc}")

    def _save_locked(self) -> None:
        try:
            data = {
                "best_sig": self._best_sig,
                "updated_ts": time.time(),
                "records": [asdict(r) for r in self._records.values()],
            }
            with open(self._path, "w", encoding="utf-8") as fp:
                json.dump(data, fp, ensure_ascii=False, indent=2)
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"[RLLM-StrategyPool] 保存失败: {exc}")


# 单例
_pool_singleton: Optional[RLLM-StrategyPool] = None
_pool_lock = threading.Lock()


def get_strategy_pool(**kwargs) -> StrategyPool:
    global _pool_singleton
    if _pool_singleton is None:
        with _pool_lock:
            if _pool_singleton is None:
                _pool_singleton = StrategyPool(**kwargs)
    return _pool_singleton
