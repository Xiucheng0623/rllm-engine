# File: D:\AI_RLLM\rllm_auto_evo\metrics\metrics_collector.py
"""磁盘推理指标采集器

采集清单（对应复盘引擎消费）：
  - 单层磁盘读取耗时 (layer_read_ms)
  - 内存峰值 (peak_memory_mb)
  - 批量吞吐 (throughput_tps)
  - SSD缓存命中率 (ssd_cache_hit_ratio)
  - IO阻塞时长 (io_block_ms)
  - 磁盘碎片率 (disk_frag_score)
  - 量化精度-耗时平衡值 (quant_score)

持久化D盘 auto_evo 目录，供复盘引擎批量拉取。
"""
from __future__ import annotations

import asyncio
import json
import threading
import time
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger

from rllm_agent_core import LOG_DIR

logger.add(
    LOG_DIR / "metrics_{time}.log",
    rotation="50 MB",
    retention="7 days",
    encoding="utf-8",
)

METRICS_DIR: Path = Path(r"D:\AI_RLLM\rllm_auto_evo\metrics\data")
METRICS_DIR.mkdir(parents=True, exist_ok=True)


class MetricName(str, Enum):
    LAYER_READ_MS = "layer_read_ms"
    PEAK_MEMORY_MB = "peak_memory_mb"
    THROUGHPUT_TPS = "throughput_tps"
    SSD_CACHE_HIT = "ssd_cache_hit_ratio"
    IO_BLOCK_MS = "io_block_ms"
    DISK_FRAG_SCORE = "disk_frag_score"
    QUANT_SCORE = "quant_score"
    KV_SPILL_COUNT = "kv_spill_count"
    FAILURE = "failure_flag"


@dataclass
class MetricPoint:
    """单条指标点"""
    ts: float = field(default_factory=time.time)
    round_id: int = 0
    task_id: str = ""
    strategy_sig: str = ""
    metric: str = ""
    value: float = 0.0
    tags: Dict[str, str] = field(default_factory=dict)


class MetricsCollector:
    """指标采集器（线程安全，批量落盘）"""

    def __init__(self, flush_every: int = 500, data_dir: Path = METRICS_DIR) -> None:
        self._data_dir = Path(data_dir)
        self._data_dir.mkdir(parents=True, exist_ok=True)
        self._buffer: List[MetricPoint] = []
        self._flush_every = flush_every
        self._lock = threading.RLock()
        self._round_counter: int = 0
        self._current_strategy_sig: str = ""
        self._file_index: int = 0
        logger.info(f"[RLLM-MetricsCollector] 初始化: flush={flush_every}, dir={self._data_dir}")

    # ----------------------------------------------------------------
    # 对外：记录
    # ----------------------------------------------------------------
    def set_strategy(self, sig: str) -> None:
        with self._lock:
            self._current_strategy_sig = sig

    def set_round(self, rid: int) -> None:
        with self._lock:
            self._round_counter = rid

    def record(
        self,
        metric: str,
        value: float,
        task_id: str = "",
        tags: Optional[Dict[str, str]] = None,
    ) -> None:
        """记录一条指标"""
        point = MetricPoint(
            round_id=self._round_counter,
            task_id=task_id,
            strategy_sig=self._current_strategy_sig,
            metric=str(metric),
            value=float(value),
            tags=tags or {},
        )
        with self._lock:
            self._buffer.append(point)
            if len(self._buffer) >= self._flush_every:
                self._flush_locked()

    def record_batch(
        self,
        metrics_dict: Dict[str, float],
        task_id: str = "",
        tags: Optional[Dict[str, str]] = None,
    ) -> None:
        """批量记录"""
        for k, v in metrics_dict.items():
            self.record(k, v, task_id, tags)

    # ----------------------------------------------------------------
    # 对外：查询与统计
    # ----------------------------------------------------------------
    def tail_values(
        self, metric: str, n: int = 100, strategy_sig: Optional[str] = None
    ) -> List[float]:
        """拉取最近N个某指标值"""
        self.flush()
        out: List[float] = []
        files = sorted(self._data_dir.glob("metrics_*.jsonl"), reverse=True)
        for fp in files:
            if len(out) >= n:
                break
            try:
                with open(fp, "r", encoding="utf-8") as f:
                    lines = f.readlines()
                for line in reversed(lines):
                    if len(out) >= n:
                        break
                    try:
                        raw = json.loads(line)
                        if raw["metric"] != metric:
                            continue
                        if strategy_sig and raw.get("strategy_sig") != strategy_sig:
                            continue
                        out.append(float(raw["value"]))
                    except Exception:  # noqa: BLE001
                        pass
            except Exception:  # noqa: BLE001
                continue
        return out

    def summarize_window(
        self, metric: str, n: int = 500
    ) -> Dict[str, float]:
        """最近N个值的统计摘要"""
        vals = self.tail_values(metric, n)
        if not vals:
            return {"count": 0, "avg": 0.0, "p95": 0.0, "max": 0.0, "min": 0.0}
        s = sorted(vals)
        def _pct(p: float) -> float:
            k = (len(s) - 1) * p
            f = int(k); c = min(f+1, len(s)-1)
            return s[f] if f == c else s[f] + (s[c]-s[f])*(k-f)
        return {
            "count": len(vals),
            "avg": sum(vals) / len(vals),
            "p95": _pct(0.95),
            "max": s[-1],
            "min": s[0],
        }

    # ----------------------------------------------------------------
    def flush(self) -> None:
        with self._lock:
            self._flush_locked()

    def _flush_locked(self) -> None:
        if not self._buffer:
            return
        file_path = self._data_dir / f"metrics_{self._file_index:06d}.jsonl"
        try:
            with open(file_path, "a", encoding="utf-8") as fp:
                for p in self._buffer:
                    fp.write(json.dumps(asdict(p), ensure_ascii=False) + "\n")
            # 文件切分：超100MB换下一个
            if file_path.stat().st_size > 100 * 1024 * 1024:
                self._file_index += 1
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"[RLLM-MetricsCollector] 落盘失败: {exc}")
        finally:
            self._buffer = []


# 单例
_collector_singleton: Optional[RLLM-MetricsCollector] = None
_collector_lock = threading.Lock()


def get_metrics_collector(**kwargs) -> MetricsCollector:
    global _collector_singleton
    if _collector_singleton is None:
        with _collector_lock:
            if _collector_singleton is None:
                _collector_singleton = MetricsCollector(**kwargs)
    return _collector_singleton
