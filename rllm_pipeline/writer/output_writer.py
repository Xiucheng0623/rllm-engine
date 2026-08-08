# File: D:\AI_RLLM\rllm_pipeline\writer\output_writer.py
"""数据集输出写入器

策略：
  - 生成结果直接追加到D盘JSONL文件，不保留在内存中
  - 每写入N条flush一次，避免频繁IO
  - 支持自动按时间/条数切分文件（最大500MB/100万条）
  - 失败重试3次写入，避免丢数
"""
from __future__ import annotations

import asyncio
import json
import threading
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger

from rllm_agent_core import OUTPUT_DATASET_DIR, LOG_DIR

logger.add(
    LOG_DIR / "pipeline_writer_{time}.log",
    rotation="50 MB",
    retention="7 days",
    encoding="utf-8",
)


@dataclass
class OutputRecord:
    """单条输出记录（对应JSONL一行）"""
    idx: int
    source_file: str
    keyword: str
    category: str
    style: str
    generated_text: str
    prompt: str = ""
    task_id: str = ""
    success: bool = True
    error_msg: str = ""
    latency_sec: float = 0.0
    peak_memory_mb: float = 0.0
    tokens_generated: int = 0
    io_metrics: Dict[str, float] = field(default_factory=dict)
    skill_config_sig: str = ""
    created_ts: float = field(default_factory=time.time)


class OutputDatasetWriter:
    """数据集输出写入器（D盘JSONL，不驻留内存）"""

    MAX_BYTES_PER_FILE: int = 500 * 1024 * 1024  # 500MB
    MAX_LINES_PER_FILE: int = 1_000_000
    FLUSH_EVERY: int = 32

    def __init__(
        self,
        output_dir: Path = OUTPUT_DATASET_DIR,
        flush_every: int = FLUSH_EVERY,
    ) -> None:
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._flush_every = flush_every
        self._lock = threading.RLock()
        self._current_path: Optional[Path] = None
        self._current_fp = None
        self._lines_in_current: int = 0
        self._bytes_in_current: int = 0
        self._file_index: int = self._detect_next_index()
        self._pending_flush: int = 0
        self._total_written: int = 0
        self._open_new_file()
        logger.info(
            f"[RLLM-OutputWriter] 初始化: dir={self._output_dir}, "
            f"flush_every={flush_every}, next_index={self._file_index}"
        )

    # ----------------------------------------------------------------
    # 对外：写入
    # ----------------------------------------------------------------
    async def write_async(self, rec: OutputRecord) -> bool:
        """异步写入（线程安全，失败重试3次）"""
        line = json.dumps(asdict(rec), ensure_ascii=False)
        for attempt in range(3):
            try:
                with self._lock:
                    if self._current_fp is None:
                        self._open_new_file()
                    assert self._current_fp is not None
                    self._current_fp.write(line + "\n")
                    self._lines_in_current += 1
                    self._bytes_in_current += len(line) + 1
                    self._total_written += 1
                    self._pending_flush += 1
                    if (
                        self._pending_flush >= self._flush_every
                        or self._bytes_in_current >= self.MAX_BYTES_PER_FILE
                        or self._lines_in_current >= self.MAX_LINES_PER_FILE
                    ):
                        self._flush_and_rotate_locked()
                return True
            except Exception as exc:  # noqa: BLE001
                wait = 0.1 * (attempt + 1)
                logger.warning(
                    f"[RLLM-OutputWriter] 写入失败 (尝试{attempt+1}/3) "
                    f"idx={rec.idx} err={exc}，等待{wait}s"
                )
                await asyncio.sleep(wait)
        logger.error(f"[RLLM-OutputWriter] 丢数 idx={rec.idx}")
        return False

    def write_sync(self, rec: OutputRecord) -> bool:
        """同步写入（兼容）"""
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(self.write_async(rec))
        finally:
            loop.close()

    def flush_all(self) -> None:
        with self._lock:
            self._flush_locked()

    def close(self) -> None:
        with self._lock:
            self._flush_locked()
            if self._current_fp is not None:
                try:
                    self._current_fp.close()
                except Exception:  # noqa: BLE001
                    pass
                self._current_fp = None
        logger.info(f"[RLLM-OutputWriter] 关闭，累计写入 {self._total_written} 条")

    def total_written(self) -> int:
        with self._lock:
            return self._total_written

    def current_file(self) -> Optional[Path]:
        return self._current_path

    # ----------------------------------------------------------------
    # 内部
    # ----------------------------------------------------------------
    def _detect_next_index(self) -> int:
        indices: List[int] = []
        for f in self._output_dir.glob("output_*.jsonl"):
            try:
                n = int(f.stem.split("_")[-1])
                indices.append(n)
            except ValueError:
                continue
        return (max(indices) + 1) if indices else 0

    def _open_new_file(self) -> None:
        ts = time.strftime("%Y%m%d")
        name = f"output_{ts}_{self._file_index:06d}.jsonl"
        self._current_path = self._output_dir / name
        self._current_fp = open(self._current_path, "a", encoding="utf-8", buffering=1)
        self._lines_in_current = 0
        self._bytes_in_current = 0
        logger.info(f"[RLLM-OutputWriter] 新建输出文件: {self._current_path}")

    def _flush_locked(self) -> None:
        if self._current_fp is not None:
            try:
                self._current_fp.flush()
            except Exception:  # noqa: BLE001
                pass
        self._pending_flush = 0

    def _flush_and_rotate_locked(self) -> None:
        self._flush_locked()
        if (
            self._bytes_in_current >= self.MAX_BYTES_PER_FILE
            or self._lines_in_current >= self.MAX_LINES_PER_FILE
        ):
            if self._current_fp is not None:
                try:
                    self._current_fp.close()
                except Exception:  # noqa: BLE001
                    pass
                self._current_fp = None
            self._file_index += 1
            self._open_new_file()


# 单例
_writer_singleton: Optional[OutputDatasetWriter] = None
_writer_lock = threading.Lock()


def get_output_writer(**kwargs) -> OutputDatasetWriter:
    global _writer_singleton
    if _writer_singleton is None:
        with _writer_lock:
            if _writer_singleton is None:
                _writer_singleton = OutputDatasetWriter(**kwargs)
    return _writer_singleton
