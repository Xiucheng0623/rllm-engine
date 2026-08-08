# File: D:\AI_RLLM\rllm_pipeline\batch_reader\keyword_reader.py
"""关键词批量输入读取器

从 D:\\AI_RLLM\\input_data 读取关键词文件：
  - .txt 每行一个关键词，逗号分隔类别
  - .jsonl {"keyword": "...", "category": "...", "style": "..."}

支持批处理、断点续跑（CheckpointManager 标记已读取的行号）。
"""
from __future__ import annotations

import asyncio
import csv
import json
import threading
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional

from loguru import logger

from rllm_agent_core import INPUT_DATA_DIR, LOG_DIR

logger.add(
    LOG_DIR / "pipeline_reader_{time}.log",
    rotation="50 MB",
    retention="7 days",
    encoding="utf-8",
)


@dataclass
class BatchInput:
    """单条批量输入（强类型）"""
    idx: int                       # 全局行号（用于断点续跑）
    source_file: str
    keyword: str
    category: str = "general"
    style: str = "default"
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_prompt(self, template: Optional[str] = None) -> str:
        """拼接推理Prompt"""
        if template is None:
            template = (
                "请基于以下关键词，生成一段小红书风格的图文内容文案与配图描述：\n"
                "关键词：{keyword}\n分类：{category}\n风格：{style}\n"
                "要求：包含标题(emoji+吸睛)、正文(3-5段)、标签(#xxx x5)、配图描述(2-3句)"
            )
        return template.format(
            keyword=self.keyword,
            category=self.category,
            style=self.style,
            **self.extra,
        )


class KeywordBatchReader:
    """关键词批量输入读取器"""

    SUPPORTED_SUFFIX: tuple = (".txt", ".jsonl", ".csv")

    def __init__(
        self,
        input_dir: Path = INPUT_DATA_DIR,
        batch_size: int = 8,
        start_line: int = 0,
    ) -> None:
        self._input_dir = Path(input_dir)
        self._input_dir.mkdir(parents=True, exist_ok=True)
        self._batch_size = batch_size
        self._start_line = start_line
        self._lock = threading.RLock()
        logger.info(
            f"[RLLM-BatchReader] 初始化: dir={self._input_dir}, "
            f"batch={batch_size}, start_line={start_line}"
        )

    # ----------------------------------------------------------------
    # 文件发现
    # ----------------------------------------------------------------
    def list_files(self) -> List[Path]:
        files: List[Path] = []
        for suf in self.SUPPORTED_SUFFIX:
            files.extend(sorted(self._input_dir.glob(f"*{suf}")))
        return files

    # ----------------------------------------------------------------
    # 逐行扫描 (同步)
    # ----------------------------------------------------------------
    def iter_all(self) -> List[BatchInput]:
        """同步读取全部输入到列表（注意：百万级建议用iter_rows_async）"""
        outs: List[BatchInput] = []
        line_no = 0
        for fp in self.list_files():
            for rec in self._iter_file(fp):
                if line_no >= self._start_line:
                    outs.append(rec)
                line_no += 1
        logger.info(f"[RLLM-BatchReader] 读取 {len(outs)} 条输入 (跳过 {self._start_line})")
        return outs

    async def iter_rows_async(
        self,
        skip_if_done: Optional[set] = None,
    ) -> AsyncIterator[BatchInput]:
        """异步逐行读取，支持跳过已完成集合"""
        skip_done = skip_if_done or set()
        line_no = 0
        for fp in self.list_files():
            for rec in self._iter_file(fp):
                try:
                    if line_no < self._start_line:
                        continue
                    if skip_done and rec.idx in skip_done:
                        continue
                    yield rec
                    await asyncio.sleep(0)  # 让出
                finally:
                    line_no += 1

    async def iter_batches_async(
        self,
        skip_if_done: Optional[set] = None,
    ) -> AsyncIterator[List[BatchInput]]:
        """按批yield"""
        buf: List[BatchInput] = []
        async for rec in self.iter_rows_async(skip_if_done=skip_if_done):
            buf.append(rec)
            if len(buf) >= self._batch_size:
                yield buf
                buf = []
        if buf:
            yield buf

    # ----------------------------------------------------------------
    # 内部：单文件解析
    # ----------------------------------------------------------------
    def _iter_file(self, fp: Path) -> List[BatchInput]:
        records: List[BatchInput] = []
        suffix = fp.suffix.lower()
        try:
            if suffix == ".txt":
                with open(fp, "r", encoding="utf-8") as f:
                    for lineno, raw in enumerate(f):
                        line = raw.strip()
                        if not line or line.startswith("#"):
                            continue
                        parts = [p.strip() for p in line.split(",")]
                        keyword = parts[0]
                        cat = parts[1] if len(parts) > 1 else "general"
                        style = parts[2] if len(parts) > 2 else "default"
                        records.append(BatchInput(
                            idx=len(records),
                            source_file=fp.name,
                            keyword=keyword,
                            category=cat,
                            style=style,
                        ))
            elif suffix == ".jsonl":
                with open(fp, "r", encoding="utf-8") as f:
                    for lineno, raw in enumerate(f):
                        line = raw.strip()
                        if not line:
                            continue
                        try:
                            obj = json.loads(line)
                            records.append(BatchInput(
                                idx=len(records),
                                source_file=fp.name,
                                keyword=str(obj.get("keyword", obj.get("text", ""))),
                                category=str(obj.get("category", "general")),
                                style=str(obj.get("style", "default")),
                                extra={k: v for k, v in obj.items() if k not in {"keyword", "category", "style"}},
                            ))
                        except Exception as exc:  # noqa: BLE001
                            logger.warning(f"[RLLM-BatchReader] JSONL行 {lineno} 失败: {exc}")
            elif suffix == ".csv":
                with open(fp, "r", encoding="utf-8-sig", newline="") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        records.append(BatchInput(
                            idx=len(records),
                            source_file=fp.name,
                            keyword=str(row.get("keyword", row.get("text", ""))),
                            category=str(row.get("category", "general")),
                            style=str(row.get("style", "default")),
                            extra={k: v for k, v in row.items() if k not in {"keyword", "category", "style"}},
                        ))
        except Exception as exc:  # noqa: BLE001
            logger.error(f"[RLLM-BatchReader] 读取文件 {fp} 失败: {exc}")
        # 修正idx为全局偏移（暂简化为单文件递增）
        for i, r in enumerate(records):
            r.idx = i
        return records
