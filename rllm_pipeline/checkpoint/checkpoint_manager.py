# File: D:\AI_RLLM\rllm_pipeline\checkpoint\checkpoint_manager.py
"""断点续跑检查点管理器

记录：
  - 已成功完成的输入idx集合（去重）
  - 当前处理到的最新idx
  - 累计成功率、累计耗时、当前策略签名
持久化：D盘 JSON 原子写（tmp->rename），避免断点文件损坏。
"""
from __future__ import annotations

import json
import os
import tempfile
import threading
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from loguru import logger

from rllm_agent_core import LOG_DIR

logger.add(
    LOG_DIR / "checkpoint_{time}.log",
    rotation="50 MB",
    retention="7 days",
    encoding="utf-8",
)

DEFAULT_CKPT_PATH: Path = Path(r"D:\AI_RLLM\rllm_pipeline\checkpoint\pipeline_ckpt.json")


@dataclass
class PipelineCheckpoint:
    """检查点快照（强类型）"""
    completed_idx: List[int] = field(default_factory=list)
    last_processed_idx: int = -1
    total_submitted: int = 0
    total_success: int = 0
    total_failed: int = 0
    current_strategy_sig: str = ""
    updated_ts: float = field(default_factory=time.time)
    extra_stats: Dict[str, Any] = field(default_factory=dict)


class CheckpointManager:
    """断点续跑检查点管理器（原子写）"""

    def __init__(
        self,
        checkpoint_path: Path = DEFAULT_CKPT_PATH,
        save_every: int = 50,
        max_completed_in_memory: int = 500_000,
    ) -> None:
        self._path = Path(checkpoint_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._save_every = save_every
        self._max_mem = max_completed_in_memory
        self._lock = threading.RLock()
        self._ckpt = PipelineCheckpoint()
        self._completed_set: Set[int] = set()
        self._dirty: int = 0
        self._load()
        logger.info(
            f"[RLLM-CheckpointMgr] 初始化: path={self._path}, "
            f"已完成={len(self._completed_set)}, last_idx={self._ckpt.last_processed_idx}"
        )

    # ----------------------------------------------------------------
    # 对外：标记完成/失败
    # ----------------------------------------------------------------
    def mark_success(self, idx: int) -> None:
        with self._lock:
            if idx not in self._completed_set:
                self._completed_set.add(idx)
                if len(self._ckpt.completed_idx) < self._max_mem:
                    self._ckpt.completed_idx.append(idx)
            self._ckpt.total_submitted += 1
            self._ckpt.total_success += 1
            self._ckpt.last_processed_idx = max(self._ckpt.last_processed_idx, idx)
            self._ckpt.updated_ts = time.time()
            self._dirty += 1
            if self._dirty >= self._save_every:
                self._save_locked()

    def mark_failed(self, idx: int, reason: str = "") -> None:
        with self._lock:
            self._ckpt.total_submitted += 1
            self._ckpt.total_failed += 1
            self._ckpt.last_processed_idx = max(self._ckpt.last_processed_idx, idx)
            if reason:
                errs = self._ckpt.extra_stats.setdefault("recent_errors", [])
                errs.append({"idx": idx, "ts": time.time(), "reason": reason[:200]})
                if len(errs) > 100:
                    del errs[:-100]
            self._ckpt.updated_ts = time.time()
            self._dirty += 1
            if self._dirty >= self._save_every:
                self._save_locked()

    def update_strategy_sig(self, sig: str) -> None:
        with self._lock:
            self._ckpt.current_strategy_sig = sig
            self._dirty += 1

    def update_extra_stats(self, key: str, value: Any) -> None:
        with self._lock:
            self._ckpt.extra_stats[key] = value
            self._dirty += 1

    # ----------------------------------------------------------------
    # 对外：查询
    # ----------------------------------------------------------------
    def is_done(self, idx: int) -> bool:
        with self._lock:
            return idx in self._completed_set

    def completed_set(self) -> Set[int]:
        with self._lock:
            return set(self._completed_set)

    def snapshot(self) -> PipelineCheckpoint:
        with self._lock:
            data = asdict(self._ckpt)
            return PipelineCheckpoint(**data)

    def save(self) -> None:
        with self._lock:
            self._save_locked()

    def reset(self) -> None:
        """清空所有状态（重新跑整个数据集）"""
        with self._lock:
            self._ckpt = PipelineCheckpoint()
            self._completed_set = set()
            self._dirty = 1
            self._save_locked()
        logger.info("[RLLM-CheckpointMgr] 检查点已清空重置")

    # ----------------------------------------------------------------
    # 内部：持久化（原子写）
    # ----------------------------------------------------------------
    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            with open(self._path, "r", encoding="utf-8") as fp:
                raw = json.load(fp)
            completed = list(raw.get("completed_idx", []))
            # 去重
            self._completed_set = set(int(x) for x in completed)
            # 截断过长列表
            if len(completed) > self._max_mem:
                completed = completed[-self._max_mem:]
            raw["completed_idx"] = completed
            self._ckpt = PipelineCheckpoint(**{
                k: raw.get(k, v)
                for k, v in asdict(PipelineCheckpoint()).items()
            })
        except Exception as exc:  # noqa: BLE001
            logger.error(f"[RLLM-CheckpointMgr] 加载失败，新建空白检查点: {exc}")
            self._backup_broken()
            self._ckpt = PipelineCheckpoint()

    def _save_locked(self) -> None:
        payload = asdict(self._ckpt)
        # 截断completed_idx为最大内存量，避免json巨大
        if len(payload["completed_idx"]) > self._max_mem:
            kept = payload["completed_idx"][-self._max_mem:]
            payload["completed_idx"] = kept
            self._ckpt.completed_idx = list(kept)
        tmp_dir = str(self._path.parent)
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                prefix=f"{self._path.stem}_",
                suffix=".tmp",
                dir=tmp_dir,
                delete=False,
                encoding="utf-8",
            ) as tmp:
                json.dump(payload, tmp, ensure_ascii=False, indent=2)
                tmp_path = tmp.name
            # 原子替换
            os.replace(tmp_path, self._path)
            self._dirty = 0
        except Exception as exc:  # noqa: BLE001
            logger.error(f"[RLLM-CheckpointMgr] 保存失败: {exc}")
            try:
                os.unlink(tmp_path)  # type: ignore[possibly-undefined]
            except Exception:  # noqa: BLE001
                pass

    def _backup_broken(self) -> None:
        try:
            backup = self._path.with_suffix(self._path.suffix + f".broken_{int(time.time())}")
            os.rename(self._path, backup)
            logger.warning(f"[RLLM-CheckpointMgr] 已备份损坏文件到 {backup}")
        except Exception:  # noqa: BLE001
            pass


# 单例
_ckpt_singleton: Optional[CheckpointManager] = None
_ckpt_lock = threading.Lock()


def get_checkpoint_manager(**kwargs) -> CheckpointManager:
    global _ckpt_singleton
    if _ckpt_singleton is None:
        with _ckpt_lock:
            if _ckpt_singleton is None:
                _ckpt_singleton = CheckpointManager(**kwargs)
    return _ckpt_singleton
