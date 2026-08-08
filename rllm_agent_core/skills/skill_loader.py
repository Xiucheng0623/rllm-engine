# File: D:\AI_RLLM\rllm_agent_core\skills\skill_loader.py
"""Hermes 可插拔 Skill 体系

核心新增 DiskOffloadInferSkill：封装磁盘分页推理完整调用链，
自动记录调用指标，触发Hermes复盘引擎的自进化逻辑。
"""
from __future__ import annotations

import abc
import asyncio
import json
import hashlib
import threading
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from loguru import logger

from rllm_agent_core import SKILL_STORAGE_DIR, SKILL_ARCHIVE_DIR, LOG_DIR
from rllm_agent_core.workers.worker_registry import (
    get_worker,
    WorkerContext,
    WorkerResult,
)

logger.add(
    LOG_DIR / "skills_{time}.log",
    rotation="50 MB",
    retention="7 days",
    encoding="utf-8",
)


# ============================================================
# Skill 基础定义
# ============================================================
@dataclass
class SkillInvocation:
    """Skill调用记录（强类型）"""
    skill_id: str
    task_id: str
    input_hash: str
    start_ts: float
    end_ts: float = 0.0
    success: bool = False
    output_size_bytes: int = 0
    metrics: Dict[str, float] = field(default_factory=dict)
    error: str = ""


class SkillBase(abc.ABC):
    """Skill抽象基类"""
    skill_id: str = "base"
    skill_name: str = "基础技能"
    version: str = "1.0.0"

    @abc.abstractmethod
    async def execute(self, **kwargs: Any) -> Dict[str, Any]:
        """执行技能"""
        raise NotImplementedError

    def description(self) -> str:
        return f"[{self.skill_id} v{self.version}] {self.skill_name}"


# ============================================================
# Skill 注册中心
# ============================================================
class SkillRegistry:
    """Skill单例注册中心 + 存档落D盘"""
    _instance: Optional["SkillRegistry"] = None

    def __new__(cls) -> "SkillRegistry":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        if not hasattr(self, "_lock"):
            self._lock = threading.RLock()
            self._skills: Dict[str, SkillBase] = {}
            self._invocation_log: List[SkillInvocation] = []

    def register(self, skill: SkillBase) -> None:
        """注册Skill"""
        with self._lock:
            self._skills[skill.skill_id] = skill
            self._persist_skill_manifest()
            logger.info(f"[RLLM-SkillRegistry] 注册Skill: {skill.description()}")

    def get(self, skill_id: str) -> Optional[SkillBase]:
        return self._skills.get(skill_id)

    def list_skills(self) -> List[str]:
        return list(self._skills.keys())

    def record_invocation(self, inv: SkillInvocation) -> None:
        """记录一次Skill调用，由复盘引擎消费"""
        with self._lock:
            self._invocation_log.append(inv)
            self._flush_invocation_log()

    def drain_invocations(self, limit: int = 1000) -> List[SkillInvocation]:
        """复盘引擎拉取指标"""
        with self._lock:
            data = self._invocation_log[:limit]
            self._invocation_log = self._invocation_log[limit:]
            return data

    # ----------------------------------------------------------------
    def _persist_skill_manifest(self) -> None:
        """落D盘技能清单"""
        manifest_path = SKILL_STORAGE_DIR / "manifest.json"
        manifest = {
            skill_id: {
                "name": s.skill_name,
                "version": s.version,
            }
            for skill_id, s in self._skills.items()
        }
        with open(manifest_path, "w", encoding="utf-8") as fp:
            json.dump(manifest, fp, ensure_ascii=False, indent=2)

    def _flush_invocation_log(self) -> None:
        """调用日志落D盘"""
        if len(self._invocation_log) > 5000:
            ts = int(time.time())
            archive = SKILL_ARCHIVE_DIR / f"invocations_{ts}.jsonl"
            archive.parent.mkdir(parents=True, exist_ok=True)
            try:
                with open(archive, "w", encoding="utf-8") as fp:
                    for inv in self._invocation_log:
                        fp.write(json.dumps(asdict(inv), ensure_ascii=False) + "\n")
                self._invocation_log = []
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"[RLLM-SkillRegistry] 调用日志落盘失败: {exc}")


def register_default_skills() -> None:
    """注册默认Skill"""
    registry = SkillRegistry()
    registry.register(DiskOffloadInferSkill())
    logger.info(f"[RLLM-SkillRegistry] 默认Skill已注册: {registry.list_skills()}")


def load_skill(skill_id: str = "disk_offload_infer") -> SkillBase:
    """加载Skill，懒注册"""
    registry = SkillRegistry()
    skill = registry.get(skill_id)
    if skill is None:
        register_default_skills()
        skill = registry.get(skill_id)
    if skill is None:
        raise KeyError(f"Skill不存在: {skill_id}")
    return skill


# ============================================================
# 核心 Skill：磁盘分页推理封装
# ============================================================
@dataclass
class DiskOffloadSkillConfig:
    """磁盘推理Skill的可调配置（自进化闭环修改对象）"""
    prefetch_layers_ahead: int = 2
    prefetch_threads: int = 4
    quantization_bits: int = 8
    kv_spill_threshold_mb: int = 512
    shard_size_mb: int = 512
    cpu_buffer_gb: float = 2.0

    def signature(self) -> str:
        """配置指纹，用于去重/对比"""
        raw = json.dumps(asdict(self), sort_keys=True)
        return hashlib.md5(raw.encode("utf-8")).hexdigest()[:12]


class DiskOffloadInferSkill(SkillBase):
    """RLLM 磁盘分页推理专属Skill (disk_offload_infer.skill)

    封装：
      1. 调用 DiskLLMInferWorker 完成推理
      2. 采集并归档调用指标（复盘引擎消费）
      3. 支持配置热更新（由自进化调优器写入）
    """
    skill_id: str = "disk_offload_infer"
    skill_name: str = "磁盘分页低内存大模型推理技能"
    version: str = "1.0.0-DiskOffload"

    _CONFIG_PATH: Path = SKILL_STORAGE_DIR / "disk_offload_infer_config.json"

    def __init__(self) -> None:
        self._config: DiskOffloadSkillConfig = self._load_config()
        self._last_save_ts: float = 0.0
        self._save_config()

    # ----------------------------------------------------------------
    # 配置加载/保存（D盘）
    # ----------------------------------------------------------------
    @classmethod
    def _load_config(cls) -> DiskOffloadSkillConfig:
        if cls._CONFIG_PATH.exists():
            try:
                with open(cls._CONFIG_PATH, "r", encoding="utf-8") as fp:
                    raw = json.load(fp)
                return DiskOffloadSkillConfig(**raw)
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"[RLLM-DiskSkill] 配置加载失败，使用默认: {exc}")
        return DiskOffloadSkillConfig()

    def _save_config(self) -> None:
        SKILL_STORAGE_DIR.mkdir(parents=True, exist_ok=True)
        with open(self._CONFIG_PATH, "w", encoding="utf-8") as fp:
            json.dump(asdict(self._config), fp, ensure_ascii=False, indent=2)
        self._last_save_ts = time.time()

    def apply_config(self, new_cfg: DiskOffloadSkillConfig) -> None:
        """由自进化调优器热更新配置（核心闭环入口）"""
        old_sig = self._config.signature()
        self._config = new_cfg
        new_sig = self._config.signature()
        self._save_config()
        # 存档历史策略
        if old_sig != new_sig:
            archive_path = SKILL_ARCHIVE_DIR / f"skill_cfg_{new_sig}_{int(time.time())}.json"
            archive_path.parent.mkdir(parents=True, exist_ok=True)
            with open(archive_path, "w", encoding="utf-8") as fp:
                json.dump(asdict(self._config), fp, ensure_ascii=False, indent=2)
        logger.info(
            f"[RLLM-DiskSkill] 配置已更新 sig={new_sig} "
            f"prefetch={new_cfg.prefetch_layers_ahead} "
            f"quant={new_cfg.quantization_bits}bit "
            f"kv_limit={new_cfg.kv_spill_threshold_mb}MB"
        )

    def get_config(self) -> DiskOffloadSkillConfig:
        return DiskOffloadSkillConfig(**asdict(self._config))

    # ----------------------------------------------------------------
    # 核心执行入口
    # ----------------------------------------------------------------
    async def execute(self, **kwargs: Any) -> Dict[str, Any]:
        """执行磁盘分页推理

        Kwargs:
            task_id (str): 任务ID
            prompt (str): 输入提示词
            max_new_tokens (int): 最大生成token数
            temperature (float): 采样温度
            top_p (float): top-p
            extra_params (dict): 额外参数

        Returns:
            {
              "task_id": str,
              "success": bool,
              "generated_text": str,
              "latency_sec": float,
              "peak_memory_mb": float,
              "tokens_generated": int,
              "io_metrics": {...},
              "skill_config_sig": str,
              "error": str,
            }
        """
        task_id: str = str(kwargs.get("task_id", f"tsk_{int(time.time()*1000)}"))
        prompt: str = str(kwargs.get("prompt", ""))
        max_new_tokens: int = int(kwargs.get("max_new_tokens", 512))
        temperature: float = float(kwargs.get("temperature", 0.7))
        top_p: float = float(kwargs.get("top_p", 0.9))
        extra_params: Dict[str, Any] = dict(kwargs.get("extra_params", {}))
        extra_params["skill_config"] = asdict(self._config)

        inv = SkillInvocation(
            skill_id=self.skill_id,
            task_id=task_id,
            input_hash=hashlib.md5(prompt.encode("utf-8")).hexdigest(),
            start_ts=time.time(),
        )
        result_dict: Dict[str, Any] = {}
        try:
            worker = get_worker("disk_llm_infer")
            ctx = WorkerContext(
                task_id=task_id,
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                extra_params=extra_params,
            )
            wres: WorkerResult = await worker.execute(ctx)

            inv.end_ts = time.time()
            inv.success = wres.success
            inv.output_size_bytes = len(wres.generated_text.encode("utf-8"))
            inv.metrics = {
                "latency_sec": wres.latency_sec,
                "peak_memory_mb": wres.peak_memory_mb,
                "tokens_generated": float(wres.tokens_generated),
                "throughput_tps": wres.tokens_generated / max(0.001, wres.latency_sec),
                **wres.io_metrics,
            }
            inv.error = wres.error_msg

            result_dict = {
                "task_id": wres.task_id,
                "success": wres.success,
                "generated_text": wres.generated_text,
                "latency_sec": wres.latency_sec,
                "peak_memory_mb": wres.peak_memory_mb,
                "tokens_generated": wres.tokens_generated,
                "io_metrics": dict(wres.io_metrics),
                "skill_config_sig": self._config.signature(),
                "error": wres.error_msg,
            }
            return result_dict
        except Exception as exc:  # noqa: BLE001
            logger.exception(f"[RLLM-DiskSkill] 调用失败 {task_id}: {exc}")
            inv.end_ts = time.time()
            inv.success = False
            inv.error = str(exc)
            result_dict = {
                "task_id": task_id,
                "success": False,
                "generated_text": "",
                "latency_sec": inv.end_ts - inv.start_ts,
                "peak_memory_mb": 0.0,
                "tokens_generated": 0,
                "io_metrics": {},
                "skill_config_sig": self._config.signature(),
                "error": str(exc),
            }
            return result_dict
        finally:
            SkillRegistry().record_invocation(inv)
