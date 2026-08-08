# File: D:\AI_RLLM\rllm_agent_core\config\hermes_config.py
"""Rebirth LLM(RLLM) 全局配置模块

强类型配置，D盘路径硬编码，Google风格文档字符串。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional

from rllm_agent_core import HERMES_ROOT, SKILL_STORAGE_DIR


# ============================================================
# 1. 磁盘卸载推理配置
# ============================================================
@dataclass
class DiskOffloadConfig:
    """磁盘分页推理调度参数

    Attributes:
        model_shards_dir: 模型分片存储目录 (D:\\AI_RLLM\\rllm_model_shards)
        offload_temp_dir: 磁盘卸载临时缓存目录
        kv_spill_threshold_mb: KV缓存溢出阈值(MB)，超过即写入磁盘
        cpu_buffer_limit_gb: CPU内存缓冲区硬上限(GB)，固定2GB
        prefetch_layers_ahead: 异步预取层数（下N层提前读入内存窗口）
        quantization_bits: 权重量化位数 (4 或 8)
        shard_size_mb: 单分片目标大小(MB)，按Transformer层对齐
        enable_mmap: 是否启用mmap映射加速磁盘IO
        prefetch_threads: 异步预取线程池大小
    """
    model_shards_dir: Path = Path("D:/AI_RLLM/rllm_model_shards")
    offload_temp_dir: Path = Path("D:/AI_RLLM/rllm_offload_temp")
    kv_spill_threshold_mb: int = 512
    cpu_buffer_limit_gb: float = 2.0
    prefetch_layers_ahead: int = 2
    quantization_bits: int = 8
    shard_size_mb: int = 512
    enable_mmap: bool = True
    prefetch_threads: int = 4

    def __post_init__(self) -> None:
        """类型转换 & 目录创建"""
        if isinstance(self.model_shards_dir, str):
            self.model_shards_dir = Path(self.model_shards_dir)
        if isinstance(self.offload_temp_dir, str):
            self.offload_temp_dir = Path(self.offload_temp_dir)
        self.model_shards_dir.mkdir(parents=True, exist_ok=True)
        self.offload_temp_dir.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. 内存限制策略（32GB系统内存 + 2GB CPU推理缓冲硬锁）
# ============================================================
@dataclass
class MemoryLimitConfig:
    """全局内存硬锁策略

    Attributes:
        system_ram_limit_gb: 整机内存上限 (默认32GB)
        cpu_infer_buffer_gb: CPU推理缓冲区封顶 (强制2GB，不可修改)
        gpu_vram_reserve_ratio: GPU显存预留比例 (防止OOM)
        force_swap_when_exceed: 超限是否强制执行swap磁盘 (True)
        monitor_interval_ms: 内存监控采样间隔(毫秒)
    """
    system_ram_limit_gb: float = 32.0
    cpu_infer_buffer_gb: float = 2.0
    gpu_vram_reserve_ratio: float = 0.15
    force_swap_when_exceed: bool = True
    monitor_interval_ms: int = 100

    def __post_init__(self) -> None:
        """强制CPU缓冲硬限2GB"""
        if self.cpu_infer_buffer_gb > 2.0:
            raise ValueError(
                f"CPU推理缓冲区硬限2GB，禁止设置为 {self.cpu_infer_buffer_gb}GB"
            )


# ============================================================
# 3. 自进化自动调优配置
# ============================================================
@dataclass
class SelfEvolutionConfig:
    """自进化闭环触发与调参规则

    Attributes:
        enable_self_evo: 是否启用自进化闭环
        latency_increase_threshold: 延迟上涨触发阈值 (默认20%)
        io_block_threshold_sec: IO阻塞触发阈值(秒) (默认30s)
        failure_rate_threshold: 生成失败率阈值 (默认0.5%)
        min_rounds_before_evo: 进化前最少采集轮次
        tuning_params_space: 可调参数搜索空间
        skill_archive_max: 技能库最大存档数量 (默认100)
    """
    enable_self_evo: bool = True
    latency_increase_threshold: float = 0.20
    io_block_threshold_sec: float = 30.0
    failure_rate_threshold: float = 0.005
    min_rounds_before_evo: int = 10
    tuning_params_space: Dict[str, List[object]] = field(default_factory=lambda: {
        "prefetch_layers_ahead": [1, 2, 3, 4],
        "prefetch_threads": [2, 4, 6, 8],
        "quantization_bits": [4, 8],
        "kv_spill_threshold_mb": [256, 512, 1024],
        "shard_size_mb": [256, 512, 1024],
    })
    skill_archive_max: int = 100


# ============================================================
# 4. 全局聚合配置
# ============================================================
@dataclass
class GlobalHermesConfig:
    """Rebirth LLM(RLLM) 全局配置聚合

    Attributes:
        disk_offload: 磁盘卸载参数
        memory_limit: 内存硬锁参数
        self_evolution: 自进化参数
        offline_mode: 是否强制离线模式 (True)
        skill_storage_dir: Hermes技能持久化目录 (D:\\AI_RLLM\\rllm_skill_storage)
    """
    disk_offload: DiskOffloadConfig = field(default_factory=DiskOffloadConfig)
    memory_limit: MemoryLimitConfig = field(default_factory=MemoryLimitConfig)
    self_evolution: SelfEvolutionConfig = field(default_factory=SelfEvolutionConfig)
    offline_mode: bool = True
    skill_storage_dir: Path = SKILL_STORAGE_DIR
    current_best_strategy_id: Optional[str] = None

    def to_json(self) -> str:
        """序列化为JSON字符串"""
        def _path_to_str(obj: object) -> object:
            if isinstance(obj, Path):
                return str(obj)
            raise TypeError(f"无法序列化类型: {type(obj)}")
        return json.dumps(asdict(self), ensure_ascii=False, indent=2, default=_path_to_str)


# ============================================================
# 配置文件持久化
# ============================================================
_CONFIG_PATH: Path = HERMES_ROOT / "hermes_core" / "config" / "global_config.json"


def load_global_config() -> GlobalHermesConfig:
    """从D盘加载全局配置，若不存在则写入默认值"""
    if _CONFIG_PATH.exists():
        try:
            with open(_CONFIG_PATH, "r", encoding="utf-8") as fp:
                raw = json.load(fp)
            cfg = GlobalHermesConfig(
                disk_offload=DiskOffloadConfig(**raw.get("disk_offload", {})),
                memory_limit=MemoryLimitConfig(**raw.get("memory_limit", {})),
                self_evolution=SelfEvolutionConfig(**raw.get("self_evolution", {})),
                offline_mode=raw.get("offline_mode", True),
                skill_storage_dir=Path(raw.get("skill_storage_dir", str(SKILL_STORAGE_DIR))),
                current_best_strategy_id=raw.get("current_best_strategy_id"),
            )
            return cfg
        except Exception as exc:  # noqa: BLE001
            import warnings
            warnings.warn(f"加载配置失败，回退默认值: {exc}")
    cfg = GlobalHermesConfig()
    save_global_config(cfg)
    return cfg


def save_global_config(cfg: GlobalHermesConfig) -> None:
    """持久化全局配置到D盘"""
    _CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(_CONFIG_PATH, "w", encoding="utf-8") as fp:
        fp.write(cfg.to_json())
