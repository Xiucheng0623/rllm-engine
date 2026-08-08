# -*- coding: utf-8 -*-
# File: D:\AI_RLLM\rllm_disk_engine\vram_pool\hot_cold_evictor.py
"""冷热分层自动置换器

策略:
  1. 实时统计每层访问频次 (LayerFreqTracker)
  2. 显存水位超阈值 → 选最低频非 pinned 层淘汰 → 回写 D 盘
  3. 下次该层 forward 时 → 从 D 盘读回 (走 ZeroCopyShardLoader)
  4. 自进化参数: evict_threshold_pct / lru_weight_vs_freq

设计要点:
  - LRU + 频次混合评分, 避免纯 LRU 误淘汰高频层
  - pinned 层不可淘汰 (高频层锁定)
  - 回写文件命名: layer_{idx:03d}.pt (单文件, 便于直接 torch.load)
"""
from __future__ import annotations

import asyncio
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from loguru import logger

from rllm_agent_core import LOG_DIR

logger.add(
    LOG_DIR / "evictor_{time}.log",
    rotation="50 MB",
    retention="7 days",
    encoding="utf-8",
)


class LayerFreqTracker:
    """层访问频次跟踪器 (线程安全)

    混合评分公式:
        score = w * freq_norm + (1-w) * recency_norm
        其中 freq_norm = count / max_count
             recency_norm = 1 / (1 + minutes_since_last_access)
    """

    def __init__(
        self,
        num_layers: int,
        lru_weight_vs_freq: float = 0.5,
    ) -> None:
        """初始化频次跟踪器

        Args:
            num_layers: 总层数
            lru_weight_vs_freq: LRU 权重 (0=纯频次, 1=纯LRU, 默认 0.5)
        """
        self._counts: List[int] = [0] * num_layers
        self._last_ts: List[float] = [0.0] * num_layers
        self._lru_weight: float = lru_weight_vs_freq
        self._lock: threading.Lock = threading.Lock()

    def record_access(self, layer_idx: int) -> None:
        """记录一次层访问"""
        with self._lock:
            self._counts[layer_idx] += 1
            self._last_ts[layer_idx] = time.time()

    def frequency_score(self, layer_idx: int) -> float:
        """计算频次评分 (0-1, 越高越热)"""
        with self._lock:
            max_cnt = max(self._counts) if self._counts else 1
            max_cnt = max(max_cnt, 1)
            cnt_score: float = self._counts[layer_idx] / max_cnt
            now = time.time()
            last_ts = self._last_ts[layer_idx]
            if last_ts > 0:
                minutes_diff = (now - last_ts) / 60.0
            else:
                minutes_diff = 9999.0
            recency_score: float = 1.0 / (1.0 + minutes_diff)
            return self._lru_weight * recency_score + (
                1 - self._lru_weight
            ) * cnt_score

    def snapshot(self) -> Dict[int, float]:
        """获取所有层评分快照"""
        with self._lock:
            return {i: self.frequency_score(i) for i in range(len(self._counts))}


class HotColdEvictor:
    """冷热置换器

    Attributes:
        _pool: 绑定的 VRAMCachePool
        _evict_dir: 冷层回写目录 (D 盘独立 SSD 分区)
        _freq_tracker: 频次跟踪器
        _lock: 线程锁
    """

    def __init__(
        self,
        vram_pool,
        evict_dir: Path = Path(
            r"D:\AI_RLLM\rllm_offload_temp\evicted_layers"
        ),
        lru_weight_vs_freq: float = 0.5,
        num_layers: int = 32,
        direct_to_disk: bool = False,
    ) -> None:
        """初始化冷热置换器

        Args:
            vram_pool: VRAMCachePool 实例
            evict_dir: 冷层回写目录
            lru_weight_vs_freq: LRU 权重
            num_layers: 总层数
            direct_to_disk: 直写 D 盘模式 (跳过 CPU RAM 中转)
                - True: VRAM ↔ D 盘直接交互 (适合 FP16/8bit 大层)
                - False: VRAM → CPU RAM → D 盘三级 (适合 4bit 小层, fetch 快)
        """
        self._pool = vram_pool
        self._evict_dir: Path = Path(evict_dir)
        self._evict_dir.mkdir(parents=True, exist_ok=True)
        self._freq_tracker: LayerFreqTracker = LayerFreqTracker(
            num_layers=num_layers,
            lru_weight_vs_freq=lru_weight_vs_freq,
        )
        self._lock: threading.RLock = threading.RLock()
        self._layer_module_factory = None  # 延迟注入, 用于 fetch_back 重建模块
        self._quant_bits: int = 4

        # 直写 D 盘模式开关
        self._direct_to_disk: bool = direct_to_disk

        # CPU RAM 二级缓存: {layer_idx: state_dict (pinned CPU tensors)}
        # 淘汰时先放 CPU RAM, 仅当 CPU RAM 超限时才落盘
        # 7B FP16: 32层 × 416MB = 13.3GB, 需要至少 10GB CPU RAM 缓存
        # direct_to_disk=True 时此缓存不使用
        self._cpu_cache: Dict[int, Dict[str, torch.Tensor]] = {}
        self._cpu_cache_max_bytes: int = 12 * 1024**3  # 12GB CPU RAM 缓存上限
        self._cpu_cache_current_bytes: int = 0

        mode_label = "直写D盘" if direct_to_disk else "CPU_RAM中转"
        logger.info(
            f"[RLLM-Evictor] 初始化: evict_dir={self._evict_dir} "
            f"lru_weight={lru_weight_vs_freq} num_layers={num_layers} "
            f"mode={mode_label} "
            f"cpu_cache_limit={self._cpu_cache_max_bytes/1024**3:.0f}GB"
        )

    def set_direct_to_disk(self, enabled: bool) -> None:
        """动态切换直写 D 盘模式

        Args:
            enabled: True 启用直写 D 盘, False 用 CPU RAM 中转
        """
        self._direct_to_disk = enabled
        mode_label = "直写D盘" if enabled else "CPU_RAM中转"
        logger.info(
            f"[RLLM-Evictor] 模式切换 → {mode_label}"
        )

    def attach_factory(
        self,
        factory,
        quant_bits: int = 4,
    ) -> None:
        """注入模块工厂 (供 fetch_back 重建模块用)"""
        self._layer_module_factory = factory
        self._quant_bits = quant_bits

    # ----------------------------------------------------------------
    # 对外接口
    # ----------------------------------------------------------------
    async def evict_coldest(self) -> Optional[int]:
        """淘汰最低频非 pinned 层

        - direct_to_disk=True: VRAM → 直接原子写 D 盘 (跳过 CPU RAM)
        - direct_to_disk=False: VRAM → CPU RAM 缓存 (超限则落盘 D 盘)

        Returns:
            被淘汰的 layer_idx, 若无可淘汰层则返回 None
        """
        with self._lock:
            resident_layers = self._pool.list_resident_layers()
            if not resident_layers:
                return None

            # 1. 候选 = 所有非 pinned 层
            candidates: List[int] = []
            for idx in resident_layers:
                entry = self._pool.get_layer_entry(idx)
                if entry is not None and not entry.pinned:
                    candidates.append(idx)
            if not candidates:
                logger.warning(
                    "[RLLM-Evictor] 所有层均 pinned, 无可淘汰"
                )
                return None

            # 2. 评分: 冷度 = 1 - 频次评分 (越冷越优先淘汰)
            scored: List[tuple] = []
            for idx in candidates:
                freq_score = self._freq_tracker.frequency_score(idx)
                entry = self._pool.get_layer_entry(idx)
                cold_score = (1 - freq_score, -entry.last_access_ts, idx)
                scored.append(cold_score)
            scored.sort(reverse=True)
            victim_idx: int = scored[0][2]

            # 3. 从池中移除
            victim_entry = self._pool.remove_layer(victim_idx)
            if victim_entry is None:
                return None

        sd_bytes = victim_entry.size_bytes

        # 分流: 直写 D 盘 vs CPU RAM 中转
        if self._direct_to_disk:
            # === 直写 D 盘模式 ===
            # VRAM → CPU (取 state_dict) → 原子写 D 盘, 不进 CPU RAM 缓存
            await self._write_layer_to_disk_direct(victim_idx, victim_entry.module)
            # 释放模块 (state_dict 已落盘, 模块对象可回收)
            del victim_entry
            self._pool.increment_evict_count()
            logger.warning(
                f"[RLLM-Evictor] 淘汰层 {victim_idx} → D盘直写, "
                f"释放 VRAM {sd_bytes/1024**2:.0f}MB, "
                f"stats={self._pool.stats()}"
            )
        else:
            # === CPU RAM 中转模式 (原逻辑, 4bit 用) ===
            victim_entry.module.cpu()
            with self._lock:
                if self._cpu_cache_current_bytes + sd_bytes > self._cpu_cache_max_bytes:
                    await self._spill_cpu_cache_to_disk(sd_bytes)
                self._cpu_cache[victim_idx] = victim_entry.module
                self._cpu_cache_current_bytes += sd_bytes
            self._pool.increment_evict_count()
            logger.warning(
                f"[RLLM-Evictor] 淘汰层 {victim_idx} → CPU RAM, "
                f"释放 VRAM {sd_bytes/1024**2:.0f}MB, "
                f"CPU缓存={self._cpu_cache_current_bytes/1024**3:.1f}GB, "
                f"stats={self._pool.stats()}"
            )
        return victim_idx

    async def _write_layer_to_disk_direct(
        self,
        layer_idx: int,
        module: torch.nn.Module,
    ) -> None:
        """直写模式: 把模块 state_dict 原子写入 D 盘

        流程: module → 提取 state_dict (CPU) → 写 .pt.tmp → os.replace 到 .pt
        写完后 module 对象由调用方释放, 不占用 CPU RAM 缓存.

        Args:
            layer_idx: 层索引
            module: 要落盘的模块 (可能在 VRAM 或 CPU)
        """
        import os as _os

        # 提取 state_dict 到 CPU (从 VRAM 取出权重)
        sd: Dict[str, torch.Tensor] = {}
        for name, param in module.named_parameters():
            sd[name] = param.data.detach().cpu()

        fp: Path = self._evict_dir / f"layer_{layer_idx:03d}.pt"
        tmp_fp: Path = fp.with_suffix(".pt.tmp")

        await asyncio.get_event_loop().run_in_executor(
            None,
            lambda f_final=fp, f_tmp=tmp_fp, d=sd: self._atomic_save(f_final, f_tmp, d),
        )

    @staticmethod
    def _atomic_save(
        f_final: Path,
        f_tmp: Path,
        d: Dict[str, torch.Tensor],
    ) -> None:
        """原子保存: 写临时文件, 完成后 rename

        Args:
            f_final: 最终文件路径
            f_tmp: 临时文件路径
            d: 要保存的 state_dict
        """
        import os as _os
        torch.save(d, f_tmp)
        _os.replace(f_tmp, f_final)

    async def fetch_back(self, layer_idx: int):
        """从 CPU RAM 缓存 (或 D 盘) 读回冷层到 VRAM

        - direct_to_disk=True: 直接从 D 盘读取 (跳过 CPU RAM 查找)
        - direct_to_disk=False: 优先从 CPU RAM pinned memory 异步 DMA 拷贝 (快 ~26ms/层),
          仅当 CPU RAM 缓存未命中时, 才从 D 盘读取 (慢 ~150ms/层).

        Args:
            layer_idx: 层索引

        Returns:
            重建后的 DecoderLayer 模块 (已在 VRAM), 或 None
        """
        if self._layer_module_factory is None:
            logger.error(
                "[RLLM-Evictor] factory 未注入, 无法 fetch_back"
            )
            return None

        # 直写 D 盘模式: 跳过 CPU RAM, 直接从 D 盘读
        if self._direct_to_disk:
            return await self._fetch_back_from_disk(layer_idx)

        # CPU RAM 中转模式: 优先查 CPU RAM 缓存
        with self._lock:
            cached_module = self._cpu_cache.pop(layer_idx, None)
            if cached_module is not None:
                # 估算大小
                sd_bytes = sum(
                    p.element_size() * p.nelement()
                    for p in cached_module.parameters()
                )
                self._cpu_cache_current_bytes -= sd_bytes
                source = "CPU_RAM"
            else:
                source = "DISK"

        if source == "CPU_RAM":
            # 把模块搬回 VRAM (CUDA 异步传输)
            module = cached_module.to(self._pool._device)
            size_bytes = sd_bytes
        else:
            # CPU RAM 未命中, 从 D 盘读取
            module, size_bytes = await self._fetch_back_from_disk(layer_idx)
            if module is None:
                return None

        # 入池前检查 VRAM, 不足则淘汰冷层
        while not self._pool._can_fit(size_bytes):
            evicted_idx = await self.evict_coldest()
            if evicted_idx is None:
                logger.warning(
                    f"[RLLM-Evictor] fetch_back 层 {layer_idx} 前无冷层可淘汰"
                )
                break

        # 入池
        from rllm_disk_engine.vram_pool.vram_cache_pool import VRAMLayerEntry
        entry = VRAMLayerEntry(
            layer_idx=layer_idx,
            module=module,
            size_bytes=size_bytes,
            quant_bits=self._quant_bits,
        )
        self._pool.add_layer(entry)

        logger.info(
            f"[RLLM-Evictor] 层 {layer_idx} 读回 VRAM (源={source}), "
            f"size={size_bytes/1024**2:.0f}MB"
        )
        return module

    async def _fetch_back_from_disk(self, layer_idx: int):
        """从 D 盘读取层 state_dict, 重建模块并搬到 VRAM

        Args:
            layer_idx: 层索引

        Returns:
            (module, size_bytes) 元组, 失败返回 (None, 0)
        """
        fp: Path = self._evict_dir / f"layer_{layer_idx:03d}.pt"
        if not fp.exists():
            logger.error(
                f"[RLLM-Evictor] 冷层 {layer_idx} 在 D 盘不存在: {fp}"
            )
            return None, 0

        # 从 D 盘读取 state_dict (CPU)
        state_dict = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: torch.load(fp, map_location="cpu"),
        )
        # 重建模块并搬到 VRAM
        module = self._layer_module_factory(layer_idx)
        module = module.to(self._pool._device)
        size_bytes = self._pool._bind_weights_inplace(module, state_dict, self._quant_bits)
        # 安全网: 确保所有参数和 buffer 都在 CUDA 上
        module = module.to(self._pool._device)
        torch.cuda.synchronize()

        # 入池前检查 VRAM, 不足则淘汰冷层
        while not self._pool._can_fit(size_bytes):
            evicted_idx = await self.evict_coldest()
            if evicted_idx is None:
                logger.warning(
                    f"[RLLM-Evictor] fetch_back 层 {layer_idx} 前无冷层可淘汰"
                )
                break

        # 入池
        from rllm_disk_engine.vram_pool.vram_cache_pool import VRAMLayerEntry
        entry = VRAMLayerEntry(
            layer_idx=layer_idx,
            module=module,
            size_bytes=size_bytes,
            quant_bits=self._quant_bits,
        )
        self._pool.add_layer(entry)

        logger.info(
            f"[RLLM-Evictor] 层 {layer_idx} 读回 VRAM (源=DISK直读), "
            f"size={size_bytes/1024**2:.0f}MB"
        )
        return module, size_bytes

    def record_access(self, layer_idx: int) -> None:
        """记录层访问 (供 VRAMPool.get_layer 调用)"""
        self._freq_tracker.record_access(layer_idx)

    # ----------------------------------------------------------------
    # 内部
    # ----------------------------------------------------------------
    async def _spill_cpu_cache_to_disk(self, needed_bytes: int) -> None:
        """当 CPU RAM 缓存超限时, 把最旧的层落盘到 D 盘

        原子写入: 先写到 .pt.tmp, 完成后 os.replace 到 .pt
        避免 fetch_back 读到写一半的文件 (竞态条件修复)

        Args:
            needed_bytes: 需要腾出的字节数
        """
        import os as _os

        freed = 0
        while freed < needed_bytes and self._cpu_cache:
            # 取最早的 key (FIFO)
            oldest_idx = next(iter(self._cpu_cache))
            module = self._cpu_cache.pop(oldest_idx)
            sd_bytes = sum(
                p.element_size() * p.nelement() for p in module.parameters()
            )
            self._cpu_cache_current_bytes -= sd_bytes
            freed += sd_bytes

            # 提取 state_dict 并落盘
            sd: Dict[str, torch.Tensor] = {}
            for name, param in module.named_parameters():
                sd[name] = param.data.detach().cpu()
            fp: Path = self._evict_dir / f"layer_{oldest_idx:03d}.pt"
            tmp_fp: Path = fp.with_suffix(".pt.tmp")

            def _atomic_save(f_final: Path, f_tmp: Path, d: Dict[str, torch.Tensor]) -> None:
                """原子保存: 写临时文件, 完成后 rename

                Args:
                    f_final: 最终文件路径
                    f_tmp: 临时文件路径
                    d: 要保存的 state_dict
                """
                torch.save(d, f_tmp)
                # Windows 上 os.replace 是原子的 (同盘内)
                _os.replace(f_tmp, f_final)

            await asyncio.get_event_loop().run_in_executor(
                None,
                _atomic_save, fp, tmp_fp, sd,
            )
            logger.debug(
                f"[RLLM-Evictor] CPU RAM → D 盘溢出: 层 {oldest_idx} "
                f"({sd_bytes/1024**2:.0f}MB)"
            )

    def _write_back_sync(
        self,
        layer_idx: int,
        module: torch.nn.Module,
    ) -> None:
        """同步回写模块权重到 D 盘"""
        fp: Path = self._evict_dir / f"layer_{layer_idx:03d}.pt"
        # 提取原始 state_dict (反量化到 bf16)
        sd: Dict[str, torch.Tensor] = {}
        for name, param in module.named_parameters():
            if hasattr(param, "data") and param.data is not None:
                # 4bit/8bit 量化参数需要反量化
                try:
                    # bitsandbytes 4bit/8bit 直接 .to(torch.bfloat16) 反量化
                    sd[name] = param.data.to(
                        torch.bfloat16
                    ).cpu()
                except Exception:
                    sd[name] = param.data.detach().cpu()
        torch.save(sd, fp)
        logger.debug(
            f"[RLLM-Evictor] 层 {layer_idx} 已回写: {fp} "
            f"({fp.stat().st_size/1024**2:.0f}MB)"
        )

    def stats(self) -> Dict[str, Any]:
        """获取置换器统计"""
        return {
            "freq_snapshot": self._freq_tracker.snapshot(),
            "evict_dir": str(self._evict_dir),
            "evicted_files": len(list(self._evict_dir.glob("*.pt"))),
        }
