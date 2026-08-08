# -*- coding: utf-8 -*-
# File: D:\AI_RLLM\rllm_disk_engine\router\router_prefetcher.py
"""路由器预取器

职责:
  1. 接收 RouterPredictor 的 Top-K 候选专家列表
  2. 过滤掉已在 VRAM 的专家
  3. 并行从 D 盘预取到 VRAM (与当前 token 计算重叠)
  4. 限制并发数 (避免 I/O 风暴, 默认 4 并发)

工作流:
  decode_token N:
    ├─ 前台: MoELayerRunner forward (用当前 VRAM 中的专家)
    └─ 后台: RouterPrefetcher.prefetch_candidates(预测的 N+1 候选)
  decode_token N+1:
    └─ 大部分候选已在 VRAM → 零 I/O 等待
"""
from __future__ import annotations

import asyncio
import time
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger

from rllm_agent_core import LOG_DIR
from rllm_disk_engine.expert_pool.expert_vram_pool import ExpertVRAMPool, ExpertKey

logger.add(
    LOG_DIR / "router_prefetcher_{time}.log",
    rotation="50 MB",
    retention="7 days",
    encoding="utf-8",
)


class RouterPrefetcher:
    """路由器预取器

    Args:
        vram_pool: ExpertVRAMPool 实例
        max_concurrent: 最大并发预取数 (默认 4)
    """

    def __init__(
        self,
        vram_pool: ExpertVRAMPool,
        max_concurrent: int = 4,
    ) -> None:
        self._pool: ExpertVRAMPool = vram_pool
        self._max_concurrent: int = max_concurrent
        self._semaphore: asyncio.Semaphore = asyncio.Semaphore(
            max_concurrent
        )

        # 统计
        self._prefetch_total: int = 0
        self._prefetch_skipped: int = 0  # 已在 VRAM, 跳过
        self._prefetch_bytes: int = 0

        logger.info(
            f"[RouterPrefetcher] 初始化: max_concurrent={max_concurrent}"
        )

    # ----------------------------------------------------------------
    # 对外主接口
    # ----------------------------------------------------------------
    async def prefetch_candidates(
        self,
        candidates: List[ExpertKey],
    ) -> int:
        """并行预取候选专家到 VRAM

        Args:
            candidates: 候选专家 key 列表 [(layer, expert), ...]

        Returns:
            实际触发预取的数量 (已在 VRAM 的不计)
        """
        if not candidates:
            return 0

        t0 = time.time()

        # 过滤: 已在 VRAM 的跳过
        resident = set(self._pool.list_resident_experts())
        to_prefetch = [
            key for key in candidates if key not in resident
        ]

        if not to_prefetch:
            self._prefetch_skipped += len(candidates)
            return 0

        # 并行预取 (限制并发数)
        tasks = [
            self._prefetch_one(key) for key in to_prefetch
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        success = sum(1 for r in results if r is not None)
        elapsed_ms = (time.time() - t0) * 1000

        self._prefetch_total += success
        self._prefetch_skipped += len(candidates) - len(to_prefetch)

        logger.info(
            f"[RouterPrefetcher] 预取 {success}/{len(to_prefetch)} 专家, "
            f"跳过 {len(candidates) - len(to_prefetch)} (已在VRAM), "
            f"耗时 {elapsed_ms:.0f}ms"
        )
        return success

    async def _prefetch_one(self, key: ExpertKey) -> Optional[ExpertKey]:
        """预取单个专家 (受信号量限制)

        Args:
            key: (layer_idx, expert_idx)

        Returns:
            成功则返回 key, 失败返回 None
        """
        async with self._semaphore:
            try:
                await self._pool.prefetch_expert(key)
                return key
            except Exception as exc:
                logger.warning(
                    f"[RouterPrefetcher] 预取 {key} 失败: {exc}"
                )
                return None

    # ----------------------------------------------------------------
    # 统计
    # ----------------------------------------------------------------
    def stats(self) -> Dict[str, Any]:
        """获取预取统计"""
        return {
            "prefetch_total": self._prefetch_total,
            "prefetch_skipped": self._prefetch_skipped,
            "max_concurrent": self._max_concurrent,
            "vram_pool_stats": self._pool.stats(),
        }
