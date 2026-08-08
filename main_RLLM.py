# File: D:\AI_RLLM\main.py
"""Rebirth LLM(RLLM) 图文批量离线生成主入口

执行流程：
  1. 初始化环境变量 (D盘路径 + 离线模式)
  2. 加载全局配置 & 检查点
  3. 注册 Hermes Worker / Skill / 复盘引擎
  4. 批量读取关键词输入（支持断点续跑）
  5. 异步调用磁盘分页推理 Skill
  6. 直接写结果到 D 盘 output_dataset (不驻留内存)
  7. 每 N 条触发复盘 → 自进化调优 → 写回 Skill 配置
  8. 打印内存/IO/吞吐实时监控面板

运行：
  call D:\AI_RLLM\.venv\Scripts\activate.bat
  python D:\AI_RLLM\main.py --max-tasks 1000 --batch-size 8
"""
from __future__ import annotations
import env_config  # RLLM全局环境变量自动注入（硬锁定D:\AI_RLLM）

import argparse
import asyncio
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

# ======== 强制D盘环境变量注入（最顶部，先于任何torch/transformers导入） ========
ROOT: Path = Path(r"D:\AI_RLLM")
os.environ["HF_HOME"] = str(ROOT / "hf_cache")
os.environ["TRANSFORMERS_CACHE"] = str(ROOT / "hf_cache" / "hub")
os.environ["HUGGINGFACE_HUB_CACHE"] = str(ROOT / "hf_cache" / "hub")
os.environ["TORCH_HOME"] = str(ROOT / "hf_cache" / "torch")
os.environ["HF_DATASETS_CACHE"] = str(ROOT / "hf_cache" / "datasets")
os.environ["HF_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

for _sub in ("hermes_core", "disk_engine", "auto_evo", "pipeline"):
    sys.path.insert(0, str(ROOT / _sub))

from loguru import logger

from rllm_agent_core.config.hermes_config import load_global_config, save_global_config
from rllm_agent_core.workers.worker_registry import register_default_workers
from rllm_agent_core.skills.skill_loader import (
    register_default_skills,
    load_skill,
    DiskOffloadInferSkill,
    DiskOffloadSkillConfig,
)
from rllm_agent_core.review.review_engine import get_review_engine

from rllm_disk_engine.memory_lock.global_memory_lock import get_memory_lock
from rllm_disk_engine.mmap_io.mmap_wrapper import get_mmap_manager

from rllm_pipeline.batch_reader.keyword_reader import KeywordBatchReader, BatchInput
from rllm_pipeline.writer.output_writer import OutputDatasetWriter, OutputRecord, get_output_writer
from rllm_pipeline.checkpoint.checkpoint_manager import CheckpointManager, get_checkpoint_manager

from rllm_auto_evo.metrics.metrics_collector import get_metrics_collector


LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
logger.add(
    LOG_DIR / "main_{time}.log",
    rotation="50 MB",
    retention="14 days",
    level="INFO",
    encoding="utf-8",
)


# ============================================================
# 主流水线
# ============================================================
class DiskOffloadPipeline:
    """磁盘分页离线批量生成流水线"""

    def __init__(
        self,
        batch_size: int = 8,
        review_every: int = 100,
        max_new_tokens: int = 768,
        temperature: float = 0.7,
    ) -> None:
        self.batch_size = batch_size
        self.review_every = review_every
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

        # 加载配置
        self.config = load_global_config()
        logger.info(f"[RLLM-Pipeline] 全局配置加载完成，离线模式={self.config.offline_mode}")

        # 初始化 Hermes
        register_default_workers()
        register_default_skills()

        # 全局组件（D盘路径均已在包内封装）
        self.skill: DiskOffloadInferSkill = load_skill("disk_offload_infer")  # type: ignore[assignment]
        self.review_engine = get_review_engine()
        self.mem_lock = get_memory_lock(limit_gb=self.config.memory_limit.cpu_infer_buffer_gb)
        self.mmap_mgr = get_mmap_manager()
        self.metrics = get_metrics_collector()
        self.reader = KeywordBatchReader(batch_size=batch_size)
        self.writer: OutputDatasetWriter = get_output_writer()
        self.ckpt: CheckpointManager = get_checkpoint_manager()

        # 运行统计
        self.task_counter: int = 0
        self.start_ts: float = 0.0
        self.last_review_count: int = 0

    # ----------------------------------------------------------------
    def __repr__(self) -> str:
        return (
            f"DiskOffloadPipeline(batch={self.batch_size}, "
            f"review_every={self.review_every}, "
            f"max_tokens={self.max_new_tokens})"
        )

    # ----------------------------------------------------------------
    async def process_one(self, item: BatchInput) -> OutputRecord:
        """处理单条输入 -> 写输出记录（不驻留内存）"""
        task_id = f"t_{int(time.time()*1000)}_{item.idx}"
        prompt = item.to_prompt()
        try:
            result: Dict[str, Any] = await self.skill.execute(
                task_id=task_id,
                prompt=prompt,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                extra_params={"source_file": item.source_file, "idx": item.idx},
            )
            success = bool(result.get("success", False))
            rec = OutputRecord(
                idx=item.idx,
                source_file=item.source_file,
                keyword=item.keyword,
                category=item.category,
                style=item.style,
                generated_text=str(result.get("generated_text", "")),
                prompt=prompt,
                task_id=task_id,
                success=success,
                error_msg=str(result.get("error", "")),
                latency_sec=float(result.get("latency_sec", 0.0)),
                peak_memory_mb=float(result.get("peak_memory_mb", 0.0)),
                tokens_generated=int(result.get("tokens_generated", 0)),
                io_metrics=dict(result.get("io_metrics", {})),
                skill_config_sig=str(result.get("skill_config_sig", "")),
            )
            # 指标采集
            self.metrics.record("peak_memory_mb", rec.peak_memory_mb, task_id)
            self.metrics.record(
                "throughput_tps",
                rec.tokens_generated / max(0.001, rec.latency_sec),
                task_id,
            )
            io_block_ms = float(rec.io_metrics.get("io_block_ms", 0.0))
            self.metrics.record("io_block_ms", io_block_ms, task_id)
            if not success:
                self.metrics.record("failure_flag", 1.0, task_id)
            else:
                self.metrics.record("failure_flag", 0.0, task_id)

            # 标记完成
            if success:
                self.ckpt.mark_success(item.idx)
            else:
                self.ckpt.mark_failed(item.idx, rec.error_msg)
        except Exception as exc:  # noqa: BLE001
            logger.exception(f"[RLLM-Pipeline] 处理失败 idx={item.idx}: {exc}")
            self.ckpt.mark_failed(item.idx, str(exc))
            rec = OutputRecord(
                idx=item.idx,
                source_file=item.source_file,
                keyword=item.keyword,
                category=item.category,
                style=item.style,
                generated_text="",
                prompt=prompt,
                task_id=task_id,
                success=False,
                error_msg=str(exc),
                latency_sec=0.0,
            )

        # 直接写D盘
        await self.writer.write_async(rec)
        self.task_counter += 1
        return rec

    # ----------------------------------------------------------------
    async def run(self, max_tasks: Optional[int] = None) -> Dict[str, Any]:
        """启动批量生成主循环"""
        self.start_ts = time.time()
        self.metrics.set_strategy(self.skill.get_config().signature())
        self.ckpt.update_strategy_sig(self.skill.get_config().signature())
        done_set = self.ckpt.completed_set()
        logger.info(
            f"[RLLM-Pipeline] 启动 {self}，断点跳过={len(done_set)}，"
            f"max_tasks={max_tasks or '无限'}"
        )

        processed = 0
        t0 = time.time()
        async for batch in self.reader.iter_batches_async(skip_if_done=done_set):
            # 并发批量
            tasks = [self.process_one(it) for it in batch]
            results: List[OutputRecord] = list(await asyncio.gather(*tasks))
            processed += len(results)

            # 进度打印
            if processed % 50 == 0:
                self._print_progress(processed, t0)

            # 触发复盘自进化
            if self.task_counter - self.last_review_count >= self.review_every:
                await self._trigger_review()

            if max_tasks and processed >= max_tasks:
                break

        # 收尾
        await self._trigger_review()
        self.writer.flush_all()
        self.ckpt.save()
        save_global_config(self.config)

        elapsed = time.time() - self.start_ts
        summary = {
            "total_processed": processed,
            "total_elapsed_sec": round(elapsed, 2),
            "avg_tps_per_item": round(processed / max(0.001, elapsed), 3),
            "mem_breach_count": self.mem_lock.breach_count,
            "best_strategy_sig": self.skill.get_config().signature(),
        }
        logger.info(f"[RLLM-Pipeline] 完成 summary={json.dumps(summary, ensure_ascii=False)}")
        return summary

    # ----------------------------------------------------------------
    async def _trigger_review(self) -> None:
        """触发复盘-自进化"""
        self.last_review_count = self.task_counter
        sig_before = self.skill.get_config().signature()
        try:
            review_res = await self.review_engine.run_review_cycle(min_samples=10)
            sig_after = self.skill.get_config().signature()
            self.metrics.set_round(self.review_engine._round_counter)  # type: ignore[attr-defined]
            self.metrics.set_strategy(sig_after)
            self.ckpt.update_strategy_sig(sig_after)
            self.config.current_best_strategy_id = sig_after
            save_global_config(self.config)
            if sig_before != sig_after:
                logger.info(
                    f"[RLLM-Pipeline] 复盘触发自进化: {sig_before[:8]} -> {sig_after[:8]}, "
                    f"review={review_res}"
                )
            else:
                logger.debug(f"[RLLM-Pipeline] 复盘完成，无需改配置 review={review_res}")
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"[RLLM-Pipeline] 复盘异常: {exc}")

    # ----------------------------------------------------------------
    def _print_progress(self, processed: int, start: float) -> None:
        snap = self.mem_lock.latest_snapshot()
        rss = f"{snap.process_rss_mb:.1f}MB" if snap else "N/A"
        ckpt = self.ckpt.snapshot()
        rate = processed / max(0.001, time.time() - start)
        logger.info(
            f"[进度] 已处理 {processed} | 速率 {rate:.2f}条/s | "
            f"进程RSS {rss} / 硬限2048MB | "
            f"成功={ckpt.total_success} 失败={ckpt.total_failed} "
            f"策略={ckpt.current_strategy_sig[:8] if ckpt.current_strategy_sig else '-'}"
        )


# ============================================================
# CLI
# ============================================================
def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="Rebirth LLM(RLLM)",
        description="Rebirth LLM(RLLM) 磁盘分页低内存大模型离线批量生成引擎 (D盘全隔离)",
    )
    p.add_argument("--max-tasks", type=int, default=None, help="最大处理条数，默认全量")
    p.add_argument("--batch-size", type=int, default=8, help="并发批大小（默认8）")
    p.add_argument("--review-every", type=int, default=100, help="每N条复盘自进化一次（默认100）")
    p.add_argument("--max-new-tokens", type=int, default=768, help="单条最大生成token数")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--reset-ckpt", action="store_true", help="清空检查点，从零开始跑")
    return p


async def main_async() -> int:
    args = build_arg_parser().parse_args()
    if args.reset_ckpt:
        ckpt = get_checkpoint_manager()
        ckpt.reset()
        logger.warning("[RLLM-Main] 检查点已清空重置")

    pipeline = DiskOffloadPipeline(
        batch_size=args.batch_size,
        review_every=args.review_every,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )
    summary = await pipeline.run(max_tasks=args.max_tasks)
    logger.success(f"[RLLM-Main] 全流程结束: {summary}")
    print("=" * 60)
    print("Rebirth LLM(RLLM) RLLM 批量生成结束")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print(f"  输出目录: {ROOT / 'output_dataset'}")
    print(f"  日志目录: {LOG_DIR}")
    print("=" * 60)
    return 0


def main() -> int:
    try:
        return asyncio.run(main_async())
    except KeyboardInterrupt:
        logger.warning("[RLLM-Main] 收到Ctrl+C，安全退出，断点已保存")
        try:
            w = get_output_writer()
            w.flush_all()
            w.close()
            c = get_checkpoint_manager()
            c.save()
        except Exception:  # noqa: BLE001
            pass
        return 130


if __name__ == "__main__":
    sys.exit(main())


# ============================================================
# 版权声明
# 本项目 Rebirth LLM(RLLM) 基于开源项目 Nous Hermes-Agent（MIT License）二次深度开发，项目内保留完整原始开源协议文件；智能体自迭代调度逻辑复用开源代码，磁盘分层加载、全局内存锁、D盘隔离部署、自动IO调优模块为自研闭源模块，分发时附带完整MIT协议文件。
# 商标隔离免责声明
# 项目名称 Rebirth LLM（简称RLLM）为独立软件项目代号，与奢侈品品牌Hermes、开源项目Hermes-Agent无品牌合作、隶属关联；仅代码内部功能性调用开源框架，不会使用Hermes相关名称开展商业宣传，无品牌混淆意图。
# ============================================================
