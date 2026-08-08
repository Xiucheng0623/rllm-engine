# File: D:\AI_RLLM\rllm_tests\stress_test.py
"""Rebirth LLM(RLLM) Rebirth LLM(RLLM) 压力测试脚本

场景：模拟百万级循环批量生成，持续采集指标驱动 Hermes 自动优化
功能：
  1. 动态生成虚拟关键词（模拟关键词输入）
  2. 持续循环跑 main.py 的处理逻辑
  3. 内存超限、IO阻塞、延迟抖动都将触发自进化
  4. 打印实时监控面板（吞吐、内存、延迟、触发次数）

运行：
  call D:\AI_RLLM\.venv\Scripts\activate.bat
  python D:\AI_RLLM\rllm_tests\stress_test.py --duration-sec 3600
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
from typing import Any, Dict, List

# 注入D盘环境
ROOT = Path(r"D:\AI_RLLM")
os.environ["HF_HOME"] = str(ROOT / "hf_cache")
os.environ["TRANSFORMERS_CACHE"] = str(ROOT / "hf_cache" / "hub")
os.environ["HF_OFFLINE"] = "1"
for _s in ("hermes_core", "disk_engine", "auto_evo", "pipeline"):
    sys.path.insert(0, str(ROOT / _s))

from loguru import logger

from rllm_agent_core.workers.worker_registry import register_default_workers
from rllm_agent_core.skills.skill_loader import register_default_skills, load_skill, DiskOffloadInferSkill
from rllm_agent_core.review.review_engine import get_review_engine
from rllm_disk_engine.memory_lock.global_memory_lock import get_memory_lock
from rllm_auto_evo.metrics.metrics_collector import get_metrics_collector
from rllm_pipeline.batch_reader.keyword_reader import BatchInput
from rllm_pipeline.writer.output_writer import OutputRecord, get_output_writer
from rllm_pipeline.checkpoint.checkpoint_manager import get_checkpoint_manager

LOG_DIR = ROOT / "logs"
logger.add(
    LOG_DIR / "stress_{time}.log",
    rotation="50 MB",
    retention="3 days",
    encoding="utf-8",
)


# 模拟关键词池
KEYWORD_POOL: List[str] = [
    "夏日穿搭", "通勤彩妆", "护肤心得", "居家好物", "美食探店",
    "旅行攻略", "健身日常", "职场穿搭", "发型教程", "收纳整理",
    "母婴好物", "宠物日常", "数码评测", "咖啡探店", "读书分享",
    "极简主义", "复古风", "多巴胺配色", "citywalk", "手作DIY",
]
CATEGORY_POOL = ["时尚", "美妆", "美食", "家居", "旅行", "健身", "母婴", "数码"]
STYLE_POOL = ["温柔风", "甜酷风", "ins风", "治愈系", "学院风", "轻奢感"]


def generate_fake_input(idx: int) -> BatchInput:
    kw = KEYWORD_POOL[idx % len(KEYWORD_POOL)]
    cat = CATEGORY_POOL[idx % len(CATEGORY_POOL)]
    stl = STYLE_POOL[idx % len(STYLE_POOL)]
    return BatchInput(
        idx=idx,
        source_file="stress_test_fake.txt",
        keyword=f"{kw}_{idx:08d}",
        category=cat,
        style=stl,
        extra={"round": idx // 1000},
    )


async def stress_loop(
    duration_sec: int,
    concurrency: int,
    max_new_tokens: int,
    review_every: int,
) -> Dict[str, Any]:
    """压测主循环"""
    logger.info(
        f"[Stress] 启动压测: duration={duration_sec}s, "
        f"concurrency={concurrency}, review_every={review_every}"
    )
    register_default_workers()
    register_default_skills()
    skill: DiskOffloadInferSkill = load_skill("disk_offload_infer")  # type: ignore[assignment]
    review = get_review_engine()
    mem_lock = get_memory_lock()
    metrics = get_metrics_collector()
    writer = get_output_writer()
    ckpt = get_checkpoint_manager()

    start = time.time()
    deadline = start + duration_sec
    idx = 0
    processed = 0
    success_cnt = 0
    fail_cnt = 0
    total_latency = 0.0
    total_mem_peak = 0.0
    last_report = start

    while time.time() < deadline:
        # 构造并发批次
        batch: List[BatchInput] = [generate_fake_input(idx + i) for i in range(concurrency)]
        tasks = [
            skill.execute(
                task_id=f"s_{int(time.time()*1000)}_{it.idx}",
                prompt=it.to_prompt(),
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                extra_params={"stress_test": True, "round": it.extra.get("round")},
            )
            for it in batch
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for it, res in zip(batch, results):
            if isinstance(res, Exception):
                fail_cnt += 1
                ckpt.mark_failed(it.idx, str(res))
                logger.warning(f"[Stress] 任务异常 it={it.idx}: {res}")
                continue
            ok = bool(res.get("success", False))
            latency = float(res.get("latency_sec", 0.0))
            peak_mb = float(res.get("peak_memory_mb", 0.0))
            total_latency += latency
            total_mem_peak += peak_mb
            processed += 1
            if ok:
                success_cnt += 1
                ckpt.mark_success(it.idx)
            else:
                fail_cnt += 1
                ckpt.mark_failed(it.idx, str(res.get("error", "")))
            metrics.record("peak_memory_mb", peak_mb)
            metrics.record(
                "throughput_tps",
                int(res.get("tokens_generated", 0)) / max(0.001, latency),
            )
            metrics.record("io_block_ms", float(res.get("io_metrics", {}).get("io_block_ms", 0.0)))
            if not ok:
                metrics.record("failure_flag", 1.0)
            # 写输出
            await writer.write_async(OutputRecord(
                idx=it.idx,
                source_file=it.source_file,
                keyword=it.keyword,
                category=it.category,
                style=it.style,
                generated_text=str(res.get("generated_text", "")),
                prompt=it.to_prompt(),
                task_id=str(res.get("task_id", "")),
                success=ok,
                error_msg=str(res.get("error", "")),
                latency_sec=latency,
                peak_memory_mb=peak_mb,
                tokens_generated=int(res.get("tokens_generated", 0)),
                io_metrics=dict(res.get("io_metrics", {})),
                skill_config_sig=str(res.get("skill_config_sig", "")),
            ))
        idx += concurrency

        # 复盘
        if processed - (review.last_processed or 0) >= review_every:  # type: ignore[attr-defined]
            try:
                await review.run_review_cycle()
                review.last_processed = processed  # type: ignore[attr-defined]
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"[Stress] 复盘异常: {exc}")

        # 每5秒打印面板
        if time.time() - last_report >= 5:
            elapsed = time.time() - start
            rate = processed / max(0.001, elapsed)
            snap = mem_lock.latest_snapshot()
            avg_lat = total_latency / max(1, processed)
            avg_mem = total_mem_peak / max(1, processed)
            metrics.set_strategy(skill.get_config().signature())
            ckpt.update_strategy_sig(skill.get_config().signature())

            panel = (
                f"[Stress 面板] 已跑={elapsed:.0f}s / 总{duration_sec}s | "
                f"处理={processed} 速率={rate:.1f}/s | "
                f"成功={success_cnt} 失败={fail_cnt} 失败率={fail_cnt/max(1,processed)*100:.2f}% | "
                f"RSS={snap.process_rss_mb:.1f}MB/2048MB 超限#={mem_lock.breach_count} | "
                f"平均延迟={avg_lat*1000:.0f}ms 平均内存={avg_mem:.0f}MB | "
                f"策略={skill.get_config().signature()[:8]}"
            )
            logger.info(panel)
            print(panel, flush=True)
            last_report = time.time()

    writer.flush_all()
    writer.close()
    ckpt.save()
    elapsed_total = time.time() - start
    summary = {
        "duration_sec": round(elapsed_total, 2),
        "processed": processed,
        "success": success_cnt,
        "failed": fail_cnt,
        "failure_rate": round(fail_cnt / max(1, processed) * 100, 3),
        "avg_latency_sec": round(total_latency / max(1, processed), 4),
        "avg_peak_memory_mb": round(total_mem_peak / max(1, processed), 2),
        "tps": round(processed / max(0.001, elapsed_total), 2),
        "mem_breach_count": mem_lock.breach_count,
        "final_strategy_sig": skill.get_config().signature(),
    }
    return summary


def main() -> int:
    p = argparse.ArgumentParser(description="Hermes-DiskOffload 压测脚本")
    p.add_argument("--duration-sec", type=int, default=3600, help="压测时长秒数(默认1小时)")
    p.add_argument("--concurrency", type=int, default=16, help="并发批大小(默认16)")
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--review-every", type=int, default=200, help="每N条复盘进化一次")
    args = p.parse_args()

    summary = asyncio.run(stress_loop(
        duration_sec=args.duration_sec,
        concurrency=args.concurrency,
        max_new_tokens=args.max_new_tokens,
        review_every=args.review_every,
    ))
    print("\n======= 压测结果 =======")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print(f"输出文件: {ROOT / 'output_dataset'}")
    print("=======================")
    # 持久化压测报告
    report = ROOT / "logs" / f"stress_report_{int(time.time())}.json"
    with open(report, "w", encoding="utf-8") as fp:
        json.dump(summary, fp, ensure_ascii=False, indent=2)
    print(f"报告: {report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())


# ============================================================
# 版权声明
# 本项目 Rebirth LLM(RLLM) 基于开源项目 Nous Hermes-Agent（MIT License）二次深度开发，项目内保留完整原始开源协议文件；智能体自迭代调度逻辑复用开源代码，磁盘分层加载、全局内存锁、D盘隔离部署、自动IO调优模块为自研闭源模块，分发时附带完整MIT协议文件。
# 商标隔离免责声明
# 项目名称 Rebirth LLM（简称RLLM）为独立软件项目代号，与奢侈品品牌Hermes、开源项目Hermes-Agent无品牌合作、隶属关联；仅代码内部功能性调用开源框架，不会使用Hermes相关名称开展商业宣传，无品牌混淆意图。
# ============================================================
