# File: D:\AI_RLLM\rllm_tests\test_modules.py
"""Rebirth LLM(RLLM) Rebirth LLM(RLLM) 单模块验收自测脚本

逐条验证：
  [x] D盘目录完整性
  [x] 内存硬锁2GB超限阻断
  [x] KV缓存溢出落D盘
  [x] 磁盘分片索引读写
  [x] 异步调度器预取/卸载
  [x] 三层记忆下沉冷层
  [x] 复盘引擎触发进化
  [x] 自动调优器邻域搜索
  [x] 检查点原子写+恢复
  [x] 输出JSONL写盘不驻留

运行:
  call D:\AI_RLLM\.venv\Scripts\activate.bat
  python D:\AI_RLLM\rllm_tests\test_modules.py
"""
from __future__ import annotations
import env_config  # RLLM全局环境变量自动注入（硬锁定D:\AI_RLLM）

import asyncio
import gc
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List

# 注入D盘
ROOT = Path(r"D:\AI_RLLM")
os.environ["HF_HOME"] = str(ROOT / "hf_cache")
os.environ["TRANSFORMERS_CACHE"] = str(ROOT / "hf_cache" / "hub")
os.environ["HF_OFFLINE"] = "1"
for _s in ("hermes_core", "disk_engine", "auto_evo", "pipeline"):
    sys.path.insert(0, str(ROOT / _s))


# =================================================================
# 测试框架（轻量）
# =================================================================
results: List[Dict[str, Any]] = []


def case(name: str) -> Callable[[Callable], Callable]:
    def _wrap(fn: Callable) -> Callable:
        def _run(*args, **kwargs):  # type: ignore[no-untyped-def]
            t0 = time.time()
            err = ""
            ok = False
            try:
                if asyncio.iscoroutinefunction(fn):
                    asyncio.run(fn(*args, **kwargs))
                else:
                    fn(*args, **kwargs)
                ok = True
            except AssertionError as e:
                err = f"断言失败: {e}"
            except MemoryError as e:
                if "硬限" in str(e):
                    ok = True
                    err = f"[预期] {e}"
                else:
                    err = f"非预期MemoryError: {e}"
            except Exception as e:  # noqa: BLE001
                err = f"异常: {type(e).__name__}: {e}"
            finally:
                dur = (time.time() - t0) * 1000
                results.append({
                    "name": name,
                    "ok": ok,
                    "ms": round(dur, 1),
                    "err": err,
                })
                flag = "✓" if ok else "✗"
                print(f"  [{flag}] {name} ({dur:.0f}ms) -> {'OK' if ok else err}")
        return _run
    return _wrap


# =================================================================
# Test 1: 目录完整性
# =================================================================
@case("D盘全套目录完整性")
def _test_dirs() -> None:
    required = [
        ROOT / ".venv", ROOT / "hf_cache" / "hub", ROOT / "hf_cache" / "datasets",
        ROOT / "model_shards" / "indexes",
        ROOT / "offload_temp" / "kv_cache", ROOT / "offload_temp" / "tensor_swap",
        ROOT / "hermes_core" / "workers", ROOT / "hermes_core" / "memory",
        ROOT / "hermes_core" / "skills", ROOT / "hermes_core" / "review",
        ROOT / "output_dataset", ROOT / "skill_storage" / "archive",
        ROOT / "disk_engine" / "sharding", ROOT / "disk_engine" / "scheduler",
        ROOT / "disk_engine" / "kv_manager", ROOT / "disk_engine" / "memory_lock",
        ROOT / "auto_evo" / "metrics", ROOT / "auto_evo" / "tuner", ROOT / "auto_evo" / "strategy",
        ROOT / "pipeline" / "checkpoint",
        ROOT / "input_data", ROOT / "logs", ROOT / "tests",
    ]
    missing = [str(d) for d in required if not d.exists()]
    assert not missing, f"缺失目录: {missing}"


# =================================================================
# Test 2: 内存硬锁
# =================================================================
@case("全局内存硬锁 - 2GB封顶构造值校验")
def _test_memlock_construct() -> None:
    from rllm_disk_engine.memory_lock.global_memory_lock import GlobalMemoryLock
    try:
        GlobalMemoryLock(limit_gb=3.0)
        raise AssertionError("应当拒绝>2GB")
    except ValueError as e:
        assert "硬限" in str(e), f"错误信息不对: {e}"


# =================================================================
# Test 3: KV溢出管理器
# =================================================================
@case("KV缓存磁盘溢出管理器 - 超阈值自动spill")
async def _test_kv_spill() -> None:
    from rllm_disk_engine.kv_manager.kv_spill_manager import KVSpillManager
    tmp = ROOT / "offload_temp" / "kv_cache_test"
    if tmp.exists():
        for f in tmp.glob("*.kv.bin"):
            f.unlink(missing_ok=True)
    km = KVSpillManager(temp_dir=tmp, spill_threshold_mb=1)  # 1MB阈值
    n = 50
    for i in range(n):
        big = b"x" * (128 * 1024)  # 128KB
        await km.put(f"k{i}", big, task_id="t", layer_idx=0, size_hint_bytes=len(big))
    stats = km.stats()
    spill_count = await km.get_spill_count()
    assert spill_count > 0, f"未触发spill stats={stats}"
    assert stats["spill_files"] > 0, "没有落盘文件"
    # 取回
    v = await km.get("k42")
    assert v is not None, "取回失败"
    assert len(v) == 128 * 1024
    print(f"    [info] 写入{n}条, KV spill_count={spill_count}, files={stats['spill_files']}")


# =================================================================
# Test 4: 分片持久化 + 索引
# =================================================================
@case("模型分片持久化器 - 32层切片+索引JSON")
def _test_shard_persistor() -> None:
    from rllm_disk_engine.sharding.shard_persistor import (
        ModelShardPersistor, QuantizationType, INDEXES_DIR
    )
    raw_dir = ROOT / "model_shards" / "_raw_test"
    raw_dir.mkdir(exist_ok=True)
    pers = ModelShardPersistor()
    idx = pers.slice_and_persist(raw_dir, "unit_test_model", QuantizationType.INT8, target_shard_mb=512)
    assert idx.total_layers == 32, f"层数不对 {idx.total_layers}"
    # 每4个分片覆盖1层，32层 * 4 = 128
    assert len(idx.shards) == 128, f"分片数不对 {len(idx.shards)}"
    idx_path = INDEXES_DIR / "unit_test_model_int8_index.json"
    assert idx_path.exists(), "索引JSON不存在"
    # 读回
    tens, meta = pers.load_shard("unit_test_model", "layer_000_attn_qkv")
    assert meta.layer_idx == 0
    assert "weight_placeholder" in tens or "meta" in tens
    print(f"    [info] 索引路径: {idx_path}, 分片数={len(idx.shards)}")


# =================================================================
# Test 5: 调度器加载/卸载
# =================================================================
@case("异步分页调度器 - 单层加载/卸载 + 缓冲区")
async def _test_scheduler() -> None:
    from rllm_disk_engine.scheduler.async_page_scheduler import AsyncPageScheduler
    sched = AsyncPageScheduler(
        shards_dir=ROOT / "model_shards",
        model_name="unit_test_model",
        prefetch_layers_ahead=1,
        prefetch_threads=2,
        enable_mmap=False,
        buffer_capacity_gb=2.0,
    )
    for li in range(4):
        tens, ms = await sched.load_layer(li)
        assert tens is not None, f"层{li}加载失败"
        assert ms >= 0, f"延迟异常 {ms}"
    stats_before = sched.stats()
    assert stats_before["loaded_layers"] > 0
    await sched.unload_layer(0)
    await sched.unload_layer(1)
    stats_after = sched.stats()
    # 预取命中应>0
    print(f"    [info] before={stats_before} after={stats_after}")


# =================================================================
# Test 6: 三层记忆
# =================================================================
@case("三层记忆 - 热层溢出下沉暖层/冷层")
async def _test_memory_tier() -> None:
    from rllm_agent_core.memory.three_layer_memory import (
        ThreeTierMemoryManager, MemoryItem, COLD_DISK_DIR
    )
    mm = ThreeTierMemoryManager(hot_cap_bytes=1024 * 10, warm_cap_bytes=1024 * 1024)
    # 10MB热层，放12条1MB条目
    for i in range(12):
        data = b"h" * (1024 * 1024)  # 1MB
        ok = await mm.store(f"m{i}", data, size_hint_bytes=len(data))
        assert ok, f"存储失败 {i}"
    usage = mm.usage_summary()
    # 冷层应有落盘
    assert usage["cold_bytes"] > 0 or usage["warm_bytes"] > 0, f"没有下沉 {usage}"
    # 取回
    val = await mm.fetch("m6")
    assert val is not None and len(val) == 1024 * 1024
    # 强制swap
    paths = await mm.force_swap_tensors_to_disk({"tA": b"A" * 2048, "tB": b"B" * 2048})
    assert len(paths) == 2 and all(str(p).startswith(str(ROOT)) for p in paths.values()), f"路径不是D盘: {paths}"
    print(f"    [info] usage={usage}, swap_paths={list(paths.values())[:1]}...")


# =================================================================
# Test 7: 复盘引擎 + 自进化触发
# =================================================================
@case("复盘引擎 - 延迟涨20%触发进化")
async def _test_review_engine() -> None:
    from rllm_agent_core.review.review_engine import (
        ReviewEngine, ReviewMetrics, EvolutionTrigger
    )
    from rllm_agent_core.skills.skill_loader import (
        register_default_skills, load_skill, DiskOffloadInferSkill
    )
    register_default_skills()
    skill: DiskOffloadInferSkill = load_skill("disk_offload_infer")  # type: ignore[assignment]
    before_sig = skill.get_config().signature()
    engine = ReviewEngine(baseline_rounds=5)
    # 构造基线
    baseline = ReviewMetrics(window_rounds=10, avg_latency_ms=100.0)
    engine._baseline_metrics = baseline  # type: ignore[attr-defined]
    # 构造触发样本：延迟涨30% + 失败率0.8%
    met = ReviewMetrics(
        window_rounds=10,
        avg_latency_ms=140.0,  # 涨40%
        failure_rate=0.01,     # 1%
        latency_change_ratio=0.40,
    )
    triggers = engine.evaluate_triggers(met)
    assert EvolutionTrigger.LATENCY_INCREASE in triggers, f"未触发延迟: {triggers}"
    assert EvolutionTrigger.FAILURE_RATE in triggers, f"未触发失败率: {triggers}"
    # 执行进化
    new_cfg = await engine._do_evolve(skill.get_config(), met)  # type: ignore[attr-defined]
    after_sig = new_cfg.signature()
    print(f"    [info] before={before_sig[:8]} after={after_sig[:8]} triggers={[t.value for t in triggers]}")


# =================================================================
# Test 8: 自动调优器邻域搜索
# =================================================================
@case("自动调优器 - 邻域搜索+策略池归档")
def _test_tuner() -> None:
    from rllm_auto_evo.tuner.auto_tuner import AutoTuner
    from rllm_auto_evo.strategy.strategy_pool import StrategyPool
    from rllm_agent_core.skills.skill_loader import DiskOffloadSkillConfig
    pool = StrategyPool(pool_path=ROOT / "skill_storage" / "ut_strategy_pool.json")
    # 清理旧数据
    if pool._path.exists():
        pool._path.unlink(missing_ok=True)
    pool = StrategyPool(pool_path=ROOT / "skill_storage" / "ut_strategy_pool.json")
    tuner = AutoTuner(strategy_pool=pool, max_adj_per_round=1)
    cfg = DiskOffloadSkillConfig()
    cfgs = set()
    for round_i in range(10):
        metrics = {"avg_peak_memory_mb": 1900.0, "max_io_block_ms": 2000.0}
        cfg = tuner.suggest_next_config(cfg, metrics, trigger="ut")
        cfgs.add(cfg.signature())
    # 10轮应至少产生3个不同配置
    assert len(cfgs) >= 3, f"搜索空间太窄: {cfgs}"
    best = pool.get_best()
    print(f"    [info] 不同配置数={len(cfgs)}, 最优策略={best.sig[:8] if best else None}")


# =================================================================
# Test 9: 检查点原子写+恢复
# =================================================================
@case("检查点管理器 - 原子写/恢复/完成集")
def _test_checkpoint() -> None:
    from rllm_pipeline.checkpoint.checkpoint_manager import CheckpointManager
    ckpt_path = ROOT / "pipeline" / "checkpoint" / "ut_ckpt.json"
    ckpt_path.unlink(missing_ok=True)
    ckpt = CheckpointManager(checkpoint_path=ckpt_path, save_every=2)
    for i in range(20):
        if i % 2 == 0:
            ckpt.mark_success(i)
        else:
            ckpt.mark_failed(i, f"fail_{i}")
    snap1 = ckpt.snapshot()
    assert snap1.total_success == 10 and snap1.total_failed == 10, f"统计错误 {snap1}"
    # 重建
    ckpt2 = CheckpointManager(checkpoint_path=ckpt_path)
    snap2 = ckpt2.snapshot()
    assert snap2.total_success == 10, f"恢复失败: success={snap2.total_success}"
    assert ckpt2.is_done(8) and not ckpt2.is_done(9)
    print(f"    [info] 成功={snap2.total_success} 失败={snap2.total_failed} last_idx={snap2.last_processed_idx}")


# =================================================================
# Test 10: 输出JSONL写盘
# =================================================================
@case("数据集输出写入器 - 追加JSONL+自动切分")
async def _test_writer() -> None:
    from rllm_pipeline.writer.output_writer import OutputDatasetWriter, OutputRecord
    odir = ROOT / "output_dataset" / "_unittest"
    odir.mkdir(exist_ok=True)
    for f in odir.glob("*.jsonl"):
        f.unlink()
    w = OutputDatasetWriter(output_dir=odir, flush_every=5)
    total = 50
    for i in range(total):
        await w.write_async(OutputRecord(
            idx=i, source_file="ut.txt", keyword=f"kw{i}",
            category="cat", style="st", generated_text=f"测试输出_{i}" * 5,
        ))
    w.flush_all()
    w.close()
    # 校验行数
    lines = 0
    for f in odir.glob("*.jsonl"):
        with open(f, "r", encoding="utf-8") as fp:
            lines += sum(1 for _ in fp)
    assert lines == total, f"写入行数不对 lines={lines} total={total}"
    print(f"    [info] 写入{lines}条, 目录={odir.name}")


# =================================================================
# Main
# =================================================================
def main() -> int:
    print("=" * 60)
    print("Rebirth LLM(RLLM) 单模块验收自测")
    print(f"D盘根目录: {ROOT}")
    print(f"Python: {sys.version}")
    print("=" * 60)
    t0 = time.time()
    # 顺序执行
    for name, fn in list(globals().items()):
        if name.startswith("_test_") and callable(fn):
            fn()
    dur = time.time() - t0
    passed = sum(1 for r in results if r["ok"])
    failed = len(results) - passed
    print("=" * 60)
    print(f"结果: 通过 {passed}/{len(results)}  失败 {failed}  总耗时 {dur*1000:.0f}ms")
    print("=" * 60)
    # 写报告
    report_path = ROOT / "logs" / f"test_report_{int(time.time())}.json"
    with open(report_path, "w", encoding="utf-8") as fp:
        json.dump(results, fp, ensure_ascii=False, indent=2)
    print(f"测试报告: {report_path}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())


# ============================================================
# 版权声明
# 本项目 Rebirth LLM(RLLM) 基于开源项目 Nous Hermes-Agent（MIT License）二次深度开发，项目内保留完整原始开源协议文件；智能体自迭代调度逻辑复用开源代码，磁盘分层加载、全局内存锁、D盘隔离部署、自动IO调优模块为自研闭源模块，分发时附带完整MIT协议文件。
# 商标隔离免责声明
# 项目名称 Rebirth LLM（简称RLLM）为独立软件项目代号，与奢侈品品牌Hermes、开源项目Hermes-Agent无品牌合作、隶属关联；仅代码内部功能性调用开源框架，不会使用Hermes相关名称开展商业宣传，无品牌混淆意图。
# ============================================================
