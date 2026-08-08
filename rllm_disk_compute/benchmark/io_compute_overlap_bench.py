# -*- coding: utf-8 -*-
# File: D:\AI_RLLM\rllm_disk_compute\benchmark\io_compute_overlap_bench.py
"""I/O vs 计算重叠基准测试 — 验证"计算下推"架构的物理根基

核心问题:
    纯软件版"计算下推"能否通过双缓冲让瓶颈从 I/O 转移到计算?

理论分析 (7B FP16 单层):
    - 层权重大小: ~416 MB (gate_proj + up_proj + down_proj + attn)
    - NVMe 顺序读带宽: 3-7 GB/s (典型笔记本 SSD)
    - 预期单层 I/O 时间: 416MB / 5GB/s ≈ 83 ms
    - CPU SIMD matmul 单层时间: 待实测
    - 若 matmul_time < io_time, 则双缓冲可完全掩盖 I/O, 瓶颈转移至计算

测试矩阵:
    1. NVMe 顺序读带宽 (416MB 文件, 计时)
    2. CPU 单层 matmul 时间 (7B FFN 等价: 4096×14336 + 14336×4096)
    3. 双缓冲流水线吞吐 (生产者读 + 消费者算, 重叠)
    4. 单缓冲基线对比 (顺序: 读→算→读→算)
    5. 结论: 瓶颈位置 + 双缓冲效率

运行:
    python -m rllm_disk_compute.benchmark.io_compute_overlap_bench
"""
from __future__ import annotations

import argparse
import gc
import io as _io
import os
import statistics
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch

# 日志
from loguru import logger

try:
    from rllm_agent_core import LOG_DIR
except Exception:
    LOG_DIR = Path(r"D:\AI_RLLM\logs")
    LOG_DIR.mkdir(parents=True, exist_ok=True)

logger.remove()
logger.add(sys.stderr, level="INFO", format="<level>{level: <7}</level> | {message}")
logger.add(
    LOG_DIR / "disk_compute_bench_{time}.log",
    rotation="50 MB",
    retention="7 days",
    encoding="utf-8",
    level="DEBUG",
)


# ============================================================
# 7B 模型单层形状常量 (Mistral-7B)
# ============================================================
HIDDEN_SIZE: int = 4096
INTERMEDIATE_SIZE: int = 14336
NUM_HEADS: int = 32
NUM_KV_HEADS: int = 8
HEAD_DIM: int = 128
# 单层 FP16 权重总字节 (近似): gate+up+down+attn ≈ 416 MB
LAYER_FP16_BYTES: int = (
    # gate_proj: [I, H]
    INTERMEDIATE_SIZE * HIDDEN_SIZE * 2
    + # up_proj: [I, H]
    INTERMEDIATE_SIZE * HIDDEN_SIZE * 2
    + # down_proj: [H, I]
    HIDDEN_SIZE * INTERMEDIATE_SIZE * 2
    + # q_proj: [H, H]
    HIDDEN_SIZE * HIDDEN_SIZE * 2
    + # k_proj: [H_kv, H]
    NUM_KV_HEADS * HEAD_DIM * HIDDEN_SIZE * 2
    + # v_proj: [H_kv, H]
    NUM_KV_HEADS * HEAD_DIM * HIDDEN_SIZE * 2
    + # o_proj: [H, H]
    HIDDEN_SIZE * HIDDEN_SIZE * 2
)
BENCH_DIR: Path = Path(r"D:\AI_RLLM\rllm_offload_temp\disk_compute_bench")


# ============================================================
# 测试结果数据结构
# ============================================================
@dataclass
class BenchResult:
    """单项测试结果

    Attributes:
        name: 测试名称
        samples: 每次采样的耗时 (秒)
        bytes_processed: 处理的总字节数 (用于算带宽)
        extra: 附加指标 (如 tok/s)
    """
    name: str
    samples: List[float] = field(default_factory=list)
    bytes_processed: int = 0
    extra: Dict[str, float] = field(default_factory=dict)

    def median_ms(self) -> float:
        """中位数耗时 (毫秒)"""
        if not self.samples:
            return 0.0
        return statistics.median(self.samples) * 1000.0

    def mean_ms(self) -> float:
        """平均耗时 (毫秒)"""
        if not self.samples:
            return 0.0
        return statistics.mean(self.samples) * 1000.0

    def bandwidth_gbps(self) -> float:
        """等效带宽 (GB/s)"""
        if not self.samples or self.bytes_processed == 0:
            return 0.0
        mean_s = statistics.mean(self.samples)
        return (self.bytes_processed / (1024**3)) / mean_s


# ============================================================
# 工具: 准备测试用权重文件
# ============================================================
def prepare_bench_files(
    layer_size_bytes: int = LAYER_FP16_BYTES,
    num_layers: int = 8,
) -> List[Path]:
    """准备 benchmark 用的权重文件 (随机数据, 仅测 I/O 不测数值)

    Args:
        layer_size_bytes: 单层权重大小 (字节)
        num_layers: 要准备的层数

    Returns:
        文件路径列表
    """
    BENCH_DIR.mkdir(parents=True, exist_ok=True)
    files: List[Path] = []
    for i in range(num_layers):
        fp = BENCH_DIR / f"bench_layer_{i:03d}.bin"
        if not fp.exists() or fp.stat().st_size != layer_size_bytes:
            logger.info(f"[准备] 生成测试文件 {fp.name} ({layer_size_bytes/1024**2:.0f}MB)")
            # 用 /dev/urandom 等价方式写随机字节 (不构造张量, 纯 I/O 测试)
            chunk_size = 64 * 1024 * 1024  # 64MB 一块
            remaining = layer_size_bytes
            with open(fp, "wb") as f:
                while remaining > 0:
                    write_size = min(chunk_size, remaining)
                    f.write(os.urandom(write_size))
                    remaining -= write_size
        files.append(fp)
    logger.info(f"[准备] 共 {len(files)} 个测试文件, 每个 {layer_size_bytes/1024**2:.0f}MB")
    return files


def cleanup_bench_files() -> None:
    """清理 benchmark 文件"""
    if BENCH_DIR.exists():
        for fp in BENCH_DIR.glob("*.bin"):
            try:
                fp.unlink()
            except OSError:
                pass
        logger.info("[清理] 已删除 benchmark 测试文件")


# ============================================================
# 测试 1: NVMe 顺序读带宽
# ============================================================
def bench_sequential_read(
    files: List[Path],
    warmup: int = 2,
    rounds: int = 5,
) -> BenchResult:
    """测试 NVMe 顺序读带宽

    每轮: 顺序读取所有文件, 每次 4MB 块, 累加字节数防优化
    跳过前 warmup 轮 (页缓存预热), 取后 rounds 轮统计

    Args:
        files: 要读取的文件列表
        warmup: 预热轮数 (页缓存)
        rounds: 实际计时轮数

    Returns:
        BenchResult
    """
    result = BenchResult(
        name="NVMe 顺序读",
        bytes_processed=len(files) * LAYER_FP16_BYTES,
    )
    total_bytes_per_round = len(files) * LAYER_FP16_BYTES
    block_size = 4 * 1024 * 1024  # 4MB 块

    logger.info(
        f"[{result.name}] 开始: {len(files)} 文件 × "
        f"{LAYER_FP16_BYTES/1024**2:.0f}MB, warmup={warmup} rounds={rounds}"
    )

    for r in range(warmup + rounds):
        t0 = time.perf_counter()
        checksum = 0
        for fp in files:
            with open(fp, "rb") as f:
                while True:
                    block = f.read(block_size)
                    if not block:
                        break
                    # 防止编译器优化掉读操作
                    checksum += len(block)
        elapsed = time.perf_counter() - t0
        # 确保checksum被使用
        assert checksum == total_bytes_per_round, "读取字节数不匹配"
        if r >= warmup:
            result.samples.append(elapsed)
            logger.debug(
                f"[{result.name}] 轮{r}: {elapsed*1000:.1f}ms "
                f"({total_bytes_per_round/elapsed/1024**3:.2f} GB/s)"
            )

    logger.success(
        f"[{result.name}] 中位 {result.median_ms():.1f}ms, "
        f"均值 {result.mean_ms():.1f}ms, "
        f"带宽 {result.bandwidth_gbps():.2f} GB/s"
    )
    return result


# ============================================================
# 测试 2: CPU 单层 matmul 时间
# ============================================================
def bench_cpu_matmul(
    warmup: int = 2,
    rounds: int = 5,
    threads: Optional[int] = None,
) -> BenchResult:
    """测试 CPU 单层 matmul 时间 (7B FFN 等价计算)

    计算: hidden(1, 4096) → gate(14336) → silu → up(14336) → down(4096)
    权重用 FP32 (CPU 上 FP32 比 FP16 快, 因 AVX2 主要是 FP32)
    测试纯计算时间, 不含 I/O

    Args:
        warmup: 预热轮数
        rounds: 计时轮数
        threads: torch CPU 线程数 (None=默认)

    Returns:
        BenchResult
    """
    if threads is not None:
        torch.set_num_threads(threads)

    result = BenchResult(name=f"CPU matmul (threads={torch.get_num_threads()})")
    logger.info(
        f"[{result.name}] 开始: hidden={HIDDEN_SIZE} intermediate={INTERMEDIATE_SIZE}"
    )

    # 预分配权重 (FP32, 模拟已在 CPU RAM 的场景)
    hidden = torch.randn(1, HIDDEN_SIZE, dtype=torch.float32)
    gate_w = torch.randn(INTERMEDIATE_SIZE, HIDDEN_SIZE, dtype=torch.float32)
    up_w = torch.randn(INTERMEDIATE_SIZE, HIDDEN_SIZE, dtype=torch.float32)
    down_w = torch.randn(HIDDEN_SIZE, INTERMEDIATE_SIZE, dtype=torch.float32)

    # 处理字节数: 三个权重矩阵 (读一遍)
    result.bytes_processed = (
        gate_w.nelement() * 4 + up_w.nelement() * 4 + down_w.nelement() * 4
    )

    for r in range(warmup + rounds):
        t0 = time.perf_counter()
        # FFN 等价: down(silu(gate(x)) * up(x))
        gate_out = torch.mm(hidden, gate_w.t())
        up_out = torch.mm(hidden, up_w.t())
        fused = torch.nn.functional.silu(gate_out) * up_out
        out = torch.mm(fused, down_w.t())
        elapsed = time.perf_counter() - t0
        # 防优化
        assert out.shape == (1, HIDDEN_SIZE)
        if r >= warmup:
            result.samples.append(elapsed)
            logger.debug(
                f"[{result.name}] 轮{r}: {elapsed*1000:.2f}ms"
            )

    # 释放
    del hidden, gate_w, up_w, down_w, gate_out, up_out, fused, out
    gc.collect()

    logger.success(
        f"[{result.name}] 中位 {result.median_ms():.2f}ms, "
        f"均值 {result.mean_ms():.2f}ms"
    )
    return result


# ============================================================
# 测试 3: CPU matmul 含 I/O 加载 (模拟"从磁盘读权重并计算")
# ============================================================
def bench_matmul_with_io(
    files: List[Path],
    warmup: int = 2,
    rounds: int = 5,
    threads: Optional[int] = None,
) -> BenchResult:
    """测试"读权重 → 计算"的串行耗时 (单缓冲基线)

    每层: read 416MB → frombuffer → mm 计算一次
    这是"先读再算"的传统模式, 作为双缓冲的对比基线

    Args:
        files: 权重文件列表
        warmup: 预热轮数
        rounds: 计时轮数
        threads: CPU 线程数

    Returns:
        BenchResult
    """
    if threads is not None:
        torch.set_num_threads(threads)

    result = BenchResult(
        name=f"串行 读+算 (threads={torch.get_num_threads()})",
        bytes_processed=LAYER_FP16_BYTES,
    )
    logger.info(f"[{result.name}] 开始")

    # 预分配输出张量 (避免分配时间计入)
    hidden = torch.randn(1, HIDDEN_SIZE, dtype=torch.float32)
    # 用 reshape 视图模拟加载后的权重 (FP32 字节数是 FP16 的 2 倍, 但我们只测流程)
    # 简化: 直接把读到的 bytes 当 FP16 张量
    # gate_w: [I, H] FP16
    gate_shape = (INTERMEDIATE_SIZE, HIDDEN_SIZE)
    up_shape = (INTERMEDIATE_SIZE, HIDDEN_SIZE)
    down_shape = (HIDDEN_SIZE, INTERMEDIATE_SIZE)
    gate_bytes = INTERMEDIATE_SIZE * HIDDEN_SIZE * 2
    up_bytes = INTERMEDIATE_SIZE * HIDDEN_SIZE * 2
    down_bytes = HIDDEN_SIZE * INTERMEDIATE_SIZE * 2

    for r in range(warmup + rounds):
        t0 = time.perf_counter()
        for fp in files[:1]:  # 每轮测一层即可
            with open(fp, "rb") as f:
                data = f.read()
            # 把 bytes 解释为 FP16 张量 (不拷贝, 用 frombuffer)
            gate_arr = torch.frombuffer(data[:gate_bytes], dtype=torch.float16)
            gate_w = gate_arr.reshape(gate_shape)
            up_arr = torch.frombuffer(data[gate_bytes:gate_bytes+up_bytes], dtype=torch.float16)
            up_w = up_arr.reshape(up_shape)
            down_arr = torch.frombuffer(
                data[gate_bytes+up_bytes:gate_bytes+up_bytes+down_bytes],
                dtype=torch.float16,
            )
            down_w = down_arr.reshape(down_shape)

            # FFN: hidden → gate → silu → up → down
            h_half = hidden.to(torch.float16)
            gate_out = torch.mm(h_half, gate_w.t())
            up_out = torch.mm(h_half, up_w.t())
            fused = torch.nn.functional.silu(gate_out) * up_out
            out = torch.mm(fused, down_w.t())

        elapsed = time.perf_counter() - t0
        assert out.shape == (1, HIDDEN_SIZE)
        if r >= warmup:
            result.samples.append(elapsed)
            logger.debug(f"[{result.name}] 轮{r}: {elapsed*1000:.1f}ms")

    del hidden, gate_w, up_w, down_w, gate_out, up_out, fused, out
    gc.collect()

    logger.success(
        f"[{result.name}] 中位 {result.median_ms():.1f}ms, "
        f"带宽 {result.bandwidth_gbps():.2f} GB/s"
    )
    return result


# ============================================================
# 测试 4: 双缓冲流水线 (I/O 与计算重叠)
# ============================================================
@dataclass
class DoubleBufferState:
    """双缓冲状态

    Attributes:
        buf: 两个缓冲区 [buf_a, buf_b]
        ready: 缓冲区是否就绪 (可被消费者使用)
        idx: 缓冲区对应的层索引 (-1 表示无效)
        lock: 互斥锁
        cond_ready: 缓冲区就绪条件变量
        cond_empty: 缓冲区空闲条件变量
        stop: 停止信号
    """
    buf: List[Optional[bytes]]
    ready: List[bool]
    idx: List[int]
    lock: threading.Lock
    cond_ready: threading.Condition
    cond_empty: threading.Condition
    stop: bool = False


def bench_double_buffer_pipeline(
    files: List[Path],
    warmup: int = 2,
    rounds: int = 5,
    threads: Optional[int] = None,
) -> BenchResult:
    """测试双缓冲流水线吞吐 (I/O 与计算重叠)

    生产者线程: 读 layer_{i+1} → 写 buf_b, 同时消费者算 buf_a (layer_i)
    消费者线程: 从 buf 读权重 → matmul → 释放 buf, 同时生产者读下一层

    流水线稳态: max(io_time, compute_time) per layer
    理想情况: min(io_time, compute_time) 被完全掩盖

    Args:
        files: 权重文件列表
        warmup: 预热轮数
        rounds: 计时轮数
        threads: CPU 线程数

    Returns:
        BenchResult
    """
    if threads is not None:
        torch.set_num_threads(threads)

    result = BenchResult(
        name=f"双缓冲流水线 (threads={torch.get_num_threads()})",
        bytes_processed=LAYER_FP16_BYTES,
    )
    num_layers = len(files)
    logger.info(
        f"[{result.name}] 开始: {num_layers} 层, 双缓冲"
    )

    # 形状常量
    gate_shape = (INTERMEDIATE_SIZE, HIDDEN_SIZE)
    up_shape = (INTERMEDIATE_SIZE, HIDDEN_SIZE)
    down_shape = (HIDDEN_SIZE, INTERMEDIATE_SIZE)
    gate_bytes = INTERMEDIATE_SIZE * HIDDEN_SIZE * 2
    up_bytes = INTERMEDIATE_SIZE * HIDDEN_SIZE * 2
    down_bytes = HIDDEN_SIZE * INTERMEDIATE_SIZE * 2

    hidden = torch.randn(1, HIDDEN_SIZE, dtype=torch.float32)

    def producer(
        state: DoubleBufferState,
        files_list: List[Path],
    ) -> None:
        """生产者: 顺序读取文件到缓冲区"""
        layer_i = 0
        while layer_i < len(files_list) and not state.stop:
            # 找一个空闲槽
            with state.cond_empty:
                while not state.stop:
                    # 找一个 not ready 的槽
                    free_slots = [i for i, r in enumerate(state.ready) if not r]
                    if free_slots:
                        slot = free_slots[0]
                        break
                    state.cond_empty.wait(timeout=0.01)
                if state.stop:
                    return

            # 读文件到 buf[slot] (不持锁, IO 可与消费并行)
            with open(files_list[layer_i], "rb") as f:
                data = f.read()

            with state.cond_ready:
                state.buf[slot] = data
                state.idx[slot] = layer_i
                state.ready[slot] = True
                state.cond_ready.notify_all()
            layer_i += 1

        # 标记结束
        with state.cond_ready:
            state.stop = True
            state.cond_ready.notify_all()

    def consumer_compute(data: bytes) -> float:
        """消费者: 从 bytes 计算一层 FFN, 返回耗时"""
        t0 = time.perf_counter()
        gate_arr = torch.frombuffer(data[:gate_bytes], dtype=torch.float16)
        gate_w = gate_arr.reshape(gate_shape)
        up_arr = torch.frombuffer(data[gate_bytes:gate_bytes+up_bytes], dtype=torch.float16)
        up_w = up_arr.reshape(up_shape)
        down_arr = torch.frombuffer(
            data[gate_bytes+up_bytes:gate_bytes+up_bytes+down_bytes],
            dtype=torch.float16,
        )
        down_w = down_arr.reshape(down_shape)

        h_half = hidden.to(torch.float16)
        gate_out = torch.mm(h_half, gate_w.t())
        up_out = torch.mm(h_half, up_w.t())
        fused = torch.nn.functional.silu(gate_out) * up_out
        out = torch.mm(fused, down_w.t())
        elapsed = time.perf_counter() - t0
        assert out.shape == (1, HIDDEN_SIZE)
        return elapsed

    # 跑 warmup + rounds
    for r in range(warmup + rounds):
        # 重置状态 (先建 lock, 再建 Condition 复用 lock, 避免循环引用)
        state_lock = threading.Lock()
        state = DoubleBufferState(
            buf=[None, None],
            ready=[False, False],
            idx=[-1, -1],
            lock=state_lock,
            cond_ready=threading.Condition(state_lock),
            cond_empty=threading.Condition(state_lock),
            stop=False,
        )

        t0 = time.perf_counter()
        # 启动生产者
        prod_thread = threading.Thread(
            target=producer, args=(state, files), daemon=True
        )
        prod_thread.start()

        # 消费者: 主线程
        compute_times: List[float] = []
        consumed = 0
        while consumed < num_layers:
            with state.cond_ready:
                while not any(state.ready) and not state.stop:
                    state.cond_ready.wait(timeout=0.01)
                if not any(state.ready) and state.stop:
                    break
                # 取一个 ready 的槽
                ready_slots = [i for i, rdy in enumerate(state.ready) if rdy]
                slot = ready_slots[0]
                data = state.buf[slot]
                state.buf[slot] = None
                state.ready[slot] = False
                state.idx[slot] = -1
                state.cond_empty.notify_all()

            # 计算 (不持锁, 可与生产者 IO 并行)
            ct = consumer_compute(data)
            compute_times.append(ct)
            consumed += 1

        prod_thread.join(timeout=5.0)
        elapsed = time.perf_counter() - t0

        if r >= warmup:
            result.samples.append(elapsed)
            # 附加: 平均计算时间和总吞吐
            avg_compute_ms = (sum(compute_times) / len(compute_times)) * 1000
            result.extra[f"avg_compute_ms_round{r}"] = avg_compute_ms
            result.extra[f"total_compute_ms_round{r}"] = sum(compute_times) * 1000
            logger.debug(
                f"[{result.name}] 轮{r}: 总 {elapsed*1000:.1f}ms, "
                f"平均每层 {elapsed*1000/num_layers:.1f}ms, "
                f"纯计算 {avg_compute_ms:.2f}ms/层"
            )

    del hidden
    gc.collect()

    # 计算每层平均时间
    per_layer_ms = result.median_ms() / num_layers
    logger.success(
        f"[{result.name}] 总中位 {result.median_ms():.1f}ms ({num_layers}层), "
        f"每层 {per_layer_ms:.1f}ms, "
        f"带宽 {LAYER_FP16_BYTES*num_layers/(result.median_ms()/1000)/1024**3:.2f} GB/s"
    )
    return result


# ============================================================
# 主入口: 跑所有测试并输出结论
# ============================================================
def run_full_benchmark(
    num_layers: int = 8,
    warmup: int = 2,
    rounds: int = 5,
    keep_files: bool = False,
) -> Dict[str, Any]:
    """运行完整 benchmark 并输出结论

    Args:
        num_layers: 测试层数 (建议 8-32)
        warmup: 预热轮数
        rounds: 计时轮数
        keep_files: 是否保留测试文件

    Returns:
        完整结果字典
    """
    logger.info("=" * 70)
    logger.info("I/O vs 计算重叠基准测试 — 验证计算下推架构根基")
    logger.info(f"单层 FP16 大小: {LAYER_FP16_BYTES/1024**2:.1f} MB")
    logger.info(f"测试层数: {num_layers}, warmup={warmup}, rounds={rounds}")
    logger.info("=" * 70)

    # CPU 信息
    cpu_threads = torch.get_num_threads()
    logger.info(f"CPU 线程数: {cpu_threads}")

    # 1. 准备文件
    files = prepare_bench_files(num_layers=num_layers)

    results: Dict[str, BenchResult] = {}

    # 2. NVMe 顺序读
    results["io_read"] = bench_sequential_read(files, warmup=warmup, rounds=rounds)

    # 3. CPU matmul (纯计算, 权重在内存)
    results["compute_pure"] = bench_cpu_matmul(warmup=warmup, rounds=rounds, threads=cpu_threads)

    # 4. 串行 读+算 (基线)
    results["serial"] = bench_matmul_with_io(files, warmup=warmup, rounds=rounds, threads=cpu_threads)

    # 5. 双缓冲流水线
    results["pipeline"] = bench_double_buffer_pipeline(
        files, warmup=warmup, rounds=rounds, threads=cpu_threads
    )

    # 6. 结论分析
    io_ms = results["io_read"].median_ms() / num_layers
    compute_ms = results["compute_pure"].median_ms()
    serial_ms = results["serial"].median_ms()
    pipeline_ms = results["pipeline"].median_ms() / num_layers

    logger.info("\n" + "=" * 70)
    logger.info("结论分析")
    logger.info("=" * 70)
    logger.info(f"每层 I/O 时间 (NVMe 读):     {io_ms:8.2f} ms")
    logger.info(f"每层计算时间 (CPU matmul):   {compute_ms:8.2f} ms")
    logger.info(f"  → I/O / 计算 比值:         {io_ms/compute_ms:8.2f} x")
    logger.info("")
    logger.info(f"串行 读+算 (单缓冲):         {serial_ms:8.2f} ms/层")
    logger.info(f"双缓冲流水线:                {pipeline_ms:8.2f} ms/层")
    logger.info(f"  → 加速比:                  {serial_ms/pipeline_ms:8.2f} x")
    logger.info("")

    # 判断瓶颈
    if compute_ms < io_ms:
        ratio = io_ms / compute_ms
        logger.success(
            f"✓ 假设成立: CPU 算一层 ({compute_ms:.2f}ms) < NVMe 读一层 ({io_ms:.2f}ms), "
            f"I/O 是瓶颈 (比值 {ratio:.1f}x)"
        )
        logger.success(
            f"  双缓冲理论加速: {1 + compute_ms/io_ms:.2f}x "
            f"(掩盖计算时间), 实测 {serial_ms/pipeline_ms:.2f}x"
        )
        if pipeline_ms > io_ms * 1.2:
            logger.warning(
                "  但双缓冲未达到 I/O 理论上限, 可能原因: "
                "(1) Python GIL 限制 (2) torch mm 抢占内存带宽 (3) 页缓存污染"
            )
        logger.info(
            "  → 结论: 单 NVMe 瓶颈仍在 I/O, 需多 NVMe 并行 (路径二) 或量化减小 I/O"
        )
    else:
        ratio = compute_ms / io_ms
        logger.success(
            f"✓ 假设更成立: CPU 算一层 ({compute_ms:.2f}ms) > NVMe 读一层 ({io_ms:.2f}ms), "
            f"计算是瓶颈 (比值 {ratio:.1f}x)"
        )
        logger.success(
            f"  双缓冲可完全掩盖 I/O, 等效于'从磁盘直接算'"
        )
        logger.info(
            "  → 结论: 纯软件即可让磁盘扮演慢速模型仓库, "
            "瓶颈在 CPU 算力, 可加多核/量化/投机解码提速"
        )

    # 清理
    if not keep_files:
        cleanup_bench_files()

    return {
        "io_read_ms_per_layer": io_ms,
        "compute_ms_per_layer": compute_ms,
        "io_compute_ratio": io_ms / compute_ms if compute_ms > 0 else float("inf"),
        "serial_ms_per_layer": serial_ms,
        "pipeline_ms_per_layer": pipeline_ms,
        "speedup": serial_ms / pipeline_ms if pipeline_ms > 0 else 0,
        "is_io_bound": compute_ms < io_ms,
        "results": {k: {
            "name": v.name,
            "median_ms": v.median_ms(),
            "mean_ms": v.mean_ms(),
            "bandwidth_gbps": v.bandwidth_gbps(),
            "samples": v.samples,
        } for k, v in results.items()},
    }


def main() -> int:
    """命令行入口

    Returns:
        退出码
    """
    parser = argparse.ArgumentParser(description="I/O vs 计算重叠 benchmark")
    parser.add_argument("--layers", type=int, default=8, help="测试层数 (默认 8)")
    parser.add_argument("--warmup", type=int, default=2, help="预热轮数")
    parser.add_argument("--rounds", type=int, default=5, help="计时轮数")
    parser.add_argument("--keep", action="store_true", help="保留测试文件")
    parser.add_argument(
        "--json", type=str, default="",
        help="结果保存到 JSON 文件 (默认不保存)"
    )
    args = parser.parse_args()

    result = run_full_benchmark(
        num_layers=args.layers,
        warmup=args.warmup,
        rounds=args.rounds,
        keep_files=args.keep,
    )

    if args.json:
        import json
        out_path = Path(args.json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False, default=str)
        logger.info(f"结果已保存: {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
