# -*- coding: utf-8 -*-
# File: D:\AI_RLLM\rllm_disk_engine\expert_pool\expert_evictor.py
"""专家级冷热置换器 (直写 D 盘模式)

与 v3 HotColdEvictor 区别:
  1. 粒度: 从 layer_idx → (layer_idx, expert_idx)
  2. 直写 D 盘: 跳过 CPU RAM 中转, VRAM → D 盘原子写 (减少一次拷贝)
  3. fetch_back: D 盘直读 → 重建 MixtralBLockSparseTop2MLP → 4bit 绑定 → 入 VRAM
  4. 配合 ExpertFreqTracker: 按混合评分 (LRU + 频次) 决定淘汰

数据流:
  evict:  VRAM 模块 → 提取 state_dict → torch.save → D 盘
  fetch:  D 盘 → torch.load → 重建空模块 → 绑定权重 → .to(cuda) → VRAM
"""
from __future__ import annotations

import asyncio
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from loguru import logger

from rllm_agent_core import LOG_DIR
from rllm_disk_engine.expert_pool.expert_freq_tracker import ExpertFreqTracker
from rllm_disk_engine.expert_pool.expert_vram_pool import (
    ExpertVRAMPool,
    ExpertEntry,
    ExpertKey,
)

logger.add(
    LOG_DIR / "expert_evictor_{time}.log",
    rotation="50 MB",
    retention="7 days",
    encoding="utf-8",
)


class ExpertEvictor:
    """专家级冷热置换器 (直写 D 盘)

    Args:
        vram_pool: ExpertVRAMPool 实例
        shard_dir: 专家分片根目录 (D:\\AI_RLLM\\rllm_model_shards\\mixtral_8x7b_4bit)
        freq_tracker: 专家频率跟踪器
        num_layers: Transformer 层数
        num_experts_per_layer: 每层专家数
        expert_quant_bits: 专家量化位宽
    """

    def __init__(
        self,
        vram_pool: ExpertVRAMPool,
        shard_dir: Path = Path(
            r"D:\AI_RLLM\rllm_model_shards\mixtral_8x7b_4bit"
        ),
        freq_tracker: Optional[ExpertFreqTracker] = None,
        num_layers: int = 32,
        num_experts_per_layer: int = 8,
        expert_quant_bits: int = 4,
    ) -> None:
        self._pool: ExpertVRAMPool = vram_pool
        self._shard_dir: Path = Path(shard_dir)
        self._freq_tracker: ExpertFreqTracker = freq_tracker or ExpertFreqTracker(
            num_layers=num_layers,
            num_experts_per_layer=num_experts_per_layer,
        )
        self._num_layers: int = num_layers
        self._num_experts: int = num_experts_per_layer
        self._quant_bits: int = expert_quant_bits
        self._lock: threading.RLock = threading.RLock()

        # 延迟注入: 专家模块工厂 (创建空 MixtralBLockSparseTop2MLP)
        self._expert_factory: Optional[Callable[[int, int], torch.nn.Module]] = None

        logger.info(
            f"[ExpertEvictor] 初始化: shard_dir={self._shard_dir} "
            f"quant={expert_quant_bits}bit "
            f"experts={num_layers}×{num_experts_per_layer}"
        )

    def attach_factory(
        self,
        factory: Callable[[int, int], torch.nn.Module],
    ) -> None:
        """注入专家模块工厂 (供 fetch_back 重建模块用)

        Args:
            factory: (layer_idx, expert_idx) → 空 MixtralBLockSparseTop2MLP 模块
        """
        self._expert_factory = factory
        logger.info("[ExpertEvictor] 专家模块工厂已注入")

    @staticmethod
    def _move_quant_state_to_cuda(
        qs: Any,
        device: Any = None,
    ) -> Any:
        """用 QuantState 构造函数重建 CUDA 版本 (递归处理 state2).

        Params4bit 经 torch.save/load 后 pickle 退化为普通 Parameter, quant_state
        的张量 (absmax/code/state2.absmax 等) 留在 CPU 上. matmul_4bit 要求
        data 和 quant_state 张量在同一设备上, 否则触发 CUDA illegal memory
        access. 不能用 Params4bit.cuda() (会对已量化数据重新量化), 需手动搬移.

        用构造函数重建比 dir() 遍历更稳健, 避免漏掉属性或拷贝 properties.
        已通过 _test_4bit_roundtrip.py 方式A 验证 (forward 成功).

        Args:
            qs: bitsandbytes.functional.QuantState 对象 (CPU)
            device: 目标设备 (默认 cuda)

        Returns:
            新的 QuantState 对象 (所有 tensor 在 CUDA 上)
        """
        if qs is None:
            return None
        from bitsandbytes.functional import QuantState

        if device is None:
            device = torch.device("cuda")
        elif not isinstance(device, torch.device):
            device = torch.device(device)

        # 递归处理嵌套的 state2 (double quantization)
        new_state2 = None
        if getattr(qs, "state2", None) is not None:
            s2 = qs.state2
            # offset 是 1-element tensor, 必须搬到 CUDA (gemm_4bit 会 .to(float32) 但不转设备)
            s2_offset_cuda = None
            if getattr(s2, "offset", None) is not None and isinstance(s2.offset, torch.Tensor):
                s2_offset_cuda = s2.offset.to(device)
            else:
                s2_offset_cuda = s2.offset
            new_state2 = QuantState(
                absmax=s2.absmax.to(device) if s2.absmax is not None else None,
                shape=s2.shape,
                code=s2.code.to(device) if s2.code is not None else None,
                blocksize=s2.blocksize,
                quant_type=s2.quant_type,
                dtype=s2.dtype,
                offset=s2_offset_cuda,
                state2=None,
            )

        # offset 是 1-element tensor (nested quantization 时), 必须搬到 CUDA
        qs_offset_cuda = None
        if getattr(qs, "offset", None) is not None and isinstance(qs.offset, torch.Tensor):
            qs_offset_cuda = qs.offset.to(device)
        else:
            qs_offset_cuda = qs.offset

        return QuantState(
            absmax=qs.absmax.to(device) if qs.absmax is not None else None,
            shape=qs.shape,
            code=qs.code.to(device) if qs.code is not None else None,
            blocksize=qs.blocksize,
            quant_type=qs.quant_type,
            dtype=qs.dtype,
            offset=qs_offset_cuda,
            state2=new_state2,
        )

    # ----------------------------------------------------------------
    # 对外接口
    # ----------------------------------------------------------------
    async def evict_coldest(
        self, skip_disk_write: bool = False
    ) -> Optional[ExpertKey]:
        """淘汰最低频非 pinned 专家

        Args:
            skip_disk_write: 跳过直写 D 盘 (推理时权重只读, D 盘已有副本,
                跳过可消除淘汰写入 I/O, 解决 decode 阶段 I/O 拥塞)

        Returns:
            被淘汰的 (layer_idx, expert_idx) 或 None
        """
        with self._lock:
            resident = self._pool.list_resident_experts()
            if not resident:
                return None

            # 1. 候选 = 所有非 pinned 专家
            candidates: List[ExpertKey] = []
            for key in resident:
                entry = self._pool.get_expert_entry(key)
                if entry is not None and not entry.pinned:
                    candidates.append(key)
            if not candidates:
                logger.warning("[ExpertEvictor] 所有专家均 pinned, 无可淘汰")
                return None

            # 2. 评分: 冷度 = 1 - 频次评分
            scored: List[Tuple[float, float, ExpertKey]] = []
            for key in candidates:
                freq = self._freq_tracker.frequency_score(key)
                entry = self._pool.get_expert_entry(key)
                cold_score = (1 - freq, -entry.last_access_ts, key)
                scored.append(cold_score)
            scored.sort(reverse=True)
            victim_key: ExpertKey = scored[0][2]

            # 3. 从池中移除
            victim_entry = self._pool.remove_expert(victim_key)
            if victim_entry is None:
                return None

        # 4. 直写 D 盘 (除非 skip_disk_write)
        # 推理时权重只读, D 盘分片由 shard_mixtral_4bit.py 预先生成,
        # 淘汰时无需重复写入 → 消除 decode 阶段写入 I/O 拥塞
        if not skip_disk_write:
            await self._write_expert_to_disk(
                victim_key, victim_entry.module
            )

        # 5. 释放 VRAM
        del victim_entry
        self._pool.increment_evict_count()

        action = "丢弃(跳过写盘)" if skip_disk_write else "直写 D 盘"
        logger.warning(
            f"[ExpertEvictor] 淘汰专家 {victim_key} → {action}, "
            f"stats={self._pool.stats()}"
        )
        return victim_key

    async def fetch_back(
        self, key: ExpertKey
    ) -> Optional[torch.nn.Module]:
        """从 D 盘读回专家到 VRAM

        流程:
          1. 检查 D 盘分片文件是否存在
          2. torch.load 读取 state_dict
          3. 用工厂创建空 MixtralBLockSparseTop2MLP
          4. 绑定量化权重 (4bit NF4)
          5. .to(cuda) + CUDA 同步
          6. 检查 VRAM 水位, 不足则先淘汰
          7. 入池

        Args:
            key: (layer_idx, expert_idx)

        Returns:
            重建后的专家模块 (已在 VRAM) 或 None
        """
        layer_idx, expert_idx = key

        if self._expert_factory is None:
            logger.error("[ExpertEvictor] factory 未注入, 无法 fetch_back")
            return None

        # 1. 定位分片文件
        shard_path = (
            self._shard_dir
            / f"layer_{layer_idx:02d}"
            / "experts"
            / f"expert_{expert_idx}.pt"
        )
        if not shard_path.exists():
            logger.error(
                f"[ExpertEvictor] 专家 {key} 分片不存在: {shard_path}"
            )
            return None

        t0 = time.time()

        # 从 D 盘读取 state_dict (异步, 不阻塞事件循环)
        # weights_only=False: 4bit 量化分片含 bitsandbytes QuantState,
        # 需要允许反序列化 (分片由本地生成, 可信)
        state_dict = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: torch.load(
                shard_path, map_location="cpu", weights_only=False
            ),
        )
        io_ms = (time.time() - t0) * 1000

        # 3. 重建空模块
        module = self._expert_factory(layer_idx, expert_idx)

        # 3.5 如果是 4bit, 把 w1/w2/w3 的 nn.Linear 换成 Linear4bit
        if self._quant_bits == 4:
            self._replace_linear_with_4bit(module, state_dict)

        # 4. 绑定权重 (4bit: Params4bit.data 已在此步搬到 CUDA)
        size_bytes = self._bind_expert_weights(module, state_dict)

        # 5. 搬到 CUDA + 同步
        # 4bit 模式下, _bind_expert_weights 已把 Params4bit.data 搬到 CUDA,
        # 不能再调用 module.to(cuda): 会触发 Params4bit.cuda() 对已量化数据
        # 重新量化 → CUDA illegal memory access. 仅非 4bit 模式才整体搬运.
        if self._quant_bits != 4:
            module = module.to(self._pool._device)
        torch.cuda.synchronize()

        # 6. 入池前检查 VRAM, 不足则淘汰
        # skip_disk_write=True: 推理时权重只读, D 盘已有副本, 跳过写入
        while not self._pool._can_fit(size_bytes):
            evicted = await self.evict_coldest(skip_disk_write=True)
            if evicted is None:
                logger.warning(
                    f"[ExpertEvictor] fetch_back {key} 前无冷专家可淘汰"
                )
                break

        # 7. 入池
        entry = ExpertEntry(
            key=key,
            module=module,
            size_bytes=size_bytes,
            quant_bits=self._quant_bits,
        )
        self._pool.add_expert(entry)

        # 记录频次
        self._freq_tracker.record_access(key)

        total_ms = (time.time() - t0) * 1000
        logger.info(
            f"[ExpertEvictor] 专家 {key} 读回 VRAM, "
            f"io={io_ms:.0f}ms total={total_ms:.0f}ms "
            f"size={size_bytes/1024**2:.0f}MB"
        )
        return module

    # ----------------------------------------------------------------
    # mmap 快速读取 (融合: 内存映射 → 跳过 pickle 反序列化)
    # ----------------------------------------------------------------
    async def _fetch_expert_mmap(
        self, key: ExpertKey
    ) -> Optional[Dict[str, Any]]:
        """mmap 直接映射读取专家 state_dict (跳过 torch.load/pickle)

        融合思路: 用 numpy.load(mmap_mode='r') 直接映射 D 盘 .npy 文件,
        跳过 Python pickle 反序列化 (当前 fetch_back 90% 时间花在
        反序列化上). numpy .npy 是原生二进制格式, mmap 内核态直接映射.

        格式: expert_N_w{1,2,3}.npy + expert_N.meta.json

        Args:
            key: (layer_idx, expert_idx)

        Returns:
            state_dict 或 None (失败时回退到 torch.load)
        """
        import json
        import numpy as np

        layer_idx, expert_idx = key
        experts_dir = (
            self._shard_dir
            / f"layer_{layer_idx:02d}"
            / "experts"
        )
        meta_path = experts_dir / f"expert_{expert_idx}.meta.json"

        if not meta_path.exists():
            return None

        # 检查 .npy 文件都存在
        for wi in range(1, 4):
            npy_path = experts_dir / f"expert_{expert_idx}_w{wi}.npy"
            if not npy_path.exists():
                return None

        try:
            # 1. 读 meta
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)

            # 2. mmap 读取 .npy (零拷贝)
            # np.load with mmap_mode='r' 只映射虚拟地址, 不复制数据
            sd: Dict[str, Any] = {}
            for wi, wname in enumerate(["w1", "w2", "w3"]):
                w_meta = meta[wname]
                npy_file = w_meta["npy_file"]
                qs_meta = w_meta["quant_state"]

                # mmap 读取 uint8 数据 (零拷贝)
                data_np = np.load(npy_file, mmap_mode="r")
                data_tensor = torch.from_numpy(
                    data_np.copy()
                )

                # 重建 quant_state
                from bitsandbytes.functional import QuantState
                from bitsandbytes.nn import Params4bit

                # mmap absmax + code
                absmax_np = np.load(qs_meta["absmax_file"], mmap_mode="r")
                code_np = np.load(qs_meta["code_file"], mmap_mode="r")
                absmax_t = torch.from_numpy(absmax_np.copy()).to(
                    torch.float32
                )
                code_t = torch.from_numpy(code_np.copy())

                # 嵌套 state2
                new_state2 = None
                if qs_meta.get("state2") is not None:
                    s2_meta = qs_meta["state2"]
                    s2_absmax_np = np.load(
                        s2_meta["absmax_file"], mmap_mode="r"
                    )
                    s2_code_np = np.load(
                        s2_meta["code_file"], mmap_mode="r"
                    )
                    new_state2 = QuantState(
                        absmax=torch.from_numpy(
                            s2_absmax_np.copy()
                        ).to(torch.float32),
                        shape=s2_meta["shape"],
                        code=torch.from_numpy(s2_code_np.copy()),
                        blocksize=s2_meta["blocksize"],
                        quant_type=s2_meta["quant_type"],
                        dtype=torch.float32,
                        offset=None,
                        state2=None,
                    )

                qs = QuantState(
                    absmax=absmax_t,
                    shape=qs_meta["shape"],
                    code=code_t,
                    blocksize=qs_meta["blocksize"],
                    quant_type=qs_meta["quant_type"],
                    dtype=torch.float32,
                    offset=None,
                    state2=new_state2,
                )

                # 重建 Params4bit (CPU)
                p = Params4bit(
                    data_tensor,
                    requires_grad=False,
                    quant_state=qs,
                    quant_type="nf4",
                    compress_statistics=True,
                    bnb_quantized=True,
                )
                sd[f"{wname}.weight"] = p
                sd[f"{wname}_shape"] = list(w_meta["orig_shape"])

            sd["quant_bits"] = meta["quant_bits"]

            return sd
        except Exception as exc:
            logger.debug(
                f"[ExpertEvictor] mmap 读取 {key} 失败: {exc}, "
                f"回退 torch.load"
            )
            return None

    def record_access(self, key: ExpertKey) -> None:
        """记录专家访问 (供 VRAMPool.get_expert 调用)"""
        self._freq_tracker.record_access(key)

    # ----------------------------------------------------------------
    # 内部: 直写 D 盘
    # ----------------------------------------------------------------
    async def _write_expert_to_disk(
        self,
        key: ExpertKey,
        module: torch.nn.Module,
    ) -> None:
        """把专家模块直写 D 盘 (原子写)

        Args:
            key: (layer_idx, expert_idx)
            module: MixtralBLockSparseTop2MLP 模块
        """
        layer_idx, expert_idx = key
        shard_path = (
            self._shard_dir
            / f"layer_{layer_idx:02d}"
            / "experts"
            / f"expert_{expert_idx}.pt"
        )
        tmp_path = shard_path.with_suffix(".tmp")

        # 提取 state_dict: 保存完整 Params4bit 对象 (含 quant_state)
        # 关键: param.detach().cpu() 会丢失 quant_state (detach 返回普通 Tensor),
        # 导致 fetch_back 时无法反量化. 必须用 __new__ 手动创建 CPU 副本,
        # 保留 quant_state 属性. Params4bit.cpu() 也不可用 (会重新量化 → CUDA 错误).
        from bitsandbytes.nn import Params4bit

        sd: Dict[str, Any] = {}
        for name, param in module.named_parameters():
            if param is None:
                continue
            has_qs = (
                hasattr(param, "quant_state")
                and param.quant_state is not None
            )
            if has_qs:
                # Params4bit: 手动搬移 data + quant_state 到 CPU
                p_cpu = Params4bit.__new__(Params4bit)
                p_cpu.data = param.data.cpu()
                p_cpu.requires_grad = False
                p_cpu.quant_state = self._move_quant_state_to_cuda(
                    param.quant_state, torch.device("cpu")
                )
                sd[name] = p_cpu
            else:
                try:
                    sd[name] = param.detach().cpu()
                except Exception:
                    sd[name] = param.detach().clone().cpu()

        # 保存原始形状信息 (从 quant_state 获取, 而非 weight.shape)
        # Params4bit.weight.shape 是 1D 量化数据 [N, 1], 不是原始 [out, in].
        for wname in ["w1", "w2", "w3"]:
            if not hasattr(module, wname):
                continue
            w = getattr(module, wname).weight
            qs = getattr(w, "quant_state", None)
            if qs is not None and hasattr(qs, "shape"):
                sd[f"{wname}_shape"] = list(qs.shape)
            elif w.data.dim() == 2:
                sd[f"{wname}_shape"] = list(w.data.shape)
            else:
                sd[f"{wname}_shape"] = [0, 0]
        sd["quant_bits"] = self._quant_bits

        # 原子写: .tmp → rename
        await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: torch.save(sd, tmp_path),
        )
        tmp_path.replace(shard_path)

        logger.debug(
            f"[ExpertEvictor] 专家 {key} 直写 D 盘: {shard_path} "
            f"({shard_path.stat().st_size/1024**2:.0f}MB)"
        )

    # ----------------------------------------------------------------
    # 内部: 绑定专家权重
    # ----------------------------------------------------------------
    def _replace_linear_with_4bit(
        self,
        module: torch.nn.Module,
        state_dict: Dict[str, Any],
    ) -> None:
        """把专家模块的 w1/w2/w3 从 nn.Linear 替换为 Linear4bit

        注意: Params4bit 经 torch.save/load 后 pickle 退化为普通 Parameter,
        其 data 是 1D 量化数据 ([29360128, 1]), 不能用 src.shape 推断形状.
        必须从 state_dict 中的 "{name}_shape" 元数据获取原始 [out, in] 形状.

        Args:
            module: MixtralBlockSparseTop2MLP 模块
            state_dict: 权重字典 (含 w1_shape/w2_shape/w3_shape 元数据)
        """
        from bitsandbytes.nn import Linear4bit

        for name in ["w1", "w2", "w3"]:
            if not hasattr(module, name):
                continue
            old_linear = getattr(module, name)

            # 优先从 state_dict 的 shape 元数据获取原始形状
            # shape = [out_features, in_features] (PyTorch Linear 权重布局)
            shape_key = f"{name}_shape"
            if shape_key in state_dict and state_dict[shape_key]:
                shape: List[int] = state_dict[shape_key]
                out_features = int(shape[0])
                in_features = int(shape[1])
            else:
                in_features = old_linear.in_features
                out_features = old_linear.out_features

            new_linear = Linear4bit(
                input_features=in_features,
                output_features=out_features,
                bias=False,
                compute_dtype=torch.bfloat16,
                quant_type="nf4",
                compress_statistics=True,
            )
            setattr(module, name, new_linear)

    @staticmethod
    def _resolve_weight_key(
        state_dict: Dict[str, Any],
        name: str,
    ) -> Optional[str]:
        """兼容两种 key 格: 优先 "w1", 回退 "w1.weight".

        新版分片脚本保存 "w1" key (Params4bit 对象).
        旧版分片脚本保存 "w1.weight" key (普通 Parameter, 无 quant_state).

        Args:
            state_dict: 从 D 盘读取的权重字典
            name: 权重名 ("w1"/"w2"/"w3")

        Returns:
            实际存在的 key 或 None
        """
        if name in state_dict:
            return name
        weight_key = f"{name}.weight"
        if weight_key in state_dict:
            return weight_key
        return None

    def _bind_expert_weights(
        self,
        module: torch.nn.Module,
        state_dict: Dict[str, Any],
    ) -> int:
        """把 state_dict 权重绑定到空模块

        支持三种格式:
          1. Params4bit 对象 (4bit 量化, 含 quant_state) — 新版分片
          2. 普通 Tensor (FP16/bf16) — 旧版分片或回退

        Args:
            module: 空 MixtralBLockSparseTop2MLP
            state_dict: 从 D 盘读取的权重字典

        Returns:
            实际占用 VRAM 字节数
        """
        size_bytes: int = 0
        from bitsandbytes.nn import Params4bit

        for name in ["w1", "w2", "w3"]:
            # 兼容 "w1" 和 "w1.weight" 两种 key 格式
            weight_key = self._resolve_weight_key(state_dict, name)
            if weight_key is None:
                continue
            if not hasattr(module, name):
                continue

            linear = getattr(module, name)
            src = state_dict[weight_key]

            if self._quant_bits == 4:
                # Params4bit 经 torch.save/load 后 pickle 退化为普通 Parameter,
                # isinstance(src, Params4bit) 会返回 False, 但 quant_state 属性
                # 被保留. 用 hasattr 检测 quant_state 来判断是否为量化参数.
                has_quant_state = (
                    hasattr(src, "quant_state")
                    and src.quant_state is not None
                )
                if has_quant_state:
                    # 用构造函数重建 Params4bit (bnb_quantized=True 避免重新量化).
                    # quant_state 的张量 (absmax/code/state2) 必须搬到 CUDA,
                    # 否则 matmul_4bit 会触发 CUDA illegal memory access.
                    # 不能用 __new__ (缺少 bnb_quantized 等属性 → forward 异常).
                    qs_cuda = self._move_quant_state_to_cuda(
                        src.quant_state, self._pool._device
                    )
                    p = Params4bit(
                        src.data.to(self._pool._device),
                        requires_grad=False,
                        quant_state=qs_cuda,
                        quant_type="nf4",
                        compress_statistics=True,
                        bnb_quantized=True,
                    )
                    linear.weight = p
                    if p.data is not None:
                        size_bytes += (
                            p.data.element_size() * p.data.nelement()
                        )
                elif isinstance(src, torch.Tensor) and src.dim() == 2 and src.dtype != torch.uint8:
                    # 普通 2D bf16/fp16 张量: 在 CUDA 上重新量化
                    # (注意: uint8 packed 数据不能重新量化, 会得到错误 quant_state)
                    w_cuda = src.to(torch.bfloat16).to(self._pool._device)
                    linear.weight = Params4bit(
                        w_cuda,
                        requires_grad=False,
                        quant_type="nf4",
                        compress_statistics=True,
                    )
                    del w_cuda
                    if linear.weight.data is not None:
                        size_bytes += (
                            linear.weight.data.element_size()
                            * linear.weight.data.nelement()
                        )
                else:
                    logger.warning(
                        f"[ExpertEvictor] 未知权重格式: {type(src)} "
                        f"dim={src.dim() if hasattr(src,'dim') else 'N/A'} "
                        f"dtype={getattr(src, 'dtype', 'N/A')}"
                    )
                    continue
            elif self._quant_bits == 8:
                from bitsandbytes.nn import Int8Params
                if isinstance(src, Int8Params):
                    linear.weight = src.to(self._pool._device)
                else:
                    w_cuda = src.to(torch.bfloat16).to(self._pool._device)
                    linear.weight = Int8Params(
                        w_cuda,
                        requires_grad=False,
                        has_fp16_weights=False,
                    )
                    del w_cuda
                if linear.weight.data is not None:
                    size_bytes += linear.weight.data.element_size() * linear.weight.data.nelement()
            else:
                # FP16/bf16: 直接加载, 强制 bfloat16 与共享层一致
                linear.weight.data = src.to(
                    self._pool._device, dtype=torch.bfloat16
                )
                size_bytes += linear.weight.data.element_size() * linear.weight.data.nelement()

        return size_bytes

    # ----------------------------------------------------------------
    # 统计
    # ----------------------------------------------------------------
    def stats(self) -> Dict[str, Any]:
        """获取置换器统计"""
        return {
            "freq_snapshot_size": len(self._freq_tracker.snapshot()),
            "shard_dir": str(self._shard_dir),
            "factory_attached": self._expert_factory is not None,
        }
