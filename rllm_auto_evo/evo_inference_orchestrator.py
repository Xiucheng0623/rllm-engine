# -*- coding: utf-8 -*-
# File: D:\AI_RLLM\rllm_auto_evo\evo_inference_orchestrator.py
"""自进化推理编排器 — Hermes 自我进化的核心

这是连接 Hermes 自进化引擎和 v3 磁盘分页推理路径的桥梁.

工作流程:
  1. AutoTuner 给出配置 (量化位宽/VRAM容量/KV阈值/prefetch层数)
  2. 按配置初始化 v3 推理路径 (VRAMCachePool + ManualLayerRunner)
  3. 跑 N 条测试 prompt, 采集指标 (吞吐/延迟/IO/内存)
  4. 计算综合得分, 回写给 StrategyPool
  5. AutoTuner 基于结果建议下一配置
  6. 重复 1-5, 进化出最优策略

进化目标:
  - 7B: 找到 4-bit 全常驻 (22 tok/s) vs FP16 分页 (0.9 tok/s) 的最优解
  - 13B: 找到 4-bit hybrid vs FP16 分页 的最优解
  - 让引擎自己决定, 不靠人调参
"""
from __future__ import annotations

import asyncio
import gc
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from loguru import logger

ROOT = Path(r"D:\AI_RLLM")
os.environ["HF_HOME"] = str(ROOT / "hf_cache")
os.environ["TRANSFORMERS_CACHE"] = str(ROOT / "hf_cache" / "hub")
os.environ["HUGGINGFACE_HUB_CACHE"] = str(ROOT / "hf_cache" / "hub")
os.environ["TORCH_HOME"] = str(ROOT / "hf_cache" / "torch")
os.environ["HF_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["BITSANDBYTES_NOWELCOME"] = "1"

sys.path.insert(0, str(ROOT))
for sub in ("rllm_agent_core", "rllm_disk_engine", "rllm_auto_evo", "rllm_pipeline"):
    sys.path.insert(0, str(ROOT / sub))

from rllm_agent_core.skills.skill_loader import DiskOffloadSkillConfig
from rllm_auto_evo.metrics.metrics_collector import MetricsCollector, get_metrics_collector
from rllm_auto_evo.strategy.strategy_pool import StrategyPool, StrategyPerformance, get_strategy_pool
from rllm_auto_evo.tuner.auto_tuner import AutoTuner, get_auto_tuner

MODEL_DIR = ROOT / "rllm_model_shards" / "_raw" / "Nous-Hermes-2-Mistral-7B-DPO"


class EvoInferenceOrchestrator:
    """自进化推理编排器

    连接 AutoTuner (策略生成) 和 v3 推理路径 (执行+采集),
    形成"配置→推理→指标→评分→新配置"的进化闭环.

    Attributes:
        tuner: AutoTuner 实例 (策略建议)
        pool: StrategyPool 实例 (策略存档)
        collector: MetricsCollector 实例 (指标采集)
        max_rounds: 最大进化轮数
        test_prompts: 测试用 prompt 列表
    """

    def __init__(
        self,
        max_rounds: int = 10,
        test_prompts: Optional[List[str]] = None,
        force_fp16_rounds: Optional[List[int]] = None,
    ) -> None:
        """初始化自进化编排器

        Args:
            max_rounds: 最大进化轮数
            test_prompts: 测试用 prompt 列表 (默认 3 条中文)
            force_fp16_rounds: 强制跑 FP16 磁盘分页的轮次编号
                (确保 Hermes 一定会尝试 FP16 全量模式)
        """
        self.tuner: AutoTuner = get_auto_tuner()
        self.pool: StrategyPool = get_strategy_pool()
        self.collector: MetricsCollector = get_metrics_collector()
        self.max_rounds: int = max_rounds

        self.test_prompts: List[str] = test_prompts or [
            "请用中文介绍一下中国的四大发明。",
            "请解释一下什么是量子计算。",
            "用简短的中文描述一下春天的景色。",
        ]

        # 强制 FP16 试验轮次: 第 1 轮和第 max//2 轮
        self.force_fp16_rounds: List[int] = force_fp16_rounds or [1, max_rounds // 2]

        # 当前配置 (从策略池加载最优, 或用默认)
        best = self.pool.get_best()
        if best is not None:
            self.current_config: DiskOffloadSkillConfig = self.pool.get_config(best.sig)
            if self.current_config is None:
                self.current_config = DiskOffloadSkillConfig()
        else:
            self.current_config = DiskOffloadSkillConfig()

        logger.info(
            f"[EvoOrchestrator] 初始化: max_rounds={max_rounds} "
            f"current_config={asdict(self.current_config)} "
            f"pool_size={len(self.pool.list_all())} "
            f"force_fp16_rounds={self.force_fp16_rounds}"
        )

    async def run_evolution(self) -> Dict[str, Any]:
        """运行自进化循环

        每轮:
          1. 用当前配置跑推理, 采集指标
          2. 计算得分, 回写策略池
          3. AutoTuner 建议下一配置
          4. 重复

        Returns:
            进化结果摘要 (最优策略/各轮得分/进化轨迹)
        """
        logger.info("=" * 60)
        logger.info("Hermes 自进化推理引擎 — 启动")
        logger.info("=" * 60)

        trajectory: List[Dict[str, Any]] = []

        for round_id in range(1, self.max_rounds + 1):
            logger.info(f"\n{'='*40} 进化轮 {round_id}/{self.max_rounds} {'='*40}")

            # 强制 FP16 试验: 在指定轮次切换到 FP16 全量磁盘分页
            if round_id in self.force_fp16_rounds and self.current_config.quantization_bits != 16:
                fp16_config = DiskOffloadSkillConfig(**asdict(self.current_config))
                fp16_config.quantization_bits = 16
                # FP16 需要更小 VRAM pool + 更大 KV 阈值
                fp16_config.shard_size_mb = 256
                fp16_config.kv_spill_threshold_mb = 256
                fp16_config.prefetch_layers_ahead = 2
                fp16_config.prefetch_threads = 4
                logger.info(
                    f"[Evo] 强制 FP16 磁盘分页试验 (轮 {round_id}): "
                    f"sig={fp16_config.signature()}"
                )
                self.current_config = fp16_config

            # 1. 执行推理试验
            sig = self.current_config.signature()
            self.collector.set_strategy(sig)
            self.collector.set_round(round_id)

            try:
                metrics = await self._run_inference_trial(
                    config=self.current_config,
                    round_id=round_id,
                )
            except Exception as exc:
                logger.error(f"轮 {round_id} 推理失败: {exc}")
                metrics = {"failure_flag": 1.0, "throughput_tps": 0.0}

            # 2. 计算得分
            score = self._compute_score(metrics)
            logger.info(
                f"[Evo] 轮 {round_id} 得分: {score:.2f} | "
                f"tps={metrics.get('throughput_tps', 0):.1f} "
                f"mem={metrics.get('peak_memory_mb', 0):.0f}MB "
                f"io={metrics.get('io_block_ms', 0):.0f}ms"
            )

            # 3. 回写策略池
            self.tuner.record_score(
                sig=sig,
                score=score,
                avg_latency_ms=metrics.get("avg_latency_ms", 0),
                avg_tps=metrics.get("throughput_tps", 0),
                avg_mem_mb=metrics.get("peak_memory_mb", 0),
                fail_rate=metrics.get("failure_flag", 0),
            )

            trajectory.append({
                "round": round_id,
                "sig": sig,
                "config": asdict(self.current_config),
                "score": score,
                "tps": metrics.get("throughput_tps", 0),
                "mem_mb": metrics.get("peak_memory_mb", 0),
            })

            # 4. 清理显存
            gc.collect()
            torch.cuda.empty_cache()
            await asyncio.sleep(1)

            # 5. AutoTuner 建议下一配置
            if round_id < self.max_rounds:
                trigger = self._detect_trigger(metrics)
                next_config = self.tuner.suggest_next_config(
                    current_cfg=self.current_config,
                    metrics_dict=metrics,
                    trigger=trigger,
                )
                logger.info(
                    f"[Evo] 下一配置: quant={next_config.quantization_bits}bit "
                    f"vram_pool={next_config.shard_size_mb}MB "
                    f"kv_thresh={next_config.kv_spill_threshold_mb}MB "
                    f"prefetch={next_config.prefetch_layers_ahead}层 "
                    f"sig={next_config.signature()}"
                )
                self.current_config = next_config

        # 输出最优策略
        best = self.pool.get_best()
        result = {
            "rounds": self.max_rounds,
            "trajectory": trajectory,
            "best_sig": best.sig if best else None,
            "best_score": best.performance.score if best else 0,
            "best_tps": best.performance.avg_throughput_tps if best else 0,
            "best_config": asdict(self.pool.get_config(best.sig)) if best else None,
        }

        logger.info("\n" + "=" * 60)
        logger.info("Hermes 自进化完成")
        logger.info("=" * 60)
        logger.info(f"最优策略: sig={result['best_sig']}")
        logger.info(f"最优得分: {result['best_score']:.2f}")
        logger.info(f"最优速度: {result['best_tps']:.1f} tok/s")
        if result["best_config"]:
            logger.info(f"最优配置: {result['best_config']}")

        # 打印进化轨迹
        logger.info("\n进化轨迹:")
        for t in trajectory:
            quant = t['config']['quantization_bits']
            quant_label = "FP16" if quant == 16 else f"{quant}bit"
            logger.info(
                f"  轮{t['round']}: score={t['score']:.1f} "
                f"tps={t['tps']:.1f} "
                f"quant={quant_label} "
                f"kv={t['config']['kv_spill_threshold_mb']}MB "
                f"prefetch={t['config']['prefetch_layers_ahead']}层"
            )

        return result

    async def _run_inference_trial(
        self,
        config: DiskOffloadSkillConfig,
        round_id: int,
    ) -> Dict[str, float]:
        """用给定配置跑一次推理试验

        Args:
            config: 策略配置
            round_id: 进化轮次

        Returns:
            指标字典
        """
        from rllm_disk_engine.quantized_model_loader import QuantizedModelLoader
        from rllm_disk_engine.vram_pool.vram_cache_pool import VRAMCachePool
        from rllm_disk_engine.vram_pool.manual_layer_runner import ManualLayerRunner

        # 清理上一轮
        gc.collect()
        torch.cuda.empty_cache()

        free_before, _ = torch.cuda.mem_get_info()
        logger.info(
            f"[Evo-Trial] 轮{round_id} 配置: "
            f"quant={config.quantization_bits}bit "
            f"shard={config.shard_size_mb}MB "
            f"kv_thresh={config.kv_spill_threshold_mb}MB "
            f"prefetch={config.prefetch_layers_ahead}层 "
            f"VRAM_free={free_before/1024**3:.1f}GB"
        )

        # 预定义 loader, 避免 FP16/8bit 路径清理阶段 del loader 报 NameError
        loader: Optional[Any] = None

        # 根据量化位宽选择加载方式
        if config.quantization_bits == 4:
            # 4-bit: 用 from_pretrained 加载
            loader = QuantizedModelLoader(
                model_dir=MODEL_DIR,
                quant_bits=4,
            )
            model, tokenizer, model_config = loader.load_model()
            components = loader.extract_layers()
            decoder_layers = components["decoder_layers"]
            embed_tokens = components["embed_tokens"]
            norm_module = components["norm"]
            lm_head = components["lm_head"]
            num_layers = len(decoder_layers)
            quant_bits = 4
        else:
            # FP16/8-bit: 用 ZeroCopyShardLoader
            from transformers.models.mistral.configuration_mistral import MistralConfig
            from transformers.models.mistral.modeling_mistral import (
                MistralDecoderLayer,
                MistralRMSNorm,
            )
            from rllm_disk_engine.zero_copy_loader.zero_copy_shard_loader import ZeroCopyShardLoader
            from transformers import AutoTokenizer

            model_config = MistralConfig.from_pretrained(str(MODEL_DIR / "config.json"))
            num_layers = model_config.num_hidden_layers

            index_path = MODEL_DIR / "model.safetensors.index.json"
            index_data = json.loads(index_path.read_text(encoding="utf-8"))
            weight_map = index_data.get("weight_map", {})

            shard_loader = ZeroCopyShardLoader()
            shard_loader.initialize(
                raw_model_dir=MODEL_DIR,
                weight_map=weight_map,
                num_layers=num_layers,
            )
            tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR), local_files_only=True)
            if tokenizer.eos_token_id is None:
                tokenizer.eos_token_id = 2

            # 提取辅助权重
            aux_weights = shard_loader.get_aux_tensors([
                "model.embed_tokens.weight",
                "model.norm.weight",
                "lm_head.weight",
            ])
            # 统一转 float16, 避免原始 bfloat16/float32 与 decoder layer .half() 冲突
            # 报错: expected mat1 and mat2 to have the same dtype, but got: float != Half
            for k in list(aux_weights.keys()):
                aux_weights[k] = aux_weights[k].to(torch.float16)

            embed_tokens = torch.nn.Embedding(
                model_config.vocab_size, model_config.hidden_size, dtype=torch.float16
            )
            embed_tokens.weight.data = aux_weights["model.embed_tokens.weight"].clone()
            embed_tokens = embed_tokens.to("cuda")

            norm_module = MistralRMSNorm(
                model_config.hidden_size, eps=model_config.rms_norm_eps
            ).half()
            norm_module.weight.data = aux_weights["model.norm.weight"].clone()
            norm_module = norm_module.to("cuda")

            lm_head = torch.nn.Linear(
                model_config.hidden_size, model_config.vocab_size, bias=False
            ).half()
            lm_head.weight.data = aux_weights["lm_head.weight"].clone()
            lm_head = lm_head.to("cuda")

            decoder_layers = None  # 用 prefill_load_all
            quant_bits = 16

        # 初始化 VRAMCachePool
        VRAMCachePool._singleton = None
        # reserve 根据量化位宽调整
        if quant_bits == 4:
            reserve_gb = 2.0  # 4-bit 3.3GB, 留 2GB 给 KV
        else:
            reserve_gb = 4.0  # FP16 13.3GB, 留 4GB 给草稿+KV
        vram_pool = VRAMCachePool(reserve_gb=reserve_gb)

        # 设置 usable 基于 shard_size_mb 配置
        usable_bytes = config.shard_size_mb * 1024 * 1024 * 4  # shard_size × 4 层
        vram_pool._usable_bytes = min(usable_bytes, vram_pool._capacity_bytes - reserve_gb * 1024**3)
        vram_pool._evict_threshold = int(vram_pool._usable_bytes * 0.8)

        # HotColdEvictor
        # FP16/8bit 层大 (416MB), CPU RAM 12GB 装不下 32 层, 启用直写 D 盘模式
        # 4bit 层小 (~100MB), 用 CPU RAM 中转 (fetch_back 快 ~26ms)
        from rllm_disk_engine.vram_pool.hot_cold_evictor import HotColdEvictor
        evict_dir = ROOT / "rllm_offload_temp" / f"evo_round_{round_id}"
        direct_to_disk = quant_bits in (16, 8)
        evictor = HotColdEvictor(
            vram_pool=vram_pool,
            evict_dir=evict_dir,
            num_layers=num_layers,
            direct_to_disk=direct_to_disk,
        )
        if quant_bits in (16, 8):
            # FP16 和 8bit 都用 .half() 加载 (8bit 暂不真量化, 避免 bitsandbytes 崩溃)
            from transformers.models.mistral.modeling_mistral import MistralDecoderLayer
            evictor.attach_factory(
                lambda idx: MistralDecoderLayer(model_config, layer_idx=idx).half(),
                quant_bits=quant_bits,
            )
        vram_pool.attach_evictor(evictor)

        # 加载层
        if quant_bits == 4:
            loaded = await vram_pool.load_from_quantized_model(
                decoder_layers=decoder_layers,
                quant_bits=4,
            )
        else:
            from transformers.models.mistral.modeling_mistral import MistralDecoderLayer
            loaded, _ = await vram_pool.prefill_load_all(
                layer_loader=shard_loader,
                layer_module_factory=lambda idx: MistralDecoderLayer(model_config, layer_idx=idx).half(),
                num_layers=num_layers,
                quant_bits=quant_bits,
            )
            await vram_pool.force_evict_to_limit()

        # ManualLayerRunner
        kv_dir = ROOT / "rllm_offload_temp" / f"evo_kv_{round_id}"
        runner = ManualLayerRunner(
            config=model_config,
            embed_tokens=embed_tokens,
            norm=norm_module,
            lm_head=lm_head,
            vram_pool=vram_pool,
            kv_spill_threshold_mb=config.kv_spill_threshold_mb,
            spill_dir=kv_dir,
        )

        # 跑测试 prompt
        # FP16 磁盘分页每层需从 D 盘读回 (~5s/层), 32 token×32 层会跑 85 分钟
        # FP16 只需验证无损质量+测速度, 用 1 条 prompt × 4 token 足够
        if quant_bits == 16:
            active_prompts: List[str] = self.test_prompts[:1]
            max_new: int = 4
        else:
            active_prompts = self.test_prompts
            max_new = 32

        tps_list: List[float] = []
        latency_list: List[float] = []
        io_block_total: float = 0.0
        failures: int = 0

        for pi, prompt in enumerate(active_prompts):
            try:
                # 重置 KV cache
                runner._kv_cache.clear()
                runner._spill_count = 0
                runner._readback_count = 0

                tokens = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")
                t0 = time.time()

                # Prefill
                first_token, prefill_time = await runner.prefill(
                    input_ids=tokens,
                    temperature=0.7,
                    top_p=0.9,
                )

                # Decode
                generated_ids = [first_token]
                t_decode = time.time()
                for step in range(max_new):
                    next_id, _ = await runner.decode_step(
                        last_token=generated_ids[-1],
                        history_tokens=generated_ids,
                        temperature=0.7,
                        top_p=0.9,
                    )
                    generated_ids.append(next_id)
                    if next_id == tokenizer.eos_token_id:
                        break
                decode_time = time.time() - t_decode
                decode_count = len(generated_ids) - 1
                tps = decode_count / max(decode_time, 0.001)

                tps_list.append(tps)
                latency_list.append(prefill_time * 1000)  # ms

                # 采集 IO 指标
                pool_stats = vram_pool.stats()
                io_block_total += pool_stats.get("evict_count", 0) * 90  # 估算 90ms/evict
                runner_stats = runner.stats()

                # 记录指标
                self.collector.record_batch(
                    {
                        "throughput_tps": tps,
                        "layer_read_ms": pool_stats.get("fetch_back_count", 0) * 90,
                        "peak_memory_mb": (torch.cuda.memory_allocated() + torch.cuda.memory_reserved()) / 1024**2,
                        "io_block_ms": io_block_total,
                        "kv_spill_count": runner_stats.get("kv_spill_count", 0),
                        "ssd_cache_hit_ratio": 1.0 if pool_stats.get("evict_count", 0) == 0 else 0.5,
                    },
                    task_id=f"round_{round_id}_prompt_{pi}",
                )

                logger.info(
                    f"[Evo-Trial] 轮{round_id} prompt{pi}: "
                    f"{decode_count}tok {tps:.1f}tok/s "
                    f"prefill={prefill_time:.2f}s "
                    f"evict={pool_stats.get('evict_count', 0)} "
                    f"fetch={pool_stats.get('fetch_back_count', 0)}"
                )

            except Exception as exc:
                logger.exception(f"[Evo-Trial] prompt{pi} 失败: {exc}")
                failures += 1

        # 汇总指标
        avg_tps = sum(tps_list) / max(len(tps_list), 1)
        avg_latency = sum(latency_list) / max(len(latency_list), 1)
        peak_mem = (torch.cuda.max_memory_allocated() + torch.cuda.max_memory_reserved()) / 1024**2
        fail_rate = failures / max(len(active_prompts), 1)

        metrics = {
            "throughput_tps": avg_tps,
            "avg_latency_ms": avg_latency,
            "peak_memory_mb": peak_mem,
            "io_block_ms": io_block_total / max(len(self.test_prompts), 1),
            "failure_flag": fail_rate,
        }

        # 清理 (loader 在 FP16/8bit 路径为 None, 需条件判断)
        del runner, vram_pool, evictor
        if loader is not None:
            loader.cleanup_model_shell()
            del loader
        del tokenizer, model_config
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        return metrics

    def _compute_score(self, metrics: Dict[str, float]) -> float:
        """计算策略综合得分

        评分公式:
          score = (吞吐 × 100 + 质量奖励) - 延迟惩罚 - 内存惩罚 - IO惩罚 - 失败惩罚

        质量奖励:
          - FP16 (16bit): +200 (全量无损, 创新模式)
          - INT8 (8bit):  +100 (近无损)
          - 4-bit  (4bit): +0   (有损, 速度快)

        这样 Hermes 会在"快但有损的 4-bit"和"慢但无损的 FP16"之间权衡,
        而不是单纯追求速度.

        Args:
            metrics: 指标字典

        Returns:
            综合得分 (越高越好)
        """
        tps = metrics.get("throughput_tps", 0)
        latency = metrics.get("avg_latency_ms", 0)
        mem = metrics.get("peak_memory_mb", 0)
        fail = metrics.get("failure_flag", 0)
        io = metrics.get("io_block_ms", 0)

        # 吞吐权重最高
        score = tps * 100

        # 质量奖励: FP16 无损 > INT8 近无损 > 4-bit 有损
        quant_bits = self.current_config.quantization_bits
        if quant_bits == 16:
            score += 200  # FP16 全量无损 + 创新模式奖励
        elif quant_bits == 8:
            score += 100  # INT8 近无损
        # 4-bit: 0 (靠速度取胜)

        # 延迟惩罚
        score -= latency / 100
        # 内存惩罚 (超过 6GB 扣分)
        if mem > 6144:
            score -= (mem - 6144) / 100
        # IO 惩罚
        score -= io / 1000
        # 失败惩罚 (严重)
        score -= fail * 1000

        return round(score, 2)

    def _detect_trigger(self, metrics: Dict[str, float]) -> str:
        """检测触发条件

        Args:
            metrics: 指标字典

        Returns:
            触发原因字符串
        """
        if metrics.get("failure_flag", 0) > 0.005:
            return "failure"
        if metrics.get("peak_memory_mb", 0) > 6144:
            return "memory"
        if metrics.get("io_block_ms", 0) > 30000:
            return "io_block"
        # 检查延迟是否比历史上涨 20%
        vals = self.collector.tail_values("throughput_tps", n=5)
        if len(vals) >= 2:
            prev_avg = sum(vals[:-1]) / len(vals[:-1])
            curr = vals[-1]
            if prev_avg > 0 and curr < prev_avg * 0.8:
                return "latency_drop"
        return "heuristic"


async def main() -> None:
    """主入口: 运行自进化

    12 轮进化:
      - 轮 1, 6: 强制 FP16 磁盘分页 (全量无损)
      - 其余轮次: AutoTuner 自动搜索 4-bit/8-bit/16-bit 参数空间
    """
    orchestrator = EvoInferenceOrchestrator(
        max_rounds=12,
        force_fp16_rounds=[1, 6],
    )
    result = await orchestrator.run_evolution()

    # 保存结果
    result_path = ROOT / "logs" / "evo_result.json"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    logger.info(f"进化结果已保存: {result_path}")


if __name__ == "__main__":
    asyncio.run(main())
