# -*- coding: utf-8 -*-
# File: D:\AI_RLLM\rllm_disk_engine\speculative_decoder.py
"""Speculative Decoding 推理引擎

核心创新:
  用 4-bit 量化小模型(草稿模型)快速生成 N 个候选 token,
  再用 FP16 全量大模型(验证模型)一次批量验证, IO 开销分摊到 N 个 token.

  草稿模型: 4-bit NF4, ~3.3GB, 全装 VRAM, 速度 22 tok/s
  验证模型: FP16 全量, ~13.3GB, CPU RAM 缓存+逐层 fetch, IO 瓶颈 0.9 tok/s
  Speculative: 草稿 N=8, 验证 1 次 IO / 8 token → 理论 5-8 tok/s

工作流程:
  1. FP16 模型 prefill prompt → 首个 token + KV cache
  2. 4-bit 草稿模型生成 N 个候选 token
  3. FP16 模型批量 verify (forward N 个 token, 复用 KV cache)
  4. 比较: 接受匹配的 token, 拒绝处用 FP16 token 替换
  5. 重复 2-4 直到达到 max_tokens

物理分析:
  - 每层 IO: 416MB / 10GB/s ≈ 42ms (PCIe 4.0)
  - 普通 decode: 32层 × 42ms = 1344ms/token → 0.74 tok/s
  - Speculative N=8: 1344ms / 8 = 168ms/token → 6.0 tok/s (理论)
  - 加草稿时间: (1344 + 364) / 8 = 214ms → 4.7 tok/s
  - 80% 接受率: 214 / 0.8 = 267ms → 3.7 tok/s
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

import torch
from loguru import logger


class SpeculativeDecoder:
    """Speculative Decoding 推理引擎

    用 4-bit 草稿模型 + FP16 验证模型实现高速全量推理.

    Attributes:
        draft_model: 4-bit 量化模型 (全装 VRAM, 快速草稿生成)
        draft_tokenizer: 草稿模型的 tokenizer
        verify_runner: FP16 全量模型的 ManualLayerRunner (磁盘分页)
        draft_size: 每轮草稿生成的 token 数 (默认 8)
        accept_count: 累计接受的 token 数
        reject_count: 累计拒绝的 token 数
    """

    def __init__(
        self,
        draft_model: Any,
        draft_tokenizer: Any,
        verify_runner: Any,
        draft_size: int = 8,
    ) -> None:
        """初始化 Speculative Decoder

        Args:
            draft_model: 4-bit 量化模型 (已加载到 VRAM)
            draft_tokenizer: 草稿模型的 tokenizer
            verify_runner: FP16 全量模型的 ManualLayerRunner
            draft_size: 每轮草稿生成的 token 数 (4-16, 默认 8)
        """
        self.draft_model = draft_model
        self.draft_tokenizer = draft_tokenizer
        self.verify_runner = verify_runner
        self.draft_size: int = draft_size

        # 草稿模型 KV cache (避免每次重新 prefill)
        self._draft_past_kv: Any = None
        self._draft_seq_len: int = 0

        # 统计
        self.accept_count: int = 0
        self.reject_count: int = 0
        self.total_rounds: int = 0
        self.total_verify_time: float = 0.0
        self.total_draft_time: float = 0.0

        logger.info(
            f"[SpecDec] 初始化: draft_size={draft_size} "
            f"draft_model={type(draft_model).__name__} "
            f"verify_runner={type(verify_runner).__name__}"
        )

    @torch.no_grad()
    def draft_generate(
        self,
        input_ids: torch.Tensor,
        n: int,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> List[int]:
        """用 4-bit 草稿模型生成 n 个候选 token (KV cache 加速)

        使用手动 forward + KV cache, 避免每次重新 prefill 整个序列.
        首次调用时 prefill, 之后只 forward 单 token.

        Args:
            input_ids: 已生成的 token 序列 [1, seq_len] (在 CUDA)
            n: 要生成的 token 数
            temperature: 采样温度
            top_p: nucleus 采样阈值

        Returns:
            n 个候选 token id 列表
        """
        draft_tokens: List[int] = []

        # 如果 KV cache 为空或序列不匹配, 重新 prefill
        current_seq_len = input_ids.shape[1]
        if self._draft_past_kv is None or self._draft_seq_len != current_seq_len:
            # Prefill 整个序列
            outputs = self.draft_model(
                input_ids=input_ids,
                use_cache=True,
            )
            self._draft_past_kv = outputs.past_key_values
            self._draft_seq_len = current_seq_len
            # 取最后一个 token 的 logits
            logits = outputs.logits[:, -1, :]
        else:
            # 只 forward 最后一个 token (KV cache 已有前面的)
            last_token = input_ids[:, -1:]
            outputs = self.draft_model(
                input_ids=last_token,
                past_key_values=self._draft_past_kv,
                use_cache=True,
            )
            self._draft_past_kv = outputs.past_key_values
            self._draft_seq_len += 1
            logits = outputs.logits[:, -1, :]

        # 生成 n 个 token
        for _ in range(n):
            # 采样
            if temperature > 0:
                probs = torch.softmax(logits / temperature, dim=-1)
                sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                cumulative = torch.cumsum(sorted_probs, dim=-1)
                cutoff = torch.searchsorted(cumulative[0], top_p)
                sorted_indices = sorted_indices[:, :cutoff + 1]
                sorted_probs = sorted_probs[:, :cutoff + 1]
                sorted_probs /= sorted_probs.sum()
                idx = torch.multinomial(sorted_probs[0], num_samples=1)
                next_token = sorted_indices[0, idx].item()
            else:
                next_token = logits[0].argmax().item()

            draft_tokens.append(next_token)
            self._draft_seq_len += 1

            # Forward 下一个 token
            next_input = torch.tensor([[next_token]], dtype=torch.long, device=input_ids.device)
            outputs = self.draft_model(
                input_ids=next_input,
                past_key_values=self._draft_past_kv,
                use_cache=True,
            )
            self._draft_past_kv = outputs.past_key_values
            logits = outputs.logits[:, -1, :]

        return draft_tokens

    def reset_draft_cache(self) -> None:
        """重置草稿模型的 KV cache (每条 prompt 前调用)"""
        self._draft_past_kv = None
        self._draft_seq_len = 0

    async def verify_drafts(
        self,
        draft_tokens: List[int],
        last_token: int,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> Tuple[List[int], int]:
        """用 FP16 验证模型批量验证草稿 token

        把 last_token + draft_tokens 一起 forward, 复用已有 KV cache.
        每层只需一次 IO, IO 开销分摊到 N 个 token.

        Args:
            draft_tokens: 草稿模型生成的候选 token 列表
            last_token: 上一个已确认的 token (作为 verify 的输入起点)
            temperature: 采样温度
            top_p: nucleus 采样阈值

        Returns:
            (验证 token 列表, 接受数量)
            验证 token 列表长度 = len(draft_tokens) + 1
            (最后一个 token 是 bonus token, 如果全部接受则额外赠送)
        """
        import torch
        from transformers.cache_utils import DynamicCache

        device = self.verify_runner._embed.weight.device

        # 构造验证输入: [last_token, draft_token_0, draft_token_1, ...]
        verify_input = [last_token] + draft_tokens
        input_ids = torch.tensor(
            [verify_input], dtype=torch.long, device=device
        )
        seq_len = len(verify_input)

        # embed
        hidden_states = self.verify_runner._embed(input_ids)

        # 构造 cache_position: 从已有 KV 长度开始
        with self.verify_runner._lock:
            first_entry = next(
                iter(self.verify_runner._kv_cache.values()), None
            )
            past_seq_len = first_entry.seq_len if first_entry else 0

        cache_position = torch.arange(
            past_seq_len, past_seq_len + seq_len, device=device
        )
        position_ids = cache_position.unsqueeze(0)

        # 恢复 KV cache
        shared_cache = DynamicCache()
        with self.verify_runner._lock:
            for layer_idx in range(self.verify_runner._num_layers):
                entry = self.verify_runner._kv_cache.get(layer_idx)
                if entry is not None and entry.in_vram and entry.key is not None:
                    shared_cache.key_cache.append(entry.key)
                    shared_cache.value_cache.append(entry.value)
                else:
                    shared_cache.key_cache.append(None)
                    shared_cache.value_cache.append(None)

        # 逐层 forward (批量, 复用 KV cache)
        for layer_idx in range(self.verify_runner._num_layers):
            layer_module = await self.verify_runner._vram_pool.get_layer(
                layer_idx
            )
            if layer_module is None:
                raise RuntimeError(
                    f"[SpecDec] 验证时无法获取层 {layer_idx}"
                )

            with torch.no_grad():
                outputs = layer_module(
                    hidden_states,
                    attention_mask=None,
                    position_ids=position_ids,
                    past_key_value=shared_cache,
                    output_attentions=False,
                    use_cache=True,
                    cache_position=cache_position,
                )
            hidden_states = outputs[0]

        # norm → lm_head
        hidden_states = self.verify_runner._norm(hidden_states)
        logits = self.verify_runner._lm_head(hidden_states)  # [1, seq_len, vocab]

        # 取每个位置的 next token (argmax 或采样)
        verify_tokens: List[int] = []
        for i in range(seq_len):
            pos_logits = logits[0, i, :]
            if temperature > 0:
                probs = torch.softmax(pos_logits / temperature, dim=-1)
                sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                cumulative = torch.cumsum(sorted_probs, dim=-1)
                cutoff = torch.searchsorted(cumulative, top_p)
                sorted_indices = sorted_indices[:cutoff + 1]
                sorted_probs = sorted_probs[:cutoff + 1]
                sorted_probs /= sorted_probs.sum()
                idx = torch.multinomial(sorted_probs, num_samples=1)
                next_tok = sorted_indices[idx].item()
            else:
                next_tok = pos_logits.argmax(dim=-1).item()
            verify_tokens.append(next_tok)

        # 更新 KV cache (shared_cache 已扩展)
        new_seq_len = past_seq_len + seq_len
        with self.verify_runner._lock:
            for layer_idx in range(self.verify_runner._num_layers):
                try:
                    new_key = shared_cache.key_cache[layer_idx]
                    new_value = shared_cache.value_cache[layer_idx]
                    entry = self.verify_runner._kv_cache.get(layer_idx)
                    if entry is not None:
                        entry.key = new_key
                        entry.value = new_value
                        entry.in_vram = True
                        entry.seq_len = new_seq_len
                except IndexError as exc:
                    logger.warning(
                        f"[SpecDec] 层 {layer_idx} KV 更新失败: {exc}"
                    )

        # 比较: draft_tokens[i] 对应 verify_tokens[i] (因为输入是 last_token + draft_tokens)
        # verify_tokens[0] = FP16 对 last_token 的预测 (应该等于 draft_tokens[0])
        # verify_tokens[1] = FP16 对 draft_tokens[0] 的预测 (应该等于 draft_tokens[1])
        # ...
        # verify_tokens[-1] = bonus token (如果全部接受)
        accepted: int = 0
        result_tokens: List[int] = []

        for i in range(len(draft_tokens)):
            if verify_tokens[i] == draft_tokens[i]:
                accepted += 1
                result_tokens.append(draft_tokens[i])
            else:
                # 拒绝, 用 FP16 的 token 替换
                result_tokens.append(verify_tokens[i])
                break

        # 如果全部接受, 加上 bonus token
        if accepted == len(draft_tokens):
            result_tokens.append(verify_tokens[-1])

        return result_tokens, accepted

    async def generate(
        self,
        prompt: str,
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> Tuple[str, Dict[str, Any]]:
        """Speculative Decoding 主生成循环

        Args:
            prompt: 输入文本
            max_new_tokens: 最大生成 token 数
            temperature: 采样温度
            top_p: nucleus 采样阈值

        Returns:
            (生成文本, 统计信息字典)
        """
        import asyncio

        t_total = time.time()

        # 1. Tokenize prompt
        tokens = self.draft_tokenizer(
            prompt, return_tensors="pt"
        ).input_ids.to("cuda")
        input_length = tokens.shape[1]

        # 2. FP16 验证模型 prefill prompt
        logger.info(
            f"[SpecDec] Prefill prompt: {input_length} tokens"
        )
        t_prefill = time.time()
        first_token, _ = await self.verify_runner.prefill(
            input_ids=tokens,
            temperature=temperature,
            top_p=top_p,
        )
        prefill_time = time.time() - t_prefill
        logger.info(
            f"[SpecDec] Prefill 完成: {prefill_time:.2f}s, "
            f"首 token={first_token}"
        )

        generated_ids: List[int] = [first_token]

        # 3. Speculative 循环
        while len(generated_ids) < max_new_tokens:
            self.total_rounds += 1

            # 3a. 草稿模型生成 N 个候选 token
            t_draft = time.time()
            draft_input = torch.tensor(
                [tokens[0].tolist() + generated_ids],
                dtype=torch.long,
                device="cuda",
            )
            draft_tokens = self.draft_generate(
                input_ids=draft_input,
                n=self.draft_size,
                temperature=temperature,
                top_p=top_p,
            )
            draft_time = time.time() - t_draft
            self.total_draft_time += draft_time

            # 3b. FP16 验证模型批量验证
            t_verify = time.time()
            result_tokens, accepted = await self.verify_drafts(
                draft_tokens=draft_tokens,
                last_token=generated_ids[-1],
                temperature=temperature,
                top_p=top_p,
            )
            verify_time = time.time() - t_verify
            self.total_verify_time += verify_time

            self.accept_count += accepted
            self.reject_count += len(draft_tokens) - accepted

            # 3c. 如果有拒绝, 重置草稿 KV cache (避免序列不匹配导致重新 prefill)
            if accepted < len(draft_tokens):
                # 拒绝时回退草稿 cache: 只保留已接受的 token 对应的 KV
                # 最简单的方式: 重置 cache, 下次 prefill 时重建
                # 但只回退到已接受的位置 (通过截断 seq_len)
                if accepted > 0:
                    # 部分接受: 截断 KV cache 到接受位置
                    # _draft_seq_len 已包含所有草稿 token, 需回退
                    rejected_count = len(draft_tokens) - accepted
                    self._draft_seq_len -= rejected_count
                    # 注意: past_key_values 的实际 KV 没有截断,
                    # 但下次 draft_generate 会检测 seq_len 不匹配并重新 prefill
                    self._draft_past_kv = None  # 强制重新 prefill
                    self._draft_seq_len = 0
                else:
                    self._draft_past_kv = None
                    self._draft_seq_len = 0

            # 3d. 添加接受的 token + 可能的 bonus token
            generated_ids.extend(result_tokens)

            # 3d. 检查 EOS
            if self.draft_tokenizer.eos_token_id in result_tokens:
                # 截断到 EOS
                eos_idx = result_tokens.index(
                    self.draft_tokenizer.eos_token_id
                )
                generated_ids = generated_ids[:-(len(result_tokens) - eos_idx - 1)]
                break

            if self.total_rounds % 5 == 0:
                accept_rate = self.accept_count / max(
                    self.accept_count + self.reject_count, 1
                )
                logger.info(
                    f"[SpecDec] Round {self.total_rounds}: "
                    f"accepted={accepted}/{len(draft_tokens)} "
                    f"verify={verify_time:.2f}s "
                    f"draft={draft_time:.2f}s "
                    f"total_toks={len(generated_ids)} "
                    f"accept_rate={accept_rate:.0%}"
                )

        total_time = time.time() - t_total
        generated_text = self.draft_tokenizer.decode(
            generated_ids, skip_special_tokens=True
        )

        # 统计
        accept_rate = self.accept_count / max(
            self.accept_count + self.reject_count, 1
        )
        tps = len(generated_ids) / max(total_time, 0.001)

        stats = {
            "total_tokens": len(generated_ids),
            "total_time": total_time,
            "prefill_time": prefill_time,
            "tps": tps,
            "rounds": self.total_rounds,
            "accept_count": self.accept_count,
            "reject_count": self.reject_count,
            "accept_rate": accept_rate,
            "avg_verify_time": self.total_verify_time / max(self.total_rounds, 1),
            "avg_draft_time": self.total_draft_time / max(self.total_rounds, 1),
        }

        logger.info(
            f"[SpecDec] 生成完成: {len(generated_ids)} tok / {total_time:.1f}s "
            f"= {tps:.1f} tok/s | accept_rate={accept_rate:.0%}"
        )

        return generated_text, stats
