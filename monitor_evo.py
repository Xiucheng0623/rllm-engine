# -*- coding: utf-8 -*-
# File: D:\AI_RLLM\monitor_evo.py
"""Hermes 自进化实时监控面板

实时读取进化日志, 显示:
  - 当前进化轮次/进度
  - 各轮得分/速度/配置
  - FP16 vs 4-bit vs 8-bit 对比
  - VRAM / CPU RAM / D 盘三层存储状态
  - 当前阶段 (Prefill / Decode / 评分 / 策略入库)

两种用法:
  1) 实时循环刷新 (默认, 给用户在终端看):
        python D:\\AI_RLLM\\monitor_evo.py
  2) 单次快照 (一次性打印, 适合被其他程序调用):
        python D:\\AI_RLLM\\monitor_evo.py --once --no-color
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import time
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

LOG_FILE: Path = Path(r"D:\AI_RLLM\logs\evo_run_v5.log")
RESULT_FILE: Path = Path(r"D:\AI_RLLM\logs\evo_result.json")
STRATEGY_FILE: Path = Path(r"D:\AI_RLLM\skill_storage\strategy_pool.json")
EVO_RESULT_FILE: Path = Path(r"D:\AI_RLLM\logs\evo_result.json")


class ColorCodes:
    """ANSI 颜色码集合

    Attributes:
        enabled: 是否启用颜色 (False 时所有属性返回空字符串)
    """

    def __init__(self, enabled: bool = True) -> None:
        """初始化颜色开关

        Args:
            enabled: True 启用 ANSI 颜色, False 全部返回空串
        """
        self.enabled: bool = enabled

    def _g(self, code: str) -> str:
        return code if self.enabled else ""

    @property
    def RESET(self) -> str:
        return self._g("\033[0m")

    @property
    def BOLD(self) -> str:
        return self._g("\033[1m")

    @property
    def DIM(self) -> str:
        return self._g("\033[2m")

    @property
    def RED(self) -> str:
        return self._g("\033[91m")

    @property
    def GREEN(self) -> str:
        return self._g("\033[92m")

    @property
    def YELLOW(self) -> str:
        return self._g("\033[93m")

    @property
    def BLUE(self) -> str:
        return self._g("\033[94m")

    @property
    def MAGENTA(self) -> str:
        return self._g("\033[95m")

    @property
    def CYAN(self) -> str:
        return self._g("\033[96m")


# 剥离 ANSI 颜色码的正则
_ANSI_RE: re.Pattern = re.compile(r"\033\[[0-9;]*m")


def strip_ansi(text: str) -> str:
    """剥离 ANSI 颜色码

    Args:
        text: 含 ANSI 码的字符串

    Returns:
        剥离后的纯文本
    """
    return _ANSI_RE.sub("", text)


def clear_screen() -> None:
    """清屏 (Windows 用 cls, 其他用 clear)"""
    os.system("cls" if os.name == "nt" else "clear")


def parse_round_info(lines: List[str]) -> Dict:
    """从日志行解析当前进化状态

    解析的事件:
      - 进化轮 X/Y
      - 强制 FP16 磁盘分页试验
      - 当前配置 quant=Xbit shard=YMB kv_thresh=ZMB prefetch=N层
      - Prefill 进度 / 完成
      - 淘汰层 → CPU RAM / D 盘溢出
      - 读回 (源=CPU_RAM / DISK)
      - 单 prompt 结果 (tok/s)
      - 轮得分 / 新最优策略
      - 下一配置 / 自进化完成

    Args:
        lines: 日志行列表

    Returns:
        解析后的状态字典
    """
    rounds: "OrderedDict[int, Dict]" = OrderedDict()
    current_round: int = 0
    total_rounds: int = 0
    current_config: Dict = {}
    current_quant: int = 0
    latest_evict: int = 0
    latest_fetch: int = 0
    latest_vram_used: float = 0.0
    latest_vram_resident: int = 0
    latest_cpu_cache_gb: float = 0.0
    latest_tps: float = 0.0
    latest_score: float = 0.0
    latest_prefill_progress: Tuple[int, int] = (0, 0)
    latest_prefill_elapsed: float = 0.0
    spill_to_disk_count: int = 0
    status: str = "等待中..."
    last_ts: str = ""

    # 时间戳正则 (匹配行首 loguru 格式: 2026-08-07 21:20:59.388)
    ts_re: re.Pattern = re.compile(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+)")

    for raw in lines:
        line: str = strip_ansi(raw)

        m = ts_re.search(line)
        if m:
            last_ts = m.group(1)

        # 进化轮
        m = re.search(r"进化轮 (\d+)/(\d+)", line)
        if m:
            current_round = int(m.group(1))
            total_rounds = int(m.group(2))
            if current_round not in rounds:
                rounds[current_round] = {
                    "round": current_round,
                    "total": total_rounds,
                    "tps": 0.0,
                    "score": 0.0,
                    "quant": 0,
                    "config": {},
                    "status": "运行中",
                    "prefill_elapsed": 0.0,
                }
            status = "进化轮启动"

        # 强制 FP16
        if "强制 FP16 磁盘分页试验" in line:
            current_quant = 16
            status = "FP16 磁盘分页强制试验"
            if current_round in rounds:
                rounds[current_round]["quant"] = 16

        # 配置信息: quant=16bit shard=256MB kv_thresh=256MB prefetch=2层
        m = re.search(
            r"quant=(\d+)bit\s+shard=(\d+)MB\s+kv_thresh=(\d+)MB\s+prefetch=(\d+)层",
            line,
        )
        if m:
            current_quant = int(m.group(1))
            current_config = {
                "quant": int(m.group(1)),
                "shard": int(m.group(2)),
                "kv_thresh": int(m.group(3)),
                "prefetch": int(m.group(4)),
            }
            if current_round in rounds:
                rounds[current_round]["quant"] = current_quant
                rounds[current_round]["config"] = current_config
            status = f"配置就绪: {current_quant}bit prefetch={current_config['prefetch']}层"

        # Prefill 进度
        m = re.search(r"Prefill 进度: (\d+)/(\d+)", line)
        if m:
            loaded = int(m.group(1))
            total = int(m.group(2))
            latest_prefill_progress = (loaded, total)
            status = f"Prefill 装入 {loaded}/{total} 层"

        # Prefill 完成
        m = re.search(r"Prefill 完成: (\d+)/(\d+) 层装入.*耗时=([\d.]+)s", line)
        if m:
            loaded = int(m.group(1))
            total = int(m.group(2))
            elapsed = float(m.group(3))
            latest_prefill_progress = (loaded, total)
            latest_prefill_elapsed = elapsed
            if current_round in rounds:
                rounds[current_round]["prefill_elapsed"] = elapsed
            status = f"Prefill 完成 {loaded}/{total} 层 ({elapsed:.1f}s)"

        # 淘汰层 → CPU RAM: "淘汰层 0 → CPU RAM, 释放 VRAM 416MB, CPU缓存=0.4GB"
        m = re.search(
            r"淘汰层 (\d+) → CPU RAM.*CPU缓存=([\d.]+)GB.*evict_count'?: (\d+)",
            line,
        )
        if m:
            latest_cpu_cache_gb = float(m.group(2))
            latest_evict = int(m.group(3))
            if "Prefill" not in status:
                status = f"淘汰层到 CPU RAM (缓存 {latest_cpu_cache_gb:.1f}GB)"

        # 淘汰层 → D盘直写: "淘汰层 0 → D盘直写, 释放 VRAM 416MB"
        m = re.search(r"淘汰层 (\d+) → D盘直写.*evict_count'?: (\d+)", line)
        if m:
            latest_evict = int(m.group(2))
            if "Prefill" not in status:
                status = f"淘汰层直写 D 盘 (跳过 CPU RAM)"

        # CPU RAM → D 盘溢出
        if "CPU RAM → D 盘溢出" in line:
            spill_to_disk_count += 1
            status = f"CPU RAM → D 盘溢出 (累计 {spill_to_disk_count} 层)"

        # 读回
        if "读回 VRAM" in line:
            m2 = re.search(r"源=(\w+)", line)
            if m2:
                src = m2.group(1)
                if "fetch_back_count" in line:
                    m3 = re.search(r"fetch_back_count'?: (\d+)", line)
                    if m3:
                        latest_fetch = int(m3.group(1))
                # 显示源: CPU_RAM / DISK / DISK直读
                src_label = {"CPU_RAM": "CPU RAM", "DISK": "D 盘", "DISK直读": "D 盘直读"}.get(src, src)
                status = f"从 {src_label} 读回层到 VRAM"

        # 单 prompt 结果
        # "轮1 prompt0: 32tok 23.6tok/s prefill=0.5s evict=0 fetch=0"
        m = re.search(
            r"轮(\d+) prompt(\d+): (\d+)tok\s+([\d.]+)tok/s.*prefill=([\d.]+)s.*evict=(\d+).*fetch=(\d+)",
            line,
        )
        if m:
            r = int(m.group(1))
            tps = float(m.group(4))
            evict = int(m.group(6))
            fetch = int(m.group(7))
            latest_tps = tps
            latest_evict = evict
            latest_fetch = fetch
            if r in rounds:
                rounds[r]["tps"] = tps
                rounds[r]["status"] = "Decode 完成"
            status = f"Decode 完成 prompt{int(m.group(2))} → {tps:.1f} tok/s"

        # 得分 (支持负分, 失败轮会是 -800 / -900 等)
        m = re.search(r"轮 (\d+) 得分: (-?[\d.]+).*tps=([\d.]+)", line)
        if m:
            r = int(m.group(1))
            score = float(m.group(2))
            tps = float(m.group(3))
            latest_score = score
            latest_tps = tps
            if r in rounds:
                rounds[r]["score"] = score
                rounds[r]["tps"] = tps
                # 负分表示失败
                rounds[r]["status"] = "失败" if score < 0 else "已评分"
            status = (
                f"失败 score={score:.0f}" if score < 0
                else f"评分完成: score={score:.0f}"
            )

        # 推理失败 (捕获异常堆栈行)
        if "推理失败" in line:
            m2 = re.search(r"轮 (\d+) 推理失败: (.+)", line)
            if m2:
                r = int(m2.group(1))
                err_msg = m2.group(2).strip()
                if r in rounds:
                    rounds[r]["status"] = "失败"
                    rounds[r]["error"] = err_msg
                status = f"轮{r} 推理失败: {err_msg[:40]}"

        # 新最优策略
        if "新最优策略" in line:
            m = re.search(r"score=([\d.]+)", line)
            if m:
                status = f"★ 新最优策略! score={m.group(1)}"

        # 下一配置
        m = re.search(r"下一配置: quant=(\d+)bit", line)
        if m:
            status = f"准备下一轮: quant={m.group(1)}bit"

        # VRAM 状态: "vram_used_gb': 0.41 ... vram_layers_resident': 1"
        m = re.search(
            r"vram_used_gb'?: ([\d.]+).*vram_layers_resident'?: (\d+)",
            line,
        )
        if m:
            latest_vram_used = float(m.group(1))
            latest_vram_resident = int(m.group(2))

        # 自进化完成
        if "自进化完成" in line:
            status = "自进化完成"

    return {
        "current_round": current_round,
        "total_rounds": total_rounds,
        "rounds": rounds,
        "status": status,
        "last_ts": last_ts,
        "latest_tps": latest_tps,
        "latest_score": latest_score,
        "latest_evict": latest_evict,
        "latest_fetch": latest_fetch,
        "latest_vram_used": latest_vram_used,
        "latest_vram_resident": latest_vram_resident,
        "latest_cpu_cache_gb": latest_cpu_cache_gb,
        "latest_prefill_progress": latest_prefill_progress,
        "latest_prefill_elapsed": latest_prefill_elapsed,
        "spill_to_disk_count": spill_to_disk_count,
        "current_quant": current_quant,
        "current_config": current_config,
    }


def render_dashboard(info: Dict, C: ColorCodes) -> str:
    """渲染监控面板

    Args:
        info: parse_round_info 返回的状态字典
        C: 颜色码实例

    Returns:
        渲染后的面板字符串
    """
    lines: List[str] = []
    rounds = info["rounds"]

    # 标题
    lines.append(f"{C.BOLD}{C.CYAN}{'=' * 64}")
    lines.append(f"  Hermes 自进化推理引擎 — 实时监控  (RLLM DiskOffload)")
    lines.append(f"{'=' * 64}{C.RESET}")

    # 时间戳
    if info["last_ts"]:
        lines.append(f"{C.DIM}日志最新时间: {info['last_ts']}{C.RESET}")

    # 当前状态
    quant = info["current_quant"]
    if quant == 16:
        quant_label = "FP16"
        quant_color = C.YELLOW
    elif quant == 8:
        quant_label = "8bit"
        quant_color = C.BLUE
    elif quant == 4:
        quant_label = "4bit"
        quant_color = C.GREEN
    else:
        quant_label = "未启动"
        quant_color = C.DIM

    cur_r = info["current_round"]
    total_r = info["total_rounds"]
    round_str = f"{cur_r}/{total_r}" if total_r > 0 else str(cur_r)

    lines.append(f"\n{C.BOLD}进化轮次:{C.RESET} {C.CYAN}{round_str}{C.RESET}  "
                 f"{C.BOLD}当前模式:{C.RESET} {quant_color}{quant_label}{C.RESET}")
    lines.append(f"{C.BOLD}当前状态:{C.RESET} {C.GREEN}{info['status']}{C.RESET}")

    # 当前配置
    cfg = info["current_config"]
    if cfg:
        lines.append(
            f"{C.BOLD}当前配置:{C.RESET} quant={cfg.get('quant', '?')}bit "
            f"shard={cfg.get('shard', '?')}MB "
            f"kv_thresh={cfg.get('kv_thresh', '?')}MB "
            f"prefetch={cfg.get('prefetch', '?')}层"
        )

    # 三层存储状态
    lines.append(f"\n{C.BOLD}{C.MAGENTA}── 三层存储状态 ──{C.RESET}")
    vram_used = info["latest_vram_used"]
    vram_res = info["latest_vram_resident"]
    cpu_gb = info["latest_cpu_cache_gb"]
    spill_cnt = info["spill_to_disk_count"]

    vram_color = C.RED if vram_used > 6.0 else (C.YELLOW if vram_used > 3.0 else C.GREEN)
    cpu_color = C.RED if cpu_gb > 10.0 else (C.YELLOW if cpu_gb > 5.0 else C.GREEN)
    spill_color = C.RED if spill_cnt > 0 else C.GREEN

    lines.append(
        f"  {C.BOLD}VRAM{C.RESET}:    {vram_color}{vram_used:.2f}GB{C.RESET} "
        f"(常驻层 {vram_res}/32)  [8GB 上限]"
    )
    lines.append(
        f"  {C.BOLD}CPU RAM{C.RESET}: {cpu_color}{cpu_gb:.2f}GB{C.RESET} "
        f"(淘汰缓存)  [12GB 上限]"
    )
    lines.append(
        f"  {C.BOLD}D 盘 SSD{C.RESET}: {spill_color}{spill_cnt} 层已溢出{C.RESET} "
        f"(KV/权重 spill)"
    )

    # Prefill 进度
    pf_loaded, pf_total = info["latest_prefill_progress"]
    pf_elapsed = info["latest_prefill_elapsed"]
    if pf_total > 0:
        if pf_elapsed > 0:
            lines.append(
                f"  {C.BOLD}Prefill{C.RESET}:  {pf_loaded}/{pf_total} 层 "
                f"({pf_elapsed:.1f}s)"
            )
        else:
            lines.append(f"  {C.BOLD}Prefill{C.RESET}:  {pf_loaded}/{pf_total} 层 (进行中)")

    # 淘汰/回读计数
    lines.append(
        f"  {C.BOLD}淘汰/回读:{C.RESET} {info['latest_evict']}/{info['latest_fetch']}"
    )

    # 进化轨迹表格
    lines.append(f"\n{C.BOLD}{C.CYAN}{'─' * 64}")
    lines.append(
        f"  {'轮':>3} | {'模式':>6} | {'速度':>12} | {'得分':>8} | "
        f"{'Prefill':>8} | {'状态':>12}"
    )
    lines.append(f"{'─' * 64}{C.RESET}")

    for r_data in rounds.values():
        r = r_data["round"]
        q = r_data["quant"]
        tps = r_data["tps"]
        score = r_data["score"]
        st = r_data["status"]
        pf = r_data.get("prefill_elapsed", 0.0)
        err = r_data.get("error", "")

        if q == 16:
            q_label = f"{C.YELLOW}FP16{C.RESET}"
        elif q == 8:
            q_label = f"{C.BLUE}8bit{C.RESET}"
        elif q == 4:
            q_label = f"{C.GREEN}4bit{C.RESET}"
        else:
            q_label = "??"

        if tps > 0:
            tps_str = f"{tps:.1f} tok/s"
        else:
            tps_str = "..."

        # 分数显示: 正分绿色, 负分红色 (失败)
        if score > 0:
            score_str = f"{C.GREEN}{score:.0f}{C.RESET}"
        elif score < 0:
            score_str = f"{C.RED}{score:.0f}{C.RESET}"
        else:
            score_str = "-"

        if pf > 0:
            pf_str = f"{pf:.1f}s"
        else:
            pf_str = "-"

        if st == "失败":
            st_color = C.RED
            st_display = f"失败 ({err[:20]})" if err else "失败"
        elif st == "已评分":
            st_color = C.GREEN
            st_display = st
        elif st in ("Decode 完成",):
            st_color = C.CYAN
            st_display = st
        elif st == "运行中":
            st_color = C.YELLOW
            st_display = st
        else:
            st_color = C.DIM
            st_display = st

        lines.append(
            f"  {r:>3} | {q_label} | {tps_str:>12} | {score_str:>8} | "
            f"{pf_str:>8} | {st_color}{st_display}{C.RESET}"
        )

    lines.append(f"{C.DIM}{'─' * 64}{C.RESET}")

    # FP16 vs 4bit 对比
    fp16_rounds = [r for r in rounds.values() if r["quant"] == 16 and r["tps"] > 0]
    bit4_rounds = [r for r in rounds.values() if r["quant"] == 4 and r["tps"] > 0]
    bit8_rounds = [r for r in rounds.values() if r["quant"] == 8 and r["tps"] > 0]

    if fp16_rounds or bit4_rounds or bit8_rounds:
        lines.append(f"\n{C.BOLD}{C.MAGENTA}── 模式对比 (平均 tok/s) ──{C.RESET}")
        if fp16_rounds:
            avg = sum(r["tps"] for r in fp16_rounds) / len(fp16_rounds)
            best = max(r["tps"] for r in fp16_rounds)
            lines.append(
                f"  {C.YELLOW}FP16{C.RESET}  avg={avg:.1f}  best={best:.1f}  "
                f"({len(fp16_rounds)} 轮, 无损质量)"
            )
        if bit8_rounds:
            avg = sum(r["tps"] for r in bit8_rounds) / len(bit8_rounds)
            best = max(r["tps"] for r in bit8_rounds)
            lines.append(
                f"  {C.BLUE}8bit{C.RESET}  avg={avg:.1f}  best={best:.1f}  "
                f"({len(bit8_rounds)} 轮, 近无损)"
            )
        if bit4_rounds:
            avg = sum(r["tps"] for r in bit4_rounds) / len(bit4_rounds)
            best = max(r["tps"] for r in bit4_rounds)
            lines.append(
                f"  {C.GREEN}4bit{C.RESET}  avg={avg:.1f}  best={best:.1f}  "
                f"({len(bit4_rounds)} 轮, 有损但快)"
            )

    # 最优策略
    if info["latest_score"] > 0:
        lines.append(
            f"\n{C.BOLD}最新得分:{C.RESET} {C.GREEN}{info['latest_score']:.0f}{C.RESET}  "
            f"{C.BOLD}最新速度:{C.RESET} {C.GREEN}{info['latest_tps']:.1f} tok/s{C.RESET}"
        )

    return "\n".join(lines)


def load_strategy_pool_summary() -> Optional[Dict]:
    """加载策略池摘要 (最优策略 + 数量)

    兼容三种 JSON 结构:
      - {"records": [{sig, config, performance:{score,...}}, ...]}
      - {"strategies": [...]} / {"pool": [...]}
      - [ {..., "score": ...}, ... ]

    Returns:
        包含 best_strategy / pool_size / best_score 的字典, 失败返回 None
    """
    import json

    if not STRATEGY_FILE.exists():
        return None
    try:
        with open(STRATEGY_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            strategies = (
                data.get("records")
                or data.get("strategies")
                or data.get("pool")
                or []
            )
            best_sig = data.get("best_sig")
        else:
            strategies = data if isinstance(data, list) else []
            best_sig = None
        if not isinstance(strategies, list) or not strategies:
            return None

        def _score_of(item: Dict) -> float:
            """从策略条目提取分数 (兼容 performance.score 和顶层 score)"""
            if not isinstance(item, dict):
                return 0.0
            perf = item.get("performance")
            if isinstance(perf, dict):
                return float(perf.get("score", 0.0))
            return float(item.get("score", 0.0))

        def _tps_of(item: Dict) -> float:
            """从策略条目提取吞吐"""
            if not isinstance(item, dict):
                return 0.0
            perf = item.get("performance")
            if isinstance(perf, dict):
                return float(perf.get("avg_throughput_tps", 0.0))
            return float(item.get("avg_throughput_tps", 0.0))

        best = max(strategies, key=_score_of)
        return {
            "pool_size": len(strategies),
            "best_score": _score_of(best),
            "best_tps": _tps_of(best),
            "best_strategy": best,
            "best_sig": best_sig,
        }
    except Exception:
        return None


def main() -> int:
    """主入口

    Returns:
        退出码 (0 正常, 1 日志不存在)
    """
    parser = argparse.ArgumentParser(
        description="Hermes 自进化实时监控面板"
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="只输出一次快照, 不循环刷新",
    )
    parser.add_argument(
        "--no-color",
        action="store_true",
        help="禁用 ANSI 颜色 (适合管道/重定向)",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=2.0,
        help="循环刷新间隔秒 (默认 2.0)",
    )
    parser.add_argument(
        "--log",
        type=str,
        default=str(LOG_FILE),
        help=f"日志文件路径 (默认 {LOG_FILE})",
    )
    args = parser.parse_args()

    log_file: Path = Path(args.log)
    C: ColorCodes = ColorCodes(enabled=not args.no_color)

    if args.once:
        # 单次快照模式
        if not log_file.exists():
            print(f"[错误] 日志文件不存在: {log_file}")
            return 1
        try:
            with open(log_file, "r", encoding="utf-8", errors="replace") as f:
                lines = f.readlines()
        except OSError as exc:
            print(f"[错误] 读取日志失败: {exc}")
            return 1

        recent = lines[-3000:] if len(lines) > 3000 else lines
        info = parse_round_info(recent)
        print(render_dashboard(info, C))

        # 附加策略池摘要
        summary = load_strategy_pool_summary()
        if summary is not None:
            best = summary["best_strategy"]
            print(f"\n{C.BOLD}{C.MAGENTA}── 策略池摘要 (历史最优) ──{C.RESET}")
            print(f"  池中策略数: {summary['pool_size']}")
            print(
                f"  历史最优得分: {C.GREEN}{summary['best_score']:.0f}{C.RESET}  "
                f"速度: {C.GREEN}{summary.get('best_tps', 0.0):.1f} tok/s{C.RESET}"
            )
            if isinstance(best, dict):
                cfg = best.get("config", best)
                print(
                    f"  最优配置: quant={cfg.get('quantization_bits', '?')}bit "
                    f"prefetch={cfg.get('prefetch_layers_ahead', '?')}层 "
                    f"threads={cfg.get('prefetch_threads', '?')} "
                    f"kv_thresh={cfg.get('kv_spill_threshold_mb', '?')}MB"
                )
                sig = best.get("sig", "?")
                rnd = best.get("evo_round", "?")
                print(f"  策略 sig: {sig}  (来自第 {rnd} 轮)")
        return 0

    # 循环刷新模式
    print(f"{C.CYAN}启动 Hermes 自进化监控 (Ctrl+C 退出)...{C.RESET}")
    try:
        while True:
            try:
                with open(log_file, "r", encoding="utf-8", errors="replace") as f:
                    lines = f.readlines()
            except FileNotFoundError:
                print(f"{C.RED}日志文件不存在: {log_file}{C.RESET}")
                print(f"{C.DIM}等待进化引擎启动...{C.RESET}")
                time.sleep(3)
                continue
            except OSError as exc:
                print(f"{C.RED}读取日志失败: {exc}{C.RESET}")
                time.sleep(3)
                continue

            recent = lines[-3000:] if len(lines) > 3000 else lines
            info = parse_round_info(recent)
            clear_screen()
            print(render_dashboard(info, C))
            print(f"\n{C.DIM}刷新中... (每 {args.interval:.0f}s, Ctrl+C 退出){C.RESET}")
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print(f"\n{C.YELLOW}监控已停止{C.RESET}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
