# -*- coding: utf-8 -*-
# File: D:\AI_RLLM\rllm_chat.py
"""RLLM Engine 本地聊天 CLI

在低配电脑上运行 47B 大模型的命令行聊天工具.

用法:
    D:\\AI_RLLM\\.venv\\Scripts\\python.exe D:\\AI_RLLM\\rllm_chat.py
    D:\\AI_RLLM\\.venv\\Scripts\\python.exe D:\\AI_RLLM\\rllm_chat.py --model mixtral-8x7b
    D:\\AI_RLLM\\.venv\\Scripts\\python.exe D:\\AI_RLLM\\rllm_chat.py --max-tokens 256 --temp 0.8

参数:
    --model MODEL_NAME      模型名称 (默认: mixtral-8x7b)
    --max-tokens N          最大生成 token 数 (默认: 128)
    --temp F                采样温度 (默认: 0.7)
    --top-p F               nucleus 采样阈值 (默认: 0.9)
    --no-chat               单次生成模式 (非对话)
    --prompt TEXT           单次生成时的 prompt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_RLLM_ROOT = Path(r"D:\AI_RLLM")
if str(_RLLM_ROOT) not in sys.path:
    sys.path.insert(0, str(_RLLM_ROOT))

from rllm_engine import RLLMEngine


def parse_args() -> argparse.Namespace:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="RLLM Engine - 本地大模型推理引擎",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  %(prog)s                                    # 交互式对话
  %(prog)s --no-chat --prompt "写一首诗"       # 单次生成
  %(prog)s --model mixtral-8x7b-instruct       # 指定模型
  %(prog)s --max-tokens 512 --temp 0.8         # 调整参数
        """,
    )
    parser.add_argument(
        "--model",
        default="mixtral-8x7b",
        help="模型名称 (默认: mixtral-8x7b)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=128,
        help="最大生成 token 数 (默认: 128)",
    )
    parser.add_argument(
        "--temp",
        type=float,
        default=0.7,
        help="采样温度 (默认: 0.7)",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="nucleus 采样阈值 (默认: 0.9)",
    )
    parser.add_argument(
        "--no-chat",
        action="store_true",
        help="单次生成模式",
    )
    parser.add_argument(
        "--prompt",
        default="",
        help="单次生成的 prompt",
    )
    return parser.parse_args()


def main() -> int:
    """主入口"""
    args = parse_args()

    print(f"\n{'#'*60}")
    print(f"#  RLLM Engine - 47B大模型在8GB显卡上跑起来")
    print(f"#  专家级磁盘分页 + 热冷驱逐 + decode容错回退")
    print(f"{'#'*60}\n")

    # 初始化引擎
    engine = RLLMEngine(
        model_name=args.model,
        max_new_tokens=args.max_tokens,
        temperature=args.temp,
        top_p=args.top_p,
    )

    try:
        engine.load()
    except FileNotFoundError as e:
        print(f"\n[错误] {e}\n")
        print("请先运行模型管理工具:")
        print(f"  python rllm_manager.py download {args.model}")
        print(f"  python rllm_manager.py shard {args.model}")
        return 1
    except RuntimeError as e:
        print(f"\n[错误] {e}")
        return 1

    if args.no_chat:
        # 单次生成
        prompt = args.prompt or input("输入 prompt: ").strip()
        if not prompt:
            print("未输入 prompt, 退出。")
            return 0
        result = engine.generate(prompt)
        print(f"\n{result}")
    else:
        # 交互式对话
        engine.chat()

    # 清理
    engine.unload()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
