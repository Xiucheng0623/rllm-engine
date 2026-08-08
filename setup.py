# -*- coding: utf-8 -*-
# File: D:\AI_RLLM\setup.py
"""RLLM Engine 安装配置

本地大模型推理引擎 - 在消费级 GPU 上运行 7B+ 模型.

安装:
    pip install -e .
    # 或
    python setup.py install

使用:
    from rllm_engine import RLLMEngine
    engine = RLLMEngine("Nous-Hermes-2-Mistral-7B-DPO")
    engine.load()
    print(engine.generate("你好"))
"""
from setuptools import setup, find_packages

setup(
    name="rllm-engine",
    version="1.0.0",
    author="RLLM Team",
    description="本地大模型推理引擎 - 消费级 GPU 运行大模型",
    long_description=open("README.md", encoding="utf-8").read()
    if __import__("os").path.exists("README.md")
    else __doc__,
    long_description_content_type="text/markdown",
    url="https://github.com/rllm-org/rllm-engine",
    packages=find_packages(
        include=[
            "rllm_engine",
            "rllm_engine.*",
            "rllm_disk_engine",
            "rllm_disk_engine.*",
            "rllm_agent_core",
            "rllm_agent_core.*",
            "rllm_auto_evo",
            "rllm_auto_evo.*",
            "rllm_disk_compute",
            "rllm_disk_compute.*",
            "rllm_pipeline",
            "rllm_pipeline.*",
        ]
    ),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.4.0",
        "transformers>=4.40.0",
        "accelerate>=0.30.0",
        "bitsandbytes>=0.43.0",
        "aiofiles>=23.2.0",
        "diskcache>=5.6.0",
        "loguru>=0.7.0",
        "numpy>=1.24.0",
        "safetensors>=0.4.0",
        "tqdm>=4.66.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.21.0",
        ],
    },
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: Other/Proprietary License",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    entry_points={
        "console_scripts": [
            "rllm-demo=rllm_engine.demo:main",
            "rllm-chat=rllm_chat:main",
            "rllm-manager=rllm_manager:main",
        ],
    },
)
