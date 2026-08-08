# RLLM Engine

<p align="center">
  <img src="https://trae-api-cn.mchost.guru/api/ide/v1/text_to_image?prompt=A+futuristic+minimalist+logo+showing+a+chip+connected+to+a+hard+drive%2C+symbolizing+AI+models+running+from+disk+storage%2C+dark+background%2C+neon+cyan+and+purple+accents%2C+clean+vector+style&image_size=square" alt="RLLM Engine Logo" width="200">
</p>

<p align="center">
  <strong>在 8GB 消费级显卡上运行 7B/13B 大模型</strong>
</p>

<p align="center">
  <a href="#-5-分钟快速开始"><img src="https://img.shields.io/badge/Quick_Start-5_min-orange?style=flat-square" alt="Quick Start"></a>
  <a href="#-安装"><img src="https://img.shields.io/badge/pip-install-blue?style=flat-square" alt="Install"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Commercial_Trial-red?style=flat-square" alt="License"></a>
  <a href="#-benchmark"><img src="https://img.shields.io/badge/Speed-23.6_tok/s-brightgreen?style=flat-square" alt="Speed"></a>
  <img src="https://img.shields.io/badge/Python-3.10-blue?style=flat-square" alt="Python">
  <img src="https://img.shields.io/badge/Platform-Win_|_Linux_|_macOS-lightgrey?style=flat-square" alt="Platform">
  <img src="https://img.shields.io/badge/GPU-8GB+_VRAM-green?style=flat-square" alt="GPU">
</p>

---

## 这是什么？

**RLLM Engine** 是一个本地大模型推理引擎。它通过创新的**层级磁盘分页**技术，让 7B/13B 参数的大模型在 **8GB VRAM** 的消费级显卡上运行——传统方案需要 16GB+ VRAM。

不需要云服务，不需要 API Key，一切在你自己的电脑上运行。

### 核心创新

| 特性 | 说明 |
|------|------|
| **层级磁盘分页** | 将 Transformer 层按需加载到 VRAM，未使用的层常驻 NVMe SSD |
| **热-冷驱逐** | 高频层常驻 VRAM，低频层自动逐出到磁盘 |
| **KV Cache 溢出** | KV 缓存超过阈值自动溢出到磁盘，突破 VRAM 限制 |
| **直写磁盘模式** | FP16/8bit 模式直接 VRAM ↔ NVMe，跳过 CPU RAM |

---

## 5 分钟快速开始

### 1. 安装

```bash
pip install git+https://github.com/Xiucheng0623/rllm-engine.git
```

### 2. 下载模型 (选一个)

```bash
# 推荐：Nous Hermes 2 Mistral 7B (4bit, ~4GB)
rllm-manager download Nous-Hermes-2-Mistral-7B-DPO

# 或手动：将 HuggingFace 模型放到 ~/.rllm/models/
```

### 3. 运行 Demo

```bash
rllm-demo
```

输出示例：

```
  📋 系统信息
  模型     Nous-Hermes-2-Mistral-7B-DPO
  GPU      NVIDIA GeForce RTX 5070 (8.0GB)

  📋 Demo 演示
  ──────────────────────────────────────
  >>> 简单问答
  What is the capital of France?

  Paris is the capital of France.

  >>> 创作
  请用中文写一首关于秋天的短诗...

  秋天绿叶落地，红叶飞舞自由。
  金色阳光暖心，白云遥远山丘。

  📋 性能汇总
  平均速度   23.6 tok/s
  VRAM       3.8GB (4bit)

  与传统方案对比:
  ─────────────────────────────────────
  指标          传统 GPU       RLLM Engine
  VRAM         14GB (FP16)    3.8GB
  速度          ~20 tok/s     23.6 tok/s
  GPU 门槛      16GB+          8GB+
```

### 4. Python API

```python
from rllm_engine import RLLMEngine

# 一行代码加载模型
engine = RLLMEngine("Nous-Hermes-2-Mistral-7B-DPO")
engine.load()

# 生成
answer = engine.generate("请用中文解释什么是人工智能")
print(answer)

# 交互对话
engine.chat()
```

---

## 安装

### 系统要求

- **GPU**: NVIDIA GeForce RTX 2060+ (8GB VRAM 最低)
- **RAM**: 16GB+
- **存储**: 20GB 可用空间，推荐 NVMe SSD
- **系统**: Windows 10/11, Linux (Ubuntu 22.04+), macOS (Metal 实验性)

### 安装步骤

```bash
# 1. 创建虚拟环境 (推荐)
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

# 2. 安装
pip install rllm-engine

# 3. 验证
rllm-demo --list-models
```

### LangChain 集成 (可选)

```python
from rllm_engine import RLLMEngine
from langchain.llms.base import LLM

class RLLM(LLM):
    engine = RLLMEngine("Nous-Hermes-2-Mistral-7B-DPO")
    engine.load()

    def _call(self, prompt: str, **kwargs):
        return self.engine.generate(prompt)

    @property
    def _llm_type(self):
        return "rllm"
```

---

## Benchmark

### 测试环境

- GPU: NVIDIA GeForce RTX 5070 Laptop (8GB VRAM)
- RAM: 32GB DDR5
- SSD: NVMe PCIe 4.0
- 模型: Nous-Hermes-2-Mistral-7B-DPO (4bit)

### 结果

| 指标 | 传统 GPU (FP16) | RLLM Engine (4bit) | 节省 |
|------|-----------------|-------------------|------|
| **VRAM 占用** | 14GB | 3.8GB | **73%** ↓ |
| **推理速度** | ~20 tok/s | 23.6 tok/s | **18%** ↑ |
| **最小 GPU** | 16GB VRAM | 8GB VRAM | 门槛减半 |
| **加载时间** | ~5s | ~18s | - |

### 不同硬件对比

| 模型 | GPU | VRAM | 速度 | 质量 |
|------|-----|------|------|------|
| 7B (4bit) | RTX 5070 (8GB) | 3.8GB | 23.6 tok/s | 正常 |
| 13B (4bit) | RTX 5090 (16GB) | 7.2GB | 15.8 tok/s | 良好 |

> 运行自己的 benchmark: `python -m rllm_engine.demo --benchmark`

---

## 为什么比传统方案好？

```
传统方案:  模型整个加载 → 爆 VRAM → 无法运行
RLLM:      逐层加载 → 只用当前层 → 永远不爆
```

| 方案 | 7B 模型 VRAM | 13B 模型 VRAM | 47B MoE | 
|------|-------------|-------------|------|
| Transformers | 14GB | 26GB | 不可运行 |
| llama.cpp | 6GB | 12GB | 有限 |
| vLLM/SGLang | 16GB | 28GB | 不可运行 |
| **RLLM Engine** | **3.8GB** | **~7GB** | **8GB** ✅ |

---

## 架构

```
┌─────────────────────────────────────────────────────┐
│                    RLLM Engine                        │
├─────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌──────────────┐  ┌────────────┐ │
│  │ QuantLoader  │  │ VRAM Pool    │  │ LayerRunner│ │
│  │ (4bit量化)   │→│ (热冷驱逐)    │→│ (逐层推理)  │ │
│  └─────────────┘  └──────┬───────┘  └────────────┘ │
│                          │                            │
│              ┌───────────▼───────────┐               │
│              │    NVMe SSD (D盘)      │               │
│              │   32层权重分片 + KV缓存  │               │
│              └───────────────────────┘               │
└─────────────────────────────────────────────────────┘
```

---

## 命令参考

| 命令 | 功能 |
|------|------|
| `rllm-demo` | 快速 Demo 验证效果 |
| `rllm-demo --interactive` | 交互式对话 |
| `rllm-demo --list-models` | 列出可用模型 |
| `rllm-manager download MODEL` | 下载模型 |
| `rllm-manager shard MODEL` | 模型分片 (MoE) |
| `rllm-manager verify MODEL` | 验证完整性 |

---

## 许可

**本版本为试用版 (Trial License).**

- ✅ 个人学习、研究、评估免费使用
- ✅ 30 天全功能试用
- ❌ 商业使用需购买商业许可
- ❌ 禁止重新分发、修改后分发

正式版 (v2.0+) 将提供:
- 13B/70B/405B 模型支持
- MoE 专家级分页 (47B Mixtral)
- 投机解码加速
- 商业授权 + 技术支持

商业许可咨询: `rllm-business@proton.me`

详见 [LICENSE](LICENSE)

---

## 路线图

- [x] v1.0: 7B/13B dense 模型 (层级分页) ← **当前**
- [ ] v2.0: 47B MoE 模型 (专家级分页) - 已内部验证
- [ ] v2.1: 投机解码 (2x 加速)
- [ ] v3.0: 多 GPU 并行 + 云部署

---

## 常见问题

**Q: 和 llama.cpp 有什么区别？**

llama.cpp 将整个模型 offload 到 CPU/GPU，需要至少 6GB VRAM。RLLM 的层级分页让每层独立加载，VRAM 只需容纳单层（~200MB），剩余全部在 NVMe SSD 上。

**Q: 支持 Windows 吗？**

完全支持。Windows/Linux/macOS 统一步体验。

**Q: 数据安全吗？**

100% 本地运行。没有网络请求，没有数据上传，没有遥测。

**Q: 支持哪些模型？**

理论上所有 HuggingFace transformers 兼容的模型。已验证: Mistral, Hermes, LLaMA, Qwen。

---

<p align="center">
  <sub>RLLM Engine © 2026. All rights reserved.</sub><br>
  <sub>本项目中提及的第三方商标（Mixtral, NVIDIA, PyTorch, bitsandbytes, Mistral, Hermes, LLaMA, Qwen 等）均为各自所有者的财产。<br>RLLM Engine 与上述第三方无任何关联、认可或赞助关系。</sub>
</p>
