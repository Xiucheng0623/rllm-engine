# 贡献指南

感谢你对 RLLM Engine 的关注！

## 当前状态

RLLM Engine v1.0 处于 **试用阶段**。核心推理引擎为闭源商业软件，
本仓库提供 Python API 封装、CLI 工具和 demo 脚本。

## 可以贡献什么？

| 类型 | 欢迎程度 | 说明 |
|------|---------|------|
| Bug 报告 | ✅ 欢迎 | 通过 GitHub Issues |
| 文档改进 | ✅ 欢迎 | README, API 文档 |
| 测试用例 | ✅ 欢迎 | benchmark, 兼容性测试 |
| 示例代码 | ✅ 欢迎 | LangChain/LlamaIndex 集成 |
| 核心引擎 | ❌ 暂不接受 | 闭源商业代码 |
| 新模型支持 | 联系讨论 | 需评估技术可行性 |

## Bug 报告

请通过 [GitHub Issues](https://github.com/rllm-org/rllm-engine/issues) 提交，
并包含以下信息：

```
环境:
  - 系统版本: Windows 11 / Ubuntu 22.04
  - GPU: NVIDIA RTX 5070 (8GB)
  - Python: 3.10.12
  - rllm-engine: 1.0.0

复现步骤:
  1. pip install rllm-engine
  2. rllm-demo
  3. [错误信息]

错误日志:
  [粘贴完整错误输出]
```

## 开发环境设置

```bash
git clone https://github.com/rllm-org/rllm-engine
cd rllm-engine
python -m venv .venv
source .venv/bin/activate
pip install -e .
pip install -e ".[dev]"
```

## 许可

提交代码即表示同意将您的贡献按照项目许可（商业试用许可）分发。
