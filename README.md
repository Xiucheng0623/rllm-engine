# Rebirth LLM（RLLM）— 磁盘卸载低内存大模型离线批量生成引擎

> 全 D 盘隔离部署 | 离线优先 | 自进化（自动调参）闭环
> 目标场景：普通笔记本（32GB 内存 + 独显）离线跑大模型批量图文/文本生成

---

## 一、项目简介

RLLM 是一套面向**资源受限 Windows 机器**的大模型离线批量生成工具链，核心思路是：
把模型权重放到磁盘，按需分页加载 + 量化，配合全局内存锁与异步预取调度，
并尝试用一个“复盘 → 自进化”闭环在运行中自动调优推理参数。

⚠️ **诚实声明（请先读第五节）**：仓库里同时存在“已经能跑的逻辑”和“只搭了骨架、尚未真正落地的模块”。
本 README 只记录**已实现 / 经代码核查可用**的部分；脚手架与未兑现承诺单列在第五节，避免误判项目成熟度。

---

## 二、已实现 / 经核查可用的模块

### 1. 推理（真实可用：v2 路径）
- 文件：`rllm_agent_core/workers/worker_registry.py` → `DiskLLMInferWorker`
- 当前**真正在跑的推理**走 HuggingFace `transformers`：
  - `MistralForCausalLM.from_pretrained(..., load_in_4bit=True, device_map="auto")`
  - 采样：temperature + top_p（nucleus）+ repetition penalty，避免贪心退化
  - 离线加载 tokenizer、KV-cache 在 decode 阶段复用、任务结束清理显存
- 这是项目里**唯一经代码确认能产出真实生成文本**的路径。

### 2. 磁盘引擎基础设施（真实可用，作为独立组件）
- `rllm_disk_engine/memory_lock/global_memory_lock.py`
  - 基于 `psutil` 的进程 RSS 实时监测；CPU 缓冲硬限校验；超限计数；
    强制 swap 回调；仍超限则 `raise MemoryError` 阻断加载。监控线程独立运行。
- `rllm_disk_engine/scheduler/async_page_scheduler.py`
  - 异步预取线程池、单层 load/unload、缓冲区超容强制驱逐、可选 mmap；
  - **“真实模式”下直接从 safetensors 按层读取真实 bf16 权重**（绕过 .shard 占位文件）。
- `rllm_disk_engine/kv_manager/kv_spill_manager.py`：KV 缓存超阈值落盘。
- `rllm_disk_engine/mmap_io/mmap_wrapper.py`：Windows mmap 窗口池封装。
- `rllm_disk_engine/sharding/shard_persistor.py`：索引骨架生成 + 分片元数据管理（**落盘内容见第五节**）。

### 3. 自进化闭环（真实可用的逻辑）
- `rllm_agent_core/review/review_engine.py`
  - 聚合指标（延迟/内存/吞吐/IO阻塞/KV溢出/失败率）、触发条件判定、综合评分、
    调用调优器产出新配置并写回 Skill、保存最优策略与复盘报告。
- `rllm_auto_evo/metrics/metrics_collector.py`：关键指标批量采集。
- `rllm_auto_evo/strategy/strategy_pool.py`：策略池（最优/回放/淘汰）。
- `rllm_auto_evo/tuner/auto_tuner.py`：强干预 + 邻域搜索 + 历史最优回放 + 随机探索的参数搜索。
- `rllm_agent_core/skills/skill_loader.py`：`DiskOffloadInferSkill` 封装 worker + 调用指标记录 + 配置热更新（自进化入口）。

### 4. 业务流水线（真实可用）
- `rllm_pipeline/batch_reader/keyword_reader.py`：TXT / JSONL / CSV 输入读取。
- `rllm_pipeline/writer/output_writer.py`：结果异步写 D 盘 JSONL（不驻留内存）。
- `rllm_pipeline/checkpoint/checkpoint_manager.py`：原子写断点、续跑跳过已完成、重置。

### 5. 入口与支撑
- `main_RLLM.py`：批量生成主入口（CLI：`--max-tasks / --batch-size / --review-every / --max-new-tokens / --reset-ckpt`）。
- `rllm_chat.py`：单条对话入口。
- `rllm_manager.py`：模块/组件管理入口。
- `monitor_evo.py`：自进化过程监控面板。
- `env_config.py` / `validate_env_RLLM.py`：D 盘路径注入 + 离线模式 + 环境校验。
- `setup.py` / `requirements.txt` / `install.bat` / `init_RLLM_env.bat`：依赖与初始化。
- `LICENSE`：保留原始开源协议（基于 Nous Hermes-Agent MIT 二次开发）。

### 6. 测试
- `rllm_tests/rllm_test_modules.py`：模块验收自测。
- `rllm_tests/rllm_stress_test.py`：压测驱动自进化。
- `tests/`：v3 / v4 各阶段验证脚本（4bit、MoE 分页、Mixtral 真实加载等）。

---

## 三、目录结构（实际代码，非 README 旧版的理想布局）

```
D:\AI_RLLM\
├── main_RLLM.py            # 批量生成主入口
├── rllm_chat.py            # 对话入口
├── rllm_manager.py         # 组件管理
├── monitor_evo.py          # 自进化监控
├── env_config.py           # D盘路径/离线模式注入
├── validate_env_RLLM.py    # 环境校验
├── setup.py / requirements.txt / install.bat / init_RLLM_env.bat
├── LICENSE
├── rllm_agent_core/        # 推理Worker + Skill + 复盘引擎 + 配置 + 三层记忆
├── rllm_disk_engine/       # 内存锁 / 调度器 / KV溢出 / mmap / 分片 / MoE / VRAM / 投机解码
├── rllm_auto_evo/          # 指标采集 / 策略池 / 自动调优器
├── rllm_pipeline/          # 批量读取 / JSONL写出 / 断点续跑
├── rllm_engine/            # 引擎封装 + 平台路径
├── rllm_disk_compute/      # IO-计算重叠基准
├── rllm_tests/             # 验收自测 + 压测
└── tests/                  # 各阶段验证脚本
```

运行时产生的大体积目录（**不纳入版本库**）：`.venv/`、`hf_cache/`、`model_shards/`、`rllm_model_shards/`、
`offload_temp/`、`rllm_offload_temp/`、`output_dataset/`、`rllm_output_dataset/`、`input_data/`、`logs/`。
（见 `.gitignore`）

---

## 四、快速开始

```bat
:: 1. 初始化环境（创建目录 + 虚拟环境 + 依赖）
D:\AI_RLLM\install.bat

:: 2. 激活并校验
call D:\AI_RLLM\.venv\Scripts\activate.bat
python D:\AI_RLLM\validate_env_RLLM.py

:: 3. 手动放置模型权重（离线，不联网下载）
::    把 Mistral / Hermes 等 safetensors 模型整目录拷到
::    D:\AI_RLLM\rllm_model_shards\_raw\<模型名>\  （需含 config.json / tokenizer / *.safetensors）

:: 4. 单条对话
python D:\AI_RLLM\rllm_chat.py

:: 5. 批量生成（断点可续跑）
python D:\AI_RLLM\main_RLLM.py --max-tasks 100 --batch-size 8 --review-every 20

:: 6. 模块验收
python D:\AI_RLLM\rllm_tests\rllm_test_modules.py
```

> 前置：Python 3.10+、NVIDIA 驱动（CUDA 可用时走 4bit GPU 路径）、D 盘 ≥ 300GB 可用空间。

---

## 五、当前为脚手架 / 尚未真正落地的部分（务必知悉）

为避免误判，以下**目前并非真实可用**，请勿对外宣称已具备相应能力：

1. **自定义磁盘逐层分页推理（v1 路径）未真正接入 Worker**
   - `DiskLLMInferWorker` 的 `_prefill` / `_decode_step` 实际调用的是 `transformers` 全量模型
     （`device_map="auto"` 自行负责分片/卸载），并非自研的逐层分页 compute。
   - 自研逐层 forward 只剩一个 `_make_decoder_layer_factory` 工厂（注释标明“保留供未来大模型场景”）。
2. **`ModelShardPersistor` 的自定义 `.shard` 格式写的是占位（全 0）权重**
   - `_write_placeholder_or_real` 固定写零填充；`_read_shard_file` 返回 `{"weight_placeholder": ...}`。
   - 文档声称的 bitsandbytes 4/8bit 量化 + 真实序列化到 `.shard` **未实现**。
   - 调度器另有“真实模式”直接读 safetensors，绕开了 `.shard`，所以真实权重读取不依赖该占位逻辑。
3. **2GB 内存硬限承诺未兑现**
   - v2 推理初始化时会把内存锁放宽到 8GB 并停止监控线程（`_lazy_init_v2` 内），
     因此“全程 RSS < 2GB”只对未真正加载模型的内存锁演示成立，对真实大模型不保证。
4. **三层记忆 `three_layer_memory.py`** 部分方法仍为 `raise NotImplementedError`。
5. **MoE / Expert Pool / VRAM Pool / 投机解码（speculative）** 等模块目前为初期脚手架，
   未经端到端验证，不可作为成品能力。
6. 仓库根目录的旧版 `README.md` 描述的是理想化 `disk_engine/` 布局与“50B 跑在 2GB”等目标，
   与实际 `rllm_*` 代码及上述限制不一致，请以本文件为准。

---

## 六、自进化闭环说明

1. 每处理 N 条（默认 `--review-every 100`）触发一次复盘（`review_engine.run_review_cycle`）。
2. 复盘聚合延迟/内存/吞吐/IO/失败率，判定触发条件（延迟涨 20% / 内存突破 / IO 阻塞 >30s / 失败率 >0.5%）。
3. 触发后调用 `auto_tuner.suggest_next_config` 产出新参数，写回 `DiskOffloadInferSkill` 配置并落盘。
4. 最优策略签名保存在 `rllm_skill_storage/review_engine_state.json`，下次启动自动加载。

> 注：自进化调优的是 `DiskOffloadSkillConfig`（预取层数/线程数/量化位/分片大小/KV阈值等），
> 并不直接改写模型权重或自研分页 compute（参见第五节）。

---

## 七、商业合作

- **商务合作电话：13504091457**
- 欢迎就大模型离线部署、低资源推理优化、批量内容生成等方向洽谈合作。

---

## 八、版权与商标声明

本项目基于开源项目 **Nous Hermes-Agent（MIT License）** 二次深度开发，仓库内保留完整原始开源协议文件。
磁盘分层加载、全局内存锁、D 盘隔离部署、自动 IO 调优为自研模块。

项目代号 **Rebirth LLM（RLLM）** 为独立软件项目名，与奢侈品品牌 Hermès、开源项目 Hermes-Agent
无品牌合作或隶属关联，仅内部功能性调用开源框架，不用于商业品牌宣传。

---

**文档版本**：v2.0（代码实测对齐）  ·  **更新**：2026-08-08
