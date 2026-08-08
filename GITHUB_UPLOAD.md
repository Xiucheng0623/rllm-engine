# 推送到 GitHub 说明（私有仓库）

## 已完成的部分（在本机 D:\AI_RLLM 已就绪）

- ✅ `git init` + 初始提交（110 个源码/配置/文档文件）
- ✅ `.gitignore` 已排除：`.venv/`、模型权重缓存（`hf_cache/`、`model_shards/`、`rllm_model_shards/`、`offload_temp/`、`rllm_offload_temp/`）、`output_dataset/`、`input_data/`、`logs/`、`*.log`、备份文件 `*.bak`、调试临时脚本 `_*.py`
- ✅ `README.md` 已重写为“已实测对齐”的版本，**明确区分了“已实现”与“脚手架/未落地”模块**，并包含商业合作电话 **13504091457**

> 注意：当前仓库默认分支为 `main`。

## 为什么没有自动上传

自动化上传依赖 WorkBuddy 连接的 GitHub 连接器，但它**没有“创建仓库”的权限**（API 返回 403 `Resource not accessible by integration`），
且本机 git 没有缓存的 GitHub 凭证、也未安装 `gh`。因此无法在本会话内直接推送。

下面两种方式任选其一即可完成上传（仓库均为**私有**）。

---

## 方式 A（推荐，最快）：用 PAT 一键推送

1. 在 GitHub 生成一个 Personal Access Token：
   - 经典令牌：勾选 `repo`（全部）；或
   - Fine-grained：对该账号授予 `Administration` + `Contents` 写权限
2. 在本仓库目录打开终端，运行：

   ```bat
   push_to_github.bat <你的PAT>
   ```

   脚本会：① 用 API 创建私有仓库（已存在则跳过）② 配置 remote ③ 推送 `main` 分支。
   脚本不会在屏幕上回显 PAT，git 推送时也会自动隐藏凭证。

## 方式 B：手动创建仓库后推送

1. 打开 https://github.com/new ，仓库名填 `RLLM-Rebirth-LLM`，**勾选 Private**，不要初始化 README。
2. 在本仓库目录打开终端，运行：

   ```bat
   git remote add origin https://github.com/Xiucheng0623/RLLM-Rebirth-LLM.git
   git branch -M main
   git push -u origin main
   ```

   若提示认证，按提示用 GitHub 账号密码 / PAT 登录即可（Windows 可用 Git Credential Manager）。

---

## 上传后建议

- 在仓库 Settings → Manage access 中按需添加协作者（保持私有）。
- 模型权重、缓存、运行日志**不会**进入版本库（已被 `.gitignore` 排除）。
- 如需公开部分内容再单独调整。
