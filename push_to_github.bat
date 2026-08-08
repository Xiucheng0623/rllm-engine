@echo off
chcp 65001 >nul
REM ============================================================
REM  RLLM 一键推送到 GitHub（私有仓库）
REM  用法:  push_to_github.bat <GITHUB_PAT>
REM
REM  说明:
REM   1) 通过 GitHub API 用 PAT 创建私有仓库（已存在则跳过）
REM   2) 配置 remote 并推送本地 main 分支
REM  需要的 PAT 权限: 经典令牌需勾选 "repo" 全部；
REM                    或 Fine-grained 令牌对该账号有 "Administration + Contents" 写权限
REM  本脚本不会在屏幕上回显你的 PAT（git 推送时会自动隐藏凭证）
REM ============================================================

set "PAT=%~1"
if "%PAT%"=="" (
  echo [错误] 缺少参数。用法: push_to_github.bat ^<GITHUB_PAT^>
  exit /b 1
)

set "OWNER=Xiucheng0623"
set "REPO=RLLM-Rebirth-LLM"

echo [1/3] 通过 GitHub API 创建私有仓库（若已存在则忽略）...
curl -s -o "_create_repo_resp.txt" -w "HTTP %%{http_code}" ^
  -X POST ^
  -H "Authorization: token %PAT%" ^
  -H "Content-Type: application/json" ^
  -d "{\"name\":\"%REPO%\",\"private\":true,\"description\":\"Rebirth LLM (RLLM) - 磁盘卸载低内存大模型离线批量生成引擎（私有）。商业合作：13504091457\"}" ^
  https://api.github.com/user/repos
echo.
findstr /i "already_exists" "_create_repo_resp.txt" >nul && echo   仓库已存在，继续推送... || echo   （若上方提示 422/already_exists 属正常，仓库已存在）
del "_create_repo_resp.txt" 2>nul

echo [2/3] 配置 remote（含 PAT，git 会自动隐藏凭证）...
git remote remove origin >nul 2>nul
git remote add origin https://%PAT%@github.com/%OWNER%/%REPO%.git
git branch -M main

echo [3/3] 推送本地 main 分支...
git push -u origin main
echo.
echo 完成。私有仓库地址: https://github.com/%OWNER%/%REPO%
pause
