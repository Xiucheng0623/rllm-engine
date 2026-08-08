@echo off
chcp 65001 >nul

REM ============================================================
REM File: D:\AI_RLLM\init_env.bat
REM 功能：Rebirth LLM(RLLM) 一键D盘环境初始化脚本
REM 约束：全部资源隔离至D:\AI_RLLM，零C盘占用
REM ============================================================

setlocal enabledelayedexpansion

echo ============================================================
echo [Rebirth LLM(RLLM)] Rebirth LLM(RLLM) D盘全隔离环境初始化
echo 硬件约束：RTX5070Ti / 32GB内存 / RLLM CPU缓冲区硬限2GB
echo ============================================================
echo.

REM ========== Step1: 强制校验D盘可用空间 ==========
echo [1/7] 检测D盘可用空间...
for /f "tokens=3" %%a in ('dir d:\ ^| find "可用字节"') do set FREE_BYTES=%%a
if "%FREE_BYTES%"=="" (
    for /f "usebackq delims=" %%a in (`powershell -NoProfile -Command "(Get-PSDrive D).Free"`) do set FREE_BYTES=%%a
)
set /a FREE_GB=%FREE_BYTES:~0,-9%
echo D盘可用空间: %FREE_GB% GB
if %FREE_GB% LSS 200 (
    echo [警告] D盘空间不足200GB，大模型推理可能失败，建议至少预留300GB
    pause
)

REM ========== Step2: 创建D盘全套目录 ==========
echo.
echo [2/7] 创建D盘全套隔离目录...
set ROOT=D:\AI_RLLM
set DIRS=!ROOT!\.venv;!ROOT!\hf_cache;!ROOT!\hf_cache\hub;!ROOT!\hf_cache\datasets
set DIRS=%DIRS%;!ROOT!\model_shards;!ROOT!\model_shards\indexes
set DIRS=%DIRS%;!ROOT!\offload_temp;!ROOT!\offload_temp\kv_cache;!ROOT!\offload_temp\tensor_swap
set DIRS=%DIRS%;!ROOT!\hermes_core;!ROOT!\hermes_core\workers;!ROOT!\hermes_core\memory
set DIRS=%DIRS%;!ROOT!\hermes_core\skills;!ROOT!\hermes_core\review;!ROOT!\hermes_core\config
set DIRS=%DIRS%;!ROOT!\output_dataset;!ROOT!\skill_storage;!ROOT!\skill_storage\archive
set DIRS=%DIRS%;!ROOT!\disk_engine;!ROOT!\disk_engine\sharding;!ROOT!\disk_engine\scheduler
set DIRS=%DIRS%;!ROOT!\disk_engine\kv_manager;!ROOT!\disk_engine\memory_lock;!ROOT!\disk_engine\mmap_io
set DIRS=%DIRS%;!ROOT!\auto_evo;!ROOT!\auto_evo\metrics;!ROOT!\auto_evo\tuner;!ROOT!\auto_evo\strategy
set DIRS=%DIRS%;!ROOT!\pipeline;!ROOT!\pipeline\batch_reader;!ROOT!\pipeline\writer;!ROOT!\pipeline\checkpoint
set DIRS=%DIRS%;!ROOT!\input_data;!ROOT!\logs;!ROOT!\tests

for %%d in (%DIRS%) do (
    if not exist "%%d" (
        mkdir "%%d" 2>nul
        echo   创建: %%d
    )
)
echo D盘目录结构就绪

REM ========== Step3: 配置全局D盘环境变量 ==========
echo.
echo [3/7] 强制配置HuggingFace/Torch D盘缓存路径...
REM 永久用户环境变量配置
setx HF_HOME "D:\AI_RLLM\hf_cache" >nul
setx TRANSFORMERS_CACHE "D:\AI_RLLM\hf_cache\hub" >nul
setx HUGGINGFACE_HUB_CACHE "D:\AI_RLLM\hf_cache\hub" >nul
setx TORCH_HOME "D:\AI_RLLM\hf_cache\torch" >nul
setx HF_DATASETS_CACHE "D:\AI_RLLM\hf_cache\datasets" >nul
setx HF_OFFLINE "1" >nul
setx DISABLE_MLFLOW_INTEGRATION "TRUE" >nul

REM 当前会话即时生效
set HF_HOME=D:\AI_RLLM\hf_cache
set TRANSFORMERS_CACHE=D:\AI_RLLM\hf_cache\hub
set HUGGINGFACE_HUB_CACHE=D:\AI_RLLM\hf_cache\hub
set TORCH_HOME=D:\AI_RLLM\hf_cache\torch
set HF_DATASETS_CACHE=D:\AI_RLLM\hf_cache\datasets
set HF_OFFLINE=1

echo   HF_HOME              = %HF_HOME%
echo   TRANSFORMERS_CACHE   = %TRANSFORMERS_CACHE%
echo   TORCH_HOME           = %TORCH_HOME%
echo   HF_OFFLINE           = 1 (离线模式已启用，禁止自动下载模型)
echo 环境变量配置完成

REM ========== Step4: 创建D盘独立Python虚拟环境 ==========
echo.
echo [4/7] 创建D盘独立Python 3.10虚拟环境...
set VENV_DIR=D:\AI_RLLM\.venv

where python >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo [错误] 未检测到Python，请先安装Python 3.10到系统后重试
    pause
    exit /b 1
)

REM 检测Python版本
for /f "tokens=2 delims= " %%v in ('python --version 2^>^&1') do set PY_VER=%%v
echo 系统Python版本: %PY_VER%
echo %PY_VER% | findstr /b "3.10" >nul
if %ERRORLEVEL% NEQ 0 (
    echo [警告] 非Python 3.10版本，可能存在兼容性问题，建议使用Python 3.10.x
)

if not exist "!VENV_DIR!\Scripts\python.exe" (
    echo 正在创建虚拟环境至 !VENV_DIR! ...
    python -m venv "!VENV_DIR!"
    if errorlevel 1 (
        echo [错误] 虚拟环境创建失败
        pause
        exit /b 1
    )
    echo 虚拟环境创建成功
) else (
    echo 虚拟环境已存在，跳过创建
)

REM ========== Step5: 安装D盘PyTorch + CUDA + 全套依赖 ==========
echo.
echo [5/7] 安装PyTorch 2.4 (CUDA 11.8/12.4兼容) + 全部依赖...
call "!VENV_DIR!\Scripts\activate.bat"

REM 升级pip至D盘虚拟环境内
python -m pip install --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple

REM 先尝试CUDA12.4，失败则降级CUDA11.8
echo 尝试安装 CUDA 12.4 版 PyTorch 2.4 ...
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu124 -i https://pypi.tuna.tsinghua.edu.cn/simple
if errorlevel 1 (
    echo [CUDA12.4失败] 降级尝试 CUDA 11.8 版 PyTorch...
    pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu118 -i https://pypi.tuna.tsinghua.edu.cn/simple
    if errorlevel 1 (
        echo [错误] PyTorch安装失败，请检查网络/CUDA驱动
        pause
        exit /b 1
    )
)

REM 安装核心推理/Hermes依赖
echo 安装 transformers accelerate bitsandbytes ...
pip install transformers==4.44.0 accelerate==0.33.0 bitsandbytes==0.43.2 -i https://pypi.tuna.tsinghua.edu.cn/simple
if errorlevel 1 (
    echo [警告] bitsandbytes Windows编译版可能缺失，尝试CPU兼容模式
    pip install transformers==4.44.0 accelerate==0.33.0 -i https://pypi.tuna.tsinghua.edu.cn/simple
)

REM 磁盘IO依赖
echo 安装 aiofiles diskcache pywin32 psutil ...
pip install aiofiles==24.1.0 diskcache==5.6.3 pywin32==306 psutil==6.0.0 -i https://pypi.tuna.tsinghua.edu.cn/simple

REM Hermes智能体依赖
echo 安装 Hermes-Agent 核心依赖 ...
pip install pydantic==2.8.2 pyyaml==6.0.2 rich==13.7.1 typer==0.12.3 aiohttp==3.10.3 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install numpy==1.26.4 pandas==2.2.2 jsonlines==4.0.0 -i https://pypi.tuna.tsinghua.edu.cn/simple

REM 锁定版本导出requirements
pip freeze > "D:\AI_RLLM\requirements.txt"
echo 依赖安装完成，版本清单已写入 requirements.txt

REM ========== Step6: 部署RLLM-Agent源码骨架 (离线模式) ==========
echo.
echo [6/7] 部署 Rebirth LLM(RLLM) 智能体源码骨架 (离线模式，无git依赖)...
set HERMES_DIR=D:\AI_RLLM\rllm_agent_core

REM 1) 写临时Python初始化脚本（避免python -c多行嵌套引号的问题）
set PY_INIT=%TEMP%\_rllm_agent_init_%RANDOM%.py
(
    echo import os, sys
    echo sys.path.insert(0, r'D:\AI_RLLM')
    echo hermes_files = ['__init__.py','workers/__init__.py','memory/__init__.py','skills/__init__.py','review/__init__.py','config/__init__.py']
    echo for f in hermes_files:
    echo     fp = os.path.join(r'D:\AI_RLLM\rllm_agent_core', f)
    echo     if not os.path.exists(fp):
    echo         os.makedirs(os.path.dirname(fp), exist_ok=True)
    echo         with open(fp, 'w', encoding='utf-8') as hf:
    echo             hf.write('# Rebirth LLM(RLLM) - based on Nous Hermes-Agent open-source framework\n')
    echo print('RLLM智能体骨架就绪: 共', len(hermes_files), '个模块')
) > "%PY_INIT%"

REM 2) 执行临时脚本
if exist "%PY_INIT%" (
    python "%PY_INIT%"
    del /F /Q "%PY_INIT%" >nul 2>nul
) else (
    echo [警告] 临时脚本生成失败，跳过骨架初始化（目录已在Step2创建则不影响）
)

REM ========== Step7: 运行环境校验 ==========
echo.
echo [7/7] 运行D盘环境校验...
if exist "D:\AI_RLLM\validate_env_RLLM.py" (
    python "D:\AI_RLLM\validate_env_RLLM.py"
) else (
    echo Rebirth LLM(RLLM) 环境校验脚本待生成，执行完毕后请手动运行 validate_env_RLLM.py
)

echo.
echo ============================================================
echo [完成] Rebirth LLM(RLLM) Rebirth LLM(RLLM) D盘全隔离环境初始化完成
echo RLLM 虚拟环境: D:\AI_RLLM\.venv
echo RLLM 激活命令: call D:\AI_RLLM\.venv\Scripts\activate.bat
echo RLLM 模型放置: 手动拷贝模型至 D:\AI_RLLM\rllm_model_shards
echo ============================================================
endlocal
pause


REM ============================================================
REM 版权声明
REM 本项目 Rebirth LLM(RLLM) 基于开源项目 Nous Hermes-Agent（MIT License）二次深度开发，项目内保留完整原始开源协议文件；智能体自迭代调度逻辑复用开源代码，磁盘分层加载、全局内存锁、D盘隔离部署、自动IO调优模块为自研闭源模块，分发时附带完整MIT协议文件。
REM 商标隔离免责声明
REM 项目名称 Rebirth LLM（简称RLLM）为独立软件项目代号，与奢侈品品牌Hermes、开源项目Hermes-Agent无品牌合作、隶属关联；仅代码内部功能性调用开源框架，不会使用Hermes相关名称开展商业宣传，无品牌混淆意图。
REM ============================================================
