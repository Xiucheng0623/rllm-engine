@echo off
chcp 65001 >nul
:: ==========================================================
::  RLLM Engine v1.0 - 一键安装脚本
::  47B MoE 模型在 8GB 显卡上运行
:: ==========================================================
setlocal enabledelayedexpansion

echo.
echo ############################################################
echo #  RLLM Engine v1.0 安装
echo #  专家级磁盘分页推理引擎
echo ############################################################
echo.

:: 1. 检查 D 盘
if not exist "D:\" (
    echo [ERR] D 盘不存在, 请确保有 D 盘分区
    pause
    exit /b 1
)

:: 2. 创建目录结构
echo [1/5] 创建目录结构...

set RLLM_ROOT=D:\AI_RLLM

mkdir "%RLLM_ROOT%" 2>nul
mkdir "%RLLM_ROOT%\rllm_engine" 2>nul
mkdir "%RLLM_ROOT%\rllm_model_shards" 2>nul
mkdir "%RLLM_ROOT%\hf_cache" 2>nul
mkdir "%RLLM_ROOT%\logs" 2>nul
mkdir "%RLLM_ROOT%\offload_temp" 2>nul
mkdir "%RLLM_ROOT%\input_data" 2>nul
mkdir "%RLLM_ROOT%\output_dataset" 2>nul

echo   目录结构创建完成

:: 3. 设置环境变量
echo [2/5] 设置环境变量...

setx HF_HOME "%RLLM_ROOT%\hf_cache" >nul 2>&1
setx TRANSFORMERS_CACHE "%RLLM_ROOT%\hf_cache\transformers" >nul 2>&1
setx TORCH_HOME "%RLLM_ROOT%\hf_cache\torch" >nul 2>&1

:: 设置当前会话的环境变量
set HF_HOME=%RLLM_ROOT%\hf_cache
set TRANSFORMERS_CACHE=%RLLM_ROOT%\hf_cache\transformers
set TORCH_HOME=%RLLM_ROOT%\hf_cache\torch

echo   HF_HOME=%HF_HOME%
echo   TRANSFORMERS_CACHE=%TRANSFORMERS_CACHE%

:: 4. 创建/使用虚拟环境
echo [3/5] 配置 Python 虚拟环境...

set VENV_DIR=%RLLM_ROOT%\.venv
set VENV_PYTHON=%VENV_DIR%\Scripts\python.exe

if exist "%VENV_PYTHON%" (
    echo   虚拟环境已存在: %VENV_DIR%
) else (
    echo   创建虚拟环境...
    python -m venv "%VENV_DIR%"
    if errorlevel 1 (
        echo [ERR] 创建虚拟环境失败, 请安装 Python 3.10
        pause
        exit /b 1
    )
)

:: 升级 pip
echo   升级 pip...
call "%VENV_PYTHON%" -m pip install --upgrade pip --quiet

:: 5. 安装依赖
echo [4/5] 安装依赖...

:: 先安装 PyTorch CUDA 版
echo   安装 PyTorch 2.4 CUDA...
call "%VENV_PYTHON%" -m pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu118 --quiet

if errorlevel 1 (
    echo [WARN] CUDA 11.8 PyTorch 安装失败, 尝试 CUDA 12.4...
    call "%VENV_PYTHON%" -m pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu124 --quiet
)

:: 安装其他依赖
echo   安装其他依赖...
call "%VENV_PYTHON%" -m pip install -r "%RLLM_ROOT%\requirements.txt" --quiet

if errorlevel 1 (
    echo [WARN] 部分依赖安装失败, 尝试逐个安装...
    call "%VENV_PYTHON%" -m pip install transformers accelerate bitsandbytes aiofiles diskcache safetensors numpy loguru tqdm click --quiet
)

:: 6. 安装 RLLM Engine 包
echo [5/5] 安装 RLLM Engine...
call "%VENV_PYTHON%" -m pip install -e "%RLLM_ROOT%" --quiet

:: 验证
echo.
echo 验证安装...
call "%VENV_PYTHON%" -c "import torch; print(f'PyTorch {torch.__version__} CUDA={torch.cuda.is_available()}')"
call "%VENV_PYTHON%" -c "from rllm_engine import RLLMEngine; print('RLLM Engine 导入成功')"

echo.
echo ############################################################
echo #  RLLM Engine v1.0 安装完成!
echo #
echo #  下一步:
echo #  1. python rllm_manager.py download mixtral-8x7b
echo #  2. python rllm_manager.py shard mixtral-8x7b
echo #  3. python rllm_chat.py
echo ############################################################

endlocal
pause
