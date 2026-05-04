@echo off
REM stt-whisper 一键启动脚本 (Windows)
REM 使用方法：双击此文件

cd /d "%~dp0"

echo Setting HuggingFace mirror...
set HF_ENDPOINT=https://hf-mirror.com

echo Starting stt-whisper...
start /b python -m streamlit run src\stt_whisper\web\app.py --server.port 8501 --server.address 127.0.0.1 --server.headless

echo.
echo ✅ stt-whisper 已启动！
echo 📍 访问地址: http://127.0.0.1:8501
echo.
pause
