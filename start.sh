#!/bin/bash
# stt-whisper 一键启动脚本
# 使用方法：双击此文件，或在终端运行 ./start.sh

cd "$(dirname "$0")"

# 设置 HuggingFace 镜像（解决网络不通问题）
export HF_ENDPOINT="https://hf-mirror.com"

# 启动 Streamlit（后台运行）
nohup python3 -m streamlit run src/stt_whisper/web/app.py \
    --server.port 8501 \
    --server.address 127.0.0.1 \
    --server.headless true \
    > app.log 2>&1 &

PID=$!
echo "✅ stt-whisper 已启动 (PID: $PID)"
echo "📍 访问地址: http://127.0.0.1:8501"
echo "📝 日志文件: $(pwd)/app.log"
echo ""
echo "按 Enter 关闭此窗口（服务仍在后台运行）"
read
