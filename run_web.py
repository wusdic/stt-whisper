#!/usr/bin/env python3
"""
stt-whisper Web Interface Launcher
Usage: python run_web.py [--port PORT] [--host HOST]
"""
import sys
import argparse
from pathlib import Path

# 添加 src 到 path
SRC_DIR = Path(__file__).parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# 设置 HuggingFace 镜像（解决网络不通问题）
import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")


def main():
    parser = argparse.ArgumentParser(description="stt-whisper Web Interface")
    parser.add_argument("--port", type=int, default=8501, help="Port to run on (default: 8501)")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to (default: 0.0.0.0)")
    parser.add_argument("--browser", action="store_true", default=True, help="Open browser automatically")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    args = parser.parse_args()

    import streamlit.web.cli as stcli

    web_dir = Path(__file__).parent / "src" / "stt_whisper" / "web"
    app_path = web_dir / "app.py"

    sys.argv = [
        "streamlit",
        "run",
        str(app_path),
        "--server.port", str(args.port),
        "--server.address", args.host,
        "--browser", "false" if not args.browser else "true",
        "--server.headless", "true",
    ]

    if args.debug:
        sys.argv.append("--server.development")

    print(f"\n🎙️  stt-whisper Web Interface")
    print(f"   启动中: http://{args.host}:{args.port}")
    print(f"   按 Ctrl+C 停止\n")

    stcli.main()


if __name__ == "__main__":
    main()
