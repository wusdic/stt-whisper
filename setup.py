"""
stt-whisper 安装配置
"""

from setuptools import setup, find_packages
from pathlib import Path

# 读取 README
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

setup(
    name="stt-whisper",
    version="0.1.0",
    description="语音转写文字系统 - 用于模型训练的语料治理",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="",
    python_requires=">=3.9",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "pyyaml",
    ],
    extras_require={
        "whisper": ["openai-whisper"],
        "faster-whisper": ["faster-whisper"],
        "diarization": ["pyannote.audio"],
        "audio": ["pydub", "ffmpeg"],
        "dev": ["pytest", "pytest-cov"],
    },
    entry_points={
        "console_scripts": [
            "stt-whisper=stt_whisper.pipeline:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
    ],
)
