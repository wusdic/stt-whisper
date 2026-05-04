# stt-whisper

语音转写文字系统 — 将录音文件转写为带说话人区分的文字，提取背景信息，输出结构化训练语料。

## 特性

- 🎤 **语音转写** — 基于 OpenAI Whisper / faster-whisper，支持中文英文等多语言
- 👥 **说话人区分** — 支持 pyannote.audio 说话人识别，可回退到简化版
- 🧹 **后处理治理** — 自动清理填充词、噪声标记、口语化表达
- 📊 **质量评分** — 多维度评估转写质量（完整性/清洁度/有效性）
- 📝 **多格式输出** — 支持 Dialogue / Utterances / Plain Text 三种训练数据格式
- ⚡ **Pipeline 流水线** — 一键完成从音频到训练语料的全流程

## 安装

```bash
# 克隆项目
git clone <repo-url>
cd stt-whisper

# 创建虚拟环境
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .\.venv\Scripts\activate  # Windows

# 安装依赖
pip install -e .
```

### 依赖

- `faster-whisper` 或 `openai-whisper` — Whisper 模型
- `pyannote.audio` — 说话人区分（可选）
- `pydub`, `ffmpeg` — 音频处理
- `pyyaml` — 配置管理
- `pytest` — 测试

## 快速开始

### Python API

```python
from stt_whisper import process, STTPipeline, PipelineConfig

# 一行代码处理
result = process("meeting.mp3", background="产品规划会议")
print(f"输出: {result.output_path}")
print(f"质量: {result.quality_score}/100")
```

### 命令行

```bash
# 单文件处理
python -m stt_whisper.pipeline --input ./audio.mp3 --background "会议讨论"

# 批量处理
python -m stt_whisper.pipeline --input ./data/raw/ --output ./data/curated/
```

### 独立使用各模块

```python
# 仅转写
from stt_whisper import WhisperTranscriber
transcriber = WhisperTranscriber(model="base", language="zh")
result = transcriber.transcribe("audio.mp3", background="背景信息")

# 仅清理
from stt_whisper import TextCleaner
cleaner = TextCleaner()
cleaned = cleaner.clean_segments(segments)

# 仅格式化
from stt_whisper import to_dialogue
dialogue = to_dialogue(segments, background="会议", source_file="audio.mp3")
```

## 项目结构

```
stt-whisper/
├── docs/
│   └── SPEC.md              # 详细规格文档
├── src/
│   └── stt_whisper/        # 主包
│       ├── __init__.py
│       ├── transcribe/          # 转写模块
│       │   ├── stt_whisper.py       # Whisper 转写核心
│       │   ├── speaker_diarization.py  # 说话人区分
│       │   └── audio_utils.py        # 音频处理工具
│       ├── postprocess/         # 后处理模块
│       │   ├── text_cleaner.py       # 文字清理
│       │   ├── formatter.py          # 格式化输出
│       │   └── quality_score.py      # 质量评分
│       └── pipeline.py          # 流水线
├── config/
│   └── default.yaml         # 默认配置
├── tests/
│   ├── test_transcribe.py
│   └── test_postprocess.py
└── README.md
```

## 配置

编辑 `config/default.yaml` 或在代码中传入配置：

```python
from stt_whisper import PipelineConfig

config = PipelineConfig(
    model="large-v3",
    language="zh",
    enable_diarization=True,
    output_format="dialogue"
)

pipeline = STTPipeline(config)
result = pipeline.process_single("audio.mp3", background="会议讨论")
```

## 输出格式

### Dialogue（对话格式）

```json
{
  "conversations": [
    {"role": "user", "content": "大家好，今天我们讨论..."},
    {"role": "assistant", "content": "关于这个议题..."}
  ],
  "metadata": {
    "background": "产品规划会议",
    "source_file": "meeting.mp3"
  }
}
```

### Utterances（片段格式）

```json
{
  "segments": [
    {"speaker": "SPEAKER_00", "text": "...", "start": 0.0, "end": 5.0},
    {"speaker": "SPEAKER_01", "text": "...", "start": 5.0, "end": 10.0}
  ],
  "background": "产品规划会议",
  "source_file": "meeting.mp3"
}
```

## 测试

```bash
# 运行所有测试
pytest tests/ -v

# 运行转写模块测试
pytest tests/test_transcribe.py -v

# 运行后处理模块测试
pytest tests/test_postprocess.py -v
```

## 规格

详见 [docs/SPEC.md](docs/SPEC.md)

## License

MIT
