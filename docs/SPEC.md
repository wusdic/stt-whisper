# stt-whisper 项目规格书

> 语音转写文字 - 用于模型训练的语料治理系统

---

## 1. 项目概述

### 目标
将录音文件转写为带说话人区分的文字，提取背景信息，输出结构化语料，用于大模型训练。

### 核心流程
```
原始录音 → 语音转写(区分说话人) → 后处理治理 → 训练语料
```

---

## 2. 模块划分

### 模块 A：stt-transcribe（语音转写）

#### 2.1 录音文件读取
- 支持格式：`mp3`, `wav`, `m4a`, `ogg`, `flac`, `mp4`, `avi`
- 自动检测采样率、时长
- 校验文件完整性

#### 2.2 语音转写（Whisper）
- 使用 OpenAI Whisper API 或本地 Whisper 模型
- **说话人区分（Diarization）**
  - 标注说话人标签：`SPEAKER_00`, `SPEAKER_01`, ...
  - 记录每段话的开始/结束时间
- **背景信息**（可选）
  - 批量录音可填写"总体背景"
  - 单个录音可填写"独立背景"
  - 未填写时使用默认背景或"通用背景"
  - 背景作为元数据保存，影响转写准确性

#### 2.3 输出格式
```json
{
  "file": "录音文件名",
  "duration_seconds": 360.5,
  "background": "会议讨论关于Q2产品规划...",
  "segments": [
    {
      "speaker": "SPEAKER_00",
      "start": 0.0,
      "end": 15.3,
      "text": "大家好，今天我们讨论..."
    }
  ],
  "full_text": "SPEAKER_00: 大家好...\nSPEAKER_01: 关于这个..."
}
```

---

### 模块 B：stt-curate（语料治理/后处理）

#### 2.4 文字后处理
- 标点符号规范化
- 口语化表达转书面语（如"嗯嗯"→删除）
- 重复语句合并
- 敏感词过滤（可选）

#### 2.5 训练数据格式化
根据训练需求输出不同格式：

**格式 1：对话格式（Dialogue）**
```
{
  "conversations": [
    {"role": "user", "content": "大家好，今天我们讨论..."},
    {"role": "assistant", "content": "关于这个议题..."}
  ],
  "metadata": {
    "background": "...",
    "source_file": "..."
  }
}
```

**格式 2：片段格式（Utterances）**
```
{
  "segments": [
    {"speaker": "SPEAKER_00", "text": "..."},
    {"speaker": "SPEAKER_01", "text": "..."}
  ],
  "background": "..."
}
```

**格式 3：纯文本格式（Plain Text）**
```
每行一段话，说话人标签可选保留
```

#### 2.6 数据质量评分
- 转写置信度评估
- 有效内容比例
- 标注完整性检查

---

## 3. 目录结构

```
stt-whisper/
├── docs/
│   ├── SPEC.md          # 本规格文档
│   └── API.md           # 接口文档（待定）
├── src/
│   ├── transcribe/      # 模块A：语音转写
│   │   ├── __init__.py
│   │   ├── stt_whisper.py
│   │   ├── speaker_diarization.py
│   │   └── audio_utils.py
│   └── postprocess/     # 模块B：后处理治理
│       ├── __init__.py
│       ├── text_cleaner.py
│       ├── formatter.py
│       └── quality_score.py
├── config/
│   ├── default.yaml     # 默认配置
│   └── Whisper/         # Whisper模型配置
├── tests/
│   ├── test_transcribe.py
│   └── test_postprocess.py
├── data/
│   ├── raw/             # 原始录音文件
│   ├── transcribed/     # 转写结果(JSON)
│   └── curated/         # 治理后训练语料
└── outputs/             # 最终交付产物
```

---

## 4. 配置项

```yaml
# config/default.yaml
stt:
  model: "large-v3"              # Whisper模型大小
  language: "zh"                # 语言，null=自动检测
  device: "cuda"                # cuda 或 cpu

diarization:
  enabled: true                  # 是否启用说话人区分
  min_speakers: 1
  max_speakers: 10

background:
  use_batch_background: true     # 优先使用批量背景
  default: "通用对话"            # 未提供背景时的默认值

curation:
  remove_filler_words: true      # 删除填充词（嗯、啊、呃）
  normalize_punctuation: true     # 标点规范化
  output_format: "dialogue"      # dialogue | utterances | plain
```

---

## 5. 使用方式

### 命令行接口
```bash
# 转写
python -m stt_whisper.transcribe --input ./data/raw/ --background "会议讨论..."

# 后处理治理
python -m stt_whisper.postprocess --input ./data/transcribed/ --format dialogue

# 一键处理
python -m stt_whisper.pipeline --input ./data/raw/ --background "..." --format dialogue
```

### Python API
```python
from stt_whisper import Transcriber, Curator

# 转写
transcriber = Transcriber()
result = transcriber.transcribe("audio.mp3", background="会议讨论...")

# 治理
curator = Curator()
curated = curator.process(result, format="dialogue")
```

---

## 6. 依赖

- `openai-whisper` / `faster-whisper`
- `pyannote.audio`（说话人区分）
- `pydub` / `ffmpeg`（音频处理）
- `湖蓝` 或 `等`（配置管理）
- `pytest`（测试）

---

## 7. 开发进度

### ✅ 已完成
- [x] 说话人区分模块（Diarization）
- [x] Whisper转写核心
- [x] 音频工具（格式转换、切割）
- [x] 文字清理器
- [x] 多格式输出器
- [x] 质量评分
- [x] Pipeline 批量流水线
- [x] 单元测试（55个测试全部通过）
- [x] 配置文件
- [x] 可安装包（pip install -e .）
- [x] Dataclass兼容（Segment/TranscribeResult 与 formatters/cleaner/scorer 完全兼容）

### ⏳ 待完善
- [ ] Web界面（可选）
