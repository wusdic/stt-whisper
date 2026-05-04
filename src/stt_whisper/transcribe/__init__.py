"""
stt-whisper 转写模块
提供语音转文字功能
"""

from .stt_whisper import (
    WhisperTranscriber,
    TranscribeResult,
    Segment,
    transcribe,
)
from .speaker_diarization import (
    SpeakerDiarizer,
    SpeakerTurn,
    SimplerDiarizer,
    create_diarizer,
)
from .audio_utils import (
    AudioConverter,
    AudioInfo,
    AudioConverter,
    get_audio_info,
    convert_to_wav,
    split_audio,
)

__all__ = [
    # 转写核心
    "WhisperTranscriber",
    "TranscribeResult",
    "Segment",
    "transcribe",
    # 说话人区分
    "SpeakerDiarizer",
    "SpeakerTurn",
    "SimplerDiarizer",
    "create_diarizer",
    # 音频工具
    "AudioConverter",
    "AudioInfo",
    "get_audio_info",
    "convert_to_wav",
    "split_audio",
]
