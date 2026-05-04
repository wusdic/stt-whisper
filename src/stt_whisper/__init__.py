"""
stt-whisper
语音转写文字系统 - 用于模型训练的语料治理
"""

from .transcribe import (
    WhisperTranscriber,
    TranscribeResult,
    Segment,
    transcribe,
    SpeakerDiarizer,
    create_diarizer,
    AudioConverter,
    get_audio_info,
)
from .postprocess import (
    TextCleaner,
    AdvancedCleaner,
    clean_text,
    clean_segments,
    DialogueFormatter,
    UtteranceFormatter,
    PlainTextFormatter,
    FormatterFactory,
    to_dialogue,
    to_utterances,
    to_plain_text,
    QualityScorer,
    score_quality,
)
from .pipeline import (
    STTPipeline,
    PipelineConfig,
    PipelineResult,
    create_pipeline,
    process,
)

__all__ = [
    # Pipeline
    "STTPipeline",
    "PipelineConfig",
    "PipelineResult",
    "create_pipeline",
    "process",
    # Transcribe
    "WhisperTranscriber",
    "TranscribeResult",
    "Segment",
    "transcribe",
    "SpeakerDiarizer",
    "create_diarizer",
    "AudioConverter",
    "get_audio_info",
    # Postprocess
    "TextCleaner",
    "AdvancedCleaner",
    "clean_text",
    "clean_segments",
    "DialogueFormatter",
    "UtteranceFormatter",
    "PlainTextFormatter",
    "FormatterFactory",
    "to_dialogue",
    "to_utterances",
    "to_plain_text",
    "QualityScorer",
    "score_quality",
]

__version__ = "0.1.0"
