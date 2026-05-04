"""
stt-whisper 后处理模块
提供文字清理、格式化、质量评分功能
"""

from .text_cleaner import (
    TextCleaner,
    AdvancedCleaner,
    CleaningResult,
    clean_text,
    clean_segments,
)
from .formatter import (
    DialogueFormatter,
    UtteranceFormatter,
    PlainTextFormatter,
    FormatterFactory,
    to_dialogue,
    to_utterances,
    to_plain_text,
)
from .quality_score import (
    QualityScorer,
    QualityScore,
    score_quality,
)

__all__ = [
    # 文字清理
    "TextCleaner",
    "AdvancedCleaner",
    "CleaningResult",
    "clean_text",
    "clean_segments",
    # 格式化
    "DialogueFormatter",
    "UtteranceFormatter",
    "PlainTextFormatter",
    "FormatterFactory",
    "to_dialogue",
    "to_utterances",
    "to_plain_text",
    # 质量评分
    "QualityScorer",
    "QualityScore",
    "score_quality",
]
