"""
测试后处理模块
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch
import sys

# 添加 src 目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# 尝试导入 stt_whisper 包
try:
    from stt_whisper.postprocess import (
        TextCleaner,
        AdvancedCleaner,
        CleaningResult,
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
        QualityScore,
        score_quality,
    )
except ModuleNotFoundError:
    # 如果 stt_whisper 未安装，直接从本地模块导入
    import stt_whisper.postprocess as postprocess
    import stt_whisper.transcribe as transcribe
    TextCleaner = postprocess.TextCleaner
    AdvancedCleaner = postprocess.AdvancedCleaner
    CleaningResult = postprocess.CleaningResult
    clean_text = postprocess.clean_text
    clean_segments = postprocess.clean_segments
    DialogueFormatter = postprocess.DialogueFormatter
    UtteranceFormatter = postprocess.UtteranceFormatter
    PlainTextFormatter = postprocess.PlainTextFormatter
    FormatterFactory = postprocess.FormatterFactory
    to_dialogue = postprocess.to_dialogue
    to_utterances = postprocess.to_utterances
    to_plain_text = postprocess.to_plain_text
    QualityScorer = postprocess.QualityScorer
    QualityScore = postprocess.QualityScore
    score_quality = postprocess.score_quality


# ===== TextCleaner 测试 =====

class TestTextCleaner:
    """测试 TextCleaner"""

    def setup_method(self):
        self.cleaner = TextCleaner()

    def test_clean_fillers(self):
        """测试填充词删除"""
        result = self.cleaner.clean_text("嗯嗯大家好")
        assert "嗯" not in result.cleaned
        assert "大家好" in result.cleaned

    def test_clean_repeated_words(self):
        """测试重复词清理"""
        result = self.cleaner.clean_text("这个这个这个很好")
        assert "这个" in result.cleaned
        # 重复的 "这个" 应该被合并

    def test_clean_noise_markers(self):
        """测试噪声标记删除"""
        result = self.cleaner.clean_text("[笑声] 哈哈哈 [掌声]")
        assert "[笑声]" not in result.cleaned
        assert "[掌声]" not in result.cleaned

    def test_normalize_punctuation(self):
        """测试标点规范化"""
        result = self.cleaner.clean_text("你好！！！")
        assert "!!!" not in result.cleaned  # 合并多余感叹号

    def test_strip_whitespace(self):
        """测试空白字符处理"""
        result = self.cleaner.clean_text("  你好  ")
        assert result.cleaned == "你好"

    def test_empty_result(self):
        """测试空输入"""
        result = self.cleaner.clean_text("")
        assert result.cleaned == ""
        assert len(result.removed_items) == 0

    def test_clean_segments(self):
        """测试批量清理"""
        segments = [
            {"text": "嗯嗯好的", "speaker": "SPEAKER_00", "start": 0.0, "end": 2.0},
            {"text": "很好", "speaker": "SPEAKER_01", "start": 2.0, "end": 4.0},
        ]
        cleaned = self.cleaner.clean_segments(segments)

        assert len(cleaned) >= 1
        assert cleaned[0]["is_cleaned"] == True
        assert "original_text" in cleaned[0]

    def test_clean_segments_filters_short(self):
        """测试过短片段过滤"""
        cleaner = TextCleaner(min_word_count=5)
        segments = [
            {"text": "啊", "speaker": "SPEAKER_00", "start": 0.0, "end": 1.0},
            {"text": "这是一段正常的话", "speaker": "SPEAKER_01", "start": 1.0, "end": 5.0},
        ]
        cleaned = cleaner.clean_segments(segments)
        # 过短的片段应该被过滤
        assert len(cleaned) <= len(segments)


class TestAdvancedCleaner:
    """测试 AdvancedCleaner"""

    def setup_method(self):
        self.cleaner = AdvancedCleaner()

    def test_colloquial_to_formal(self):
        """测试口语化转书面语"""
        result = self.cleaner.clean_text("干啥呢")
        # 应该转换为 "做什么"
        assert "干啥" not in result.cleaned

    def test_inherited_cleaning(self):
        """测试继承的清理功能"""
        result = self.cleaner.clean_text("嗯嗯干啥呢")
        assert "嗯" not in result.cleaned


# ===== Formatter 测试 =====

class TestDialogueFormatter:
    """测试 DialogueFormatter"""

    def setup_method(self):
        self.formatter = DialogueFormatter()

    def test_format_basic(self):
        """测试基本对话格式"""
        segments = [
            {"speaker": "SPEAKER_00", "start": 0.0, "end": 5.0, "text": "你好"},
            {"speaker": "SPEAKER_01", "start": 5.0, "end": 10.0, "text": "再见"},
        ]
        result = self.formatter.format(segments, background="测试", source_file="test.mp3")

        assert len(result.conversations) == 2
        assert result.metadata["background"] == "测试"
        assert result.metadata["source_file"] == "test.mp3"

    def test_format_odd_even_roles(self):
        """测试奇偶角色分配"""
        segments = [
            {"speaker": "SPEAKER_00", "start": 0.0, "end": 5.0, "text": "第一句"},
            {"speaker": "SPEAKER_01", "start": 5.0, "end": 10.0, "text": "第二句"},
            {"speaker": "SPEAKER_00", "start": 10.0, "end": 15.0, "text": "第三句"},
        ]
        result = self.formatter.format(segments)

        # 奇数=user, 偶数=assistant
        assert result.conversations[0].role == "user"
        assert result.conversations[1].role == "assistant"
        assert result.conversations[2].role == "user"

    def test_format_empty_segments(self):
        """测试空片段"""
        result = self.formatter.format([])
        assert len(result.conversations) == 0

    def test_save(self, tmp_path):
        """测试保存"""
        segments = [
            {"speaker": "SPEAKER_00", "start": 0.0, "end": 5.0, "text": "你好"},
        ]
        result = self.formatter.format(segments)

        output_path = tmp_path / "dialogue.json"
        self.formatter.save(result, output_path)

        assert output_path.exists()


class TestUtteranceFormatter:
    """测试 UtteranceFormatter"""

    def setup_method(self):
        self.formatter = UtteranceFormatter()

    def test_format_basic(self):
        """测试基本片段格式"""
        segments = [
            {"speaker": "SPEAKER_00", "start": 0.0, "end": 5.0, "text": "你好"},
            {"speaker": "SPEAKER_01", "start": 5.0, "end": 10.0, "text": "再见"},
        ]
        result = self.formatter.format(segments, background="测试", source_file="test.mp3")

        assert len(result.segments) == 2
        assert result.background == "测试"
        assert result.source_file == "test.mp3"

    def test_format_preserves_fields(self):
        """测试保留必要字段"""
        segments = [
            {"speaker": "SPEAKER_00", "start": 0.0, "end": 5.0, "text": "你好", "extra": "数据"},
        ]
        result = self.formatter.format(segments)

        assert "speaker" in result.segments[0]
        assert "text" in result.segments[0]
        assert "start" in result.segments[0]
        assert "end" in result.segments[0]


class TestPlainTextFormatter:
    """测试 PlainTextFormatter"""

    def setup_method(self):
        self.formatter = PlainTextFormatter()

    def test_format_with_speaker(self):
        """测试带说话人的纯文本"""
        segments = [
            {"speaker": "SPEAKER_00", "start": 0.0, "end": 5.0, "text": "你好"},
            {"speaker": "SPEAKER_01", "start": 5.0, "end": 10.0, "text": "再见"},
        ]
        text = self.formatter.format(segments)

        assert "SPEAKER_00: 你好" in text
        assert "SPEAKER_01: 再见" in text

    def test_format_without_speaker(self):
        """测试不带说话人的纯文本"""
        formatter = PlainTextFormatter(include_speaker=False)
        segments = [
            {"speaker": "SPEAKER_00", "start": 0.0, "end": 5.0, "text": "你好"},
        ]
        text = formatter.format(segments)

        assert "SPEAKER_00" not in text
        assert "你好" in text

    def test_format_with_background(self):
        """测试添加背景"""
        segments = [
            {"speaker": "SPEAKER_00", "start": 0.0, "end": 5.0, "text": "你好"},
        ]
        text = self.formatter.format(segments, background="会议讨论")

        assert "[背景: 会议讨论]" in text

    def test_save(self, tmp_path):
        """测试保存"""
        text = "SPEAKER_00: 你好\nSPEAKER_01: 再见"
        output_path = tmp_path / "plain.txt"

        self.formatter.save(text, output_path)

        assert output_path.exists()
        assert output_path.read_text(encoding="utf-8") == text


class TestFormatterFactory:
    """测试 FormatterFactory"""

    def test_create_dialogue(self):
        """测试创建对话格式化器"""
        formatter = FormatterFactory.create("dialogue")
        assert isinstance(formatter, DialogueFormatter)

    def test_create_utterances(self):
        """测试创建片段格式化器"""
        formatter = FormatterFactory.create("utterances")
        assert isinstance(formatter, UtteranceFormatter)

    def test_create_plain(self):
        """测试创建纯文本格式化器"""
        formatter = FormatterFactory.create("plain")
        assert isinstance(formatter, PlainTextFormatter)

    def test_create_invalid(self):
        """测试无效格式"""
        with pytest.raises(ValueError, match="不支持的格式"):
            FormatterFactory.create("invalid_format")


# ===== QualityScore 测试 =====

class TestQualityScore:
    """测试 QualityScore 数据类"""

    def test_creation(self):
        score = QualityScore(
            overall=85.0,
            completeness=90.0,
            cleanliness=80.0,
            validity=85.0,
            details={"completeness": 90.0}
        )
        assert score.overall == 85.0
        assert score.completeness == 90.0


class TestQualityScorer:
    """测试 QualityScorer"""

    def setup_method(self):
        self.scorer = QualityScorer()

    def test_score_empty_segments(self):
        """测试空片段评分"""
        score = self.scorer.score_segments([])
        assert score.overall == 0.0

    def test_score_good_segments(self):
        """测试良好片段评分"""
        segments = [
            {"speaker": "SPEAKER_00", "start": 0.0, "end": 5.0, "text": "这是一段正常的话用于测试"},
            {"speaker": "SPEAKER_01", "start": 5.0, "end": 10.0, "text": "另一段正常的内容用于质量评估"},
        ]
        score = self.scorer.score_segments(segments)
        assert score.overall > 0
        assert score.completeness > 0

    def test_score_low_filler_ratio(self):
        """测试低填充词比例得高分"""
        segments = [
            {"speaker": "SPEAKER_00", "start": 0.0, "end": 5.0, "text": "正常文本"},
        ]
        score = self.scorer.score_segments(segments)
        assert score.cleanliness >= 80.0  # 低填充词应该得高分

    def test_score_high_filler_ratio(self):
        """测试高填充词比例扣分"""
        segments = [
            {"speaker": "SPEAKER_00", "start": 0.0, "end": 5.0, "text": "嗯嗯啊啊嗯嗯啊啊啊"},
        ]
        score = self.scorer.score_segments(segments)
        assert score.cleanliness < 50.0  # 高填充词应该得低分

    def test_score_speaker_diversity(self):
        """测试说话人多样性评分"""
        segments = [
            {"speaker": "SPEAKER_00", "start": 0.0, "end": 5.0, "text": "第一人说的"},
            {"speaker": "SPEAKER_01", "start": 5.0, "end": 10.0, "text": "第二人说的"},
        ]
        score = self.scorer.score_segments(segments)
        assert score.details.get("speaker_diversity", 0) >= 80.0  # 多人对话应该得高分

    def test_get_recommendation(self):
        """测试建议生成"""
        # 优秀
        score = QualityScore(85.0, 90.0, 80.0, 85.0, {})
        assert "优秀" in self.scorer.get_recommendation(score)

        # 良好
        score = QualityScore(65.0, 70.0, 60.0, 65.0, {})
        assert "良好" in self.scorer.get_recommendation(score)

        # 一般
        score = QualityScore(45.0, 50.0, 40.0, 45.0, {})
        assert "一般" in self.scorer.get_recommendation(score)

        # 较差
        score = QualityScore(25.0, 30.0, 20.0, 25.0, {})
        assert "较差" in self.scorer.get_recommendation(score)


# ===== 便捷函数测试 =====

class TestConvenienceFunctions:
    """测试便捷函数"""

    def test_clean_text_function(self):
        result = clean_text("嗯嗯测试")
        assert "嗯" not in result

    def test_to_dialogue_function(self):
        segments = [
            {"speaker": "SPEAKER_00", "start": 0.0, "end": 5.0, "text": "你好"},
        ]
        result = to_dialogue(segments)
        assert len(result.conversations) == 1

    def test_to_utterances_function(self):
        segments = [
            {"speaker": "SPEAKER_00", "start": 0.0, "end": 5.0, "text": "你好"},
        ]
        result = to_utterances(segments)
        assert len(result.segments) == 1

    def test_to_plain_text_function(self):
        segments = [
            {"speaker": "SPEAKER_00", "start": 0.0, "end": 5.0, "text": "你好"},
        ]
        text = to_plain_text(segments)
        assert "你好" in text

    def test_score_quality_function(self):
        segments = [
            {"speaker": "SPEAKER_00", "start": 0.0, "end": 5.0, "text": "正常文本"},
        ]
        score = score_quality(segments)
        assert score.overall >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
