"""
测试转写模块
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# 添加 src 目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from stt_whisper.transcribe import (
    WhisperTranscriber,
    TranscribeResult,
    Segment,
    transcribe,
)
from stt_whisper.transcribe.speaker_diarization import (
    SpeakerDiarizer,
    SpeakerTurn,
    SimplerDiarizer,
    create_diarizer,
)
from stt_whisper.transcribe.audio_utils import (
    AudioConverter,
    AudioInfo,
    get_audio_info,
    convert_to_wav,
    split_audio,
)


class TestSegment:
    """测试 Segment 数据类"""

    def test_segment_creation(self):
        seg = Segment(speaker="SPEAKER_00", start=0.0, end=5.0, text="测试文本")
        assert seg.speaker == "SPEAKER_00"
        assert seg.start == 0.0
        assert seg.end == 5.0
        assert seg.text == "测试文本"

    def test_segment_to_dict(self):
        seg = Segment(speaker="SPEAKER_01", start=1.0, end=6.0, text="第二段")
        d = seg.__dict__
        assert d["speaker"] == "SPEAKER_01"
        assert d["text"] == "第二段"


class TestTranscribeResult:
    """测试 TranscribeResult 数据类"""

    def test_result_creation(self):
        segments = [
            Segment(speaker="SPEAKER_00", start=0.0, end=5.0, text="第一句"),
            Segment(speaker="SPEAKER_01", start=5.0, end=10.0, text="第二句"),
        ]
        result = TranscribeResult(
            file="test.mp3",
            duration_seconds=10.0,
            background="测试背景",
            segments=segments,
            language="zh",
            model="large-v3"
        )

        assert result.file == "test.mp3"
        assert result.duration_seconds == 10.0
        assert result.background == "测试背景"
        assert len(result.segments) == 2
        assert result.language == "zh"

    def test_full_text(self):
        segments = [
            Segment(speaker="SPEAKER_00", start=0.0, end=5.0, text="你好"),
            Segment(speaker="SPEAKER_01", start=5.0, end=10.0, text="再见"),
        ]
        result = TranscribeResult(segments=segments)
        full = result.full_text

        assert "SPEAKER_00: 你好" in full
        assert "SPEAKER_01: 再见" in full

    def test_to_dict(self):
        segments = [Segment(speaker="SPEAKER_00", start=0.0, end=5.0, text="测试")]
        result = TranscribeResult(
            file="test.mp3",
            duration_seconds=5.0,
            segments=segments
        )
        d = result.to_dict()

        assert d["file"] == "test.mp3"
        assert d["duration_seconds"] == 5.0
        assert len(d["segments"]) == 1

    def test_save_and_load(self, tmp_path):
        segments = [Segment(speaker="SPEAKER_00", start=0.0, end=5.0, text="测试保存")]
        result = TranscribeResult(
            file="test.mp3",
            duration_seconds=5.0,
            segments=segments
        )

        # 保存
        save_path = tmp_path / "result.json"
        result.save(save_path)
        assert save_path.exists()

        # 加载
        loaded = TranscribeResult.load(save_path)
        assert loaded.file == "test.mp3"
        assert len(loaded.segments) == 1
        assert loaded.segments[0].text == "测试保存"


class TestWhisperTranscriber:
    """测试 WhisperTranscriber"""

    def test_supported_formats(self):
        transcriber = WhisperTranscriber()
        expected = {".mp3", ".wav", ".m4a", ".ogg", ".flac", ".mp4", ".avi", ".mkv"}
        assert transcriber.SUPPORTED_FORMATS == expected

    def test_init_defaults(self):
        transcriber = WhisperTranscriber()
        assert transcriber.model_name == "large-v3"
        assert transcriber.language == "zh"
        assert transcriber.device == "cpu"

    def test_init_custom(self):
        transcriber = WhisperTranscriber(
            model="base",
            language="en",
            device="cuda"
        )
        assert transcriber.model_name == "base"
        assert transcriber.language == "en"
        assert transcriber.device == "cuda"

    def test_unsupported_format_raises(self):
        transcriber = WhisperTranscriber()
        # Create a temp file with unsupported extension
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            temp_path = Path(f.name)
        try:
            with pytest.raises(ValueError, match="不支持的格式"):
                transcriber.transcribe(temp_path)
        finally:
            temp_path.unlink(missing_ok=True)

    def test_file_not_found_raises(self):
        transcriber = WhisperTranscriber()
        with pytest.raises(FileNotFoundError):
            transcriber.transcribe(Path("/nonexistent/file.mp3"))


class TestSpeakerDiarizer:
    """测试 SpeakerDiarizer"""

    def test_speaker_turn(self):
        turn = SpeakerTurn(speaker="SPEAKER_00", start=0.0, end=5.0)
        assert turn.speaker == "SPEAKER_00"
        assert turn.start == 0.0
        assert turn.end == 5.0

    def test_diarizer_init(self):
        diarizer = SpeakerDiarizer(model="test-model", device="cpu")
        assert diarizer.model_name == "test-model"
        assert diarizer.device == "cpu"
        assert diarizer._pipeline is None  # 延迟加载


class TestSimplerDiarizer:
    """测试 SimplerDiarizer"""

    def test_init(self):
        diarizer = SimplerDiarizer(min_silence_len=0.5, speech_threshold=0.02)
        assert diarizer.min_silence_len == 0.5
        assert diarizer.speech_threshold == 0.02


class TestAudioInfo:
    """测试 AudioInfo 数据类"""

    def test_audio_info_creation(self):
        info = AudioInfo(
            path="/path/to/audio.mp3",
            duration=120.5,
            sample_rate=44100,
            channels=2,
            codec="mp3",
            bitrate="192k"
        )
        assert info.path == "/path/to/audio.mp3"
        assert info.duration == 120.5
        assert info.sample_rate == 44100
        assert info.channels == 2
        assert info.codec == "mp3"
        assert info.bitrate == "192k"

    def test_audio_info_repr(self):
        info = AudioInfo(path="/test.mp3", duration=60.0, sample_rate=16000, channels=1)
        r = repr(info)
        assert "/test.mp3" in r
        assert "60.0" in r
        assert "16000" in r


class TestAudioConverter:
    """测试 AudioConverter"""

    def test_init(self):
        converter = AudioConverter(ffmpeg_path="/usr/bin/ffmpeg", ffprobe_path="/usr/bin/ffprobe")
        assert converter.ffmpeg == "/usr/bin/ffmpeg"
        assert converter.ffprobe == "/usr/bin/ffprobe"

    def test_init_defaults(self):
        converter = AudioConverter()
        assert converter.ffmpeg == "ffmpeg"
        assert converter.ffprobe == "ffprobe"


class TestTranscribeFunction:
    """测试便捷转写函数"""

    def test_transcribe_function_with_mock(self):
        """测试便捷转写函数（使用真实临时文件）"""
        import tempfile
        from unittest.mock import patch

        # Create a temp file
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            temp_path = f.name

        try:
            with patch('stt_whisper.transcribe.stt_whisper.WhisperTranscriber') as mock_transcriber_class:
                mock_instance = Mock()
                mock_result = TranscribeResult(
                    file="test.mp3",
                    duration_seconds=10.0,
                    segments=[Segment(speaker="SPEAKER_00", start=0.0, end=10.0, text="测试")]
                )
                mock_instance.transcribe.return_value = mock_result
                mock_transcriber_class.return_value = mock_instance

                result = transcribe(temp_path, background="测试背景")

                assert result.file == "test.mp3"
                mock_instance.transcribe.assert_called_once()
        finally:
            Path(temp_path).unlink(missing_ok=True)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
