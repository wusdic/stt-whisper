"""
pytest 配置和共享 fixtures
"""

import pytest
import sys
from pathlib import Path

# 添加 src 目录到 Python 路径
src_path = Path(__file__).parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))


@pytest.fixture
def sample_segments():
    """示例转写片段"""
    return [
        {"speaker": "SPEAKER_00", "start": 0.0, "end": 5.0, "text": "大家好，今天我们讨论产品规划。"},
        {"speaker": "SPEAKER_01", "start": 5.0, "end": 10.0, "text": "我先说一下当前进展。"},
        {"speaker": "SPEAKER_00", "start": 10.0, "end": 15.0, "text": "好的，请讲。"},
        {"speaker": "SPEAKER_01", "start": 15.0, "end": 20.0, "text": "产品A已经完成，产品B进行中。"},
        {"speaker": "SPEAKER_00", "start": 20.0, "end": 25.0, "text": "很好，那下一步计划是什么？"},
    ]


@pytest.fixture
def sample_segments_with_fillers():
    """带填充词的示例片段"""
    return [
        {"speaker": "SPEAKER_00", "start": 0.0, "end": 5.0, "text": "嗯嗯大家好，今天我们讨论这个这个产品规划。"},
        {"speaker": "SPEAKER_01", "start": 5.0, "end": 10.0, "text": "哦好的好的，那这个那个的话怎么说呢，嗯呢。"},
        {"speaker": "SPEAKER_00", "start": 10.0, "end": 15.0, "text": "[笑声] 哈哈哈这个太好笑了。"},
    ]


@pytest.fixture
def temp_audio_file(tmp_path):
    """临时音频文件路径（不实际创建）"""
    return tmp_path / "test_audio.mp3"


@pytest.fixture
def temp_output_dir(tmp_path):
    """临时输出目录"""
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    return output_dir
