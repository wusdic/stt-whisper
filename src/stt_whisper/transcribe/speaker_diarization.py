"""
说话人区分模块 (Speaker Diarization)
使用 pyannote.audio 识别音频中的不同说话人
"""

import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class SpeakerTurn:
    """说话人转换"""
    speaker: str
    start: float
    end: float


class SpeakerDiarizer:
    """说话人区分器"""

    def __init__(
        self,
        model: str = "pyannote/speaker-diarization-3.1",
        device: str = "cuda"
    ):
        """
        初始化说话人区分器

        Args:
            model: pyannote 模型名称或本地路径
            device: "cuda" 或 "cpu"
        """
        self.model_name = model
        self.device = device
        self._pipeline = None

    def _load_pipeline(self):
        """延迟加载管道"""
        if self._pipeline is not None:
            return

        try:
            from pyannote.audio import Pipeline
            logger.info(f"加载说话人区分模型: {self.model_name}")
            self._pipeline = Pipeline.from_pretrained(
                self.model_name,
                device=self.device
            )
        except ImportError as e:
            raise ImportError(
                "pyannote.audio 未安装，请运行: pip install pyannote.audio"
            ) from e
        except Exception as e:
            raise RuntimeError(f"加载模型失败: {e}") from e

    def diarize(self, audio_path: Path) -> List[SpeakerTurn]:
        """
        分析音频，返回说话人转换序列

        Args:
            audio_path: 音频文件路径

        Returns:
            SpeakerTurn 列表，每个元素表示一个说话时段
        """
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"音频文件不存在: {audio_path}")

        logger.info(f"开始说话人分析: {audio_path.name}")
        self._load_pipeline()

        # 执行说话人区分
        diarization = self._pipeline(str(audio_path))

        turns = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            turns.append(SpeakerTurn(
                speaker=speaker,
                start=turn.start,
                end=turn.end
            ))

        logger.info(f"发现 {len(set(t.speaker for t in turns))} 个不同说话人")
        return turns

    def merge_with_transcript(
        self,
        audio_path: Path,
        transcript_segments: List[Dict],
        min_overlap: float = 0.1
    ) -> List[Dict]:
        """
        将说话人信息与转写文本合并

        Args:
            audio_path: 音频文件路径
            transcript_segments: 转写片段列表，每项包含 start, end, text
            min_overlap: 最小重叠比例才分配说话人

        Returns:
            合并后的片段列表，包含 speaker 字段
        """
        turns = self.diarize(audio_path)

        if not turns:
            # 无说话人信息，全部标记为 SPEAKER_00
            for seg in transcript_segments:
                seg["speaker"] = "SPEAKER_00"
            return transcript_segments

        # 按时间顺序合并
        merged = []
        for seg in transcript_segments:
            seg_start = seg["start"]
            seg_end = seg["end"]

            # 找到重叠最多的说话人
            speaker_counts: Dict[str, float] = {}
            for turn in turns:
                overlap_start = max(seg_start, turn.start)
                overlap_end = min(seg_end, turn.end)
                if overlap_start < overlap_end:
                    overlap_duration = overlap_end - overlap_start
                    speaker_counts[turn.speaker] = speaker_counts.get(turn.speaker, 0) + overlap_duration

            if speaker_counts:
                best_speaker = max(speaker_counts, key=speaker_counts.get)
                # 归一化为 SPEAKER_00, SPEAKER_01 ...
                unique_speakers = sorted(set(t.speaker for t in turns))
                speaker_idx = unique_speakers.index(best_speaker)
                seg["speaker"] = f"SPEAKER_{speaker_idx:02d}"
            else:
                seg["speaker"] = "SPEAKER_00"

            merged.append(seg)

        return merged


class SimplerDiarizer:
    """
    简化版说话人区分器（当 pyannote 不可用时使用）
    基于能量和静音检测的启发式方法
    """

    def __init__(self, min_silence_len: float = 0.3, speech_threshold: float = 0.01):
        """
        Args:
            min_silence_len: 最小静音长度（秒），用于分割说话
            speech_threshold: 语音能量阈值
        """
        self.min_silence_len = min_silence_len
        self.speech_threshold = speech_threshold

    def diarize(self, audio_path: Path) -> List[SpeakerTurn]:
        """简化版说话人分析（仅检测说话变化，不区分具体人数）"""
        try:
            import librosa
            import numpy as np
        except ImportError:
            raise ImportError("librosa 未安装，请运行: pip install librosa")

        logger.info(f"使用简化说话人分析: {audio_path.name}")

        # 加载音频
        y, sr = librosa.load(audio_path, sr=16000, mono=True)

        # 计算短时能量
        frame_length = 2048
        hop_length = 512
        energy = librosa.feature.rms(
            y=y, frame_length=frame_length, hop_length=hop_length
        )[0]

        # 计算时间轴
        times = librosa.frames_to_time(
            np.arange(len(energy)), sr=sr, hop_length=hop_length
        )

        # 简单的说话人切换检测（能量突变）
        turns = []
        current_speaker = "SPEAKER_00"
        is_speaking = False
        turn_start = 0.0

        # 归一化能量
        energy_norm = (energy - energy.min()) / (energy.max() - energy.min() + 1e-8)

        for i, (t, e) in enumerate(zip(times, energy_norm)):
            if e > self.speech_threshold:
                if not is_speaking:
                    # 开始说话
                    is_speaking = True
                    turn_start = t
            else:
                if is_speaking:
                    # 结束说话
                    is_speaking = False
                    turns.append(SpeakerTurn(
                        speaker=current_speaker,
                        start=turn_start,
                        end=t
                    ))

        # 处理最后一段
        if is_speaking:
            turns.append(SpeakerTurn(
                speaker=current_speaker,
                start=turn_start,
                end=times[-1]
            ))

        # 如果检测到多段，尝试分配不同说话人
        if len(turns) > 1:
            for i, turn in enumerate(turns):
                turn.speaker = f"SPEAKER_{i % 2:02d}"

        logger.info(f"简化分析: 检测到 {len(turns)} 个语音段")
        return turns


# 便捷函数
def create_diarizer(
    model: str = "pyannote/speaker-diarization-3.1",
    use_simpler: bool = False,
    **kwargs
) -> SpeakerDiarizer:
    """
    创建说话人区分器

    Args:
        model: 模型名称
        use_simpler: 是否使用简化版（pyannote不可用时）
        **kwargs: 其他参数

    Returns:
        SpeakerDiarizer 或 SimplerDiarizer 实例
    """
    if use_simpler:
        return SimplerDiarizer(**kwargs)

    try:
        diarizer = SpeakerDiarizer(model=model, **kwargs)
        # 测试是否能加载
        return diarizer
    except Exception as e:
        logger.warning(f"pyannote 不可用 ({e})，使用简化版")
        return SimplerDiarizer(**kwargs)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        audio_file = sys.argv[1]
        diarizer = create_diarizer(use_simpler=True)
        turns = diarizer.diarize(Path(audio_file))

        print(f"\n说话人分布:")
        for turn in turns:
            print(f"  {turn.speaker}: {turn.start:.1f}s - {turn.end:.1f}s")
    else:
        print("用法: python speaker_diarization.py <音频文件>")
