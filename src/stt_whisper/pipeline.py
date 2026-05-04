"""
Pipeline 模块 - 一站式语音转训练语料流水线
串联转写 + 说话人区分 + 后处理治理
"""

import logging
import json
from pathlib import Path
from typing import Optional, List, Dict, Any, Literal
from dataclasses import dataclass, asdict

from .transcribe import WhisperTranscriber, SpeakerDiarizer, AudioConverter
from .postprocess import TextCleaner, FormatterFactory, QualityScorer

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """流水线配置"""
    # 转写配置
    model: str = "large-v3"
    language: str = "zh"
    device: str = "cpu"
    use_faster_whisper: bool = True

    # 说话人区分配置
    enable_diarization: bool = True
    diarization_model: str = "pyannote/speaker-diarization-3.1"

    # 音频处理配置
    convert_to_wav: bool = True
    target_sample_rate: int = 16000

    # 后处理配置
    remove_fillers: bool = True
    normalize_punctuation: bool = True
    output_format: Literal["dialogue", "utterances", "plain", "plain_no_speaker"] = "dialogue"

    # 质量过滤
    min_quality_score: float = 0.0  # 低于此分数的跳过，0=不过滤

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PipelineConfig":
        """从字典创建配置"""
        return cls(**{k: v for k, v in data.items() if k in cls.__annotations__})

    @classmethod
    def from_yaml(cls, path: Path) -> "PipelineConfig":
        """从 YAML 文件加载配置"""
        import yaml
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data.get("pipeline", {}))


@dataclass
class PipelineResult:
    """流水线处理结果"""
    source_file: str
    success: bool
    duration_seconds: float = 0.0
    quality_score: float = 0.0
    output_path: Optional[str] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class STTPipeline:
    """语音转训练语料流水线"""

    def __init__(self, config: Optional[PipelineConfig] = None):
        """
        初始化流水线

        Args:
            config: 流水线配置，None则使用默认配置
        """
        self.config = config or PipelineConfig()

        # 初始化各组件
        self.transcriber = WhisperTranscriber(
            model=self.config.model,
            language=self.config.language,
            device=self.config.device,
            use_faster_whisper=self.config.use_faster_whisper
        )

        if self.config.enable_diarization:
            self.diarizer = SpeakerDiarizer(
                model=self.config.diarization_model,
                device=self.config.device
            )
        else:
            self.diarizer = None

        self.audio_converter = AudioConverter()
        self.text_cleaner = TextCleaner(
            remove_fillers=self.config.remove_fillers,
            normalize_punctuation=self.config.normalize_punctuation
        )
        self.quality_scorer = QualityScorer()

    def process_single(
        self,
        audio_path: Path,
        background: str = "",
        output_dir: Optional[Path] = None
    ) -> PipelineResult:
        """
        处理单个音频文件

        Args:
            audio_path: 音频文件路径
            background: 背景信息
            output_dir: 输出目录，None则保存到原目录

        Returns:
            PipelineResult 对象
        """
        audio_path = Path(audio_path)
        output_dir = Path(output_dir) if output_dir else audio_path.parent

        result = PipelineResult(
            source_file=str(audio_path),
            success=False
        )

        try:
            # Step 1: 音频预处理
            logger.info(f"[{audio_path.name}] Step 1: 音频预处理")
            processed_audio = audio_path

            if self.config.convert_to_wav:
                wav_path = audio_path.with_suffix(".wav")
                processed_audio = self.audio_converter.convert_to_wav(
                    audio_path,
                    wav_path,
                    sample_rate=self.config.target_sample_rate
                )
                logger.info(f"[{audio_path.name}] 转换为 WAV 完成")

            # Step 2: 语音转写
            logger.info(f"[{audio_path.name}] Step 2: 语音转写")
            transcribe_result = self.transcriber.transcribe(
                processed_audio,
                background=background
            )
            result.duration_seconds = transcribe_result.duration_seconds

            segments = [s.to_dict() if hasattr(s, 'to_dict') else s for s in transcribe_result.segments]

            # Step 3: 说话人区分
            if self.config.enable_diarization and self.diarizer:
                logger.info(f"[{audio_path.name}] Step 3: 说话人区分")
                try:
                    segments = self.diarizer.merge_with_transcript(
                        processed_audio,
                        segments
                    )
                except Exception as e:
                    logger.warning(f"[{audio_path.name}] 说话人区分失败: {e}，使用默认标签")
                    for seg in segments:
                        if "speaker" not in seg:
                            seg["speaker"] = "SPEAKER_00"

            # Step 4: 文字清理
            logger.info(f"[{audio_path.name}] Step 4: 文字清理")
            segments = self.text_cleaner.clean_segments(segments)

            # Step 5: 质量评分
            logger.info(f"[{audio_path.name}] Step 5: 质量评分")
            score = self.quality_scorer.score_segments(segments)
            result.quality_score = score.overall

            if self.config.min_quality_score > 0 and score.overall < self.config.min_quality_score:
                result.error = f"质量分数 ({score.overall}) 低于阈值 ({self.config.min_quality_score})"
                logger.warning(f"[{audio_path.name}] {result.error}")
                return result

            # Step 6: 格式化输出
            logger.info(f"[{audio_path.name}] Step 6: 格式化输出")
            output_path = output_dir / f"{audio_path.stem}_curated.json"

            FormatterFactory.format_and_save(
                segments=segments,
                output_path=output_path,
                fmt=self.config.output_format,
                background=background,
                source_file=audio_path.name
            )

            result.success = True
            result.output_path = str(output_path)
            result.metadata = {
                "model": self.config.model,
                "language": transcribe_result.language,
                "segments_count": len(segments),
                "quality_details": score.details,
                "recommendation": self.quality_scorer.get_recommendation(score)
            }

            logger.info(f"[{audio_path.name}] 处理完成! 质量: {score.overall}/100")

        except Exception as e:
            result.error = str(e)
            logger.error(f"[{audio_path.name}] 处理失败: {e}")

        return result

    def process_batch(
        self,
        input_dir: Path,
        background: str = "",
        output_dir: Optional[Path] = None,
        recursive: bool = False,
        batch_backgrounds: Optional[Dict[str, str]] = None
    ) -> List[PipelineResult]:
        """
        批量处理音频文件

        Args:
            input_dir: 输入目录
            background: 默认背景信息
            output_dir: 输出目录，None则保存到输入目录
            recursive: 是否递归搜索子目录
            batch_backgrounds: 每个文件的独立背景，key为文件名，value为背景

        Returns:
            PipelineResult 列表
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir) if output_dir else input_dir
        batch_backgrounds = batch_backgrounds or {}

        # 查找所有音频文件
        audio_formats = {".mp3", ".wav", ".m4a", ".ogg", ".flac", ".mp4", ".avi", ".mkv"}
        audio_files = []

        for fmt in audio_formats:
            if recursive:
                audio_files.extend(input_dir.rglob(f"*{fmt}"))
            else:
                audio_files.extend(input_dir.glob(f"*{fmt}"))

        audio_files = sorted(set(audio_files))
        logger.info(f"找到 {len(audio_files)} 个音频文件")

        results = []
        for audio_path in audio_files:
            # 使用文件特定的背景或默认背景
            file_background = batch_backgrounds.get(audio_path.name, background)

            logger.info(f"处理: {audio_path.name}")
            result = self.process_single(audio_path, file_background, output_dir)
            results.append(result)

        return results

    def process_with_custom_steps(
        self,
        audio_path: Path,
        steps: List[str],
        background: str = "",
        output_dir: Optional[Path] = None
    ) -> PipelineResult:
        """
        自定义步骤处理

        Args:
            audio_path: 音频文件路径
            steps: 步骤列表，可选值: "preprocess", "transcribe", "diarize", "clean", "score", "format"
            background: 背景信息
            output_dir: 输出目录

        Returns:
            PipelineResult 对象
        """
        audio_path = Path(audio_path)
        output_dir = Path(output_dir) if output_dir else audio_path.parent

        result = PipelineResult(source_file=str(audio_path), success=False)

        try:
            processed_audio = audio_path
            segments = []
            transcribe_result = None

            if "preprocess" in steps:
                if self.config.convert_to_wav:
                    wav_path = audio_path.with_suffix(".wav")
                    processed_audio = self.audio_converter.convert_to_wav(
                        audio_path, wav_path, sample_rate=self.config.target_sample_rate
                    )

            if "transcribe" in steps:
                transcribe_result = self.transcriber.transcribe(processed_audio, background=background)
                result.duration_seconds = transcribe_result.duration_seconds
                segments = [s.to_dict() if hasattr(s, 'to_dict') else s for s in transcribe_result.segments]

            if "diarize" in steps and self.diarizer:
                if not segments:
                    raise ValueError("需要先执行 transcribe 步骤")
                segments = self.diarizer.merge_with_transcript(processed_audio, segments)

            if "clean" in steps:
                if not segments:
                    raise ValueError("需要先执行 transcribe 步骤")
                segments = self.text_cleaner.clean_segments(segments)

            if "score" in steps:
                if not segments:
                    raise ValueError("需要先执行 transcribe 步骤")
                score = self.quality_scorer.score_segments(segments)
                result.quality_score = score.overall

            if "format" in steps:
                if not segments:
                    raise ValueError("需要先执行 transcribe 步骤")
                output_path = output_dir / f"{audio_path.stem}_curated.json"
                FormatterFactory.format_and_save(
                    segments=segments,
                    output_path=output_path,
                    fmt=self.config.output_format,
                    background=background,
                    source_file=audio_path.name
                )
                result.success = True
                result.output_path = str(output_path)

        except Exception as e:
            result.error = str(e)
            logger.error(f"自定义处理失败: {e}")

        return result


# 便捷函数
def create_pipeline(config: Optional[PipelineConfig] = None) -> STTPipeline:
    """创建流水线实例"""
    return STTPipeline(config)


def process(
    audio_path: str,
    background: str = "",
    output_format: str = "dialogue",
    output_path: Optional[str] = None,
    **kwargs
) -> PipelineResult:
    """
    便捷处理函数

    Args:
        audio_path: 音频文件路径
        background: 背景信息
        output_format: 输出格式
        output_path: 输出路径
        **kwargs: 其他配置参数

    Returns:
        PipelineResult 对象
    """
    config = PipelineConfig(output_format=output_format, **kwargs)
    pipeline = STTPipeline(config)

    result = pipeline.process_single(Path(audio_path), background)

    if output_path and result.success:
        # 移动到指定输出路径
        import shutil
        shutil.move(result.output_path, output_path)
        result.output_path = output_path

    return result


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        audio_file = sys.argv[1]
        background = sys.argv[2] if len(sys.argv) > 2 else ""
        result = process(audio_file, background=background)

        print(f"\n处理结果: {'成功' if result.success else '失败'}")
        if result.success:
            print(f"输出文件: {result.output_path}")
            print(f"质量评分: {result.quality_score}/100")
        else:
            print(f"错误: {result.error}")
    else:
        print("用法: python pipeline.py <音频文件> [背景信息]")
