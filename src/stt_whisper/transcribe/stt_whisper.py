"""
stt-whisper 转写核心模块
使用 Whisper 进行语音转文字
"""

import json
import logging
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Segment:
    """转写片段"""
    speaker: str = "SPEAKER_00"
    start: float = 0.0
    end: float = 0.0
    text: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典（兼容 formatters 的 .get() 接口）"""
        return asdict(self)


@dataclass
class TranscribeResult:
    """转写结果"""
    file: str = ""
    duration_seconds: float = 0.0
    background: str = ""
    segments: List[Segment] = field(default_factory=list)
    language: str = ""
    model: str = ""

    @property
    def full_text(self) -> str:
        """生成带说话人的完整文本"""
        lines = []
        for seg in self.segments:
            lines.append(f"{seg.speaker}: {seg.text}")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "file": self.file,
            "duration_seconds": self.duration_seconds,
            "background": self.background,
            "segments": [asdict(s) for s in self.segments],
            "language": self.language,
            "model": self.model
        }

    def save(self, output_path: Path) -> None:
        """保存为JSON文件"""
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
        logger.info(f"转写结果已保存: {output_path}")

    @classmethod
    def load(cls, input_path: Path) -> "TranscribeResult":
        """从JSON文件加载"""
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        segments = [Segment(**s) for s in data.get("segments", [])]
        return cls(
            file=data.get("file", ""),
            duration_seconds=data.get("duration_seconds", 0.0),
            background=data.get("background", ""),
            segments=segments,
            language=data.get("language", ""),
            model=data.get("model", "")
        )


class WhisperTranscriber:
    """Whisper转写器"""

    SUPPORTED_FORMATS = {".mp3", ".wav", ".m4a", ".ogg", ".flac", ".mp4", ".avi", ".mkv"}

    def __init__(
        self,
        model: str = "large-v3",
        language: Optional[str] = "zh",
        device: str = "cpu",
        model_path: Optional[str] = None,
        use_faster_whisper: bool = True
    ):
        """
        初始化转写器

        Args:
            model: Whisper模型名称 (tiny/base/small/medium/large-v2/large-v3)
            language: 语言代码，None=自动检测
            device: "cuda" 或 "cpu"
            model_path: 本地模型路径，None=自动下载
            use_faster_whisper: 是否使用 faster-whisper (更快)
        """
        self.model_name = model
        self.language = language
        self.device = device
        self.model_path = model_path
        self.use_faster_whisper = use_faster_whisper
        self._model = None
        self._has_av = None

    def _check_av(self):
        """检查 PyAV 是否可用"""
        if self._has_av is not None:
            return self._has_av
        try:
            import av
            self._has_av = True
        except ImportError:
            self._has_av = False
            logger.info("av (PyAV) 未安装，将使用 pydub+ffmpeg 解码")
        return self._has_av

    def _ensure_wav(self, audio_path: Path) -> Path:
        """确保音频为 WAV 格式（pydub+ffmpeg 转换）"""
        suffix = audio_path.suffix.lower()
        if suffix == ".wav":
            return audio_path
        
        from pydub import AudioSegment
        audio = AudioSegment.from_file(str(audio_path))
        wav_path = audio_path.with_suffix(".wav")
        audio.export(str(wav_path), format="wav")
        logger.info(f"已转换音频为 WAV: {wav_path}")
        return wav_path

    def _load_model(self):
        """延迟加载模型"""
        if self._model is not None:
            return

        if self.use_faster_whisper:
            try:
                from faster_whisper import WhisperModel
                logger.info(f"使用 faster-whisper 加载模型: {self.model_name}")
                self._model = WhisperModel(
                    self.model_name,
                    device=self.device,
                    download_root=self.model_path
                )
            except ImportError:
                logger.warning("faster-whisper 未安装，回退到 openai-whisper")
                self.use_faster_whisper = False

        if not self.use_faster_whisper:
            import whisper
            logger.info(f"使用 openai-whisper 加载模型: {self.model_name}")
            self._model = whisper.load_model(self.model_name, device=self.device)

    def transcribe(
        self,
        audio_path: Path,
        background: str = "",
        initial_prompt: Optional[str] = None,
        vad_filter: bool = True,
        beam_size: int = 5,
        best_of: int = 5,
        temperature: float = 0.0,
        condition_on_previous_text: bool = True,
        without_timestamps: bool = False,
        _on_segment: Optional[callable] = None,
        _progress_callback: Optional[callable] = None,
    ) -> TranscribeResult:
        """
        转写单个音频文件

        Args:
            audio_path: 音频文件路径
            background: 背景信息（用于提升转写准确性）
            initial_prompt: 初始提示词
            vad_filter: 是否使用语音活动检测滤波
            _on_segment: 流式回调，每识别一个片段立即调用，签名 f(seg_dict)
            _progress_callback: 进度回调，签名 f(current_seconds, total_seconds)

        Returns:
            TranscribeResult 对象
        """
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"音频文件不存在: {audio_path}")

        suffix = audio_path.suffix.lower()
        if suffix not in self.SUPPORTED_FORMATS:
            raise ValueError(f"不支持的格式: {suffix}，支持: {self.SUPPORTED_FORMATS}")

        logger.info(f"开始转写: {audio_path.name}")
        self._load_model()
        self._check_av()

        # 无 PyAV 时，先转换为 WAV（pydub + ffmpeg）
        if not self._has_av:
            audio_path = self._ensure_wav(audio_path)

        # 如果有背景信息，用作 initial_prompt
        if background and not initial_prompt:
            initial_prompt = f"背景：{background}"

        # 执行转写
        if self.use_faster_whisper:
            # faster-whisper 路径：segments 是生成器，支持流式消费
            segments_iterable, info = self._model.transcribe(
                str(audio_path),
                language=self.language,
                initial_prompt=initial_prompt,
                vad_filter=vad_filter,
                beam_size=beam_size,
                best_of=best_of,
                temperature=temperature,
                condition_on_previous_text=condition_on_previous_text,
                without_timestamps=without_timestamps,
            )

            duration = info.duration or 0.0
            language_detected = info.language or self.language or "unknown"

            # 流式消费：每 yield 一个片段立即回调，不等待全部完成
            segment_list = []
            for seg in segments_iterable:
                seg_dict = {
                    "start": seg.start,
                    "end": seg.end,
                    "text": seg.text.strip(),
                    "speaker": "SPEAKER_00",
                }
                segment_list.append(Segment(**seg_dict))
                if _on_segment:
                    _on_segment(seg_dict)
                if _progress_callback:
                    _progress_callback(seg.end, duration)

        else:
            # openai-whisper 路径（阻塞）
            result = self._model.transcribe(
                str(audio_path),
                language=self.language,
                initial_prompt=initial_prompt
            )

            duration = result.get("segments", [{}])[-1].get("end", 0.0) if result.get("segments") else 0.0
            language_detected = result.get("language", self.language or "unknown")

            segment_list = []
            for seg in result.get("segments", []):
                seg_obj = Segment(
                    start=seg["start"],
                    end=seg["end"],
                    text=seg["text"].strip(),
                    speaker="SPEAKER_00",
                )
                segment_list.append(seg_obj)
                if _on_segment:
                    _on_segment(seg_obj.to_dict())

        return TranscribeResult(
            file=audio_path.name,
            duration_seconds=duration,
            background=background,
            segments=segment_list,
            language=language_detected,
            model=self.model_name
        )

    def transcribe_batch(
        self,
        audio_dir: Path,
        background: str = "",
        output_dir: Optional[Path] = None,
        recursive: bool = False
    ) -> List[TranscribeResult]:
        """
        批量转写目录中的音频文件

        Args:
            audio_dir: 音频目录
            background: 批量背景信息（可被单个文件背景覆盖）
            output_dir: 输出目录，None则保存到原目录
            recursive: 是否递归搜索子目录

        Returns:
            TranscribeResult 列表
        """
        audio_dir = Path(audio_dir)
        output_dir = Path(output_dir) if output_dir else audio_dir

        # 查找所有支持的音频文件
        audio_files = []
        for fmt in self.SUPPORTED_FORMATS:
            if recursive:
                audio_files.extend(audio_dir.rglob(f"*{fmt}"))
            else:
                audio_files.extend(audio_dir.glob(f"*{fmt}"))

        audio_files = sorted(set(audio_files))
        logger.info(f"找到 {len(audio_files)} 个音频文件")

        results = []
        for audio_path in audio_files:
            try:
                result = self.transcribe(audio_path, background=background)
                
                # 保存结果
                output_path = output_dir / f"{audio_path.stem}_transcribed.json"
                result.save(output_path)
                
                results.append(result)
            except Exception as e:
                logger.error(f"转写失败 {audio_path.name}: {e}")

        return results


# 便捷函数
def transcribe(
    audio_path: str,
    model: str = "large-v3",
    language: Optional[str] = "zh",
    background: str = "",
    output_path: Optional[str] = None,
    **kwargs
) -> TranscribeResult:
    """
    便捷转写函数

    Args:
        audio_path: 音频文件路径
        model: Whisper模型
        language: 语言
        background: 背景信息
        output_path: 输出JSON路径
        **kwargs: 其他参数

    Returns:
        TranscribeResult 对象
    """
    transcriber = WhisperTranscriber(model=model, language=language, **kwargs)
    result = transcriber.transcribe(Path(audio_path), background=background)
    
    if output_path:
        result.save(Path(output_path))
    
    return result


if __name__ == "__main__":
    # 测试代码
    import sys
    
    if len(sys.argv) > 1:
        audio_file = sys.argv[1]
        background = sys.argv[2] if len(sys.argv) > 2 else ""
        
        result = transcribe(audio_file, background=background)
        print(f"\n转写结果 ({result.duration_seconds:.1f}秒):")
        print(result.full_text)
    else:
        print("用法: python stt_whisper.py <音频文件> [背景信息]")
