"""
音频处理工具模块
提供音频格式转换、切割、合并等工具
"""

import logging
import subprocess
from pathlib import Path
from typing import Optional, Tuple, List

logger = logging.getLogger(__name__)


class AudioInfo:
    """音频文件信息"""
    def __init__(
        self,
        path: str,
        duration: float = 0.0,
        sample_rate: int = 0,
        channels: int = 0,
        codec: str = "",
        bitrate: str = ""
    ):
        self.path = path
        self.duration = duration
        self.sample_rate = sample_rate
        self.channels = channels
        self.codec = codec
        self.bitrate = bitrate

    def __repr__(self):
        return (f"AudioInfo(path={self.path}, duration={self.duration:.1f}s, "
                f"sr={self.sample_rate}, ch={self.channels})")


class AudioConverter:
    """音频格式转换器"""

    SUPPORTED_FORMATS = {"mp3", "wav", "m4a", "ogg", "flac", "mp4", "avi", "mkv", "wma", "aac"}

    def __init__(self, ffmpeg_path: str = "ffmpeg", ffprobe_path: str = "ffprobe"):
        """
        初始化转换器

        Args:
            ffmpeg_path: ffmpeg 程序路径
            ffprobe_path: ffprobe 程序路径
        """
        self.ffmpeg = ffmpeg_path
        self.ffprobe = ffprobe_path

    def get_audio_info(self, audio_path: Path) -> AudioInfo:
        """
        获取音频文件信息

        Args:
            audio_path: 音频文件路径

        Returns:
            AudioInfo 对象
        """
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"文件不存在: {audio_path}")

        cmd = [
            self.ffprobe,
            "-v", "quiet",
            "-print_format", "json",
            "-show_format",
            "-show_streams",
            str(audio_path)
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            import json
            data = json.loads(result.stdout)

            # 提取音频流信息
            audio_stream = None
            for stream in data.get("streams", []):
                if stream.get("codec_type") == "audio":
                    audio_stream = stream
                    break

            format_info = data.get("format", {})

            duration = float(format_info.get("duration", 0))
            sample_rate = int(audio_stream.get("sample_rate", 0)) if audio_stream else 0
            channels = int(audio_stream.get("channels", 0)) if audio_stream else 0
            codec = audio_stream.get("codec_name", "") if audio_stream else ""
            bitrate = format_info.get("bit_rate", "")

            return AudioInfo(
                path=str(audio_path),
                duration=duration,
                sample_rate=sample_rate,
                channels=channels,
                codec=codec,
                bitrate=bitrate
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"获取音频信息失败: {e}")
        except json.JSONDecodeError:
            raise RuntimeError("解析音频信息失败")

    def convert(
        self,
        input_path: Path,
        output_path: Path,
        format: str = "wav",
        sample_rate: Optional[int] = None,
        channels: int = 1,
        bitrate: str = "192k"
    ) -> Path:
        """
        转换音频格式

        Args:
            input_path: 输入文件路径
            output_path: 输出文件路径
            format: 目标格式
            sample_rate: 目标采样率，None则保持原采样率
            channels: 目标声道数，1=单声道，2=立体声
            bitrate: 比特率

        Returns:
            输出文件路径
        """
        input_path = Path(input_path)
        output_path = Path(output_path)

        if not input_path.exists():
            raise FileNotFoundError(f"输入文件不存在: {input_path}")

        output_path.parent.mkdir(parents=True, exist_ok=True)

        cmd = [self.ffmpeg, "-y", "-i", str(input_path)]

        # 音频处理选项
        if sample_rate:
            cmd.extend(["-ar", str(sample_rate)])
        if channels:
            cmd.extend(["-ac", str(channels)])

        cmd.extend(["-ab", bitrate, "-f", format, str(output_path)])

        logger.info(f"转换音频: {input_path.name} -> {output_path.name}")
        try:
            subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=300)
            logger.info(f"转换完成: {output_path}")
            return output_path
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"音频转换失败: {e.stderr}")
        except subprocess.TimeoutExpired:
            raise RuntimeError("音频转换超时")

    def convert_to_wav(
        self,
        input_path: Path,
        output_path: Optional[Path] = None,
        sample_rate: int = 16000,
        channels: int = 1
    ) -> Path:
        """
        转换为 WAV 格式（适合 Whisper 处理）

        Args:
            input_path: 输入文件路径
            output_path: 输出文件路径，None则在与输入文件同目录
            sample_rate: 目标采样率（Whisper 推荐 16kHz）
            channels: 声道数（Whisper 推荐单声道）

        Returns:
            输出文件路径
        """
        input_path = Path(input_path)
        if output_path is None:
            output_path = input_path.with_suffix(".wav")

        return self.convert(
            input_path,
            output_path,
            format="wav",
            sample_rate=sample_rate,
            channels=channels
        )

    def split_by_duration(
        self,
        input_path: Path,
        output_dir: Path,
        chunk_duration: float = 60.0,
        overlap: float = 0.0
    ) -> List[Path]:
        """
        按时长分割音频

        Args:
            input_path: 输入文件路径
            output_dir: 输出目录
            chunk_duration: 每个片段的时长（秒）
            overlap: 片段之间的重叠时长（秒）

        Returns:
            输出文件路径列表
        """
        input_path = Path(input_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        info = self.get_audio_info(input_path)
        total_duration = info.duration

        output_files = []
        start_time = 0.0

        while start_time < total_duration:
            end_time = min(start_time + chunk_duration, total_duration)
            output_path = output_dir / f"{input_path.stem}__{start_time:.0f}_{end_time:.0f}{input_path.suffix}"

            cmd = [
                self.ffmpeg, "-y",
                "-i", str(input_path),
                "-ss", str(start_time),
                "-to", str(end_time),
                "-c", "copy",  # 无重新编码，快速分割
                str(output_path)
            ]

            try:
                subprocess.run(cmd, capture_output=True, text=True, check=True)
                output_files.append(output_path)
                logger.info(f"切割: {output_path.name} ({start_time:.0f}s - {end_time:.0f}s)")
            except subprocess.CalledProcessError as e:
                logger.error(f"切割失败: {e.stderr}")

            start_time = end_time - overlap  # 重叠则往回退

        return output_files

    def split_by_silence(
        self,
        input_path: Path,
        output_dir: Path,
        silence_threshold: float = -40.0,
        min_silence_len: float = 1.0
    ) -> List[Path]:
        """
        按静音分割音频（使用 ffmpeg 的 silencedetect）

        Args:
            input_path: 输入文件路径
            output_dir: 输出目录
            silence_threshold: 静音阈值（dB）
            min_silence_len: 最小静音时长（秒）

        Returns:
            输出文件路径列表
        """
        import re

        input_path = Path(input_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # 先检测静音区域
        cmd = [
            self.ffmpeg, "-i", str(input_path),
            "-af", f"silencedetect=noise={silence_threshold}dB:d={min_silence_len}",
            "-f", "null", "-"
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            output = result.stderr
        except subprocess.TimeoutExpired:
            raise RuntimeError("静音检测超时")
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"静音检测失败: {e.stderr}")

        # 解析静音区间
        silence_ranges = []
        start_pattern = re.compile(r"silence_start: ([\d.]+)")
        end_pattern = re.compile(r"silence_end: ([\d.]+)")

        for match in start_pattern.finditer(output):
            silence_ranges.append({"start": float(match.group(1)), "end": None})
        for match in end_pattern.finditer(output):
            if silence_ranges and silence_ranges[-1]["end"] is None:
                silence_ranges[-1]["end"] = float(match.group(1))

        # 根据静音区间分割
        output_files = []
        segments = []
        last_end = 0.0

        for silence in silence_ranges:
            if silence["end"] is not None:
                # 非静音段
                if silence["start"] > last_end:
                    segments.append((last_end, silence["start"]))
                last_end = silence["end"]

        # 最后一个非静音段
        info = self.get_audio_info(input_path)
        if last_end < info.duration:
            segments.append((last_end, info.duration))

        # 提取每个非静音段
        for i, (start, end) in enumerate(segments):
            output_path = output_dir / f"{input_path.stem}_seg{i:03d}{input_path.suffix}"
            cmd = [
                self.ffmpeg, "-y",
                "-i", str(input_path),
                "-ss", str(start),
                "-to", str(end),
                "-c", "copy",
                str(output_path)
            ]
            subprocess.run(cmd, capture_output=True, text=True, check=True)
            output_files.append(output_path)
            logger.info(f"静音分割: {output_path.name} ({start:.1f}s - {end:.1f}s)")

        return output_files

    def merge_audio(self, input_paths: List[Path], output_path: Path) -> Path:
        """
        合并多个音频文件

        Args:
            input_paths: 输入文件路径列表
            output_path: 输出文件路径

        Returns:
            输出文件路径
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # 创建临时文件列表
        list_file = output_path.with_suffix(".txt")
        with open(list_file, "w") as f:
            for path in input_paths:
                f.write(f"file '{path.absolute()}'\n")

        cmd = [
            self.ffmpeg, "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", str(list_file),
            "-c", "copy",
            str(output_path)
        ]

        try:
            subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=600)
            list_file.unlink()  # 删除临时列表
            logger.info(f"合并完成: {output_path}")
            return output_path
        except subprocess.CalledProcessError as e:
            list_file.unlink(missing_ok=True)
            raise RuntimeError(f"音频合并失败: {e.stderr}")

    def normalize_volume(
        self,
        input_path: Path,
        output_path: Path,
        target_level: float = -20.0
    ) -> Path:
        """
        音量标准化

        Args:
            input_path: 输入文件路径
            output_path: 输出文件路径
            target_level: 目标音量级别（dB）

        Returns:
            输出文件路径
        """
        input_path = Path(input_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        cmd = [
            self.ffmpeg, "-y",
            "-i", str(input_path),
            "-af", f"volume={target_level}dB",
            str(output_path)
        ]

        try:
            subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=300)
            logger.info(f"音量标准化完成: {output_path}")
            return output_path
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"音量标准化失败: {e.stderr}")


# 便捷函数
def get_audio_info(path: str) -> AudioInfo:
    """获取音频信息"""
    converter = AudioConverter()
    return converter.get_audio_info(Path(path))


def convert_to_wav(input_path: str, output_path: str = None, sample_rate: int = 16000) -> str:
    """转换为 WAV"""
    converter = AudioConverter()
    result = converter.convert_to_wav(Path(input_path), Path(output_path) if output_path else None, sample_rate)
    return str(result)


def split_audio(path: str, output_dir: str, chunk_duration: float = 60.0) -> List[str]:
    """分割音频"""
    converter = AudioConverter()
    results = converter.split_by_duration(Path(path), Path(output_dir), chunk_duration)
    return [str(p) for p in results]


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        audio_file = sys.argv[1]
        info = get_audio_info(audio_file)
        print(f"音频信息: {info}")
    else:
        print("用法: python audio_utils.py <音频文件>")
