"""
数据格式化模块 - 将转写结果转换为训练数据格式
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Optional, Literal
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class DialogueTurn:
    """对话回合"""
    role: str  # "user" 或 "assistant"
    content: str


@dataclass
class DialogueData:
    """对话格式数据"""
    conversations: List[DialogueTurn]
    metadata: Dict


@dataclass
class UtteranceData:
    """片段格式数据"""
    segments: List[Dict]
    background: str
    source_file: str


class DialogueFormatter:
    """对话格式转换器"""

    def __init__(self, min_turns: int = 1):
        """
        Args:
            min_turns: 最小对话回合数
        """
        self.min_turns = min_turns

    def format(
        self,
        segments: List[Dict],
        background: str = "",
        source_file: str = ""
    ) -> DialogueData:
        """
        将转写片段转换为对话格式

        Args:
            segments: 转写片段列表
            background: 背景信息
            source_file: 源文件名

        Returns:
            DialogueData 对象
        """
        if len(segments) < self.min_turns:
            logger.warning(f"片段数 ({len(segments)}) 少于最小要求 ({self.min_turns})")

        # 按时间排序（支持 dict 和 dataclass）
        sorted_segments = sorted(
            [s.to_dict() if hasattr(s, 'to_dict') else s for s in segments],
            key=lambda x: x.get("start", 0) if isinstance(x, dict) else getattr(x, 'start', 0)
        )

        conversations = []
        for seg in sorted_segments:
            speaker = seg.get("speaker", "SPEAKER_00")
            text = seg.get("text", "").strip()

            if not text:
                continue

            # 简单策略：奇数片段为user，偶数为assistant
            # 实际项目中可根据说话人ID分配
            role = "user" if len(conversations) % 2 == 0 else "assistant"

            conversations.append(DialogueTurn(role=role, content=text))

        metadata = {
            "background": background,
            "source_file": source_file,
            "total_segments": len(conversations)
        }

        return DialogueData(
            conversations=conversations,
            metadata=metadata
        )

    def save(self, data: DialogueData, output_path: Path) -> None:
        """保存为JSON"""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump({
                "conversations": [asdict(d) for d in data.conversations],
                "metadata": data.metadata
            }, f, ensure_ascii=False, indent=2)
        
        logger.info(f"对话格式已保存: {output_path}")


class UtteranceFormatter:
    """片段格式转换器"""

    def format(
        self,
        segments: List[Dict],
        background: str = "",
        source_file: str = ""
    ) -> UtteranceData:
        """
        将转写片段转换为片段格式

        Args:
            segments: 转写片段列表
            background: 背景信息
            source_file: 源文件名

        Returns:
            UtteranceData 对象
        """
        # 只保留必要字段（支持 dict 和 dataclass）
        clean_segments = []
        for seg in segments:
            seg_dict = seg.to_dict() if hasattr(seg, 'to_dict') else seg
            clean_segments.append({
                "speaker": seg_dict.get("speaker", "UNKNOWN"),
                "text": seg_dict.get("text", "").strip(),
                "start": seg_dict.get("start", 0.0),
                "end": seg_dict.get("end", 0.0)
            })

        return UtteranceData(
            segments=clean_segments,
            background=background,
            source_file=source_file
        )

    def save(self, data: UtteranceData, output_path: Path) -> None:
        """保存为JSON"""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(asdict(data), f, ensure_ascii=False, indent=2)
        
        logger.info(f"片段格式已保存: {output_path}")


class PlainTextFormatter:
    """纯文本格式转换器"""

    def __init__(self, include_speaker: bool = True, separator: str = "\n"):
        """
        Args:
            include_speaker: 是否包含说话人标签
            separator: 分隔符
        """
        self.include_speaker = include_speaker
        self.separator = separator

    def format(
        self,
        segments: List[Dict],
        background: str = "",
        source_file: str = ""
    ) -> str:
        """
        将转写片段转换为纯文本

        Args:
            segments: 转写片段列表
            background: 背景信息（可作为文本头部）
            source_file: 源文件名

        Returns:
            纯文本字符串
        """
        lines = []

        # 可选：添加背景头部
        if background:
            lines.append(f"[背景: {background}]")
            lines.append("")

        # 按时间排序（支持 dict 和 dataclass）
        sorted_segments = sorted(
            [s.to_dict() if hasattr(s, 'to_dict') else s for s in segments],
            key=lambda x: x.get("start", 0) if isinstance(x, dict) else getattr(x, 'start', 0)
        )

        for seg in sorted_segments:
            text = seg.get("text", "").strip()
            if not text:
                continue

            if self.include_speaker:
                speaker = seg.get("speaker", "SPEAKER_00")
                lines.append(f"{speaker}: {text}")
            else:
                lines.append(text)

        return self.separator.join(lines)

    def save(self, text: str, output_path: Path) -> None:
        """保存为文本文件"""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(text)
        
        logger.info(f"纯文本已保存: {output_path}")


class FormatterFactory:
    """格式化器工厂"""

    FORMATTERS = {
        "dialogue": DialogueFormatter,
        "utterances": UtteranceFormatter,
        "plain": PlainTextFormatter,
        "plain_no_speaker": lambda: PlainTextFormatter(include_speaker=False)
    }

    @classmethod
    def create(cls, format_type: str, **kwargs):
        """创建格式化器

        Args:
            format_type: 格式类型 ("dialogue", "utterances", "plain", "plain_no_speaker")
            **kwargs: 其他参数
        """
        formatter_cls = cls.FORMATTERS.get(format_type)
        if formatter_cls is None:
            raise ValueError(f"不支持的格式: {format_type}")
        return formatter_cls(**kwargs)

    @classmethod
    def format_and_save(
        cls,
        segments: List[Dict],
        output_path: Path,
        fmt: Literal["dialogue", "utterances", "plain", "plain_no_speaker"],
        background: str = "",
        source_file: str = ""
    ) -> None:
        """格式化并保存"""
        formatter = cls.create(fmt)

        if fmt == "plain" or fmt == "plain_no_speaker":
            text = formatter.format(segments, background, source_file)
            formatter.save(text, output_path)
        else:
            data = formatter.format(segments, background, source_file)
            formatter.save(data, output_path)


# 便捷函数
def to_dialogue(
    segments: List[Dict],
    background: str = "",
    source_file: str = ""
) -> DialogueData:
    """转换为对话格式"""
    formatter = DialogueFormatter()
    return formatter.format(segments, background, source_file)


def to_utterances(
    segments: List[Dict],
    background: str = "",
    source_file: str = ""
) -> UtteranceData:
    """转换为片段格式"""
    formatter = UtteranceFormatter()
    return formatter.format(segments, background, source_file)


def to_plain_text(
    segments: List[Dict],
    background: str = "",
    include_speaker: bool = True
) -> str:
    """转换为纯文本"""
    formatter = PlainTextFormatter(include_speaker=include_speaker)
    return formatter.format(segments, background)


if __name__ == "__main__":
    # 测试
    test_segments = [
        {"speaker": "SPEAKER_00", "start": 0.0, "end": 5.0, "text": "大家好，今天我们讨论Q2产品规划。"},
        {"speaker": "SPEAKER_01", "start": 5.0, "end": 10.0, "text": "我先说一下当前进展。"},
        {"speaker": "SPEAKER_00", "start": 10.0, "end": 15.0, "text": "好的，请讲。"},
    ]

    # 对话格式
    dialogue = to_dialogue(test_segments, background="产品规划会议", source_file="meeting.mp3")
    print("=== 对话格式 ===")
    print(json.dumps([asdict(d) for d in dialogue.conversations], ensure_ascii=False, indent=2))

    # 片段格式
    utterances = to_utterances(test_segments, background="产品规划会议")
    print("\n=== 片段格式 ===")
    print(json.dumps(asdict(utterances), ensure_ascii=False, indent=2))

    # 纯文本
    plain = to_plain_text(test_segments, include_speaker=True)
    print("\n=== 纯文本 ===")
    print(plain)
