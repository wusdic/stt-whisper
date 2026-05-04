"""
文字清理模块 - 清洗转写文本中的噪声
"""

import re
import logging
from typing import List, Dict, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class CleaningResult:
    """清理结果"""
    original: str
    cleaned: str
    removed_items: List[str]  # 被移除的内容列表


def _seg_to_dict(seg):
    """将 dict 或 dataclass 转换为 dict"""
    return seg.to_dict() if hasattr(seg, 'to_dict') else seg


class TextCleaner:
    """文字清理器"""

    # 填充词模式（不使用 \b 词边界，因为中文不是 word characters）
    FILLER_PATTERNS = [
        r"嗯{1,}", r"啊{1,}", r"呃{1,}",
        r"哦{1,}", r"呀{1,}", r"嘛{1,}",
        r"啦{1,}", r"呢{1,}", r"吧{1,}",
        r"哈{1,}", r"呵{1,}",
        r"这个这个", r"那个那个", r"就是就是",
        r"的话的话",  # 重复词
    ]

    # 重复词模式（保留第一个）
    REPEAT_WORD_PATTERN = r"(.+?)\1{2,}"

    # 静音/杂音标记
    NOISE_PATTERNS = [
        r"\[音乐\]", r"\[噪音\]", r"\[静音\]",
        r"\[咳嗽\]", r"\[笑声\]", r"\[掌声\]",
        r"<音乐>", r"<噪音>", r"<静音>",
        r"（音乐）", r"（噪音）", r"（静音）",
    ]

    def __init__(
        self,
        remove_fillers: bool = True,
        remove_noise: bool = True,
        normalize_punctuation: bool = True,
        remove_repeats: bool = True,
        min_word_count: int = 1
    ):
        """
        Args:
            remove_fillers: 删除填充词
            remove_noise: 删除噪声标记
            normalize_punctuation: 标点规范化
            remove_repeats: 删除重复词
            min_word_count: 最小词数，低于此值的片段标记
        """
        self.remove_fillers = remove_fillers
        self.remove_noise = remove_noise
        self.normalize_punctuation = normalize_punctuation
        self.remove_repeats = remove_repeats
        self.min_word_count = min_word_count

    def clean_text(self, text: str) -> CleaningResult:
        """
        清理单段文本

        Args:
            text: 原始文本

        Returns:
            CleaningResult 对象
        """
        original = text
        removed_items = []

        # 1. 删除噪声标记
        if self.remove_noise:
            for pattern in self.NOISE_PATTERNS:
                matches = re.findall(pattern, text)
                if matches:
                    removed_items.extend(matches)
                    text = re.sub(pattern, "", text)

        # 2. 删除填充词
        if self.remove_fillers:
            for pattern in self.FILLER_PATTERNS:
                matches = re.findall(pattern, text, re.IGNORECASE)
                if matches:
                    removed_items.extend(matches)
                    text = re.sub(pattern, "", text, flags=re.IGNORECASE)

        # 3. 清理重复词（"这个这个" -> "这个"）
        if self.remove_repeats:
            text = re.sub(r"(.)\1{2,}", r"\1", text)

        # 4. 标点规范化
        if self.normalize_punctuation:
            # 移除多余空格
            text = re.sub(r"\s+", " ", text)
            # 规范引号
            text = text.replace(""", "'").replace(""", "'")
            text = text.replace(""", "'").replace(""", "'")
            # 规范破折号
            text = text.replace("——", "—").replace("--", "—")
            # 句尾标点标准化（如果没有标点则添加句号）
            text = re.sub(r"([^\u4e00-\u9fa5a-zA-Z0-9])$", r"\1", text)

        # 5. 去除首尾空白
        text = text.strip()

        # 6. 合并连续标点
        text = re.sub(r"([。！？；，、])\1+", r"\1", text)

        return CleaningResult(
            original=original,
            cleaned=text,
            removed_items=removed_items
        )

    def clean_segments(self, segments: List[Dict]) -> List[Dict]:
        """
        清理转写片段列表

        Args:
            segments: 转写片段，每项包含 text, speaker, start, end

        Returns:
            清理后的片段列表
        """
        cleaned_segments = []

        for seg in segments:
            seg = _seg_to_dict(seg)
            result = self.clean_text(seg.get("text", ""))

            # 保留有效内容
            if len(result.cleaned) >= self.min_word_count:
                cleaned_seg = seg.copy() if isinstance(seg, dict) else dict(seg)
                cleaned_seg["text"] = result.cleaned
                cleaned_seg["original_text"] = result.original
                cleaned_seg["removed_items"] = result.removed_items
                cleaned_seg["is_cleaned"] = (result.original != result.cleaned)
                cleaned_segments.append(cleaned_seg)
            else:
                # 内容太少，标记为无效
                logger.debug(f"片段内容过少，已跳过: {result.original[:30]}...")

        logger.info(f"清理完成: {len(segments)} -> {len(cleaned_segments)} 个有效片段")
        return cleaned_segments


class AdvancedCleaner(TextCleaner):
    """高级清理器 - 包含更多清洗规则"""

    # 口语化转书面语映射
    COLLOQUIAL_TO_FORMAL = {
        "干啥": "做什么",
        "咋样": "怎么样",
        "咋办": "怎么办",
        "挺好": "很好",
        "挺好的": "很好",
        "蛮好": "很好",
        "还行": "还可以",
        "不知道": "不知道",
        "不知道的": "不了解",
        "没啥": "没什么",
        "不咋": "不怎么",
        "别管": "不用管",
        "啥": "什么",
        "咋": "怎么",
        "咋的": "怎么",
        "嘛": "什么",
        "嘎": "啊",
        "唠": "聊",
        "甭": "不用",
        "别": "不要",
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._build_colloquial_pattern()

    def _build_colloquial_pattern(self):
        """构建口语化转换模式"""
        self.colloquial_pattern = re.compile(
            "|".join(re.escape(k) for k in self.COLLOQUIAL_TO_FORMAL.keys())
        )

    def clean_text(self, text: str) -> CleaningResult:
        """先清理，再转换口语化表达"""
        result = super().clean_text(text)

        # 口语化转书面语
        def replace_colloquial(m):
            return self.COLLOQUIAL_TO_FORMAL.get(m.group(), m.group())

        cleaned = self.colloquial_pattern.sub(replace_colloquial, result.cleaned)
        
        return CleaningResult(
            original=result.original,
            cleaned=cleaned,
            removed_items=result.removed_items
        )


# 便捷函数
def clean_text(text: str, **kwargs) -> str:
    """便捷文字清理"""
    cleaner = TextCleaner(**kwargs)
    return cleaner.clean_text(text).cleaned


def clean_segments(segments: List[Dict], **kwargs) -> List[Dict]:
    """便捷批量清理"""
    cleaner = TextCleaner(**kwargs)
    return cleaner.clean_segments(segments)


if __name__ == "__main__":
    # 测试
    test_texts = [
        "嗯嗯大家好，今天我们讨论这个这个Q2产品规划，啊我觉得挺好的。",
        "哦好的好的，那这个那个的话怎么说呢，嗯呢。",
        "[笑声] 哈哈哈这个太好笑了 [掌声]",
        "speech test 12345 testing 嗯嗯啊",
    ]

    cleaner = TextCleaner()
    for text in test_texts:
        result = cleaner.clean_text(text)
        print(f"原文: {result.original}")
        print(f"清理: {result.cleaned}")
        if result.removed_items:
            print(f"移除: {result.removed_items}")
        print()
