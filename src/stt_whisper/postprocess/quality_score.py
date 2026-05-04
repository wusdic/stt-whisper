"""
数据质量评分模块
"""

import logging
import re
from typing import List, Dict
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class QualityScore:
    """质量评分结果"""
    overall: float  # 0-100
    completeness: float  # 0-100
    cleanliness: float  # 0-100
    validity: float  # 0-100
    details: Dict[str, float]  # 各项详情


def _seg_to_dict(seg) -> Dict:
    """将 dict 或 dataclass 转换为 dict"""
    return seg.to_dict() if hasattr(seg, 'to_dict') else seg


class QualityScorer:
    """质量评分器"""

    def __init__(
        self,
        min_segment_length: int = 5,
        max_filler_ratio: float = 0.3,
        min_speaker_diversity: int = 1
    ):
        """
        Args:
            min_segment_length: 单片段最小字符数
            max_filler_ratio: 最大填充词比例
            min_speaker_diversity: 最小说话人数量
        """
        self.min_segment_length = min_segment_length
        self.max_filler_ratio = max_filler_ratio
        self.min_speaker_diversity = min_speaker_diversity

    def score_segments(self, segments: List[Dict]) -> QualityScore:
        """
        评估片段列表质量

        Args:
            segments: 转写片段列表

        Returns:
            QualityScore 对象
        """
        if not segments:
            return QualityScore(0, 0, 0, 0, {})

        details = {}

        # 1. 完整性评分 (completeness)
        completeness = self._score_completeness(segments)
        details["completeness"] = completeness

        # 2. 清洁度评分 (cleanliness)
        cleanliness = self._score_cleanliness(segments)
        details["cleanliness"] = cleanliness

        # 3. 有效性评分 (validity)
        validity = self._score_validity(segments)
        details["validity"] = validity

        # 4. 说话人多样性
        speaker_score = self._score_speaker_diversity(segments)
        details["speaker_diversity"] = speaker_score

        # 综合评分 (加权平均)
        overall = (
            completeness * 0.25 +
            cleanliness * 0.25 +
            validity * 0.30 +
            speaker_score * 0.20
        )

        return QualityScore(
            overall=round(overall, 1),
            completeness=round(completeness, 1),
            cleanliness=round(cleanliness, 1),
            validity=round(validity, 1),
            details=details
        )

    def _score_completeness(self, segments: List[Dict]) -> float:
        """评估完整性：片段数量、时长覆盖"""
        if not segments:
            return 0.0

        # 检查是否有合理的片段数量
        total_duration = 0.0
        for seg in segments:
            seg = _seg_to_dict(seg)
            start = seg.get("start", 0)
            end = seg.get("end", 0)
            total_duration += (end - start)

        # 时长超过60秒得高分
        if total_duration > 60:
            duration_score = 100.0
        else:
            duration_score = (total_duration / 60) * 100

        # 片段数量评分
        seg_count = len(segments)
        if seg_count >= 5:
            count_score = 100.0
        else:
            count_score = (seg_count / 5) * 100

        return (duration_score * 0.6 + count_score * 0.4)

    def _score_cleanliness(self, segments: List[Dict]) -> float:
        """评估清洁度：填充词、噪声"""
        total_chars = 0
        filler_chars = 0

        filler_pattern = re.compile(r"[嗯啊呃哦呀啦呢吧哈呵]")

        for seg in segments:
            seg = _seg_to_dict(seg)
            text = seg.get("text", "")
            total_chars += len(text)
            filler_chars += len(filler_pattern.findall(text))

        if total_chars == 0:
            return 0.0

        filler_ratio = filler_chars / total_chars
        if filler_ratio <= 0.05:
            return 100.0
        elif filler_ratio >= self.max_filler_ratio:
            return 20.0
        else:
            # 线性插值
            return 100.0 - (filler_ratio - 0.05) / (self.max_filler_ratio - 0.05) * 80

    def _score_validity(self, segments: List[Dict]) -> float:
        """评估有效性：最小长度、占位符"""
        if not segments:
            return 0.0

        valid_count = 0
        for seg in segments:
            seg = _seg_to_dict(seg)
            text = seg.get("text", "")
            # 检查是否为空或只有占位符
            if len(text) >= self.min_segment_length:
                # 检查是否有明显占位符
                if not re.match(r"^( SPEECH|无|空白|静音)+$", text):
                    valid_count += 1

        return (valid_count / len(segments)) * 100

    def _score_speaker_diversity(self, segments: List[Dict]) -> float:
        """评估说话人多样性"""
        speakers = set(_seg_to_dict(seg).get("speaker", "SPEAKER_00") for seg in segments)
        speaker_count = len(speakers)

        if speaker_count >= self.min_speaker_diversity:
            # 2人以上得高分
            if speaker_count >= 2:
                return min(100.0, 60 + speaker_count * 20)
            return 100.0
        else:
            return 30.0  # 单人对话扣分

    def get_recommendation(self, score: QualityScore) -> str:
        """根据评分获取建议"""
        if score.overall >= 80:
            return "质量优秀，可直接用于训练"
        elif score.overall >= 60:
            return "质量良好，建议少量清洗后使用"
        elif score.overall >= 40:
            return "质量一般，需要较多清洗工作"
        else:
            return "质量较差，建议重新转写或丢弃"


# 便捷函数
def score_quality(segments: List[Dict], **kwargs) -> QualityScore:
    """便捷质量评分"""
    scorer = QualityScorer(**kwargs)
    return scorer.score_segments(segments)


if __name__ == "__main__":
    # 测试
    test_segments = [
        {"speaker": "SPEAKER_00", "start": 0.0, "end": 5.0, "text": "大家好，今天我们讨论Q2产品规划。"},
        {"speaker": "SPEAKER_01", "start": 5.0, "end": 10.0, "text": "我先说一下当前进展，整体不错。"},
        {"speaker": "SPEAKER_00", "start": 10.0, "end": 15.0, "text": "好的，请讲讲具体细节。"},
        {"speaker": "SPEAKER_01", "start": 15.0, "end": 20.0, "text": "嗯嗯好的好的，产品A已经完成。"},
        {"speaker": "SPEAKER_00", "start": 20.0, "end": 25.0, "text": "很好，那下一步计划是什么？"},
    ]

    scorer = QualityScorer()
    score = scorer.score_segments(test_segments)

    print(f"综合评分: {score.overall}/100")
    print(f"  完整性: {score.completeness}/100")
    print(f"  清洁度: {score.cleanliness}/100")
    print(f"  有效性: {score.validity}/100")
    print(f"建议: {scorer.get_recommendation(score)}")
