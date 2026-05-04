"""
stt-whisper Streamlit Web Interface
语音转写 Web 界面 - 支持单文件和批量文件夹处理
"""

import streamlit as st
import tempfile
import os
import json
import time
import glob
import threading
from pathlib import Path
from typing import List, Optional, Dict

# 设置路径
import sys
WEB_DIR = Path(__file__).parent
PACKAGE_DIR = WEB_DIR.parent.parent
if str(PACKAGE_DIR) not in sys.path:
    sys.path.insert(0, str(PACKAGE_DIR))

# 设置 HuggingFace 镜像（解决网络不通问题）
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from stt_whisper import __version__
from stt_whisper.pipeline import STTPipeline, PipelineConfig, PipelineResult
from stt_whisper.transcribe import TranscribeResult, Segment
from stt_whisper.postprocess import (
    DialogueFormatter, UtteranceFormatter, PlainTextFormatter,
    to_dialogue, to_utterances, to_plain_text, QualityScorer
)

# ============================================================
# 页面配置
# ============================================================
st.set_page_config(
    page_title="stt-whisper 语音转写",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# 自定义 CSS
st.markdown("""
<style>
.stApp { background-color: #0e1117; }
.stTextInput > div > div > input { background-color: #262730; color: #fafafa; }
.stSelectbox > div > div > div { background-color: #262730; }
.stMultiSelect > div > div > div { background-color: #262730; }
.stButton > button { width: 100%; background-color: #ff6b6b; color: white; border: none; }
.stButton > button:hover { background-color: #ff5252; }
.css-1v0mbdj.avatar { width: 48px; height: 48px; }
.block-container { padding-top: 2rem; padding-bottom: 2rem; }
h1, h2, h3 { color: #fafafa; }
.info-box { background-color: #262730; padding: 1rem; border-radius: 0.5rem; margin: 0.5rem 0; }
.success-box { background-color: #1a472a; padding: 1rem; border-radius: 0.5rem; border-left: 4px solid #00c853; }
.error-box { background-color: #3d1f1f; padding: 1rem; border-radius: 0.5rem; border-left: 4px solid #ff5252; }
.queue-item { background-color: #1e2530; padding: 0.75rem; border-radius: 0.5rem; margin: 0.25rem 0; }
.progress-card { background-color: #262730; padding: 1.5rem; border-radius: 0.75rem; margin: 0.5rem 0; }
</style>
""", unsafe_allow_html=True)


# ============================================================
# 辅助函数
# ============================================================
def get_audio_files_in_dir(dir_path: Path) -> List[Path]:
    """获取目录中所有支持的音频/视频文件"""
    AUDIO_EXTS = {'.mp3', '.wav', '.m4a', '.ogg', '.flac', '.mp4', '.avi', '.mkv'}
    files = []
    for ext in AUDIO_EXTS:
        files.extend(dir_path.glob(f"*{ext}"))
        files.extend(dir_path.glob(f"*{ext.upper()}"))
    return sorted(set(files))


def format_time(seconds: float) -> str:
    """格式化时长"""
    if seconds < 60:
        return f"{seconds:.1f}秒"
    elif seconds < 3600:
        m = int(seconds // 60)
        s = int(seconds % 60)
        return f"{m}分{s}秒"
    else:
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        return f"{h}小时{m}分"


def get_quality_label(score: float) -> str:
    """获取质量等级标签"""
    if score >= 80:
        return "🟢 优秀"
    elif score >= 60:
        return "🟡 良好"
    elif score >= 40:
        return "🟠 一般"
    else:
        return "🔴 较差"


def load_results_from_dir(output_dir: Path) -> List[dict]:
    """从输出目录加载已有结果"""
    results = []
    for json_file in output_dir.glob("*.json"):
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                data["_result_file"] = str(json_file)
                results.append(data)
        except Exception:
            continue
    return sorted(results, key=lambda x: x.get("source_file", ""))


def process_single_with_progress(
    pipeline: STTPipeline,
    audio_path: Path,
    background: str,
    output_dir: Path,
    progress_state: dict
) -> PipelineResult:
    """
    带进度回调的单文件处理。
    progress_state 是共享的 dict，会被多个步骤更新。
    步骤顺序: init -> preprocess -> transcribe -> diarize -> clean -> score -> format -> done
    """
    def update_step(step: str, sub_step: str = "", progress: float = 0.0, detail: str = ""):
        """更新进度状态，Streamlit 每次 rerun 会读取最新值"""
        progress_state["step"] = step
        progress_state["sub_step"] = sub_step
        progress_state["progress"] = progress
        progress_state["detail"] = detail
        progress_state["updated_at"] = time.time()

    audio_path = Path(audio_path)
    output_dir = Path(output_dir)
    result = PipelineResult(source_file=str(audio_path), success=False)

    try:
        # Step 1: 音频预处理
        update_step("preprocess", "转换格式中...", 0.05, f"处理: {audio_path.name}")
        processed_audio = audio_path

        if pipeline.config.convert_to_wav:
            wav_path = audio_path.with_suffix(".wav")
            processed_audio = pipeline.audio_converter.convert_to_wav(
                audio_path, wav_path,
                sample_rate=pipeline.config.target_sample_rate
            )
        update_step("preprocess", "预处理完成", 0.15, "音频已就绪")

        # Step 2: 语音转写
        update_step("transcribe", "加载模型...", 0.20)
        transcribe_result = pipeline.transcriber.transcribe(
            processed_audio,
            background=background
        )
        result.duration_seconds = transcribe_result.duration_seconds
        update_step("transcribe", "转写完成", 0.50, f"识别 {len(transcribe_result.segments)} 个片段")

        segments = [s.to_dict() if hasattr(s, 'to_dict') else s for s in transcribe_result.segments]

        # Step 3: 说话人区分
        if pipeline.config.enable_diarization and pipeline.diarizer:
            update_step("diarize", "分离说话人中...", 0.55)
            try:
                segments = pipeline.diarizer.merge_with_transcript(processed_audio, segments)
                update_step("diarize", "说话人分离完成", 0.65)
            except Exception as e:
                for seg in segments:
                    if "speaker" not in seg:
                        seg["speaker"] = "SPEAKER_00"
                update_step("diarize", "说话人分离跳过（使用默认标签）", 0.65)
        else:
            for seg in segments:
                if "speaker" not in seg:
                    seg["speaker"] = "SPEAKER_00"
            update_step("diarize", "跳过（未启用）", 0.65)

        # Step 4: 文字清理
        update_step("clean", "清理文本中...", 0.70)
        segments = pipeline.text_cleaner.clean_segments(segments)
        update_step("clean", "清理完成", 0.80, f"保留 {len(segments)} 个有效片段")

        # Step 5: 质量评分
        update_step("score", "评分中...", 0.85)
        score = pipeline.quality_scorer.score_segments(segments)
        result.quality_score = score.overall
        update_step("score", "评分完成", 0.90, f"质量: {score.overall:.0f}/100")

        if pipeline.config.min_quality_score > 0 and score.overall < pipeline.config.min_quality_score:
            result.error = f"质量分数 ({score.overall}) 低于阈值 ({pipeline.config.min_quality_score})"
            return result

        # Step 6: 格式化输出
        update_step("format", "保存结果中...", 0.95)
        output_path = output_dir / f"{audio_path.stem}_curated.json"
        FormatterFactory.format_and_save(
            segments=segments,
            output_path=output_path,
            fmt=pipeline.config.output_format,
            background=background,
            source_file=audio_path.name
        )
        update_step("done", "转写完成!", 1.0, f"质量: {score.overall}/100")

        result.success = True
        result.output_path = str(output_path)
        result.metadata = {
            "model": pipeline.config.model,
            "language": transcribe_result.language,
            "segments_count": len(segments),
            "quality_details": score.details,
            "recommendation": pipeline.quality_scorer.get_recommendation(score)
        }

    except Exception as e:
        result.error = str(e)
        update_step("error", f"错误: {e}", 0.0, str(e))

    return result


# ============================================================
# 主界面
# ============================================================
def main():
    # 标题
    st.title("🎙️ stt-whisper 语音转写系统")
    st.caption(f"版本 {__version__} | 基于 Whisper + 说话人分离 + 语料治理")

    # 初始化 session state
    if "queue" not in st.session_state:
        st.session_state.queue: List[dict] = []
    if "results" not in st.session_state:
        st.session_state.results: List[dict] = []
    if "processing" not in st.session_state:
        st.session_state.processing = False   # 是否正在处理
    if "progress_state" not in st.session_state:
        st.session_state.progress_state: Dict[str, any] = {}  # 跨 rerun 共享进度

    # ============================================================
    # 侧边栏 - 设置
    # ============================================================
    with st.sidebar:
        st.header("⚙️ 设置")

        model_size = st.selectbox(
            "Whisper 模型",
            options=["base", "small", "medium", "large-v3"],
            index=1,
            help="越大越准确但越慢，base最快，large-v3最准确"
        )

        language = st.selectbox(
            "语言",
            options=["zh", "en", "auto"],
            index=0,
            help="zh=中文, en=英文, auto=自动检测"
        )
        if language == "auto":
            language = None

        output_format = st.selectbox(
            "输出格式",
            options=["dialogue", "utterances", "plain", "plain_no_speaker"],
            index=0,
            format_func=lambda x: {
                "dialogue": "对话格式 (训练用)",
                "utterances": "片段格式 (保留时间戳)",
                "plain": "纯文本 (含说话人)",
                "plain_no_speaker": "纯文本 (无说话人)"
            }[x]
        )

        device = st.selectbox(
            "计算设备",
            options=["cpu", "cuda"],
            index=0,
            help="cuda需要GPU支持"
        )

        enable_diarization = st.checkbox("启用说话人分离", value=True)

        st.divider()

        output_dir = st.text_input(
            "输出目录",
            value=str(PACKAGE_DIR / "outputs"),
            help="转写结果保存位置"
        )

        if st.button("🗑️ 清空队列", use_container_width=True):
            st.session_state.queue = []
            st.session_state.results = []
            st.session_state.processing = False
            st.session_state.progress_state = {}
            st.rerun()

    # ============================================================
    # 主内容区 - 三个标签页
    # ============================================================
    tab1, tab2, tab3 = st.tabs(["📁 批量处理", "📄 单文件处理", "📊 结果查看"])

    # ------------------------------------------------------------
    # 标签页1: 批量处理
    # ------------------------------------------------------------
    with tab1:
        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("📂 选择文件夹")
            folder_input = st.text_input(
                "文件夹路径",
                placeholder="/path/to/your/audio/folder",
                help="选择包含音视频文件的文件夹"
            )

            folder_path = Path(folder_input) if folder_input and Path(folder_input).exists() else None

            if folder_path and folder_path.is_dir():
                audio_files = get_audio_files_in_dir(folder_path)
                st.info(f"找到 {len(audio_files)} 个音视频文件")

                if st.button("➕ 全部添加到队列", use_container_width=True):
                    for f in audio_files:
                        if not any(q.get("path") == str(f) for q in st.session_state.queue):
                            st.session_state.queue.append({
                                "name": f.name,
                                "path": str(f),
                                "size": f.stat().st_size,
                                "status": "pending"
                            })
                    st.rerun()
            else:
                st.warning("请输入有效的文件夹路径")
                audio_files = []

        with col2:
            st.subheader("📤 拖拽上传文件")
            uploaded_files = st.file_uploader(
                "拖拽或点击上传",
                type=['mp3', 'wav', 'm4a', 'ogg', 'flac', 'mp4', 'avi', 'mkv'],
                accept_multiple_files=True,
                help="支持多种音频/视频格式"
            )

            if uploaded_files:
                if st.button("➕ 添加到队列", use_container_width=True):
                    for uf in uploaded_files:
                        if not any(q.get("name") == uf.name for q in st.session_state.queue):
                            st.session_state.queue.append({
                                "name": uf.name,
                                "path": uf,
                                "size": uf.size,
                                "status": "pending"
                            })
                    st.rerun()

        # -------------------- 进度显示区域 --------------------
        if st.session_state.processing and st.session_state.progress_state:
            ps = st.session_state.progress_state
            total = ps.get("total_files", 1)
            current = ps.get("current_file", 1)
            step = ps.get("step", "")
            sub_step = ps.get("sub_step", "")
            detail = ps.get("detail", "")
            progress = ps.get("progress", 0.0)

            st.divider()
            st.subheader(f"📊 批量进度 ({current}/{total})")

            # 整体文件进度
            overall = (current - 1 + progress) / total
            st.progress(overall, text=f"整体进度: {overall*100:.0f}%")

            # 当前文件 + 步骤信息
            with st.container():
                cols = st.columns([3, 1])
                cols[0].info(f"📄 {ps.get('current_file_name', '未知文件')}")
                cols[1].metric("步骤", step if step else "-")

                # 步骤进度
                col_steps = st.columns([1, 1, 1, 1, 1, 1])
                steps = [
                    ("init", "初始化"),
                    ("preprocess", "预处理"),
                    ("transcribe", "转写"),
                    ("diarize", "说话人"),
                    ("clean", "清理"),
                    ("score", "评分"),
                ]
                for idx, (s_key, s_label) in enumerate(steps):
                    active = (step == s_key)
                    col_steps[idx].progress(
                        1.0 if (steps.index((step, _)) > idx if step in [s[0] for s in steps] else False) else (progress if active else 0.0),
                        text=s_label
                    )
                    if active:
                        col_steps[idx].caption(f"🔄 {sub_step}")
                    else:
                        col_steps[idx].caption("✅" if steps.index((step, _)) > idx else "⏳")

                st.caption(f"💬 {detail}")

            st.divider()

        # -------------------- 队列显示 --------------------
        st.subheader(f"📋 处理队列 ({len(st.session_state.queue)} 个文件)")

        if st.session_state.queue:
            header_cols = st.columns([3, 1, 1, 1])
            header_cols[0].caption("文件名")
            header_cols[1].caption("大小")
            header_cols[2].caption("状态")
            header_cols[3].caption("操作")

            for i, item in enumerate(st.session_state.queue):
                cols = st.columns([3, 1, 1, 1])

                cols[0].text(item["name"])
                size_mb = item["size"] / (1024 * 1024)
                cols[1].text(f"{size_mb:.1f} MB")

                status = item.get("status", "pending")
                if status == "done":
                    cols[2].success("已完成")
                elif status == "processing":
                    cols[2].warning("处理中")
                elif status == "error":
                    cols[2].error("失败")
                else:
                    cols[2].info("等待中")

                if cols[3].button("❌", key=f"rm_{i}"):
                    st.session_state.queue.pop(i)
                    st.rerun()

                st.divider()
        else:
            st.info("队列为空，请添加文件")

        # 开始处理按钮
        st.divider()
        col_start, col_stop = st.columns([2, 1])
        auto_start = col_start.checkbox("▶️ 开启自动处理", value=True, help="开启后自动按顺序处理队列")

        queue_total = len(st.session_state.queue)
        done_count = sum(1 for q in st.session_state.queue if q.get("status") == "done")
        error_count = sum(1 for q in st.session_state.queue if q.get("status") == "error")

        if queue_total > 0:
            col_start.write(f"已完成: {done_count} | 失败: {error_count} | 待处理: {queue_total - done_count - error_count}")

        is_processing = st.session_state.processing
        col_stop.button(
            "🛑 停止" if is_processing else "🚀 开始处理",
            use_container_width=True,
            disabled=(queue_total == 0 and not is_processing)
        )

        if not is_processing and col_stop.button("🚀 开始处理", use_container_width=True, disabled=queue_total == 0, key="start_batch"):
            # 创建 pipeline
            st.session_state.processing = True
            st.session_state.progress_state = {
                "total_files": queue_total,
                "current_file": 0,
                "current_file_name": "",
                "step": "init",
                "sub_step": "正在初始化...",
                "progress": 0.0,
                "detail": "",
                "updated_at": time.time(),
            }

            config = PipelineConfig(
                model=model_size,
                language=language,
                device=device,
                enable_diarization=enable_diarization,
                output_format=output_format
            )
            pipeline = STTPipeline(config)

            for i, item in enumerate(st.session_state.queue):
                if item.get("status") in ["done", "processing"]:
                    continue

                st.session_state.progress_state["current_file"] = i + 1
                st.session_state.progress_state["current_file_name"] = item["name"]
                item["status"] = "processing"
                st.session_state.queue[i] = item

                try:
                    audio_path = item["path"]

                    if hasattr(audio_path, 'read'):
                        suffix = Path(item["name"]).suffix
                        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                            tmp.write(audio_path.read())
                            audio_path = tmp.name

                    result = process_single_with_progress(
                        pipeline=pipeline,
                        audio_path=Path(audio_path),
                        background=Path(audio_path).stem,
                        output_dir=Path(output_dir),
                        progress_state=st.session_state.progress_state
                    )

                    item["status"] = "done" if result.success else "error"
                    item["result"] = result.to_dict()
                    st.session_state.results.append(item)

                except Exception as e:
                    item["status"] = "error"
                    item["error"] = str(e)
                    st.error(f"处理失败: {item['name']} - {e}")

                st.session_state.queue[i] = item

            st.session_state.processing = False
            st.session_state.progress_state = {}
            st.rerun()

    # ------------------------------------------------------------
    # 标签页2: 单文件处理
    # ------------------------------------------------------------
    with tab2:
        st.subheader("📄 单文件快速转写")

        uploaded_file = st.file_uploader(
            "选择音视频文件",
            type=['mp3', 'wav', 'm4a', 'ogg', 'flac', 'mp4', 'avi', 'mkv'],
            help="支持多种音频/视频格式"
        )

        if uploaded_file:
            col_meta1, col_meta2 = st.columns([1, 1])
            bg_input = col_meta1.text_input("背景信息（可选）", placeholder="如：产品规划会议")
            fmt_select = col_meta2.selectbox(
                "输出格式",
                options=["dialogue", "utterances", "plain"],
                index=0,
                format_func=lambda x: {"dialogue": "对话格式", "utterances": "片段格式", "plain": "纯文本"}[x]
            )

            background = bg_input if bg_input else Path(uploaded_file.name).stem

            # 进度显示
            if "single_progress" not in st.session_state:
                st.session_state.single_progress = {}

            sp = st.session_state.single_progress
            if sp.get("active"):
                total = sp.get("total", 1)
                current = sp.get("current", 1)
                step = sp.get("step", "")
                sub_step = sp.get("sub_step", "")
                detail = sp.get("detail", "")
                progress = sp.get("progress", 0.0)

                st.divider()
                st.subheader("📊 当前进度")

                overall = progress
                st.progress(overall, text=f"整体进度: {overall*100:.0f}%")

                col_info1, col_info2 = st.columns([3, 1])
                col_info1.info(f"📄 {uploaded_file.name}")
                col_info2.metric("步骤", step if step else "-")

                col_steps = st.columns([1, 1, 1, 1, 1, 1])
                steps = [
                    ("init", "初始化"),
                    ("preprocess", "预处理"),
                    ("transcribe", "转写"),
                    ("diarize", "说话人"),
                    ("clean", "清理"),
                    ("score", "评分"),
                ]
                for idx, (s_key, s_label) in enumerate(steps):
                    step_idx = next((steps.index((s[0], s[1])) for s in steps if s[0] == step), -1)
                    done = step_idx > idx
                    active = step == s_key
                    col_steps[idx].progress(1.0 if done else (progress if active else 0.0), text=s_label)
                    if active:
                        col_steps[idx].caption(f"🔄 {sub_step}")
                    else:
                        col_steps[idx].caption("✅" if done else "⏳")

                st.caption(f"💬 {detail}")
                st.divider()

            if st.button("🎙️ 开始转写", use_container_width=True, key="single_start"):
                os.makedirs(output_dir, exist_ok=True)

                suffix = Path(uploaded_file.name).suffix
                with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                    tmp.write(uploaded_file.read())
                    tmp_path = tmp.name

                sp = st.session_state.single_progress = {
                    "active": True,
                    "total": 1,
                    "current": 1,
                    "step": "init",
                    "sub_step": "正在初始化模型...",
                    "progress": 0.0,
                    "detail": "请稍候",
                    "updated_at": time.time(),
                }

                config = PipelineConfig(
                    model=model_size,
                    language=language,
                    device=device,
                    enable_diarization=enable_diarization,
                    output_format=output_format
                )
                pipeline = STTPipeline(config)

                try:
                    result = process_single_with_progress(
                        pipeline=pipeline,
                        audio_path=Path(tmp_path),
                        background=background,
                        output_dir=Path(output_dir),
                        progress_state=sp
                    )

                    sp["active"] = False

                    if result.success:
                        st.success(f"✅ 转写完成！时长: {format_time(result.duration_seconds)}, 质量: {result.quality_score:.0f}/100")

                        st.subheader("📝 转写结果预览")
                        result_data = result.to_dict()

                        if result.output_path and Path(result.output_path).exists():
                            with open(result.output_path, "r", encoding="utf-8") as f:
                                content = f.read()
                                st.text_area("结果内容", value=content, height=300, label_visibility="collapsed")

                            with open(result.output_path, "r", encoding="utf-8") as f:
                                st.download_button(
                                    "📥 下载结果",
                                    f.read(),
                                    file_name=Path(result.output_path).name,
                                    mime="application/json",
                                    use_container_width=True
                                )
                    else:
                        st.error(f"转写失败: {result.error}")

                except Exception as e:
                    sp["active"] = False
                    st.error(f"处理出错: {e}")
                finally:
                    if Path(tmp_path).exists():
                        os.unlink(tmp_path)
                    st.rerun()

    # ------------------------------------------------------------
    # 标签页3: 结果查看
    # ------------------------------------------------------------
    with tab3:
        st.subheader("📊 转写结果查看")

        result_dir = Path(output_dir)
        if result_dir.exists():
            results = load_results_from_dir(result_dir)

            if results:
                st.info(f"共 {len(results)} 个转写结果")

                for res in reversed(results):
                    with st.container():
                        cols = st.columns([4, 1, 1, 1])

                        source = res.get("source_file", "未知")
                        cols[0].text(source)

                        dur = res.get("duration_seconds", 0)
                        cols[1].text(format_time(dur))

                        score = res.get("quality_score", 0)
                        cols[2].text(get_quality_label(score))

                        result_file = res.get("_result_file")
                        if result_file and cols[3].button("👁️", key=f"view_{result_file}"):
                            st.session_state.viewing_result = result_file
                            st.rerun()

                        st.divider()

                if "viewing_result" in st.session_state and st.session_state.viewing_result:
                    result_file = st.session_state.viewing_result
                    with open(result_file, "r", encoding="utf-8") as f:
                        content = json.load(f)

                    st.subheader(f"📄 {Path(result_file).stem}")

                    col1, col2, col3 = st.columns(3)
                    col1.metric("时长", format_time(content.get("duration_seconds", 0)))
                    col2.metric("质量", f"{content.get('quality_score', 0):.0f}/100")
                    col3.metric("片段数", content.get("metadata", {}).get("segments_count", "N/A"))

                    st.text_area("JSON内容", value=json.dumps(content, ensure_ascii=False, indent=2), height=400, label_visibility="collapsed")

                    with open(result_file, "r", encoding="utf-8") as f:
                        st.download_button(
                            "📥 下载 JSON",
                            f.read(),
                            file_name=Path(result_file).name,
                            mime="application/json",
                            use_container_width=True
                        )

                    if st.button("❌ 关闭"):
                        del st.session_state.viewing_result
                        st.rerun()
            else:
                st.info("暂无转写结果，请先进行转写")
        else:
            st.info(f"输出目录不存在: {output_dir}")


# ============================================================
# 启动入口
# ============================================================
if __name__ == "__main__":
    main()
