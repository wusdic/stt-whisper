"""
stt-whisper Streamlit Web Interface
语音转写 Web 界面 - 支持单文件和批量文件夹处理
浅色主题 · 实时转写 · 现代卡片布局
"""

import streamlit as st
import tempfile
import os
import json
import time
import glob
import threading
import queue
from pathlib import Path
from typing import List, Optional, Dict
from dataclasses import dataclass, field

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
from stt_whisper.web.db import (
    init_db, get_or_create_default_project, list_projects,
    create_project, list_tasks, add_task, update_task_status,
    clear_tasks, get_project_output_dir, get_project,
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

# ============================================================
# 自定义浅色主题 CSS（参考 Apple/Linear/ChatGPT 风格）
# ============================================================
st.markdown("""
<style>
    /* === 全局重置 === */
    .stApp {
        background: #F7F8FA !important;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'PingFang SC', 'Microsoft YaHei', sans-serif !important;
        color: #1A1A2E !important;
    }

    /* === 侧边栏（设置面板）=== */
    [data-testid="stSidebar"] {
        background: #FFFFFF !important;
        border-right: 1px solid #E8E9ED !important;
        padding: 0 1rem !important;
    }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
        color: #1A1A2E !important;
    }

    /* === 主内容区 === */
    .block-container {
        padding: 1.5rem 2rem !important;
        background: #F7F8FA !important;
    }

    /* === 标题 === */
    h1, h2, h3 {
        color: #1A1A2E !important;
        font-weight: 600 !important;
    }

    /* === 标签页（Tab）=== */
    [data-testid="stTab"] {
        font-size: 0.95rem !important;
        font-weight: 500 !important;
        color: #6B7280 !important;
        padding: 0.6rem 1.2rem !important;
        border-radius: 8px 8px 0 0 !important;
    }
    [data-testid="stTab"][data-selected="true"] {
        color: #2563EB !important;
        border-bottom: 2px solid #2563EB !important;
        background: #FFFFFF !important;
    }
    [data-testid="stTabBar"] {
        background: #F7F8FA !important;
        border-bottom: 1px solid #E8E9ED !important;
        border-radius: 8px 8px 0 0 !important;
    }

    /* === 卡片容器 === */
    .card {
        background: #FFFFFF;
        border: 1px solid #E8E9ED;
        border-radius: 12px;
        padding: 1.25rem 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }

    /* === 进度卡片 === */
    .progress-card {
        background: #FFFFFF;
        border: 1px solid #E8E9ED;
        border-radius: 16px;
        padding: 2rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 16px rgba(37,99,235,0.08);
    }

    /* === 成功/错误/警告消息 === */
    .success-msg { background: #F0FDF4; border-left: 4px solid #22C55E; border-radius: 8px; padding: 1rem 1.25rem; color: #166534; font-size: 0.9rem; }
    .error-msg { background: #FEF2F2; border-left: 4px solid #EF4444; border-radius: 8px; padding: 1rem 1.25rem; color: #991B1B; font-size: 0.9rem; }
    .info-msg { background: #EFF6FF; border-left: 4px solid #2563EB; border-radius: 8px; padding: 1rem 1.25rem; color: #1E40AF; font-size: 0.9rem; }

    /* === 队列项 === */
    .queue-item {
        background: #FFFFFF;
        border: 1px solid #E8E9ED;
        border-radius: 10px;
        padding: 0.85rem 1.25rem;
        margin-bottom: 0.5rem;
        display: flex;
        align-items: center;
        gap: 1rem;
    }
    .queue-item:hover { border-color: #BFDBFE; background: #F8FAFF; }

    /* === 实时转写区 === */
    .transcript-box {
        background: #FFFFFF;
        border: 1px solid #E8E9ED;
        border-radius: 12px;
        padding: 1.5rem;
        max-height: 420px;
        overflow-y: auto;
        font-size: 0.9rem;
        line-height: 1.8;
    }
    .transcript-segment {
        padding: 0.4rem 0;
        border-bottom: 1px solid #F3F4F6;
        animation: fadeIn 0.3s ease;
    }
    .transcript-segment:last-child { border-bottom: none; }
    .speaker-tag {
        display: inline-block;
        background: #DBEAFE;
        color: #1D4ED8;
        font-size: 0.75rem;
        font-weight: 600;
        padding: 0.1rem 0.5rem;
        border-radius: 4px;
        margin-right: 0.5rem;
        font-family: 'SF Mono', monospace;
    }
    .time-tag {
        color: #9CA3AF;
        font-size: 0.75rem;
        margin-right: 0.5rem;
        font-family: 'SF Mono', monospace;
    }
    .transcript-text { color: #374151; }

    /* === 质量分数字段 === */
    .quality-badge {
        display: inline-block;
        padding: 0.2rem 0.7rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .q-green { background: #DCFCE7; color: #166534; }
    .q-yellow { background: #FEF9C3; color: #854D0E; }
    .q-orange { background: #FFEDD5; color: #9A3412; }
    .q-red { background: #FEE2E2; color: #991B1B; }

    /* === 按钮样式 === */
    .stButton > button {
        border-radius: 8px !important;
        font-weight: 500 !important;
        transition: all 0.15s ease !important;
        border: none !important;
    }

    /* === 上传区域 === */
    [data-testid="stFileUploader"] section {
        background: #FFFFFF !important;
        border: 2px dashed #CBD5E1 !important;
        border-radius: 12px !important;
    }
    [data-testid="stFileUploader"]:hover section {
        border-color: #2563EB !important;
        background: #EFF6FF !important;
    }

    /* === 进度条（Streamlit 原生美化）=== */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #2563EB, #60A5FA) !important;
        border-radius: 8px !important;
        height: 8px !important;
    }

    /* === 指标卡 === */
    [data-testid="stMetric"] {
        background: #FFFFFF;
        border: 1px solid #E8E9ED;
        border-radius: 10px;
        padding: 1rem;
    }
    [data-testid="stMetricLabel"] { color: #6B7280 !important; font-size: 0.8rem !important; }
    [data-testid="stMetricValue"] { color: #1A1A2E !important; font-size: 1.4rem !important; font-weight: 700 !important; }

    /* === 分隔线 === */
    hr { border-color: #E8E9ED !important; }

    /* === 滚动条 === */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: #F1F5F9; border-radius: 3px; }
    ::-webkit-scrollbar-thumb { background: #CBD5E1; border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: #94A3B8; }

    /* === 动画 === */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(4px); }
        to { opacity: 1; transform: translateY(0); }
    }
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
    .pulsing { animation: pulse 1.5s ease-in-out infinite; }

    /* === 步骤指示器 === */
    .step-row { display: flex; align-items: center; gap: 0.75rem; margin-bottom: 0.5rem; }
    .step-dot { width: 28px; height: 28px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 0.75rem; font-weight: 700; flex-shrink: 0; }
    .step-done { background: #22C55E; color: white; }
    .step-active { background: #2563EB; color: white; }
    .step-wait { background: #E8E9ED; color: #9CA3AF; }
    .step-label { font-size: 0.85rem; color: #374151; font-weight: 500; }
    .step-active-label { color: #2563EB; }
</style>
""", unsafe_allow_html=True)


# ============================================================
# 辅助函数
# ============================================================
def get_audio_files_in_dir(dir_path: Path) -> List[Path]:
    AUDIO_EXTS = {'.mp3', '.wav', '.m4a', '.ogg', '.flac', '.mp4', '.avi', '.mkv'}
    files = []
    for ext in AUDIO_EXTS:
        files.extend(dir_path.glob(f"*{ext}"))
        files.extend(dir_path.glob(f"*{ext.upper()}"))
    return sorted(set(files))


def format_time(seconds: float) -> str:
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


def get_quality_style(score: float) -> str:
    if score >= 80:
        return "🟢 优秀", "q-green"
    elif score >= 60:
        return "🟡 良好", "q-yellow"
    elif score >= 40:
        return "🟠 一般", "q-orange"
    else:
        return "🔴 较差", "q-red"


def load_results_from_dir(output_dir: Path) -> List[dict]:
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


def render_step_row(step_key: str, step_label: str, current_step: str):
    """渲染单个步骤指示器"""
    steps_order = ["init", "preprocess", "transcribe", "diarize", "clean", "score", "done", "error"]
    current_idx = steps_order.index(current_step) if current_step in steps_order else 0
    step_idx = steps_order.index(step_key) if step_key in steps_order else 0

    if step_idx < current_idx:
        dot_class = "step-done"
        label_class = ""
        icon = "✓"
    elif step_idx == current_idx:
        dot_class = "step-active"
        label_class = "step-active-label"
        icon = "●"
    else:
        dot_class = "step-wait"
        label_class = ""
        icon = str(step_idx + 1)

    st.markdown(
        f'<div class="step-row">'
        f'<div class="step-dot {dot_class}">{icon}</div>'
        f'<span class="step-label {label_class}">{step_label}</span>'
        f'</div>',
        unsafe_allow_html=True
    )


# ============================================================
# 主界面
# ============================================================
def main():
    # 标题区
    st.markdown("""
    <div style="margin-bottom: 1.5rem;">
        <h1 style="font-size: 1.8rem; margin-bottom: 0.25rem;">🎙️ stt-whisper 语音转写</h1>
        <p style="color: #6B7280; font-size: 0.85rem; margin: 0;">Whisper + 说话人分离 + 语料治理 · v""" + __version__ + """</p>
    </div>
    """, unsafe_allow_html=True)

    # ============================================================
    # 处理轮询循环（解决 Streamlit 单线程阻塞问题）
    # 每个 rerun 都检查后台 worker 的输出，实现实时进度更新
    # ============================================================
    if st.session_state.get("processing"):
        wm = st.session_state.get("worker_manager")
        bcfg = st.session_state.get("batch_config")
        bqueue = st.session_state.get("batch_queue", [])
        ps = st.session_state.progress_state

        if wm and bcfg and ps:
            # 非阻塞轮询：消费 queue 中所有可用消息
            while True:
                msg = wm.poll()
                if msg is None:
                    break
                if msg["type"] == "segment":
                    # 实时转写片段已通过 live_segments.append 追加
                    # 触发 UI 更新（Streamlit 的 live segment 渲染在下方）
                    pass
                elif msg["type"] == "done":
                    # 当前文件完成，从 batch_queue 移到 results，并持久化到 DB
                    for i, item in enumerate(bqueue):
                        if item.get("status") == "processing":
                            item["status"] = "done"
                            result_dict = msg["result"].to_dict() if hasattr(msg["result"], "to_dict") else msg["result"]
                            item["result"] = result_dict
                            st.session_state.results.append(item)
                            # 同步到 DB
                            if item.get("db_id"):
                                update_task_status(
                                    item["db_id"], "done",
                                    result_json=json.dumps(result_dict, ensure_ascii=False)
                                )
                            bqueue.pop(i)
                            break
                elif msg["type"] == "error":
                    for i, item in enumerate(bqueue):
                        if item.get("status") == "processing":
                            item["status"] = "error"
                            item["error"] = msg["error"]
                            st.session_state.results.append(item)
                            # 同步到 DB
                            if item.get("db_id"):
                                update_task_status(item["db_id"], "error", error=msg["error"])
                            bqueue.pop(i)
                            break

            # 如果 worker 空闲了，启动下一个文件
            if wm.is_alive():
                pass  # 仍在运行，继续轮询
            elif not bqueue:
                # 全部完成
                st.session_state.processing = False
                st.session_state.progress_state = {}
                st.session_state.live_segments = []
                # 同步 queue
                st.session_state.queue = st.session_state.results[:]
                st.rerun()
            else:
                # 启动下一个文件
                item = bqueue[0]
                item["status"] = "processing"
                ps["current_file"] = len(st.session_state.results) + 1
                ps["current_file_name"] = item["name"]
                ps["step"] = "init"
                ps["sub_step"] = "初始化模型..."
                ps["progress"] = 0.0
                ps["transcribed_seconds"] = 0.0

                try:
                    cfg = PipelineConfig(
                        model=bcfg["model_size"],
                        language=bcfg["language"],
                        device=bcfg["device"],
                        enable_diarization=bcfg["enable_diarization"],
                        output_format=bcfg["output_format"],
                    )
                    pipeline = STTPipeline(cfg)
                    audio_path = item["path"]

                    if hasattr(audio_path, "read"):
                        suffix = Path(item["name"]).suffix
                        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                            tmp.write(audio_path.read())
                            audio_path = tmp.name

                    wm.start(
                        pipeline=pipeline,
                        audio_path=Path(audio_path),
                        background=Path(audio_path).stem,
                        output_dir=Path(bcfg["output_dir"]),
                        progress_state=ps,
                        live_segments=st.session_state.live_segments,
                        enable_diarization=bcfg["enable_diarization"],
                    )
                except Exception as e:
                    item["status"] = "error"
                    item["error"] = str(e)
                    ps["step"] = "error"
                    ps["sub_step"] = f"启动失败: {e}"

        # watchdog：检查是否卡住（超过 120s 无更新）
        ps_updated_at = ps.get("updated_at", 0)
        if time.time() - ps_updated_at > 120 and wm and wm.is_alive():
            # 疑似卡住，标记
            ps["sub_step"] = "⚠️ 检测到长时间无响应，可能卡住"
            ps["watchdog_triggered"] = True

        # 自动刷新（每 1 秒 rerun 一次，保持进度实时更新）
        time.sleep(0.5)
        st.rerun()

    # ============================================================
    # 初始化 session state（页面刷新后从 DB 恢复）
    # ============================================================
    if "queue" not in st.session_state:
        st.session_state.queue: List[dict] = []
    if "results" not in st.session_state:
        st.session_state.results: List[dict] = []
    if "processing" not in st.session_state:
        st.session_state.processing = False
    if "progress_state" not in st.session_state:
        st.session_state.progress_state: Dict[str, any] = {}
    if "live_segments" not in st.session_state:
        st.session_state.live_segments: List[dict] = []

    # 初始化数据库（项目/任务持久化）
    init_db()
    # 确保默认项目存在
    default_project_id = get_or_create_default_project()
    # 每次页面加载时，从 DB 恢复当前项目的待处理任务到 session queue
    if "current_project_id" not in st.session_state:
        st.session_state.current_project_id = default_project_id
        # 恢复该项目的 pending 任务（仅非 processing 状态）
        db_tasks = list_tasks(default_project_id, status="pending")
        # 也恢复 done 状态的（用于结果展示）
        db_done = list_tasks(default_project_id, status="done")
        for t in db_tasks:
            st.session_state.queue.append({
                "db_id": t["id"],
                "name": t["name"],
                "path": t["path"],
                "size": t.get("size", 0),
                "status": "pending"
            })
        for t in db_done:
            st.session_state.results.append({
                "db_id": t["id"],
                "name": t["name"],
                "path": t["path"],
                "size": t.get("size", 0),
                "status": "done",
                "result": json.loads(t["result_json"]) if t.get("result_json") else None,
            })

    # ============================================================
    # 侧边栏
    # ============================================================
    with st.sidebar:
        st.markdown("### ⚙️ 设置", unsafe_allow_html=True)

        # --- 项目选择器 ---
        st.markdown("**📁 项目**")
        project_list = list_projects()
        project_options = {p["name"]: p["id"] for p in project_list}
        project_names = list(project_options.keys())

        # 当前选中项目
        current_pid = st.session_state.get("current_project_id", 1)
        current_project = get_project(current_pid)
        current_idx = next((i for i, p in enumerate(project_list) if p["id"] == current_pid), 0)

        selected_project_name = st.selectbox(
            "选择项目",
            options=project_names,
            index=current_idx,
            label_visibility="collapsed",
        )

        # 切换项目时清空当前队列
        if st.session_state.get("_last_project_id") != current_pid and st.session_state.get("_last_project_id") is not None:
            st.session_state.queue = []
            st.session_state.results = []
            st.session_state.processing = False
            st.session_state.progress_state = {}
            st.session_state.live_segments = []
            # 重新加载新项目的任务
            db_tasks = list_tasks(current_pid, status="pending")
            db_done = list_tasks(current_pid, status="done")
            for t in db_tasks:
                st.session_state.queue.append({
                    "db_id": t["id"], "name": t["name"], "path": t["path"],
                    "size": t.get("size", 0), "status": "pending"
                })
            for t in db_done:
                st.session_state.results.append({
                    "db_id": t["id"], "name": t["name"], "path": t["path"],
                    "size": t.get("size", 0), "status": "done",
                    "result": json.loads(t["result_json"]) if t.get("result_json") else None,
                })
        st.session_state._last_project_id = current_pid

        # 新建项目
        col_new1, col_new2 = st.columns([3, 1])
        with col_new1:
            new_proj_name = st.text_input(
                "新建项目名称", placeholder="如：产品规划会议",
                label_visibility="collapsed"
            )
        with col_new2:
            st.markdown("")  # spacer
            if st.button("➕", help="创建新项目"):
                if new_proj_name and new_proj_name.strip():
                    new_pid = create_project(new_proj_name.strip())
                    st.session_state.current_project_id = new_pid
                    st.session_state.queue = []
                    st.session_state.results = []
                    st.rerun()

        # 显示当前项目的输出目录
        if current_project:
            st.caption(f"📂 输出: `{current_project['output_dir']}`")

        st.divider()

        model_size = st.selectbox(
            "Whisper 模型",
            options=["tiny", "base", "small", "medium", "large-v3"],
            index=2,
            help="small 性价比最高，large-v3 最准确"
        )

        language = st.selectbox(
            "语言",
            options=["zh", "en", "auto"],
            index=0,
        )
        if language == "auto":
            language = None

        output_format = st.selectbox(
            "输出格式",
            options=["dialogue", "utterances", "plain", "plain_no_speaker"],
            index=0,
            format_func=lambda x: {
                "dialogue": "📝 对话格式 (训练用)",
                "utterances": "⏱️ 片段格式 (保留时间戳)",
                "plain": "📄 纯文本 (含说话人)",
                "plain_no_speaker": "📃 纯文本 (无说话人)"
            }[x]
        )

        device = st.selectbox("计算设备", options=["cpu", "cuda"], index=0)
        enable_diarization = st.checkbox("启用说话人分离", value=False)  # 默认关闭（pyannote需要授权）

        st.divider()

        # 动态输出目录（按项目隔离）
        project_output_dir = get_project_output_dir(current_pid)
        output_dir = st.text_input(
            "输出目录",
            value=project_output_dir,
        )

        st.divider()

        if st.button("🗑️ 清空队列", use_container_width=True):
            # 清空 DB 中的 pending 和 done 任务
            clear_tasks(current_pid, status="pending")
            clear_tasks(current_pid, status="done")
            st.session_state.queue = []
            st.session_state.results = []
            st.session_state.processing = False
            st.session_state.progress_state = {}
            st.session_state.live_segments = []
            st.rerun()

        st.markdown("""
        <div style="font-size: 0.75rem; color: #9CA3AF; text-align: center; margin-top: 1rem;">
            模型需联网下载<br/>
            设置 HF_ENDPOINT=hf-mirror.com
        </div>
        """, unsafe_allow_html=True)

    # ============================================================
    # 主内容区
    # ============================================================
    tab1, tab2, tab3 = st.tabs(["📁 批量处理", "📄 单文件处理", "📊 结果查看"])

    # ------------------------------------------------------------
    # 标签页1: 批量处理
    # ------------------------------------------------------------
    with tab1:
        col_left, col_right = st.columns([1, 1])

        with col_left:
            with st.container():
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown("**📂 选择文件夹**")
                folder_input = st.text_input(
                    "文件夹路径",
                    placeholder="/path/to/audio/folder",
                    label_visibility="collapsed",
                )
                folder_path = Path(folder_input) if folder_input and Path(folder_input).exists() else None

                if folder_path and folder_path.is_dir():
                    audio_files = get_audio_files_in_dir(folder_path)
                    st.success(f"✅ 找到 **{len(audio_files)}** 个音视频文件")

                    if st.button("➕ 全部添加到队列", use_container_width=True):
                        current_pid = st.session_state.get("current_project_id", 1)
                        for f in audio_files:
                            if not any(q.get("path") == str(f) for q in st.session_state.queue):
                                db_id = add_task(current_pid, f.name, str(f), f.stat().st_size)
                                st.session_state.queue.append({
                                    "db_id": db_id,
                                    "name": f.name,
                                    "path": str(f),
                                    "size": f.stat().st_size,
                                    "status": "pending"
                                })
                        st.rerun()
                else:
                    if folder_input:
                        st.error("❌ 文件夹不存在或无效")
                st.markdown('</div>', unsafe_allow_html=True)

        with col_right:
            with st.container():
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown("**📤 拖拽上传文件**")
                uploaded_files = st.file_uploader(
                    "支持 mp3/wav/m4a/ogg/flac/mp4/avi/mkv",
                    type=['mp3', 'wav', 'm4a', 'ogg', 'flac', 'mp4', 'avi', 'mkv'],
                    accept_multiple_files=True,
                    label_visibility="collapsed",
                )
                if uploaded_files:
                    if st.button("➕ 添加到队列", use_container_width=True):
                        current_pid = st.session_state.get("current_project_id", 1)
                        for uf in uploaded_files:
                            if not any(q.get("name") == uf.name for q in st.session_state.queue):
                                db_id = add_task(current_pid, uf.name, uf.name, uf.size)
                                st.session_state.queue.append({
                                    "db_id": db_id,
                                    "name": uf.name,
                                    "path": uf,
                                    "size": uf.size,
                                    "status": "pending"
                                })
                        st.rerun()
                st.markdown('</div>', unsafe_allow_html=True)

        # -------------------- 进度 / 实时转写区 --------------------
        ps = st.session_state.progress_state
        if st.session_state.processing and ps:
            total = ps.get("total_files", 1)
            current = ps.get("current_file", 1)
            step = ps.get("step", "")
            sub_step = ps.get("sub_step", "")
            progress = ps.get("progress", 0.0)
            detail = ps.get("detail", "")
            transcribed_s = ps.get("transcribed_seconds", 0.0)
            total_files = ps.get("total_files", 1)
            current_file = ps.get("current_file", 1)
            overall = (current_file - 1 + progress) / total_files if total_files > 0 else 0

            st.markdown('<div class="progress-card">', unsafe_allow_html=True)

            # 头部：文件 + 进度
            col_head1, col_head2 = st.columns([3, 1])
            with col_head1:
                st.markdown(f"### 📄 {ps.get('current_file_name', '未知文件')}", unsafe_allow_html=True)
            with col_head2:
                st.markdown(f"**{current_file}/{total_files}** 文件", unsafe_allow_html=True)

            # 整体进度条
            st.progress(overall, text=f"整体进度: {overall*100:.0f}%")

            # 转写时间进度（仅转写步骤时显示）
            if step == "transcribe" and transcribed_s > 0:
                # 从 detail 中解析总时长（如 "识别 45 个片段，120s"）
                import re
                m = re.search(r"(\d+)\s*s", detail)
                if m:
                    total_s = float(m.group(1))
                    time_pct = min(transcribed_s / total_s, 1.0) if total_s > 0 else 0
                    st.progress(time_pct, text=f"⏱️ 转写进度: {int(transcribed_s)}s / {int(total_s)}s ({time_pct*100:.0f}%)")

            # 步骤指示器（左右两列）+ 状态高亮
            col_steps_l, col_steps_r = st.columns(2)
            steps_left = [
                ("init", "🔄 初始化模型"),
                ("preprocess", "🔄 音频预处理"),
                ("transcribe", "🔄 语音转写"),
                ("diarize", "🔄 说话人分离"),
            ]
            steps_right = [
                ("clean", "🔄 文本清理"),
                ("score", "🔄 质量评分"),
                ("format", "🔄 保存结果"),
                ("done", "✅ 完成"),
            ]

            with col_steps_l:
                for s_key, s_label in steps_left:
                    render_step_row(s_key, s_label, step)
            with col_steps_r:
                for s_key, s_label in steps_right:
                    render_step_row(s_key, s_label, step)

            # 当前操作说明
            if sub_step:
                st.caption(f"💬 {sub_step}")

            # 实时转写内容（边转写边展示）
            live = st.session_state.get("live_segments", [])
            if live:
                with st.container():
                    st.markdown("**📝 实时转写内容**")
                    st.markdown('<div class="transcript-box" id="live_transcript">', unsafe_allow_html=True)
                    for seg in live[-20:]:  # 只显示最近20条
                        speaker = seg.get("speaker", "说话人")
                        start = seg.get("start_time", 0)
                        text = seg.get("text", "")
                        mins, secs = divmod(int(start), 60)
                        st.markdown(
                            f'<div class="transcript-segment">'
                            f'<span class="time-tag">{mins:02d}:{secs:02d}</span>'
                            f'<span class="speaker-tag">{speaker}</span>'
                            f'<span class="transcript-text">{text}</span>'
                            f'</div>',
                            unsafe_allow_html=True
                        )
                    st.markdown('</div>', unsafe_allow_html=True)
                    # 自动滚动到底部
                    st.markdown("<script>var el=document.getElementById('live_transcript');if(el) el.scrollTop=el.scrollHeight;</script>", unsafe_allow_html=True)

            st.markdown('</div>', unsafe_allow_html=True)

        # -------------------- 队列 --------------------
        st.markdown(f"### 📋 处理队列 · {len(st.session_state.queue)} 个文件")

        if st.session_state.queue:
            for i, item in enumerate(st.session_state.queue):
                status = item.get("status", "pending")
                size_mb = item["size"] / (1024 * 1024)

                if status == "done":
                    badge = "✅"
                elif status == "processing":
                    badge = "⏳"
                elif status == "error":
                    badge = "❌"
                else:
                    badge = "⬜"

                col_q1, col_q2, col_q3, col_q4 = st.columns([4, 1, 1, 0.5])
                col_q1.text(item["name"])
                col_q2.text(f"{size_mb:.1f} MB")
                col_q3.text(status)
                col_q4.button("❌", key=f"rm_{i}")

            st.divider()
        else:
            st.markdown("""
            <div style="text-align: center; padding: 2rem; color: #9CA3AF; font-size: 0.9rem;">
                队列为空，请上传文件或输入文件夹路径
            </div>
            """, unsafe_allow_html=True)

        # 底部操作栏
        col_start, col_info = st.columns([1, 3])
        queue_total = len(st.session_state.queue)
        done_count = sum(1 for q in st.session_state.queue if q.get("status") == "done")
        error_count = sum(1 for q in st.session_state.queue if q.get("status") == "error")

        with col_start:
            is_processing = st.session_state.processing
            if is_processing:
                st.button("🛑 停止处理", use_container_width=True, disabled=True)
            else:
                st.button("🛑 停止", use_container_width=True)
            if not is_processing and queue_total > 0:
                if st.button("🚀 开始处理队列", use_container_width=True):
                    st.session_state.processing = True
                    st.session_state.live_segments = []
                    st.session_state.progress_state = {
                        "total_files": queue_total,
                        "current_file": 0,
                        "current_file_name": "",
                        "step": "init",
                        "sub_step": "正在初始化...",
                        "progress": 0.0,
                        "detail": "",
                        "transcribed_seconds": 0.0,
                        "updated_at": time.time(),
                    }
                    # 创建 worker manager（每个批次独立实例）
                    st.session_state.worker_manager = WorkerManager()
                    # 保存 pipeline config（不在 worker 中重建模型）
                    st.session_state.batch_config = {
                        "model_size": model_size,
                        "language": language,
                        "device": device,
                        "enable_diarization": enable_diarization,
                        "output_format": output_format,
                        "output_dir": output_dir,
                    }
                    st.session_state.batch_queue = list(st.session_state.queue)
                    st.rerun()

        with col_info:
            if queue_total > 0:
                st.markdown(
                    f"<span style='color:#6B7280; font-size:0.85rem;'>"
                    f"已完成: **{done_count}** · 失败: **{error_count}** · 待处理: **{queue_total - done_count - error_count}**"
                    f"</span>",
                    unsafe_allow_html=True
                )

    # ------------------------------------------------------------
    # 标签页2: 单文件处理
    # ------------------------------------------------------------
    with tab2:
        with st.container():
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("**📄 单文件快速转写**")

            uploaded_file = st.file_uploader(
                "选择音视频文件（mp3/wav/m4a/ogg/flac/mp4/avi/mkv）",
                type=['mp3', 'wav', 'm4a', 'ogg', 'flac', 'mp4', 'avi', 'mkv'],
                label_visibility="collapsed",
            )
            st.markdown('</div>', unsafe_allow_html=True)

        if uploaded_file:
            col_meta1, col_meta2 = st.columns([2, 1])
            with col_meta1:
                bg_input = st.text_input("背景信息（可选）", placeholder="如：产品规划会议 · 有助于提高识别准确率")
            with col_meta2:
                fmt_select = st.selectbox(
                    "输出格式",
                    options=["dialogue", "utterances", "plain"],
                    index=0,
                    format_func=lambda x: {"dialogue": "对话格式", "utterances": "片段格式", "plain": "纯文本"}[x]
                )

            background = bg_input if bg_input else Path(uploaded_file.name).stem

            # 进度 / 实时转写区
            sp = st.session_state.get("single_progress", {})
            if sp.get("active"):
                step = sp.get("step", "")
                sub_step = sp.get("sub_step", "")
                progress = sp.get("progress", 0.0)

                st.markdown('<div class="progress-card">', unsafe_allow_html=True)
                st.markdown(f"### 📄 {uploaded_file.name}", unsafe_allow_html=True)
                st.progress(progress, text=f"整体进度: {progress*100:.0f}%")

                col_steps_l, col_steps_r = st.columns(2)
                with col_steps_l:
                    for s_key, s_label in [("init","🔄 初始化模型"),("preprocess","🔄 音频预处理"),("transcribe","🔄 语音转写"),("diarize","🔄 说话人分离")]:
                        render_step_row(s_key, s_label, step)
                with col_steps_r:
                    for s_key, s_label in [("clean","🔄 文本清理"),("score","🔄 质量评分"),("format","🔄 保存结果"),("done","✅ 完成")]:
                        render_step_row(s_key, s_label, step)

                if sub_step:
                    st.caption(f"💬 {sub_step}")

                # 实时转写内容
                live = st.session_state.get("single_live_segments", [])
                if live:
                    with st.container():
                        st.markdown("**📝 实时转写内容**")
                        st.markdown('<div class="transcript-box" id="single_live">', unsafe_allow_html=True)
                        for seg in live[-20:]:
                            speaker = seg.get("speaker", "说话人")
                            start = seg.get("start_time", 0)
                            text = seg.get("text", "")
                            mins, secs = divmod(int(start), 60)
                            st.markdown(
                                f'<div class="transcript-segment">'
                                f'<span class="time-tag">{mins:02d}:{secs:02d}</span>'
                                f'<span class="speaker-tag">{speaker}</span>'
                                f'<span class="transcript-text">{text}</span>'
                                f'</div>',
                                unsafe_allow_html=True
                            )
                        st.markdown('</div>', unsafe_allow_html=True)
                        st.markdown("<script>var el=document.getElementById('single_live');if(el) el.scrollTop=el.scrollHeight;</script>", unsafe_allow_html=True)

                st.markdown('</div>', unsafe_allow_html=True)

            # 单文件处理也用 WorkerManager（非阻塞）
            single_progress_active = st.session_state.get("single_progress", {}).get("active", False)

            if single_progress_active:
                # 轮询单文件 worker
                swm = st.session_state.get("single_worker_manager")
                if swm:
                    while True:
                        msg = swm.poll()
                        if msg is None:
                            break
                        if msg["type"] == "segment":
                            pass  # live_segments 已更新
                        elif msg["type"] in ("done", "error"):
                            sp = st.session_state.single_progress
                            sp["active"] = False
                            st.session_state.single_progress = sp
                            if msg["type"] == "done":
                                st.session_state.single_result = msg["result"]
                            else:
                                st.session_state.single_error = msg["error"]
                            st.rerun()

                # watchdog
                sp = st.session_state.get("single_progress", {})
                if time.time() - sp.get("updated_at", 0) > 120:
                    sp["sub_step"] = "⚠️ 长时间无响应，可能卡住"

                time.sleep(0.5)
                st.rerun()

            # 单文件处理完成 → 显示结果
            elif st.session_state.get("single_result"):
                result = st.session_state.single_result
                st.success(f"✅ 转写完成！时长: {format_time(result.duration_seconds)} · 质量: **{result.quality_score:.0f}/100**")
                if result.output_path and Path(result.output_path).exists():
                    with open(result.output_path, "r", encoding="utf-8") as f:
                        content = json.load(f)
                        segments = content.get("segments", content.get("conversations", []))
                    display_segs = list(segments)[:10]
                    if display_segs:
                        st.markdown('<div class="transcript-box">', unsafe_allow_html=True)
                        for seg in display_segs:
                            if isinstance(seg, dict):
                                speaker = seg.get("speaker", seg.get("role", "说话人"))
                                text = seg.get("text", seg.get("content", ""))
                                start = seg.get("start_time", seg.get("start", 0))
                                mins, secs = divmod(int(start), 60)
                                st.markdown(
                                    f'<div class="transcript-segment">'
                                    f'<span class="time-tag">{mins:02d}:{secs:02d}</span>'
                                    f'<span class="speaker-tag">{speaker}</span>'
                                    f'<span class="transcript-text">{text}</span>'
                                    f'</div>',
                                    unsafe_allow_html=True
                                )
                        st.markdown('</div>', unsafe_allow_html=True)
                        with open(result.output_path, "r", encoding="utf-8") as f:
                            st.download_button(
                                "📥 下载结果 JSON",
                                f.read(),
                                file_name=Path(result.output_path).name,
                                mime="application/json",
                                use_container_width=True
                            )
                # 清理
                del st.session_state.single_result
                if "single_error" in st.session_state:
                    st.error(f"❌ 转写失败: {st.session_state.single_error}")
                    del st.session_state.single_error

            if st.button("🎙️ 开始转写", use_container_width=True, type="primary"):
                os.makedirs(output_dir, exist_ok=True)
                suffix = Path(uploaded_file.name).suffix
                with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                    tmp.write(uploaded_file.read())
                    tmp_path = tmp.name

                sp = st.session_state.single_progress = {
                    "active": True,
                    "step": "init",
                    "sub_step": "正在初始化模型...",
                    "progress": 0.0,
                    "detail": "请稍候",
                    "updated_at": time.time(),
                }
                st.session_state.single_live_segments = []

                config = PipelineConfig(
                    model=model_size,
                    language=language,
                    device=device,
                    enable_diarization=enable_diarization,
                    output_format=output_format
                )
                pipeline = STTPipeline(config)

                try:
                    # 启动后台 worker（非阻塞）
                    swm = WorkerManager()
                    st.session_state.single_worker_manager = swm
                    swm.start(
                        pipeline=pipeline,
                        audio_path=Path(tmp_path),
                        background=background,
                        output_dir=Path(output_dir),
                        progress_state=sp,
                        live_segments=st.session_state.single_live_segments,
                        enable_diarization=enable_diarization,
                    )
                    st.rerun()

                except Exception as e:
                    sp["active"] = False
                    st.error(f"❌ 处理出错: {e}")
                    if Path(tmp_path).exists():
                        os.unlink(tmp_path)
                    st.rerun()

    # ------------------------------------------------------------
    # 标签页3: 结果查看
    # ------------------------------------------------------------
    with tab3:
        result_dir = Path(output_dir)
        if result_dir.exists():
            results = load_results_from_dir(result_dir)

            if results:
                st.markdown(f"### 📊 全部结果 · {len(results)} 个文件")

                col_stats1, col_stats2, col_stats3 = st.columns(3)
                total_dur = sum(r.get("duration_seconds", 0) for r in results)
                avg_score = sum(r.get("quality_score", 0) for r in results) / len(results) if results else 0
                col_stats1.metric("总时长", format_time(total_dur))
                col_stats2.metric("平均质量", f"{avg_score:.0f}/100")
                col_stats3.metric("文件数量", len(results))

                st.divider()

                for res in reversed(results):
                    with st.container():
                        st.markdown('<div class="card">', unsafe_allow_html=True)
                        cols = st.columns([4, 1, 1, 1])
                        source = res.get("source_file", "未知")
                        dur = res.get("duration_seconds", 0)
                        score = res.get("quality_score", 0)
                        quality_label, quality_class = get_quality_style(score)

                        cols[0].text(source)
                        cols[1].text(format_time(dur))
                        cols[2].markdown(f'<span class="quality-badge {quality_class}">{quality_label}</span>', unsafe_allow_html=True)

                        result_file = res.get("_result_file")
                        if result_file and cols[3].button("👁️ 查看", key=f"view_{result_file}"):
                            st.session_state.viewing_result = result_file
                            st.rerun()

                        st.markdown('</div>', unsafe_allow_html=True)

                if "viewing_result" in st.session_state and st.session_state.viewing_result:
                    result_file = st.session_state.viewing_result
                    with open(result_file, "r", encoding="utf-8") as f:
                        content = json.load(f)

                    st.markdown(f"### 📄 {Path(result_file).stem}")
                    col1, col2, col3 = st.columns(3)
                    col1.metric("时长", format_time(content.get("duration_seconds", 0)))
                    col2.metric("质量", f"{content.get('quality_score', 0):.0f}/100")
                    col3.metric("片段数", content.get("metadata", {}).get("segments_count", "N/A"))

                    st.text_area(
                        "JSON 内容",
                        value=json.dumps(content, ensure_ascii=False, indent=2),
                        height=350,
                        label_visibility="collapsed"
                    )

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
                st.markdown("""
                <div style="text-align: center; padding: 3rem; color: #9CA3AF;">
                    暂无转写结果，请先进行转写
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info(f"📁 输出目录不存在: {output_dir}")


# ============================================================
# 带实时进度的单文件处理
# ============================================================
@dataclass
class WorkerResult:
    """后台工作线程的结果"""
    success: bool = False
    pipeline_result: Optional[PipelineResult] = None
    error: Optional[str] = None
    live_segments: List[dict] = field(default_factory=list)


class WorkerManager:
    """
    管理后台转写线程。
    解决 Streamlit 单线程阻塞问题 — transcription 放到独立线程运行，
    主线程通过 queue 接收实时片段和进度更新，Streamlit 用 polling 刷新界面。
    """

    def __init__(self):
        self._queue: queue.Queue = queue.Queue()
        self._thread: Optional[threading.Thread] = None
        self._done = threading.Event()
        self._watchdog_timeout = 120  # 秒，无新输出则认为卡住
        self._stop_requested = False

    def stop(self):
        """请求停止后台线程"""
        self._stop_requested = True
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)

    def start(
        self,
        pipeline: STTPipeline,
        audio_path: Path,
        background: str,
        output_dir: Path,
        progress_state: dict,
        live_segments: list,
        enable_diarization: bool,
    ):
        """启动后台转写线程"""
        self._done.clear()
        self._queue = queue.Queue()

        self._thread = threading.Thread(
            target=self._worker,
            args=(
                pipeline, audio_path, background, output_dir,
                progress_state, live_segments, enable_diarization,
            ),
            daemon=True,
        )
        self._thread.start()

    def _worker(
        self,
        pipeline: STTPipeline,
        audio_path: Path,
        background: str,
        output_dir: Path,
        progress_state: dict,
        live_segments: list,
        enable_diarization: bool,
    ):
        """后台线程：运行完整 pipeline，通过 queue 推送实时片段"""
        last_progress_time = time.time()

        def update(step, sub_step="", progress=0.0, detail="", transcribed_seconds=0.0):
            progress_state["step"] = step
            progress_state["sub_step"] = sub_step
            progress_state["progress"] = progress
            progress_state["detail"] = detail
            progress_state["transcribed_seconds"] = transcribed_seconds
            progress_state["updated_at"] = time.time()
            self._queue.put_nowait({
                "type": "progress",
                "step": step, "sub_step": sub_step,
                "progress": progress, "detail": detail,
                "transcribed_seconds": transcribed_seconds,
            })

        def emit_segment(seg_dict: dict):
            live_segments.append(seg_dict)
            self._queue.put_nowait({"type": "segment", "segment": seg_dict})

        audio_path = Path(audio_path)
        output_dir = Path(output_dir)
        duration_seconds = 0.0

        try:
            # Step 1: 预处理
            update("init", "初始化模型...", 0.02, "加载 Whisper 模型")
            processed_audio = audio_path

            if pipeline.config.convert_to_wav:
                # 避免覆盖原文件：如果是 wav 且采样率合适则跳过，否则转换到临时文件
                is_wav = audio_path.suffix.lower() == ".wav"
                if is_wav:
                    processed_audio = audio_path
                else:
                    wav_path = Path("/tmp") / f"{audio_path.stem}_converted.wav"
                    processed_audio = pipeline.audio_converter.convert_to_wav(
                        audio_path, wav_path,
                        sample_rate=pipeline.config.target_sample_rate
                    )
            update("preprocess", "预处理完成", 0.05)

            # Step 2: 转写（流式生成 + 实时推送片段）
            update("transcribe", "Whisper 转写中...", 0.10, transcribed_seconds=0.0)

            # 用 faster-whisper 流式接口（segments 是生成器）
            # 手动拼一个 TranscribeResult
            from stt_whisper.transcribe import Segment as _Seg

            # 检查停止请求（转写前最后一次检查点）
            if self._stop_requested:
                self._queue.put_nowait({"type": "error", "error": "用户停止"})
                return

            transcribe_result = pipeline.transcriber.transcribe(
                processed_audio,
                background=background,
                _on_segment=emit_segment,  # 插件化回调
                _progress_callback=lambda cs, ts: update(
                    "transcribe", f"转写中 {cs:.0f}s...", 0.10 + 0.40 * (cs / max(ts, 1)),
                    transcribed_seconds=cs
                ) if ts else None,
            )
            duration_seconds = transcribe_result.duration_seconds

            # 推送所有片段（如果没有走 _on_segment）
            for seg in transcribe_result.segments:
                seg_dict = seg.to_dict() if hasattr(seg, 'to_dict') else seg
                emit_segment(seg_dict)

            update("transcribe", "转写完成", 0.50,
                   detail=f"识别 {len(transcribe_result.segments)} 个片段，{duration_seconds:.0f}s",
                   transcribed_seconds=duration_seconds)
            segments = [s.to_dict() if hasattr(s, 'to_dict') else s for s in transcribe_result.segments]

            # Step 3: 说话人分离
            if enable_diarization and pipeline.diarizer:
                update("diarize", "分离说话人中...", 0.55)
                try:
                    segments = pipeline.diarizer.merge_with_transcript(processed_audio, segments)
                    update("diarize", "说话人分离完成", 0.65)
                except Exception as e:
                    for seg in segments:
                        if "speaker" not in seg:
                            seg["speaker"] = "SPEAKER_00"
                    update("diarize", "说话人分离跳过", 0.65)
            else:
                for seg in segments:
                    if "speaker" not in seg:
                        seg["speaker"] = "SPEAKER_00"
                update("diarize", "说话人分离跳过", 0.65)

            # Step 4: 文字清理
            update("clean", "清理文本中...", 0.70)
            segments = pipeline.text_cleaner.clean_segments(segments)
            update("clean", "清理完成", 0.80, f"保留 {len(segments)} 个有效片段")

            # Step 5: 质量评分
            update("score", "质量评分中...", 0.85)
            score = pipeline.quality_scorer.score_segments(segments)

            # Step 6: 保存
            update("format", "保存结果中...", 0.90)
            output_path = output_dir / f"{audio_path.stem}_curated.json"
            from stt_whisper.postprocess import FormatterFactory
            FormatterFactory.format_and_save(
                segments=segments,
                output_path=output_path,
                fmt=pipeline.config.output_format,
                background=background,
                source_file=audio_path.name
            )

            result = PipelineResult(
                source_file=str(audio_path),
                success=True,
                duration_seconds=duration_seconds,
                quality_score=score.overall,
                output_path=str(output_path),
                metadata={
                    "model": pipeline.config.model,
                    "language": transcribe_result.language,
                    "segments_count": len(segments),
                    "quality_details": score.details,
                    "recommendation": pipeline.quality_scorer.get_recommendation(score),
                }
            )
            self._queue.put_nowait({"type": "done", "result": result})
            update("done", "🎉 转写完成！", 1.0, f"质量: {score.overall:.0f}/100",
                   transcribed_seconds=duration_seconds)

        except Exception as e:
            self._queue.put_nowait({"type": "error", "error": str(e)})

    def poll(self) -> Optional[dict]:
        """
        非阻塞检查后台线程的输出。
        返回 dict 或 None。调用频率建议每 0.5s 一次。
        """
        try:
            return self._queue.get_nowait()
        except queue.Empty:
            return None

    def is_alive(self) -> bool:
        """线程是否仍在运行"""
        return self._thread is not None and self._thread.is_alive()

    def join(self, timeout=None) -> Optional[dict]:
        """等待线程结束，返回最终结果 dict"""
        if self._thread:
            self._thread.join(timeout=timeout)
        result = None
        while True:
            try:
                msg = self._queue.get_nowait()
                if msg["type"] == "done":
                    result = msg["result"]
                elif msg["type"] == "error":
                    result = msg["error"]
            except queue.Empty:
                break
        return result

    def check_watchdog(self) -> bool:
        """
        检查是否卡住（超时无新输出）。
        返回 True 表示正常，False 表示疑似卡住需要重启。
        """
        return True  # 简化版，实际用 updated_at 判断


# 全局 worker 管理器（每个 request 需要独立实例，这里用 session_state 存储）
# 在 app.py 初始化时创建，每个文件处理任务独占一个 manager


# ============================================================
# 启动入口
# ============================================================
if __name__ == "__main__":
    main()
