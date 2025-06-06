import streamlit as st
import time
import json
from typing import Dict, Any
import logging
from src.ui.api_client import (
    get_jobs_list,
    get_job_details,
    get_job_statistics,
    api_request
)
from src.ui.session_init import initialize_session_state

logger = logging.getLogger(__name__)

initialize_session_state()

st.title("📋 后台任务")
st.markdown("查看和管理您的处理任务")

# === JOB STATISTICS OVERVIEW ===
job_stats = get_job_statistics()

if any(job_stats.values()):
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("已完成", job_stats.get("completed", 0))
    with col2:
        processing_count = job_stats.get("processing", 0)
        st.metric("处理中", processing_count)
    with col3:
        st.metric("等待中", job_stats.get("pending", 0))
    with col4:
        st.metric("失败", job_stats.get("failed", 0))

    st.markdown("---")

# === JOB LIST WITH FILTERS ===
jobs = get_jobs_list()

if not jobs:
    st.info("📭 暂无处理任务")
    if st.button("🔄 刷新", use_container_width=True):
        st.rerun()
    st.stop()

# Filter tabs
tab1, tab2, tab3 = st.tabs(["⏳ 处理中", "✅ 已完成", "📋 全部任务"])

def format_job_type(job_type: str) -> str:
    """Format job type for display"""
    type_names = {
        "video_processing": "🎬 视频处理",
        "pdf_processing": "📄 PDF处理",
        "text_processing": "✍️ 文字处理",
        "llm_inference": "🔍 查询处理",
        "batch_video_processing": "🎬 批量视频"
    }
    return type_names.get(job_type, job_type)

def format_time(timestamp: float) -> str:
    """Format timestamp for display"""
    if not timestamp:
        return "未知时间"
    try:
        return time.strftime("%m-%d %H:%M", time.localtime(timestamp))
    except:
        return "时间格式错误"

def display_job_card(job: Dict[str, Any], context: str, index: int):
    """
    Display a job card with progress and actions.

    SIMPLIFIED: No manual Unicode decoding needed - data is already clean.
    """
    job_id = job.get("job_id", "")
    job_type = job.get("job_type", "")
    status = job.get("status", "")
    created_at = job.get("created_at", 0)

    if not job_id:
        st.error("无效任务数据")
        return

    # Status styling
    status_config = {
        "pending": {"icon": "⏳", "color": "#FFA500"},
        "processing": {"icon": "🔄", "color": "#1E90FF"},
        "completed": {"icon": "✅", "color": "#32CD32"},
        "failed": {"icon": "❌", "color": "#FF4500"}
    }

    config = status_config.get(status, {"icon": "❓", "color": "#808080"})

    # Create unique keys with context and index
    job_short_id = job_id[:8]
    expand_key = f"expand_{context}_{index}_{job_short_id}"

    # Job card container
    with st.container():
        # Header row
        col1, col2, col3, col4 = st.columns([1, 3, 2, 1])

        with col1:
            st.markdown(f"<span style='font-size: 2em'>{config['icon']}</span>",
                        unsafe_allow_html=True)

        with col2:
            st.markdown(f"**{format_job_type(job_type)}**")
            st.caption(f"ID: {job_id[:12]}...")

        with col3:
            st.markdown(f"**状态: {status}**")
            st.caption(f"创建: {format_time(created_at)}")

        with col4:
            if st.button("📄 详情", key=f"detail_{context}_{index}_{job_short_id}"):
                # Toggle the expansion state for this specific job
                if expand_key not in st.session_state:
                    st.session_state[expand_key] = False
                st.session_state[expand_key] = not st.session_state[expand_key]
                st.rerun()

        # Progress bar for processing jobs
        if status == "processing":
            progress_info = job.get("progress_info", {})
            progress = progress_info.get("progress")
            message = progress_info.get("message", "")

            if progress is not None:
                st.progress(progress / 100.0)
                st.caption(f"进度: {progress}% - {message}")
            else:
                st.progress(0.0)
                st.caption("处理中...")

        # Show details inline with unique keys
        if st.session_state.get(expand_key, False):
            st.markdown("---")
            st.markdown("### 📋 任务详情")

            # Get full job details
            job_detail = get_job_details(job_id)

            if job_detail:
                # Basic information in columns
                detail_col1, detail_col2 = st.columns(2)

                with detail_col1:
                    st.write(f"**任务ID:** {job_id}")
                    st.write(f"**类型:** {format_job_type(job_detail.get('job_type', ''))}")
                    st.write(f"**状态:** {job_detail.get('status', '')}")

                with detail_col2:
                    created = job_detail.get('created_at', 0)
                    updated = job_detail.get('updated_at', 0)

                    if created:
                        st.write(f"**创建:** {time.strftime('%m-%d %H:%M:%S', time.localtime(created))}")
                    if updated:
                        st.write(f"**更新:** {time.strftime('%m-%d %H:%M:%S', time.localtime(updated))}")

                # Progress information
                progress_info = job_detail.get('progress_info', {})
                if progress_info:
                    progress = progress_info.get('progress')
                    message = progress_info.get('message', '')

                    st.write("**当前进度:**")
                    if progress is not None:
                        st.progress(progress / 100.0)
                        st.caption(f"{progress}% - {message}")
                    else:
                        st.caption(message or "处理中...")

                # Enhanced Metadata Display
                st.markdown("**📋 任务元数据:**")
                metadata = job_detail.get('metadata', {})
                result = job_detail.get('result', {})

                if metadata and isinstance(metadata, dict):
                    # Basic job metadata - no decoding needed, data is already clean
                    if metadata.get('url'):
                        st.write(f"**🔗 URL:** {metadata['url']}")
                    if metadata.get('query'):
                        st.write(f"**❓ 查询:** {metadata['query']}")
                    if metadata.get('platform'):
                        st.write(f"**📺 平台:** {metadata['platform']}")

                # Parse result properly
                if isinstance(result, str):
                    try:
                        import json
                        result = json.loads(result)
                    except:
                        result = {}

                # Enhanced Results Display for Video Processing
                if job_detail.get('status') == 'completed':
                    if result and isinstance(result, dict):

                        # Show video metadata - NO MANUAL UNICODE DECODING NEEDED
                        video_metadata = result.get('video_metadata', {})

                        if video_metadata and isinstance(video_metadata, dict):
                            st.markdown("**🎬 视频信息:**")

                            video_col1, video_col2 = st.columns(2)
                            with video_col1:
                                # Data is already clean thanks to global patch
                                if video_metadata.get('title'):
                                    st.write(f"**标题:** {video_metadata['title']}")
                                if video_metadata.get('author'):
                                    st.write(f"**作者:** {video_metadata['author']}")
                                if video_metadata.get('published_date'):
                                    pub_date = video_metadata['published_date']
                                    # Format date if it's in YYYYMMDD format
                                    if isinstance(pub_date, str) and len(pub_date) == 8:
                                        formatted_date = f"{pub_date[:4]}-{pub_date[4:6]}-{pub_date[6:8]}"
                                        st.write(f"**发布日期:** {formatted_date}")
                                    else:
                                        st.write(f"**发布日期:** {pub_date}")

                                # Show URL as clickable link
                                if video_metadata.get('url'):
                                    st.write(f"**链接:** [观看视频]({video_metadata['url']})")

                            with video_col2:
                                if video_metadata.get('length'):
                                    duration_mins = video_metadata['length'] // 60
                                    duration_secs = video_metadata['length'] % 60
                                    st.write(f"**时长:** {duration_mins}分{duration_secs}秒")
                                if video_metadata.get('views'):
                                    views = video_metadata['views']
                                    st.write(f"**观看次数:** {views:,}")
                                if video_metadata.get('video_id'):
                                    st.write(f"**视频ID:** {video_metadata['video_id']}")

                                # Show language from transcription result
                                language = result.get('language') or video_metadata.get('language')
                                if language:
                                    lang_display = {"zh": "中文", "en": "英文"}.get(language, language)
                                    st.write(f"**语言:** {lang_display}")

                            # Show description if available - no decoding needed
                            if video_metadata.get('description') and video_metadata['description'] != '-':
                                description = video_metadata['description']
                                st.write("**📝 视频描述:**")

                                if len(description) > 300:
                                    desc_key = f"show_desc_{context}_{index}_{job_short_id}"
                                    if desc_key not in st.session_state:
                                        st.session_state[desc_key] = False

                                    if st.session_state[desc_key]:
                                        st.text_area("完整描述", description, height=150, disabled=True,
                                                     key=f"full_desc_{context}_{index}_{job_short_id}")
                                        if st.button("收起描述", key=f"hide_desc_{context}_{index}_{job_short_id}"):
                                            st.session_state[desc_key] = False
                                            st.rerun()
                                    else:
                                        st.text_area("描述预览", description[:300] + "...", height=80,
                                                     disabled=True, key=f"short_desc_{context}_{index}_{job_short_id}")
                                        if st.button("显示完整描述",
                                                     key=f"show_desc_btn_{context}_{index}_{job_short_id}"):
                                            st.session_state[desc_key] = True
                                            st.rerun()
                                else:
                                    st.text_area("视频描述", description, height=80, disabled=True,
                                                 key=f"desc_{context}_{index}_{job_short_id}")

                        # Show transcription with better formatting - no decoding needed
                        transcript = result.get('transcript', '')
                        if transcript:
                            st.markdown("**🎤 转录内容:**")

                            # Data is already clean from global patch
                            transcript_key = f"show_transcript_{context}_{index}_{job_short_id}"
                            if transcript_key not in st.session_state:
                                st.session_state[transcript_key] = False

                            # Show transcript stats
                            word_count = len(transcript.split())
                            char_count = len(transcript)
                            language = result.get('language', '未知')
                            duration = result.get('duration', 0)

                            trans_col1, trans_col2, trans_col3, trans_col4 = st.columns(4)
                            with trans_col1:
                                st.metric("字数", f"{word_count:,}")
                            with trans_col2:
                                st.metric("字符数", f"{char_count:,}")
                            with trans_col3:
                                lang_display = {"zh": "中文", "en": "英文"}.get(language, language)
                                st.metric("语言", lang_display)
                            with trans_col4:
                                if duration > 0:
                                    st.metric("时长", f"{duration:.1f}秒")

                            # Toggle transcript display
                            if st.button(f"{'隐藏' if st.session_state[transcript_key] else '显示'} 转录内容",
                                         key=f"toggle_transcript_{context}_{index}_{job_short_id}"):
                                st.session_state[transcript_key] = not st.session_state[transcript_key]
                                st.rerun()

                            if st.session_state[transcript_key]:
                                st.text_area(
                                    "完整转录内容",
                                    transcript,
                                    height=300,
                                    disabled=True,
                                    key=f"transcript_{context}_{index}_{job_short_id}"
                                )

                        # Document processing results - no decoding needed
                        if 'document_count' in result:
                            st.success(f"✅ 成功生成 {result['document_count']} 个文档片段")

                            documents = result.get('documents', [])
                            if documents:
                                show_docs_key = f"show_docs_{context}_{index}_{job_short_id}"

                                if show_docs_key not in st.session_state:
                                    st.session_state[show_docs_key] = False

                                # Toggle button with more details
                                if st.button(
                                        f"📄 {'隐藏' if st.session_state[show_docs_key] else '显示'} {len(documents)} 个文档片段 (向量化后)",
                                        key=f"toggle_docs_{context}_{index}_{job_short_id}"):
                                    st.session_state[show_docs_key] = not st.session_state[show_docs_key]
                                    st.rerun()

                                if st.session_state[show_docs_key]:
                                    st.markdown("**📄 向量化文档片段:**")
                                    st.caption("这些是被切分并存储到向量数据库中的文档片段")

                                    for i, doc in enumerate(documents):
                                        with st.container():
                                            st.markdown(f"**片段 {i + 1}/{len(documents)}:**")

                                            # Enhanced metadata display - no decoding needed
                                            doc_metadata = doc.get('metadata', {})
                                            if doc_metadata:
                                                meta_cols = st.columns(4)
                                                with meta_cols[0]:
                                                    if doc_metadata.get('source'):
                                                        st.caption(f"📍 来源: {doc_metadata['source']}")
                                                with meta_cols[1]:
                                                    if doc_metadata.get('chunk_id') is not None:
                                                        st.caption(f"🔢 片段: {doc_metadata['chunk_id'] + 1}")
                                                with meta_cols[2]:
                                                    if doc_metadata.get('language'):
                                                        lang_display = {"zh": "中文", "en": "英文"}.get(
                                                            doc_metadata['language'], doc_metadata['language'])
                                                        st.caption(f"🌐 语言: {lang_display}")
                                                with meta_cols[3]:
                                                    if doc_metadata.get('total_chunks'):
                                                        st.caption(f"📊 总片段: {doc_metadata['total_chunks']}")

                                                # Video-specific metadata - data is already clean
                                                if doc_metadata.get('title'):
                                                    st.caption(f"📺 标题: {doc_metadata['title']}")
                                                if doc_metadata.get('author'):
                                                    st.caption(f"👤 作者: {doc_metadata['author']}")
                                                if doc_metadata.get('url'):
                                                    st.caption(f"🔗 [视频链接]({doc_metadata['url']})")

                                            # Show content - no decoding needed
                                            content = doc.get('content', '')
                                            if content:
                                                if len(content) > 500:
                                                    st.text_area(
                                                        f"内容片段 {i + 1}",
                                                        content[:500] + "...(已截断)",
                                                        height=100,
                                                        key=f"doc_content_{context}_{index}_{job_short_id}_{i}",
                                                        disabled=True
                                                    )

                                                    full_key = f"show_full_{context}_{index}_{job_short_id}_{i}"
                                                    if full_key not in st.session_state:
                                                        st.session_state[full_key] = False

                                                    if st.button(f"显示完整内容",
                                                                 key=f"btn_full_{context}_{index}_{job_short_id}_{i}"):
                                                        st.session_state[full_key] = not st.session_state[full_key]
                                                        st.rerun()

                                                    if st.session_state[full_key]:
                                                        st.text_area(
                                                            f"完整内容",
                                                            content,
                                                            height=200,
                                                            key=f"full_content_{context}_{index}_{job_short_id}_{i}",
                                                            disabled=True
                                                        )
                                                else:
                                                    st.text_area(
                                                        f"内容片段 {i + 1}",
                                                        content,
                                                        height=100,
                                                        key=f"doc_short_{context}_{index}_{job_short_id}_{i}",
                                                        disabled=True
                                                    )

                                            st.markdown("---")

                        # Query results - clean up LLM artifacts only
                        if 'answer' in result:
                            st.write("**❓ 查询答案:**")
                            answer = result['answer']

                            # Clean up LLM thinking artifacts (not Unicode issues)
                            if "</think>" in answer:
                                answer = answer.split("</think>")[-1].strip()
                            if answer.startswith("<think>"):
                                lines = answer.split('\n')
                                clean_lines = []
                                thinking_section = True
                                for line in lines:
                                    if thinking_section and (not line.strip().startswith('<') and line.strip()):
                                        thinking_section = False
                                    if not thinking_section:
                                        clean_lines.append(line)
                                answer = '\n'.join(clean_lines).strip()
                            answer = answer.replace("<think>", "").replace("</think>", "").strip()

                            if answer:
                                st.info(answer)
                            else:
                                st.warning("答案为空或无法解析")

                # Error information (for failed jobs)
                elif job_detail.get('status') == 'failed':
                    error = job_detail.get('error', '')
                    if error:
                        st.error(f"❌ **错误:** {error}")

                # Quick actions with unique keys
                action_col1, action_col2 = st.columns(2)
                with action_col1:
                    if st.button("🔄 刷新", key=f"refresh_{context}_{index}_{job_short_id}"):
                        st.rerun()
                with action_col2:
                    if job_detail.get('status') in ['completed', 'failed']:
                        if st.button("🗑️ 删除", key=f"delete_{context}_{index}_{job_short_id}"):
                            try:
                                result = api_request(f"/ingest/jobs/{job_id}", method="DELETE")
                                if result:
                                    st.success("任务已删除")
                                    time.sleep(1)
                                    st.rerun()
                                else:
                                    st.error("删除失败")
                            except:
                                st.error("删除操作失败")
            else:
                st.error("无法获取任务详情")

        st.divider()

# Display jobs in tabs
with tab1:  # Processing jobs
    processing_jobs = [j for j in jobs if j.get("status") in ["pending", "processing"]]

    if processing_jobs:
        st.write(f"**当前有 {len(processing_jobs)} 个任务正在处理**")

        for i, job in enumerate(processing_jobs):
            display_job_card(job, f"processing", i)

        # Auto-refresh option for processing jobs
        if st.checkbox("⚡ 自动刷新 (5秒)", key="auto_refresh_processing"):
            time.sleep(5)
            st.rerun()
    else:
        st.info("✨ 当前没有正在处理的任务")

with tab2:  # Completed jobs
    completed_jobs = [j for j in jobs if j.get("status") == "completed"]

    if completed_jobs:
        st.write(f"**已完成 {len(completed_jobs)} 个任务**")

        for i, job in enumerate(completed_jobs):
            display_job_card(job, f"completed", i)
    else:
        st.info("📭 暂无已完成的任务")

with tab3:  # All jobs
    st.write(f"**显示最近 {len(jobs)} 个任务**")

    for i, job in enumerate(jobs):
        display_job_card(job, f"all", i)


# === PAGE ACTIONS ===
st.markdown("---")

action_cols = st.columns(4)

with action_cols[0]:
    if st.button("🔄 刷新列表", use_container_width=True):
        st.rerun()

with action_cols[1]:
    if st.button("📤 新建任务", use_container_width=True):
        st.switch_page("pages/数据摄取.py")

with action_cols[2]:
    if st.button("🔍 开始查询", use_container_width=True):
        st.switch_page("pages/查询.py")

with action_cols[3]:
    if st.button("📊 系统状态", use_container_width=True):
        st.switch_page("pages/系统信息.py")

# Show active processing count
processing_count = len([j for j in jobs if j.get("status") in ["pending", "processing"]])
if processing_count > 0:
    st.info(f"ℹ️ 当前有 {processing_count} 个任务正在处理中")

st.caption("后台任务 - 跟踪您的处理任务进度")