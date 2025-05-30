"""
Clean background tasks page - src/ui/pages/后台任务.py
Focus: Individual job tracking, progress, results, management
"""

import streamlit as st
import time
from typing import Dict, Any
from src.ui.api_client import (
    get_jobs_list,
    get_job_details,
    get_job_statistics,
    api_request
)
from src.ui.session_init import initialize_session_state

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
    """Display a job card with progress and actions - FIXED INLINE DETAILS"""
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

    # Create unique key for this job's expansion state
    expand_key = f"expand_{context}_{index}_{job_id[:8]}"

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
            # Toggle button for details - FIXED TO EXPAND INLINE
            if st.button("📄 详情", key=f"detail_{context}_{index}_{job_id[:8]}"):
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

        # FIXED: Show details inline immediately below this job if expanded
        if st.session_state.get(expand_key, False):
            with st.expander("任务详情", expanded=True):
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

                    # Metadata
                    metadata = job_detail.get('metadata', {})
                    if metadata and isinstance(metadata, dict):
                        if metadata.get('url'):
                            st.write(f"**URL:** {metadata['url']}")
                        if metadata.get('query'):
                            st.write(f"**查询:** {metadata['query']}")

                    # Results (for completed jobs)
                    if job_detail.get('status') == 'completed':
                        result = job_detail.get('result', {})
                        if result and isinstance(result, dict):
                            # Document processing results
                            if 'document_count' in result:
                                st.success(f"✅ 成功生成 {result['document_count']} 个文档片段")

                                # FIXED: Use checkbox instead of nested expander
                                documents = result.get('documents', [])
                                if documents:
                                    # Create a checkbox to show/hide documents
                                    show_docs_key = f"show_docs_{job_id[:8]}"
                                    if show_docs_key not in st.session_state:
                                        st.session_state[show_docs_key] = False

                                    # Toggle checkbox
                                    st.session_state[show_docs_key] = st.checkbox(
                                        f"📄 显示 {len(documents)} 个文档片段",
                                        value=st.session_state[show_docs_key],
                                        key=f"docs_toggle_{job_id[:8]}"
                                    )

                                    # Show documents if checkbox is checked
                                    if st.session_state[show_docs_key]:
                                        st.markdown("**文档片段:**")

                                        for i, doc in enumerate(documents):
                                            # Use a container with border styling instead of expander
                                            with st.container():
                                                st.markdown(f"**📄 文档 {i + 1}:**")

                                                # Show metadata if available
                                                metadata = doc.get('metadata', {})
                                                if metadata:
                                                    meta_cols = st.columns(3)
                                                    with meta_cols[0]:
                                                        if metadata.get('source'):
                                                            st.caption(f"来源: {metadata['source']}")
                                                    with meta_cols[1]:
                                                        if metadata.get('chunk_id') is not None:
                                                            st.caption(f"片段: {metadata['chunk_id'] + 1}")
                                                    with meta_cols[2]:
                                                        if metadata.get('title'):
                                                            st.caption(f"标题: {metadata['title'][:30]}...")

                                                # Show content
                                                content = doc.get('content', '')
                                                if content:
                                                    # Truncate very long content
                                                    if len(content) > 500:
                                                        # Use text_area for long content
                                                        st.text_area(
                                                            f"内容片段 {i + 1}",
                                                            content[:500] + "...(内容已截断)",
                                                            height=80,
                                                            key=f"doc_content_{job_id}_{i}",
                                                            disabled=True
                                                        )

                                                        # Button to show full content
                                                        full_content_key = f"show_full_{job_id}_{i}"
                                                        if full_content_key not in st.session_state:
                                                            st.session_state[full_content_key] = False

                                                        if st.button(f"显示完整内容 {i + 1}",
                                                                     key=f"toggle_full_{job_id}_{i}"):
                                                            st.session_state[full_content_key] = not st.session_state[
                                                                full_content_key]

                                                        if st.session_state[full_content_key]:
                                                            st.text_area(
                                                                f"完整内容 {i + 1}",
                                                                content,
                                                                height=200,
                                                                key=f"full_content_{job_id}_{i}",
                                                                disabled=True
                                                            )
                                                    else:
                                                        # Short content - show directly
                                                        st.text_area(
                                                            f"内容 {i + 1}",
                                                            content,
                                                            height=80,
                                                            key=f"doc_short_{job_id}_{i}",
                                                            disabled=True
                                                        )

                                                st.markdown("---")  # Separator between documents

                            # Query results
                            if 'answer' in result:
                                st.write("**查询答案:**")
                                answer = result['answer']

                                # FIXED: Remove </think> artifacts more thoroughly
                                if "</think>" in answer:
                                    # Remove everything before and including </think>
                                    answer = answer.split("</think>")[-1].strip()

                                # Also remove <think> tags if they exist without closing
                                if answer.startswith("<think>"):
                                    # Find the end of thinking section
                                    lines = answer.split('\n')
                                    clean_lines = []
                                    thinking_section = True
                                    for line in lines:
                                        if thinking_section and (not line.strip().startswith('<') and line.strip()):
                                            thinking_section = False
                                        if not thinking_section:
                                            clean_lines.append(line)
                                    answer = '\n'.join(clean_lines).strip()

                                # Final cleanup - remove any remaining thinking artifacts
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

                    # Quick actions
                    action_col1, action_col2 = st.columns(2)
                    with action_col1:
                        if st.button("🔄 刷新", key=f"refresh_{job_id[:8]}"):
                            st.rerun()
                    with action_col2:
                        if job_detail.get('status') in ['completed', 'failed']:
                            if st.button("🗑️ 删除", key=f"delete_{job_id[:8]}"):
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
            display_job_card(job, f"processing", i)  # ADDED index parameter

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
            display_job_card(job, f"completed", i)  # ADDED index parameter
    else:
        st.info("📭 暂无已完成的任务")

with tab3:  # All jobs
    st.write(f"**显示最近 {len(jobs)} 个任务**")

    for i, job in enumerate(jobs):
        display_job_card(job, f"all", i)  # ADDED index parameter


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