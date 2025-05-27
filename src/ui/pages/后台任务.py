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

def display_job_card(job: Dict[str, Any], context: str):
    """Display a job card with progress and actions"""
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
            if st.button("📄 详情", key=f"detail_{job_id}_{context}"):
                st.session_state.selected_job_id = job_id
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

        st.divider()

# Display jobs in tabs
with tab1:  # Processing jobs
    processing_jobs = [j for j in jobs if j.get("status") in ["pending", "processing"]]

    if processing_jobs:
        st.write(f"**当前有 {len(processing_jobs)} 个任务正在处理**")

        for i, job in enumerate(processing_jobs):
            display_job_card(job, f"processing_{i}")

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
            display_job_card(job, f"completed_{i}")
    else:
        st.info("📭 暂无已完成的任务")

with tab3:  # All jobs
    st.write(f"**显示最近 {len(jobs)} 个任务**")

    for i, job in enumerate(jobs):
        display_job_card(job, f"all_{i}")

# === JOB DETAILS MODAL ===
if hasattr(st.session_state, 'selected_job_id') and st.session_state.selected_job_id:
    job_id = st.session_state.selected_job_id
    job_detail = get_job_details(job_id)

    st.markdown("---")
    st.subheader(f"📄 任务详情")

    if job_detail:
        # Basic information
        col1, col2 = st.columns(2)

        with col1:
            st.write(f"**任务ID:** {job_id}")
            st.write(f"**类型:** {format_job_type(job_detail.get('job_type', ''))}")
            st.write(f"**状态:** {job_detail.get('status', '')}")

        with col2:
            created = job_detail.get('created_at', 0)
            updated = job_detail.get('updated_at', 0)

            if created:
                st.write(f"**创建时间:** {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(created))}")
            if updated:
                st.write(f"**更新时间:** {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(updated))}")

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

        # Job chain information
        job_chain_info = job_detail.get('job_chain', {})
        if job_chain_info:
            st.write("**处理流程:**")
            current_step = job_chain_info.get('current_step', 0)
            total_steps = job_chain_info.get('total_steps', 0)
            current_task = job_chain_info.get('current_task', '')

            if total_steps > 0:
                progress_percent = (current_step / total_steps) * 100
                st.progress(progress_percent / 100.0)
                st.caption(f"步骤 {current_step}/{total_steps}: {current_task}")

        # Metadata
        metadata = job_detail.get('metadata', {})
        if metadata:
            with st.expander("📋 任务详细信息"):
                for key, value in metadata.items():
                    if key not in ['custom_metadata'] and value:
                        st.write(f"**{key}:** {value}")

        # Results (for completed jobs)
        if job_detail.get('status') == 'completed':
            result = job_detail.get('result', {})
            if result:
                st.write("**处理结果:**")

                if isinstance(result, dict):
                    # Document processing results
                    if 'document_count' in result:
                        st.success(f"✅ 成功生成 {result['document_count']} 个文档片段")

                    # Query results
                    if 'answer' in result:
                        st.write("**查询答案:**")
                        st.info(result['answer'])

                    # Documents
                    if 'documents' in result and result['documents']:
                        docs = result['documents']
                        st.write(f"**相关文档 ({len(docs)}):**")

                        with st.expander("查看文档内容"):
                            for i, doc in enumerate(docs[:3]):  # Show first 3
                                if isinstance(doc, dict):
                                    content = doc.get('content', '')
                                    if content:
                                        st.write(f"**文档 {i+1}:**")
                                        st.text_area("", content[:300] + "..." if len(content) > 300 else content,
                                                   height=100, key=f"doc_content_{i}")

                elif isinstance(result, str):
                    st.info(result)

        # Error information (for failed jobs)
        elif job_detail.get('status') == 'failed':
            error = job_detail.get('error', '')
            if error:
                st.error(f"❌ **错误信息:** {error}")

                # Suggest actions
                st.write("**建议操作:**")
                st.write("• 检查输入数据格式是否正确")
                st.write("• 稍后重试或联系管理员")

        # Action buttons
        button_cols = st.columns(3)

        with button_cols[0]:
            if st.button("🔄 刷新详情", key="refresh_detail"):
                st.rerun()

        with button_cols[1]:
            if st.button("❌ 关闭详情", key="close_detail"):
                del st.session_state.selected_job_id
                st.rerun()

        with button_cols[2]:
            if job_detail.get('status') in ['completed', 'failed']:
                if st.button("🗑️ 删除任务", key="delete_job"):
                    try:
                        result = api_request(f"/ingest/jobs/{job_id}", method="DELETE")
                        if result:
                            st.success("任务已删除")
                            del st.session_state.selected_job_id
                            time.sleep(1)
                            st.rerun()
                        else:
                            st.error("删除失败")
                    except:
                        st.error("删除操作失败")
    else:
        st.error("❌ 无法获取任务详情")
        if st.button("关闭", key="close_error"):
            del st.session_state.selected_job_id
            st.rerun()

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