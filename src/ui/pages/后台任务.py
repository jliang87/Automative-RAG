"""
Clean status page - src/ui/pages/后台任务.py
"""

import streamlit as st
import time
from src.ui.api_client import api_request
from src.ui.session_init import initialize_session_state

initialize_session_state()

st.title("📊 处理状态")

def get_jobs():
    """Get all jobs"""
    try:
        return api_request("/ingest/jobs", method="GET", params={"limit": 20})
    except:
        return []

def get_job_status(job_id):
    """Get specific job status"""
    try:
        return api_request(f"/ingest/jobs/{job_id}", method="GET")
    except:
        return None

# Get jobs
jobs = get_jobs()

if not jobs:
    st.info("暂无处理任务")
    if st.button("刷新"):
        st.rerun()
else:
    # Filter tabs
    tab1, tab2, tab3 = st.tabs(["⏳ 处理中", "✅ 已完成", "📋 全部"])

    def display_job(job):
        """Display a single job"""
        job_id = job.get("job_id", "")
        job_type = job.get("job_type", "")
        status = job.get("status", "")
        created_at = job.get("created_at", 0)

        # Status icon
        status_icons = {
            "pending": "⏳",
            "processing": "🔄",
            "completed": "✅",
            "failed": "❌"
        }

        status_colors = {
            "pending": "orange",
            "processing": "blue",
            "completed": "green",
            "failed": "red"
        }

        icon = status_icons.get(status, "❓")
        color = status_colors.get(status, "gray")

        # Job type display
        type_names = {
            "video_processing": "视频处理",
            "pdf_processing": "PDF处理",
            "text_processing": "文字处理",
            "llm_inference": "查询处理"
        }

        type_display = type_names.get(job_type, job_type)

        # Time display
        if created_at:
            time_str = time.strftime("%m-%d %H:%M", time.localtime(created_at))
        else:
            time_str = "未知时间"

        with st.container():
            col1, col2, col3, col4 = st.columns([1, 2, 2, 1])

            with col1:
                st.markdown(f"<span style='color: {color}; font-size: 1.5em'>{icon}</span>",
                           unsafe_allow_html=True)

            with col2:
                st.write(f"**{type_display}**")
                st.caption(f"ID: {job_id[:8]}...")

            with col3:
                st.write(f"**{status}**")
                st.caption(time_str)

            with col4:
                if st.button("详情", key=f"detail_{job_id}"):
                    st.session_state.selected_job = job_id
                    st.rerun()

        st.divider()

    # Filter jobs by status
    with tab1:  # Processing
        processing_jobs = [j for j in jobs if j.get("status") in ["pending", "processing"]]
        if processing_jobs:
            for job in processing_jobs:
                display_job(job)
        else:
            st.info("当前没有处理中的任务")

    with tab2:  # Completed
        completed_jobs = [j for j in jobs if j.get("status") == "completed"]
        if completed_jobs:
            for job in completed_jobs:
                display_job(job)
        else:
            st.info("暂无已完成的任务")

    with tab3:  # All
        for job in jobs:
            display_job(job)

# Job details modal
if hasattr(st.session_state, 'selected_job'):
    job_id = st.session_state.selected_job
    job_detail = get_job_status(job_id)

    if job_detail:
        st.subheader(f"任务详情 - {job_id[:8]}...")

        # Basic info
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**类型:** {job_detail.get('job_type', '')}")
            st.write(f"**状态:** {job_detail.get('status', '')}")

        with col2:
            created = job_detail.get('created_at', 0)
            if created:
                created_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(created))
                st.write(f"**创建时间:** {created_str}")

            updated = job_detail.get('updated_at', 0)
            if updated:
                updated_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(updated))
                st.write(f"**更新时间:** {updated_str}")

        # Metadata
        metadata = job_detail.get('metadata', {})
        if metadata:
            st.write("**详细信息:**")
            for key, value in metadata.items():
                if key not in ['custom_metadata'] and value:
                    st.write(f"- {key}: {value}")

        # Result
        if job_detail.get('status') == 'completed':
            result = job_detail.get('result', {})
            if result:
                st.write("**处理结果:**")
                if isinstance(result, dict):
                    if 'document_count' in result:
                        st.write(f"- 生成文档数: {result['document_count']}")
                    if 'answer' in result:
                        st.write("**查询结果:**")
                        st.write(result['answer'])

        # Error info
        if job_detail.get('status') == 'failed':
            error = job_detail.get('error', '')
            if error:
                st.error(f"错误信息: {error}")

        if st.button("关闭详情"):
            del st.session_state.selected_job
            st.rerun()

# Auto refresh for processing jobs
if any(j.get("status") in ["pending", "processing"] for j in jobs):
    if st.checkbox("自动刷新", value=False):
        time.sleep(3)
        st.rerun()

# Manual refresh
if st.button("手动刷新", use_container_width=True):
    st.rerun()