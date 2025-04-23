"""
后台任务管理页面（Streamlit UI）
"""

import os
import time
import streamlit as st
from typing import Dict, List, Optional, Union, Any
import pandas as pd

from src.ui.components import header, api_request, loading_spinner

# API 配置
API_URL = os.environ.get("API_URL", "http://localhost:8000")
API_KEY = os.environ.get("API_KEY", "default-api-key")

# 会话状态初始化
if "api_url" not in st.session_state:
    st.session_state.api_url = API_URL
if "api_key" not in st.session_state:
    st.session_state.api_key = API_KEY
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "selected_job_id" not in st.session_state:
    st.session_state.selected_job_id = None
if "auto_refresh" not in st.session_state:
    st.session_state.auto_refresh = False
if "last_refresh" not in st.session_state:
    st.session_state.last_refresh = time.time()
if "job_filter" not in st.session_state:
    st.session_state.job_filter = "all"

# 常量定义
REFRESH_INTERVAL = 10  # 自动刷新间隔（秒）
JOB_STATUS_COLORS = {
    "pending": "🟡",
    "processing": "🔵",
    "completed": "🟢",
    "failed": "🔴",
    "timeout": "🟠"
}
JOB_TYPE_NAMES = {
    "video_processing": "视频处理",
    "pdf_processing": "PDF处理",
    "batch_video_processing": "批量视频处理",
    "manual_text": "手动文本处理"
}

header(
    "后台任务管理",
    "管理所有后台运行的数据导入任务。"
)

# 自动刷新功能
auto_refresh = st.sidebar.checkbox("自动刷新", value=st.session_state.auto_refresh)
st.session_state.auto_refresh = auto_refresh

if st.session_state.auto_refresh:
    current_time = time.time()
    if current_time - st.session_state.last_refresh > REFRESH_INTERVAL:
        st.session_state.last_refresh = current_time
        st.rerun()

# 手动刷新按钮
if st.sidebar.button("刷新数据"):
    st.session_state.last_refresh = time.time()
    st.rerun()

# 过滤选项
st.sidebar.subheader("筛选")
job_filter = st.sidebar.radio(
    "任务状态",
    options=["all", "pending", "processing", "completed", "failed", "timeout"],
    format_func=lambda x: {
        "all": "所有任务",
        "pending": "🟡 等待中",
        "processing": "🔵 处理中",
        "completed": "🟢 已完成",
        "failed": "🔴 失败",
        "timeout": "🟠 超时"
    }.get(x, x),
    index=0
)
st.session_state.job_filter = job_filter

job_type_filter = st.sidebar.radio(
    "任务类型",
    options=["all", "video_processing", "pdf_processing", "batch_video_processing"],
    format_func=lambda x: {
        "all": "所有类型",
        "video_processing": "视频处理",
        "pdf_processing": "PDF处理",
        "batch_video_processing": "批量视频处理"
    }.get(x, x),
    index=0
)

# 获取任务状态统计
with st.spinner("获取任务统计..."):
    status_info = api_request(
        endpoint="/ingest/status",
        method="GET"
    )

if status_info and "job_stats" in status_info:
    job_stats = status_info["job_stats"]

    # 显示任务统计
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("等待中", job_stats.get("pending_jobs", 0))
    with col2:
        st.metric("处理中", job_stats.get("processing_jobs", 0))
    with col3:
        st.metric("已完成", job_stats.get("completed_jobs", 0))
    with col4:
        st.metric("失败", job_stats.get("failed_jobs", 0))

# 获取任务列表
with st.spinner("获取任务列表..."):
    jobs_data = api_request(
        endpoint="/ingest/jobs",
        method="GET",
        params={"limit": 100}
    )

if jobs_data:
    # 根据筛选条件过滤任务
    filtered_jobs = []
    for job in jobs_data:
        status_match = job_filter == "all" or job.get("status") == job_filter
        job_type_match = job_type_filter == "all" or job.get("job_type") == job_type_filter

        if status_match and job_type_match:
            filtered_jobs.append(job)

    if not filtered_jobs:
        st.info("没有符合条件的任务")
    else:
        # 准备表格数据
        table_data = []
        for job in filtered_jobs:
            # 获取任务元数据信息
            metadata = job.get("metadata", {})

            # 获取资源名称（文件名或URL）
            resource_name = ""
            if "filename" in metadata:
                resource_name = metadata["filename"]
            elif "url" in metadata:
                resource_name = metadata["url"]
            elif "urls" in metadata:
                urls = metadata["urls"]
                if isinstance(urls, list) and urls:
                    resource_name = f"{len(urls)} 个视频"

            # 获取创建和更新时间
            created_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(job.get("created_at", 0)))
            updated_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(job.get("updated_at", 0)))

            # 计算任务持续时间
            if job.get("status") in ["completed", "failed", "timeout"]:
                duration = job.get("updated_at", 0) - job.get("created_at", 0)
                if duration < 60:
                    duration_str = f"{duration:.1f} 秒"
                elif duration < 3600:
                    duration_str = f"{duration / 60:.1f} 分钟"
                else:
                    duration_str = f"{duration / 3600:.1f} 小时"
            else:
                duration = time.time() - job.get("created_at", 0)
                duration_str = f"已运行 {duration / 60:.1f} 分钟"

            # 获取任务状态图标
            status_icon = JOB_STATUS_COLORS.get(job.get("status", ""), "⚪")

            # 获取任务类型名称
            job_type_name = JOB_TYPE_NAMES.get(job.get("job_type", ""), job.get("job_type", ""))

            # 添加到表格数据
            table_data.append({
                "ID": job.get("job_id", ""),
                "状态": f"{status_icon} {job.get('status', '')}",
                "任务类型": job_type_name,
                "资源名称": resource_name,
                "创建时间": created_time,
                "更新时间": updated_time,
                "持续时间": duration_str
            })

        # 创建DataFrame并显示为表格
        df = pd.DataFrame(table_data)

        # 为表格添加点击事件
        selection = st.data_editor(
            df,
            column_config={
                "ID": st.column_config.TextColumn("ID", width="small"),
                "状态": st.column_config.TextColumn("状态", width="small"),
                "任务类型": st.column_config.TextColumn("任务类型", width="small"),
                "资源名称": st.column_config.TextColumn("资源名称", width="medium"),
                "创建时间": st.column_config.TextColumn("创建时间", width="medium"),
                "更新时间": st.column_config.TextColumn("更新时间", width="medium"),
                "持续时间": st.column_config.TextColumn("持续时间", width="small")
            },
            hide_index=True,
            width=None,
            use_container_width=True
        )

        # 显示任务详情
        st.subheader("任务详情")

        # 选择任务显示详情
        selected_job_id = st.selectbox(
            "选择任务查看详情",
            options=[job.get("job_id", "") for job in filtered_jobs],
            format_func=lambda job_id: next(
                (f"{JOB_STATUS_COLORS.get(job.get('status', ''), '⚪')} {job.get('job_type', '')} - {job_id[:8]}..." for
                 job in filtered_jobs if job.get("job_id") == job_id), job_id),
            index=0
        )

        # 获取所选任务
        selected_job = next((job for job in filtered_jobs if job.get("job_id") == selected_job_id), None)

        if selected_job:
            # 显示任务详情
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**任务ID**: {selected_job.get('job_id', '')}")
                st.markdown(
                    f"**状态**: {JOB_STATUS_COLORS.get(selected_job.get('status', ''), '⚪')} {selected_job.get('status', '')}")
                st.markdown(
                    f"**任务类型**: {JOB_TYPE_NAMES.get(selected_job.get('job_type', ''), selected_job.get('job_type', ''))}")

                created_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(selected_job.get("created_at", 0)))
                updated_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(selected_job.get("updated_at", 0)))
                st.markdown(f"**创建时间**: {created_time}")
                st.markdown(f"**更新时间**: {updated_time}")

            with col2:
                # 显示元数据
                metadata = selected_job.get("metadata", {})
                st.markdown("**元数据**:")
                if "filename" in metadata:
                    st.markdown(f"- 文件名: {metadata['filename']}")
                if "url" in metadata:
                    st.markdown(f"- URL: {metadata['url']}")
                if "platform" in metadata:
                    st.markdown(f"- 平台: {metadata['platform']}")
                if "urls" in metadata:
                    urls = metadata["urls"]
                    if isinstance(urls, list):
                        st.markdown(f"- 视频数量: {len(urls)}")
                        if len(urls) <= 3:  # 只显示前3个URL
                            for url in urls:
                                st.markdown(f"  - {url}")
                        else:
                            for i, url in enumerate(urls[:3]):
                                st.markdown(f"  - {url}")
                            st.markdown(f"  - ... 共 {len(urls)} 个视频")

                # 如果任务已完成，显示删除按钮
                if selected_job.get("status") in ["completed", "failed", "timeout"]:
                    if st.button("删除任务", key=f"delete_{selected_job_id}"):
                        with st.spinner("删除任务中..."):
                            delete_result = api_request(
                                endpoint=f"/ingest/jobs/{selected_job_id}",
                                method="DELETE"
                            )
                            if delete_result:
                                st.success("任务已删除")
                                time.sleep(1)
                                st.rerun()

            # 显示结果或错误
            if selected_job.get("status") == "completed" and selected_job.get("result"):
                st.subheader("处理结果")

                result = selected_job.get("result")
                if isinstance(result, dict):
                    # 显示消息
                    if "message" in result:
                        st.success(result["message"])

                    # 显示文档ID
                    if "document_ids" in result:
                        doc_ids = result["document_ids"]
                        st.markdown(f"**文档ID**: {len(doc_ids)} 个")
                        with st.expander("查看文档ID"):
                            for doc_id in doc_ids:
                                st.text(doc_id)

                    # 显示文档数量
                    if "document_count" in result:
                        st.markdown(f"**文档数量**: {result['document_count']}")

                    # 显示批处理结果
                    if "results" in result:
                        batch_results = result["results"]
                        if isinstance(batch_results, dict):
                            st.markdown("**批处理结果**:")
                            for url, url_result in batch_results.items():
                                if isinstance(url_result, list):
                                    st.success(f"✅ {url}: {len(url_result)} 个文档")
                                elif isinstance(url_result, dict) and "error" in url_result:
                                    st.error(f"❌ {url}: {url_result['error']}")
                                else:
                                    st.info(f"ℹ️ {url}: {url_result}")
                else:
                    st.text(str(result))

            # 显示错误信息
            if selected_job.get("status") in ["failed", "timeout"] and selected_job.get("error"):
                st.subheader("错误信息")
                st.error(selected_job.get("error"))

            # 显示正在处理的任务的状态更新
            if selected_job.get("status") in ["pending", "processing"]:
                st.subheader("状态")
                status_placeholder = st.empty()

                status_text = "任务状态: "
                if selected_job.get("status") == "pending":
                    status_text += "⏳ 等待中..."
                elif selected_job.get("status") == "processing":
                    status_text += "🔄 处理中..."

                status_placeholder.info(status_text)

                # 如果开启了自动刷新，显示下次刷新时间
                if st.session_state.auto_refresh:
                    next_refresh = REFRESH_INTERVAL - (time.time() - st.session_state.last_refresh)
                    st.caption(f"⏱️ 下次自动刷新: {next_refresh:.1f} 秒后")
                else:
                    st.caption("提示: 开启侧边栏中的「自动刷新」功能可以自动更新任务状态")
else:
    st.info("当前无任务")