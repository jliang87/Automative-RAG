"""
查询任务状态页面（Streamlit UI）
"""

import streamlit as st
import time
import os
import json
import pandas as pd
from src.ui.components import header, api_request, display_document

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

# 常量定义
JOB_STATUS_COLORS = {
    "pending": "🟡",
    "processing": "🔵",
    "completed": "🟢",
    "failed": "🔴",
    "timeout": "🟠"
}

def render_task_status_page():
    """渲染任务状态页面"""
    header(
        "查询任务状态",
        "查看异步查询任务的状态和结果。"
    )

    # 任务ID输入
    job_id = st.text_input("输入任务ID查看详情")

    col1, col2 = st.columns([1, 3])

    with col1:
        check_button = st.button("查看任务详情", type="primary")

    with col2:
        refresh_button = st.button("刷新任务列表")

    # 获取所有查询任务
    with st.spinner("获取任务列表..."):
        all_jobs = api_request(
            endpoint="/ingest/jobs",
            method="GET",
            params={"job_type": "llm_inference", "limit": 50}
        )

    if not all_jobs:
        st.warning("无法获取任务列表，请检查API连接")
        return

    # 过滤掉非查询任务
    query_jobs = [job for job in all_jobs if job.get("job_type") == "llm_inference"]

    if not query_jobs:
        st.info("没有找到查询任务")
    else:
        # 准备表格数据
        table_data = []
        for job in query_jobs:
            status = job.get("status", "")
            status_icon = JOB_STATUS_COLORS.get(status, "⚪")

            created_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(job.get("created_at", 0)))
            updated_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(job.get("updated_at", 0)))

            query_text = job.get("metadata", {}).get("query", "")
            if len(query_text) > 50:
                query_text = query_text[:47] + "..."

            table_data.append({
                "ID": job.get("job_id", "")[:8] + "...",
                "完整ID": job.get("job_id", ""),
                "状态": f"{status_icon} {status}",
                "查询": query_text,
                "创建时间": created_time,
                "更新时间": updated_time
            })

        # 显示任务表格
        st.subheader("查询任务列表")

        df = pd.DataFrame(table_data)

        # 表格点击交互
        selected_row = st.dataframe(
            df[[col for col in df.columns if col != "完整ID"]],
            use_container_width=True,
            hide_index=True
        )

        if check_button and job_id:
            # 检查特定任务ID
            st.session_state.selected_job_id = job_id
        elif refresh_button:
            # 刷新时清除选择
            st.session_state.selected_job_id = None

        # 显示选中任务详情
        if "selected_job_id" in st.session_state and st.session_state.selected_job_id:
            selected_id = st.session_state.selected_job_id

            # 获取任务详情
            job_data = api_request(
                endpoint=f"/ingest/jobs/{selected_id}",
                method="GET"
            )

            if not job_data:
                st.error(f"无法获取任务 {selected_id} 的详情")
                return

            # 显示任务详情
            st.subheader("任务详情")

            status = job_data.get("status", "")
            status_icon = JOB_STATUS_COLORS.get(status, "⚪")

            col1, col2, col3 = st.columns(3)
            with col1:
                st.info(f"任务ID: {selected_id}")
            with col2:
                st.info(f"状态: {status_icon} {status}")
            with col3:
                created_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(job_data.get("created_at", 0)))
                st.info(f"创建时间: {created_time}")

            # 显示查询和结果
            st.markdown("### 查询内容")
            query = job_data.get("metadata", {}).get("query", "")
            st.markdown(f"> {query}")

            if status == "completed":
                # 显示完成的结果
                result = job_data.get("result", {})

                if isinstance(result, str):
                    try:
                        result = json.loads(result)
                    except:
                        result = {"answer": result}

                st.markdown("### 查询结果")
                st.markdown(result.get("answer", ""))

                st.markdown("### 数据来源")
                documents = result.get("documents", [])
                if documents:
                    for i, doc in enumerate(documents):
                        display_document(doc, i)
                else:
                    st.info("没有找到相关文档")

                execution_time = result.get("execution_time", 0)
                st.caption(f"处理时间: {execution_time:.2f}秒")

            elif status == "failed":
                # 显示错误信息
                st.error(f"任务失败: {job_data.get('error', '未知错误')}")

            elif status in ["pending", "processing"]:
                # 显示处理中状态
                st.info("任务正在处理中...")

                # 添加刷新按钮
                if st.button("刷新任务状态"):
                    st.rerun()

# 渲染页面
render_task_status_page()