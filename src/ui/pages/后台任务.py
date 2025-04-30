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

def retry_job(job_id: str, job_type: str, metadata: dict):
    """重试任务"""
    # 根据任务类型重新创建任务
    if job_type == "video_processing":
        # 获取视频URL和自定义元数据
        url = metadata.get("url", "")
        custom_metadata = metadata.get("custom_metadata", {})

        if not url:
            return {"success": False, "message": "无法获取视频URL"}

        # 重新提交视频处理任务
        response = api_request(
            endpoint="/ingest/video",
            method="POST",
            data={
                "url": url,
                "metadata": custom_metadata
            }
        )

        if response and "job_id" in response:
            return {"success": True, "message": f"已创建新任务", "new_job_id": response["job_id"]}
        else:
            return {"success": False, "message": "创建任务失败"}

    elif job_type == "pdf_processing":
        # PDF需要重新上传文件，不能直接重试
        return {"success": False, "message": "PDF任务需要重新上传文件"}

    elif job_type == "manual_text":
        # 获取文本内容和元数据
        content = metadata.get("content", "")
        text_metadata = metadata.get("text_metadata", {})

        if not content:
            return {"success": False, "message": "无法获取文本内容"}

        # 重新提交文本处理任务
        response = api_request(
            endpoint="/ingest/text",
            method="POST",
            data={
                "content": content,
                "metadata": text_metadata
            }
        )

        if response and "job_id" in response:
            return {"success": True, "message": f"已创建新任务", "new_job_id": response["job_id"]}
        else:
            return {"success": False, "message": "创建任务失败"}

    elif job_type == "llm_inference":
        # 获取查询内容和元数据过滤条件
        query = metadata.get("query", "")
        metadata_filter = metadata.get("metadata_filter", {})

        if not query:
            return {"success": False, "message": "无法获取查询内容"}

        # 重新提交异步查询任务
        response = api_request(
            endpoint="/query/async",
            method="POST",
            data={
                "query": query,
                "metadata_filter": metadata_filter,
                "top_k": 5
            }
        )

        if response and "job_id" in response:
            return {"success": True, "message": f"已创建新查询任务", "new_job_id": response["job_id"]}
        else:
            return {"success": False, "message": "创建任务失败"}

    elif job_type == "batch_video_processing":
        # 获取URL列表和自定义元数据
        urls = metadata.get("urls", [])
        custom_metadata = metadata.get("custom_metadata", [])

        if not urls:
            return {"success": False, "message": "无法获取视频URL列表"}

        # 重新提交批量视频处理任务
        response = api_request(
            endpoint="/ingest/batch-videos",
            method="POST",
            data={
                "urls": urls,
                "metadata": custom_metadata
            }
        )

        if response and "job_id" in response:
            return {"success": True, "message": f"已创建新批量任务", "new_job_id": response["job_id"]}
        else:
            return {"success": False, "message": "创建批量任务失败"}

    else:
        return {"success": False, "message": f"不支持重试此类型的任务: {job_type}"}

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
            params={"limit": 50}
        )

    if not all_jobs:
        st.warning("无法获取任务列表，请检查API连接")
        return

    # 过滤掉非查询任务
    query_jobs = all_jobs

    if not query_jobs:
        st.info("没有找到任务")
    else:
        # 任务列表标题
        st.subheader("任务列表")

        # 使用水平布局展示任务
        # 创建行和列以显示任务
        num_cols = 3  # 每行显示的任务数

        # 处理失败/超时任务的重试
        retry_job_id = None

        # 按创建时间对任务排序（最新的在前）
        sorted_jobs = sorted(query_jobs, key=lambda x: x.get("created_at", 0), reverse=True)

        # 准备任务列表
        for i in range(0, len(sorted_jobs), num_cols):
            cols = st.columns(num_cols)

            for j in range(num_cols):
                if i + j < len(sorted_jobs):
                    job = sorted_jobs[i + j]
                    job_id = job.get("job_id", "")
                    status = job.get("status", "")
                    status_icon = JOB_STATUS_COLORS.get(status, "⚪")
                    job_type = job.get("job_type", "")

                    # 创建唯一键
                    key_prefix = f"job_{job_id}"

                    # 获取任务的元数据
                    metadata = job.get("metadata", {})

                    # 对于不同类型的任务获取展示内容
                    display_content = ""

                    # 查询类任务
                    if job_type == "llm_inference":
                        query_text = metadata.get("query", "")
                        if len(query_text) > 30:
                            query_text = query_text[:27] + "..."
                        display_content = query_text

                    # 视频处理任务
                    elif job_type == "video_processing":
                        # 获取视频URL
                        video_url = metadata.get("url", "")

                        # 获取视频标题
                        video_title = metadata.get("video_title", "")

                        # 如果没有标题，尝试从custom_metadata获取
                        if not video_title and "custom_metadata" in metadata and metadata["custom_metadata"]:
                            video_title = metadata["custom_metadata"].get("title", "")

                        # 如果结果中包含标题，也可以从结果中获取
                        if not video_title and status == "completed" and job.get("result"):
                            result = job.get("result", {})
                            if isinstance(result, dict):
                                # 尝试从结果消息或metadata中提取标题
                                if "message" in result and "Successfully processed" in result["message"]:
                                    # 有些情况下标题可能在消息中
                                    parts = result["message"].split(": ")
                                    if len(parts) > 1:
                                        video_title = parts[1]

                        # 显示标题或URL
                        if video_title:
                            display_content = f"📹 {video_title}"
                        else:
                            display_content = f"📹 {video_url}"

                        if len(display_content) > 30:
                            display_content = display_content[:27] + "..."

                    # PDF处理任务
                    elif job_type == "pdf_processing":
                        # 从元数据中获取文件名
                        file_path = metadata.get("filepath", "")
                        file_name = os.path.basename(file_path) if file_path else "PDF文件"

                        # 尝试获取文档标题
                        pdf_title = ""
                        if "custom_metadata" in metadata and metadata["custom_metadata"]:
                            pdf_title = metadata["custom_metadata"].get("title", "")

                        # 显示标题或文件名
                        if pdf_title:
                            display_content = f"📄 {pdf_title}"
                        else:
                            display_content = f"📄 {file_name}"

                        if len(display_content) > 30:
                            display_content = display_content[:27] + "..."

                    # 手动文本输入任务
                    elif job_type == "manual_text":
                        text_title = metadata.get("title", "手动输入文本")
                        display_content = f"📝 {text_title}"

                        if len(display_content) > 30:
                            display_content = display_content[:27] + "..."

                    # 批量视频处理
                    elif job_type == "batch_video_processing":
                        # 获取视频URL列表
                        urls = metadata.get("urls", [])
                        url_count = len(urls)
                        display_content = f"📹 批量处理{url_count}个视频"

                    # 其他类型的任务
                    else:
                        # 尝试获取任何可用的描述信息
                        display_content = f"{job_type} 任务"

                    # 显示任务卡片
                    with cols[j]:
                        with st.container(border=True):
                            st.caption(f"ID: {job_id[:8]}...")
                            st.caption(f"类型: {job_type}")
                            st.markdown(f"**{display_content}**")
                            st.text(f"状态: {status_icon} {status}")

                            # 创建时间
                            created_time = time.strftime("%m-%d %H:%M", time.localtime(job.get("created_at", 0)))
                            st.caption(f"创建: {created_time}")

                            # 添加详情和重试按钮在同一行
                            detail_col, action_col = st.columns([1, 1])

                            with detail_col:
                                if st.button("详情", key=f"{key_prefix}_detail"):
                                    st.session_state.selected_job_id = job_id
                                    st.rerun()

                            with action_col:
                                # 失败或超时任务显示重试按钮
                                if status in ["failed", "timeout"]:
                                    if st.button("重试", key=f"{key_prefix}_retry"):
                                        retry_job_id = job_id

        # 如果有任务需要重试
        if retry_job_id:
            # 获取要重试的任务详情
            retry_job_data = api_request(
                endpoint=f"/ingest/jobs/{retry_job_id}",
                method="GET"
            )

            if retry_job_data:
                # 执行重试
                retry_result = retry_job(
                    job_id=retry_job_id,
                    job_type=retry_job_data.get("job_type", ""),
                    metadata=retry_job_data.get("metadata", {})
                )

                if retry_result["success"]:
                    st.success(f"{retry_result['message']}: {retry_result.get('new_job_id', '')}")
                    # 重新加载页面以显示新任务
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error(f"重试失败: {retry_result['message']}")
            else:
                st.error(f"无法获取任务 {retry_job_id} 的详情")

        if check_button and job_id:
            # 检查特定任务ID
            st.session_state.selected_job_id = job_id
            st.rerun()
        elif refresh_button:
            # 刷新时清除选择
            st.session_state.selected_job_id = None
            st.rerun()

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
            job_type = job_data.get("job_type", "")

            col1, col2, col3 = st.columns(3)
            with col1:
                st.info(f"任务ID: {selected_id}")
            with col2:
                st.info(f"状态: {status_icon} {status}")
            with col3:
                created_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(job_data.get("created_at", 0)))
                st.info(f"创建时间: {created_time}")

            # 添加重试按钮（对于失败的任务）
            if status in ["failed", "timeout"]:
                if st.button("⟲ 重试此任务", key="retry_detail"):
                    # 执行重试
                    retry_result = retry_job(
                        job_id=selected_id,
                        job_type=job_type,
                        metadata=job_data.get("metadata", {})
                    )

                    if retry_result["success"]:
                        st.success(f"{retry_result['message']}: {retry_result.get('new_job_id', '')}")
                        # 清除当前选择并重新加载页面以显示新任务
                        st.session_state.selected_job_id = retry_result.get("new_job_id", "")
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error(f"重试失败: {retry_result['message']}")

            # 根据任务类型显示不同的信息
            if job_type == "llm_inference":
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

            elif job_type == "video_processing":
                # 显示视频处理结果
                st.markdown("### 视频信息")

                # 从元数据中获取视频URL
                metadata = job_data.get("metadata", {})
                video_url = metadata.get("url", "未知URL")

                # 尝试获取视频标题
                video_title = metadata.get("video_title", "")
                if not video_title and "custom_metadata" in metadata and metadata["custom_metadata"]:
                    video_title = metadata["custom_metadata"].get("title", "")

                # 显示视频URL和标题（如果有）
                st.markdown(f"**URL**: {video_url}")
                if video_title:
                    st.markdown(f"**标题**: {video_title}")

                # 显示处理结果
                if status == "completed":
                    result = job_data.get("result", {})
                    if isinstance(result, str):
                        try:
                            result = json.loads(result)
                        except:
                            pass

                    if isinstance(result, dict):
                        # 显示文档数量
                        doc_count = result.get("document_count", 0)
                        st.markdown(f"**处理的文档数**: {doc_count}")

                        # 显示文档ID列表
                        doc_ids = result.get("document_ids", [])
                        if doc_ids:
                            st.markdown("**文档ID**:")
                            st.code("\n".join(doc_ids))

            elif job_type in ["pdf_processing", "manual_text"]:
                # 显示PDF或文本处理结果
                metadata = job_data.get("metadata", {})

                if job_type == "pdf_processing":
                    st.markdown("### PDF信息")
                    filepath = metadata.get("filepath", "")
                    filename = os.path.basename(filepath) if filepath else "未知文件"
                    st.markdown(f"**文件名**: {filename}")
                else:
                    st.markdown("### 文本信息")
                    title = metadata.get("title", "手动输入文本")
                    st.markdown(f"**标题**: {title}")

                # 显示处理结果
                if status == "completed":
                    result = job_data.get("result", {})
                    if isinstance(result, str):
                        try:
                            result = json.loads(result)
                        except:
                            pass

                    if isinstance(result, dict):
                        # 显示文档数量
                        doc_count = result.get("document_count", 0)
                        st.markdown(f"**处理的文档数**: {doc_count}")

                        # 显示文档ID列表
                        doc_ids = result.get("document_ids", [])
                        if doc_ids:
                            st.markdown("**文档ID**:")
                            st.code("\n".join(doc_ids))

            elif job_type == "batch_video_processing":
                # 显示批量视频处理信息
                st.markdown("### 批量视频处理")

                metadata = job_data.get("metadata", {})
                urls = metadata.get("urls", [])

                if urls:
                    st.markdown(f"**处理的URL数量**: {len(urls)}")

                    # 显示子任务ID和状态
                    if status == "completed" or status == "processing":
                        result = job_data.get("result", {})
                        if isinstance(result, str):
                            try:
                                result = json.loads(result)
                            except:
                                pass

                        if isinstance(result, dict):
                            sub_job_ids = result.get("sub_job_ids", [])
                            if sub_job_ids:
                                st.markdown("**子任务状态**:")

                                # 获取每个子任务的状态
                                for sub_id in sub_job_ids:
                                    sub_job_data = api_request(
                                        endpoint=f"/ingest/jobs/{sub_id}",
                                        method="GET"
                                    )

                                    if sub_job_data:
                                        sub_status = sub_job_data.get("status", "未知")
                                        sub_status_icon = JOB_STATUS_COLORS.get(sub_status, "⚪")
                                        sub_url = sub_job_data.get("metadata", {}).get("url", "未知URL")

                                        # 创建子任务容器
                                        with st.container():
                                            sub_cols = st.columns([3, 2, 1])
                                            with sub_cols[0]:
                                                st.text(f"URL: {sub_url}")
                                            with sub_cols[1]:
                                                st.text(f"状态: {sub_status_icon} {sub_status}")
                                            with sub_cols[2]:
                                                # 失败或超时任务显示重试按钮
                                                if sub_status in ["failed", "timeout"]:
                                                    if st.button("重试", key=f"retry_sub_{sub_id}"):
                                                        # 执行子任务重试
                                                        retry_result = retry_job(
                                                            job_id=sub_id,
                                                            job_type="video_processing",
                                                            metadata=sub_job_data.get("metadata", {})
                                                        )

                                                        if retry_result["success"]:
                                                            st.success(f"{retry_result['message']}: {retry_result.get('new_job_id', '')}")
                                                            # 重新加载页面
                                                            time.sleep(1)
                                                            st.rerun()
                                                        else:
                                                            st.error(f"重试失败: {retry_result['message']}")

            # 如果任务失败，显示错误信息
            if status == "failed":
                st.error(f"任务失败: {job_data.get('error', '未知错误')}")

            elif status in ["pending", "processing"]:
                # 显示处理中状态
                st.info("任务正在处理中...")

                # 添加刷新按钮
                if st.button("刷新任务状态"):
                    st.rerun()

# 渲染页面
render_task_status_page()