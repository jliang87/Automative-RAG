"""
查询任务状态页面（Streamlit UI）- 增强版
显示任务在不同处理阶段和处理器之间的完整流转过程
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

# 任务状态颜色和图标定义
JOB_STATUS_COLORS = {
    "pending": "🟡",
    "processing": "🔵",
    "completed": "🟢",
    "failed": "🔴",
    "timeout": "🟠"
}

# 任务阶段名称映射
STAGE_NAMES = {
    "cpu_tasks": "文本/PDF处理 (CPU)",
    "gpu_tasks": "向量嵌入 (GPU-Embedding)",
    "inference_tasks": "查询生成 (GPU-Inference)",
    "transcription_tasks": "语音转录 (GPU-Whisper)",
    "reranking_tasks": "文档重排序 (GPU-Inference)",
    "system_tasks": "系统任务"
}

# 任务类型与子类型映射
JOB_TYPE_MAPPING = {
    "video_processing": {"type": "ingestion", "subtype": "视频处理"},
    "batch_video_processing": {"type": "ingestion", "subtype": "批量视频处理"},
    "pdf_processing": {"type": "ingestion", "subtype": "PDF处理"},
    "manual_text": {"type": "ingestion", "subtype": "文本输入"},
    "transcription": {"type": "ingestion", "subtype": "语音转录"},
    "embedding": {"type": "ingestion", "subtype": "向量嵌入"},
    "llm_inference": {"type": "query", "subtype": "查询处理"},
    "reranking": {"type": "query", "subtype": "文档重排序"},
}

def get_job_type_info(job_type):
    """获取任务类型的主类别和子类别"""
    info = JOB_TYPE_MAPPING.get(job_type, {"type": "其他", "subtype": job_type})
    return info["type"], info["subtype"]

def get_job_stage(job_data):
    """
    根据任务的元数据和状态判断其当前处理阶段
    """
    status = job_data.get("status", "")
    job_type = job_data.get("job_type", "")
    result = job_data.get("result", {})

    # 如果任务已完成或失败，直接返回状态
    if status in ["completed", "failed", "timeout"]:
        return status, None

    # 检查是否有子任务ID（表示任务链）
    if isinstance(result, dict) and "embedding_job_id" in result:
        # 表明主任务已完成其阶段，正在等待嵌入任务
        return "processing", "gpu_tasks"

    # 检查任务类型来确定处理阶段
    if job_type == "video_processing" or job_type == "batch_video_processing":
        # 检查结果中是否有转录信息
        if isinstance(result, dict) and "transcript" in result:
            return "processing", "gpu_tasks"  # 转录完成，正在嵌入
        elif isinstance(result, dict) and "message" in result:
            message = result.get("message", "")
            if "transcription in progress" in message:
                return "processing", "transcription_tasks"
            elif "downloading" in message:
                return "processing", "cpu_tasks"
        return "processing", "transcription_tasks"  # 默认假设在转录阶段

    elif job_type == "pdf_processing":
        # 检查是否在处理PDF
        if isinstance(result, dict) and "embedding_job_id" in result:
            # PDF处理完成，等待嵌入
            return "processing", "gpu_tasks"
        return "processing", "cpu_tasks"  # 默认在CPU处理阶段

    elif job_type == "manual_text":
        # 检查是否在处理文本
        if isinstance(result, dict) and "embedding_job_id" in result:
            # 文本处理完成，等待嵌入
            return "processing", "gpu_tasks"
        return "processing", "cpu_tasks"  # 默认在CPU处理阶段

    elif job_type == "embedding":
        # 嵌入任务始终在GPU嵌入工作器上
        return "processing", "gpu_tasks"

    elif job_type == "llm_inference":
        # 查询始终在GPU推理工作器上
        return "processing", "inference_tasks"

    elif job_type == "transcription":
        # 转录始终在GPU Whisper工作器上
        return "processing", "transcription_tasks"

    # 默认返回处理中状态
    return "processing", None

def check_priority_queue_status():
    """获取优先队列状态"""
    try:
        response = api_request(
            endpoint="/query/queue-status",
            method="GET"
        )
        if response:
            return response
        return None
    except Exception as e:
        st.warning(f"无法获取优先队列状态: {str(e)}")
        return None

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
        "后台任务管理",
        "查看和管理各种任务的状态，包括查询、视频处理、PDF处理和文本处理。"
    )

    # 创建两个选项卡：任务列表和任务详情
    tab1, tab2, tab3 = st.tabs(["任务列表", "任务详情", "系统状态"])

    with tab1:
        # 筛选和排序选项
        col1, col2, col3 = st.columns(3)

        with col1:
            job_type_filter = st.selectbox(
                "任务类型",
                options=["全部", "查询", "摄取"],
                index=0
            )

        with col2:
            job_status_filter = st.selectbox(
                "任务状态",
                options=["全部", "等待中", "处理中", "已完成", "失败"],
                index=0
            )

        with col3:
            sort_option = st.selectbox(
                "排序方式",
                options=["最新的优先", "最旧的优先"],
                index=0
            )

        # 刷新按钮
        if st.button("刷新任务列表", type="primary"):
            st.rerun()

        # 获取所有任务
        with st.spinner("获取任务列表..."):
            all_jobs = api_request(
                endpoint="/ingest/jobs",
                method="GET",
                params={"limit": 100}
            )

        if not all_jobs:
            st.warning("无法获取任务列表，请检查API连接")
            return

        # 应用筛选
        filtered_jobs = []
        for job in all_jobs:
            # 获取任务主类型和子类型
            main_type, subtype = get_job_type_info(job.get("job_type", ""))

            # 按任务类型筛选
            if job_type_filter == "查询" and main_type != "query":
                continue
            elif job_type_filter == "摄取" and main_type != "ingestion":
                continue

            # 按状态筛选
            status = job.get("status", "")
            if job_status_filter == "等待中" and status != "pending":
                continue
            elif job_status_filter == "处理中" and status != "processing":
                continue
            elif job_status_filter == "已完成" and status != "completed":
                continue
            elif job_status_filter == "失败" and status not in ["failed", "timeout"]:
                continue

            # 添加到筛选结果
            filtered_jobs.append(job)

        # 应用排序
        if sort_option == "最旧的优先":
            filtered_jobs.sort(key=lambda x: x.get("created_at", 0))
        else:  # 默认最新的优先
            filtered_jobs.sort(key=lambda x: x.get("created_at", 0), reverse=True)

        # 显示统计信息
        st.subheader("任务统计")

        # 计算各种状态的任务数量
        status_counts = {
            "pending": 0,
            "processing": 0,
            "completed": 0,
            "failed": 0,
            "timeout": 0
        }

        type_counts = {
            "query": 0,
            "ingestion": 0
        }

        for job in all_jobs:
            status = job.get("status", "")
            if status in status_counts:
                status_counts[status] += 1

            # 获取任务主类型
            main_type, _ = get_job_type_info(job.get("job_type", ""))
            if main_type in type_counts:
                type_counts[main_type] += 1

        # 显示统计信息
        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            st.metric("等待中", status_counts["pending"])
        with col2:
            st.metric("处理中", status_counts["processing"])
        with col3:
            st.metric("已完成", status_counts["completed"])
        with col4:
            st.metric("失败", status_counts["failed"] + status_counts["timeout"])
        with col5:
            st.metric("总任务数", len(all_jobs))

        # 显示类型分布
        col1, col2 = st.columns(2)

        with col1:
            st.metric("查询任务", type_counts["query"])
        with col2:
            st.metric("摄取任务", type_counts["ingestion"])

        # 任务列表表格
        st.subheader(f"任务列表 ({len(filtered_jobs)})")

        if filtered_jobs:
            # 创建Pandas DataFrame以便更好地显示和交互
            job_table_data = []

            for job in filtered_jobs:
                # 获取基本信息
                job_id = job.get("job_id", "")
                job_type = job.get("job_type", "")
                status = job.get("status", "")
                status_icon = JOB_STATUS_COLORS.get(status, "⚪")
                created_at = time.strftime("%m-%d %H:%M", time.localtime(job.get("created_at", 0)))
                updated_at = time.strftime("%m-%d %H:%M", time.localtime(job.get("updated_at", 0)))

                # 获取主类型和子类型
                main_type, subtype = get_job_type_info(job_type)

                # 获取处理阶段
                stage_status, stage = get_job_stage(job)
                stage_name = STAGE_NAMES.get(stage, "未知") if stage else "未知"

                # 获取任务描述
                description = ""
                metadata = job.get("metadata", {})

                if job_type == "llm_inference":
                    description = metadata.get("query", "")[:30] + "..." if len(metadata.get("query", "")) > 30 else metadata.get("query", "")
                elif job_type in ["video_processing", "batch_video_processing"]:
                    url = metadata.get("url", "")
                    description = url[:30] + "..." if len(url) > 30 else url
                elif job_type == "pdf_processing":
                    file_path = metadata.get("filepath", "")
                    file_name = os.path.basename(file_path) if file_path else "PDF文件"
                    description = file_name
                elif job_type == "manual_text":
                    description = "手动输入文本"

                # 添加到表格数据
                job_table_data.append({
                    "任务ID": job_id[:8] + "...",
                    "完整ID": job_id,  # 用于查看详情
                    "类型": main_type,
                    "子类型": subtype,
                    "状态": f"{status_icon} {status}",
                    "处理阶段": stage_name,
                    "创建时间": created_at,
                    "更新时间": updated_at,
                    "描述": description
                })

            # 创建DataFrame
            job_df = pd.DataFrame(job_table_data)

            # 使用st.dataframe显示，添加点击查看详情功能
            selected_indices = st.dataframe(
                job_df[["任务ID", "类型", "子类型", "状态", "处理阶段", "创建时间", "更新时间", "描述"]],
                hide_index=True,
                use_container_width=True,
                column_config={
                    "任务ID": st.column_config.TextColumn("任务ID", width="small"),
                    "类型": st.column_config.TextColumn("类型", width="small"),
                    "子类型": st.column_config.TextColumn("子类型", width="small"),
                    "状态": st.column_config.TextColumn("状态", width="small"),
                    "处理阶段": st.column_config.TextColumn("处理阶段", width="medium"),
                    "创建时间": st.column_config.TextColumn("创建时间", width="small"),
                    "更新时间": st.column_config.TextColumn("更新时间", width="small"),
                    "描述": st.column_config.TextColumn("描述", width="medium"),
                }
            )

            # 如果选择了行，显示详情
            if selected_indices is not None and len(selected_indices) > 0:
                selected_index = selected_indices[0]
                selected_job_id = job_df.iloc[selected_index]["完整ID"]
                st.session_state.selected_job_id = selected_job_id
                st.rerun()
        else:
            st.info("没有找到符合筛选条件的任务")

    # 任务详情选项卡
    with tab2:
        # 手动输入任务ID
        job_id_input = st.text_input("输入任务ID查看详情", key="job_id_input")

        check_button = st.button("查看详情", key="check_detail_button")

        if check_button and job_id_input:
            st.session_state.selected_job_id = job_id_input

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

            # 获取主类型和子类型
            main_type, subtype = get_job_type_info(job_type)

            # 获取处理阶段
            stage_status, stage = get_job_stage(job_data)
            stage_name = STAGE_NAMES.get(stage, "未知") if stage else "未知"

            # 显示任务基本信息
            basic_info_cols = st.columns(3)
            with basic_info_cols[0]:
                st.info(f"任务ID: {selected_id}")
            with basic_info_cols[1]:
                st.info(f"状态: {status_icon} {status}")
            with basic_info_cols[2]:
                created_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(job_data.get("created_at", 0)))
                st.info(f"创建时间: {created_time}")

            # 显示任务类型和阶段
            type_cols = st.columns(3)
            with type_cols[0]:
                st.info(f"主类型: {main_type}")
            with type_cols[1]:
                st.info(f"子类型: {subtype}")
            with type_cols[2]:
                st.info(f"处理阶段: {stage_name}")

            if "stage_history" in job_data:
                st.subheader("阶段处理时间")
                display_stage_timing(job_data)

            # 显示任务流程图
            st.subheader("任务流程")

            # 根据任务类型展示不同的流程图
            if main_type == "ingestion":
                if job_type in ["video_processing", "batch_video_processing"]:
                    # 视频处理流程
                    ingestion_cols = st.columns(5)

                    # 各阶段状态
                    download_status = "🔵" if stage == "cpu_tasks" else ("🟢" if stage in ["transcription_tasks", "gpu_tasks"] else "⚪")
                    transcription_status = "🔵" if stage == "transcription_tasks" else ("🟢" if stage == "gpu_tasks" else "⚪")
                    embedding_status = "🔵" if stage == "gpu_tasks" else "⚪"
                    completed_status = "🟢" if status == "completed" else "⚪"

                    with ingestion_cols[0]:
                        st.markdown(f"### {download_status}")
                        st.markdown("#### 视频下载")
                        st.markdown("CPU Worker")

                    with ingestion_cols[1]:
                        st.markdown("### ➡️")

                    with ingestion_cols[2]:
                        st.markdown(f"### {transcription_status}")
                        st.markdown("#### 语音转录")
                        st.markdown("GPU-Whisper Worker")

                    with ingestion_cols[3]:
                        st.markdown("### ➡️")

                    with ingestion_cols[4]:
                        st.markdown(f"### {embedding_status}")
                        st.markdown("#### 向量嵌入")
                        st.markdown("GPU-Embedding Worker")

                elif job_type in ["pdf_processing", "manual_text"]:
                    # PDF/文本处理流程
                    ingestion_cols = st.columns(3)

                    # 各阶段状态
                    process_status = "🔵" if stage == "cpu_tasks" else ("🟢" if stage == "gpu_tasks" else "⚪")
                    embedding_status = "🔵" if stage == "gpu_tasks" else "⚪"
                    completed_status = "🟢" if status == "completed" else "⚪"

                    with ingestion_cols[0]:
                        if job_type == "pdf_processing":
                            st.markdown(f"### {process_status}")
                            st.markdown("#### PDF处理和OCR")
                        else:
                            st.markdown(f"### {process_status}")
                            st.markdown("#### 文本处理")
                        st.markdown("CPU Worker")

                    with ingestion_cols[1]:
                        st.markdown("### ➡️")

                    with ingestion_cols[2]:
                        st.markdown(f"### {embedding_status}")
                        st.markdown("#### 向量嵌入")
                        st.markdown("GPU-Embedding Worker")

            elif main_type == "query":
                # 查询处理流程
                query_cols = st.columns(5)

                # 各阶段状态
                retrieve_status = "🟢" # 初始检索总是由API进行的
                rerank_status = "🔵" if stage == "reranking_tasks" else ("🟢" if stage == "inference_tasks" else "⚪")
                inference_status = "🔵" if stage == "inference_tasks" else "⚪"
                completed_status = "🟢" if status == "completed" else "⚪"

                with query_cols[0]:
                    st.markdown(f"### {retrieve_status}")
                    st.markdown("#### 初始检索")
                    st.markdown("API Server")

                with query_cols[1]:
                    st.markdown("### ➡️")

                with query_cols[2]:
                    st.markdown(f"### {rerank_status}")
                    st.markdown("#### 文档重排序")
                    st.markdown("GPU-Inference Worker")

                with query_cols[3]:
                    st.markdown("### ➡️")

                with query_cols[4]:
                    st.markdown(f"### {inference_status}")
                    st.markdown("#### 答案生成")
                    st.markdown("GPU-Inference Worker")

            # 显示任务结果
            # 添加重试按钮（对于失败的任务）
            if status in ["failed", "timeout"]:
                if st.button("⟲ 重试此任务", key="retry_detail_button"):
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

            # 显示任务结果或错误
            if status == "completed":
                st.subheader("任务结果")
                result = job_data.get("result", {})

                if isinstance(result, str):
                    try:
                        result = json.loads(result)
                    except:
                        st.text(result)

                if job_type == "llm_inference":
                    # 显示查询结果
                    st.markdown("### 查询内容")
                    query = job_data.get("metadata", {}).get("query", "")
                    st.markdown(f"> {query}")

                    st.markdown("### 回答")
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

                elif job_type in ["video_processing", "batch_video_processing"]:
                    # 显示视频处理结果
                    if isinstance(result, dict):
                        if "transcript" in result:
                            st.markdown("### 转录结果")
                            st.text_area("转录文本", result.get("transcript", ""), height=200)

                            st.markdown("### 元信息")
                            st.json({
                                "language": result.get("language", "未知"),
                                "duration": result.get("duration", 0),
                                "processing_time": result.get("processing_time", 0)
                            })

                        # 显示文档ID
                        if "document_ids" in result:
                            st.markdown("### 生成的文档")
                            st.code("\n".join(result.get("document_ids", [])))

                            st.markdown(f"总共生成 {result.get('document_count', 0)} 个文档")

                elif job_type in ["pdf_processing", "manual_text"]:
                    # 显示PDF或文本处理结果
                    if isinstance(result, dict):
                        # 显示文档ID
                        if "document_ids" in result:
                            st.markdown("### 生成的文档")
                            st.code("\n".join(result.get("document_ids", [])))

                            st.markdown(f"总共生成 {result.get('document_count', 0)} 个文档")

                            if "processing_time" in result:
                                st.caption(f"处理时间: {result.get('processing_time', 0):.2f}秒")

            elif status in ["failed", "timeout"]:
                st.subheader("错误信息")
                st.error(job_data.get("error", "未知错误"))

            elif status in ["pending", "processing"]:
                st.subheader("处理状态")

                # 显示处理中信息
                result = job_data.get("result", {})
                if isinstance(result, dict) and "message" in result:
                    st.info(result.get("message", "任务正在处理中..."))
                else:
                    st.info("任务正在处理中...")

                # 显示处理时间
                if job_data.get("updated_at") and job_data.get("created_at"):
                    processing_time = job_data.get("updated_at") - job_data.get("created_at")
                    st.caption(f"已处理时间: {processing_time:.2f}秒")

                # 添加刷新按钮
                if st.button("刷新任务状态", key="refresh_status_button"):
                    st.rerun()

            # 显示子任务信息（如果有）
            result = job_data.get("result", {})
            if isinstance(result, dict) and "embedding_job_id" in result:
                st.subheader("子任务")
                embedding_job_id = result.get("embedding_job_id")
                st.markdown(f"向量嵌入任务ID: `{embedding_job_id}`")

                if st.button("查看嵌入任务详情", key="view_embedding_button"):
                    st.session_state.selected_job_id = embedding_job_id
                    st.rerun()

    # 系统状态选项卡
    with tab3:
        st.subheader("优先队列状态")

        # 获取优先队列状态信息
        queue_status = check_priority_queue_status()

        if queue_status:
            # 显示活动任务
            st.markdown("### 当前活动GPU任务")
            active_task = queue_status.get("active_task")
            if active_task:
                active_task_info = [
                    ["任务ID", active_task.get("task_id", "未知")],
                    ["队列", active_task.get("queue_name", "未知")],
                    ["优先级", active_task.get("priority", "未知")],
                    ["任务类型", active_task.get("job_id", "未知")]
                ]

                # 计算任务活动时间
                registered_at = active_task.get("registered_at")
                if registered_at:
                    time_active = time.time() - registered_at
                    if time_active < 60:
                        time_str = f"{time_active:.1f}秒"
                    elif time_active < 3600:
                        time_str = f"{time_active/60:.1f}分钟"
                    else:
                        time_str = f"{time_active/3600:.1f}小时"

                    active_task_info.append(["活动时间", time_str])

                # 使用数据表显示
                active_df = pd.DataFrame(active_task_info, columns=["属性", "值"])
                st.dataframe(active_df, hide_index=True, use_container_width=True)
            else:
                st.info("当前没有活动的GPU任务")

            # 显示等待任务
            st.markdown("### 等待中的任务")

            # 按优先级分组显示任务
            tasks_by_priority = queue_status.get("tasks_by_priority", {})
            if tasks_by_priority:
                priority_data = []
                for priority, count in tasks_by_priority.items():
                    # 找出优先级对应的队列名称
                    queue_name = "未知"
                    for q, p in queue_status.get("priority_levels", {}).items():
                        if p == priority:
                            queue_name = q

                    priority_data.append({
                        "优先级": int(priority),
                        "队列": queue_name,
                        "任务数": count
                    })

                # 按优先级排序
                priority_data.sort(key=lambda x: x["优先级"])

                # 显示为DataFrame
                priority_df = pd.DataFrame(priority_data)
                st.dataframe(priority_df, hide_index=True, use_container_width=True)
            else:
                st.info("当前没有等待中的任务")

            # 显示各队列统计
            st.markdown("### 队列分布")

            tasks_by_queue = queue_status.get("tasks_by_queue", {})
            if tasks_by_queue:
                # 创建饼图数据
                queue_data = []
                for queue, count in tasks_by_queue.items():
                    queue_data.append({
                        "队列": STAGE_NAMES.get(queue, queue),
                        "任务数": count
                    })

                # 按任务数排序
                queue_data.sort(key=lambda x: x["任务数"], reverse=True)

                # 显示为DataFrame
                queue_df = pd.DataFrame(queue_data)
                st.dataframe(queue_df, hide_index=True, use_container_width=True)

                # 简单的刷新按钮
                if st.button("刷新队列状态", key="refresh_queue_button"):
                    st.rerun()
            else:
                st.info("当前没有等待中的任务")
        else:
            st.warning("无法获取优先队列状态")

            # 添加刷新按钮
            if st.button("刷新状态", key="refresh_status_button"):
                st.rerun()

        # 显示GPU使用情况
        st.subheader("GPU使用情况")

        # 获取系统状态信息
        system_status = api_request(
            endpoint="/ingest/status",
            method="GET"
        )

        if system_status and "gpu_info" in system_status:
            gpu_info = system_status.get("gpu_info", {})

            if gpu_info:
                gpu_data = []

                if "device_name" in gpu_info:
                    gpu_data.append(["设备名称", gpu_info["device_name"]])

                if "memory_allocated" in gpu_info:
                    gpu_data.append(["已使用显存", gpu_info["memory_allocated"]])

                if "memory_reserved" in gpu_info:
                    gpu_data.append(["已保留显存", gpu_info["memory_reserved"]])

                if "device" in gpu_info:
                    gpu_data.append(["设备", gpu_info["device"]])

                if "fp16_enabled" in gpu_info:
                    gpu_data.append(["混合精度", "启用" if gpu_info["fp16_enabled"] else "禁用"])

                if "whisper_model" in gpu_info:
                    gpu_data.append(["Whisper模型", gpu_info["whisper_model"]])

                # 显示为DataFrame
                gpu_df = pd.DataFrame(gpu_data, columns=["属性", "值"])
                st.dataframe(gpu_df, hide_index=True, use_container_width=True)
            else:
                st.info("没有获取到GPU信息")
        else:
            st.warning("无法获取系统状态信息")

def display_stage_timing(job_data):
    """Display timing information for each processing stage."""
    stage_history = job_data.get("stage_history", [])
    if not stage_history:
        st.info("没有可用的阶段计时信息")
        return

    # Calculate time spent in each stage
    stage_timings = []
    current_time = time.time()

    for i, stage_entry in enumerate(stage_history):
        stage = stage_entry["stage"]
        stage_name = STAGE_NAMES.get(stage, stage)
        start_time = stage_entry["started_at"]

        # Calculate end time (either next stage start or current time)
        if i < len(stage_history) - 1:
            end_time = stage_history[i + 1]["started_at"]
        else:
            # For the current stage
            end_time = current_time

        duration = end_time - start_time

        # Format duration string
        if duration < 60:
            duration_str = f"{duration:.1f}秒"
        elif duration < 3600:
            duration_str = f"{duration / 60:.1f}分钟"
        else:
            duration_str = f"{duration / 3600:.1f}小时"

        stage_timings.append({
            "阶段": stage_name,
            "开始时间": time.strftime("%H:%M:%S", time.localtime(start_time)),
            "持续时间": duration_str,
            "原始时长(秒)": duration  # For sorting and calculations
        })

    # Calculate total processing time
    total_time = sum(timing["原始时长(秒)"] for timing in stage_timings)
    if total_time > 0:
        for timing in stage_timings:
            timing["占比"] = f"{(timing['原始时长(秒)'] / total_time * 100):.1f}%"

    # Display as table
    timing_df = pd.DataFrame(stage_timings)
    st.dataframe(
        timing_df[["阶段", "开始时间", "持续时间", "占比"]],
        hide_index=True,
        use_container_width=True
    )

    # Add a visual timeline
    st.subheader("处理时间线")

    # Create a horizontal bar chart showing time distribution
    chart_data = pd.DataFrame(stage_timings)
    chart_data = chart_data.sort_values("开始时间")

    # Create a simple text-based timeline (can be replaced with a proper chart)
    total_chars = 50  # Width of timeline in characters
    timeline = ""

    if total_time > 0:
        for timing in stage_timings:
            stage_width = int((timing["原始时长(秒)"] / total_time) * total_chars)
            if stage_width < 1:
                stage_width = 1

            timeline += timing["阶段"][0] * stage_width  # Use first character of stage name

        # Print the timeline
        st.text(timeline)
        st.caption(f"总处理时间: {format_duration(total_time)}")

def format_duration(timestamp):
    """Format duration in seconds to a readable string."""
    if not timestamp:
        return "N/A"

    seconds = time.time() - timestamp

    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds / 60:.1f}m"
    else:
        return f"{seconds / 3600:.1f}h"

# 渲染页面
render_task_status_page()