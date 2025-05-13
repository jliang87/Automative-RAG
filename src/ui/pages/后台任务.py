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

# Import enhanced components
from src.ui.system_notifications import display_notifications_sidebar
from src.ui.enhanced_error_handling import robust_api_status_indicator, handle_worker_dependency
from src.ui.model_loading_status_indicator import model_loading_status
from src.ui.task_progress_visualization import display_task_progress, display_stage_timeline, render_priority_queue_visualization

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
    "embedding_tasks": "向量嵌入 (GPU-Embedding)",
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
        return "processing", "embedding_tasks"

    # 检查任务类型来确定处理阶段
    if job_type == "video_processing" or job_type == "batch_video_processing":
        # 检查结果中是否有转录信息
        if isinstance(result, dict) and "transcript" in result:
            return "processing", "embedding_tasks"  # 转录完成，正在嵌入
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
            return "processing", "embedding_tasks"
        return "processing", "cpu_tasks"  # 默认在CPU处理阶段

    elif job_type == "manual_text":
        # 检查是否在处理文本
        if isinstance(result, dict) and "embedding_job_id" in result:
            # 文本处理完成，等待嵌入
            return "processing", "embedding_tasks"
        return "processing", "cpu_tasks"  # 默认在CPU处理阶段

    elif job_type == "embedding":
        # 嵌入任务始终在GPU嵌入工作器上
        return "processing", "embedding_tasks"

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
            method="GET",
            retries=2,  # Add retries for more robust error handling
            timeout=5.0  # Increased timeout
        )
        if response:
            return response
        return None
    except Exception as e:
        st.warning(f"无法获取优先队列状态: {str(e)}")
        return None

def retry_job(job_id: str, job_type: str, metadata: dict):
    """重试任务"""
    # 检查相关Worker是否可用
    if job_type == "video_processing" or job_type == "batch_video_processing":
        if not handle_worker_dependency("video"):
            return {"success": False, "message": "视频处理Worker不可用"}
    elif job_type == "pdf_processing":
        if not handle_worker_dependency("pdf"):
            return {"success": False, "message": "PDF处理Worker不可用"}
    elif job_type == "manual_text":
        if not handle_worker_dependency("text"):
            return {"success": False, "message": "文本处理Worker不可用"}
    elif job_type == "llm_inference":
        if not handle_worker_dependency("query"):
            return {"success": False, "message": "查询处理Worker不可用"}

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
            },
            retries=1  # Add retry for robustness
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
            },
            retries=1
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
            endpoint="/query",
            method="POST",
            data={
                "query": query,
                "metadata_filter": metadata_filter,
                "top_k": 5
            },
            retries=1
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
            },
            retries=1
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

    # Display notifications in sidebar
    display_notifications_sidebar(st.session_state.api_url, st.session_state.api_key)

    # Check API status in sidebar
    with st.sidebar:
        api_available = robust_api_status_indicator(show_detail=True)

        # Show model loading status
        with st.expander("模型加载状态", expanded=False):
            model_loading_status()

    # Only proceed if API is available
    if api_available:
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
                    params={"limit": 100},
                    retries=2  # Add retries for robustness
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
                    method="GET",
                    retries=2  # Add retries for robustness
                )

                if not job_data:
                    st.error(f"无法获取任务 {selected_id} 的详情")
                    return

                # 使用 task_progress_visualization 来显示任务状态和进度
                action = display_task_progress(job_data)

                # 处理用户操作（如重试、取消等）
                if action["action"] == "retry":
                    # 执行重试
                    retry_result = retry_job(
                        job_id=selected_id,
                        job_type=job_data.get("job_type", ""),
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

                elif action["action"] == "cancel":
                    # 取消任务
                    cancel_response = api_request(
                        endpoint=f"/ingest/jobs/cancel/{selected_id}",
                        method="POST"
                    )
                    if cancel_response:
                        st.success(f"已发送取消请求。任务将在安全状态下终止。")
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error("取消任务失败")

                # 显示任务阶段时间线
                if "stage_history" in job_data:
                    st.subheader("处理阶段时间线")
                    display_stage_timeline(job_data)

                # 显示子任务信息（如果有）
                result = job_data.get("result", {})
                if isinstance(result, dict) and "embedding_job_id" in result:
                    st.subheader("子任务")
                    embedding_job_id = result.get("embedding_job_id")
                    st.markdown(f"向量嵌入任务ID: `{embedding_job_id}`")

                    if st.button("查看嵌入任务详情", key="view_embedding_button"):
                        st.session_state.selected_job_id = embedding_job_id
                        st.rerun()

                # 显示相关文档（对于查询任务）
                if job_data.get("job_type") == "llm_inference" and job_data.get("status") == "completed":
                    st.subheader("相关文档")

                    result = job_data.get("result", {})
                    if isinstance(result, str):
                        try:
                            result = json.loads(result)
                        except:
                            pass

                    # 显示文档
                    documents = result.get("documents", [])
                    if documents:
                        for i, doc in enumerate(documents):
                            display_document(doc, i)
                    else:
                        st.info("没有找到相关文档")

        # 系统状态选项卡
        with tab3:
            st.subheader("优先队列状态")

            # 获取优先队列状态信息
            queue_status = check_priority_queue_status()

            if queue_status:
                # 使用 task_progress_visualization 组件渲染优先队列
                render_priority_queue_visualization(queue_status)
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
                method="GET",
                retries=2  # Add retries for robustness
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
    else:
        st.error("无法连接到API服务或所需的Worker未运行")
        st.info("请检查系统状态并确保所需服务正在运行")

        if st.button("刷新", key="refresh_error"):
            st.rerun()

# 渲染页面
render_task_status_page()