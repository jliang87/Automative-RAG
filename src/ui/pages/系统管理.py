"""
Update to src/ui/pages/系统管理.py to use enhanced worker status and notification system
"""

import streamlit as st
import pandas as pd
import time
import os
import json
import datetime
from src.ui.components import header, api_request
from typing import Dict, List, Any, Optional

# Add the enhanced components
from src.ui.enhanced_worker_status import enhanced_worker_status
from src.ui.system_notifications import display_notifications_sidebar
from src.ui.interactive_priority_queue_visualization import render_interactive_queue_visualization
from src.ui.enhanced_error_handling import robust_api_status_indicator, handle_worker_dependency


def format_bytes(size_bytes):
    """Format bytes to human-readable format."""
    if size_bytes == 0:
        return "0B"
    size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
    i = 0
    while size_bytes >= 1024 and i < len(size_name) - 1:
        size_bytes /= 1024
        i += 1
    return f"{size_bytes:.2f} {size_name[i]}"


def render_system_dashboard():
    """Render the system monitoring dashboard."""
    header(
        "系统管理控制台",
        "监控系统状态、资源使用和任务队列。"
    )

    # Display notifications in the sidebar
    display_notifications_sidebar(st.session_state.api_url, st.session_state.api_key)

    # Use robust API status indicator instead of simple status check
    with st.sidebar:
        api_available = robust_api_status_indicator(show_detail=True)

    # Only proceed if API is available
    if api_available:

        # Create tabs for different monitoring views
        tab1, tab2, tab3, tab4 = st.tabs(["系统状态", "资源监控", "队列管理", "系统配置"])

        with tab1:
            st.subheader("系统状态概览")

            # Get detailed health information
            health_info = api_request(
                endpoint="/system/health/detailed",
                method="GET"
            )

            if not health_info:
                st.error("无法获取系统状态信息")
                if st.button("重试"):
                    st.rerun()
                return

            # Display system information
            system_info = health_info.get("system", {})

            # Create a dashboard layout with metrics
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                status = health_info.get("status", "unknown")
                st.metric("系统状态", status)

            with col2:
                uptime = system_info.get("uptime", 0)
                uptime_days = uptime / (24 * 3600)
                st.metric("系统运行时间", f"{uptime_days:.1f}天")

            with col3:
                cpu = system_info.get("cpu_usage", 0)
                st.metric("CPU使用率", f"{cpu:.1f}%")

            with col4:
                mem = system_info.get("memory_usage", 0)
                st.metric("内存使用率", f"{mem:.1f}%")

            # Display worker status using enhanced component
            st.subheader("Worker状态")
            enhanced_worker_status()

            # Display model status
            st.subheader("模型状态")

            model_status = health_info.get("model_status", {})
            if model_status:
                model_data = []

                for model_type, status in model_status.items():
                    display_name = {
                        "embedding": "嵌入向量模型",
                        "llm": "大语言模型",
                        "colbert": "重排序模型",
                        "whisper": "语音识别模型"
                    }.get(model_type, model_type)

                    model_data.append({
                        "模型": display_name,
                        "加载状态": "已加载" if status.get("loaded", False) else "未加载",
                        "加载时间": f"{status.get('loading_time', 0):.1f}秒"
                    })

                # Display as DataFrame
                model_df = pd.DataFrame(model_data)
                st.dataframe(model_df, hide_index=True, use_container_width=True)
            else:
                st.warning("未获取到模型状态信息")

            # Display refresh button
            if st.button("刷新状态", key="refresh_status"):
                st.rerun()

        with tab2:
            st.subheader("资源监控")

            # Get detailed health information if not already fetched
            if not health_info:
                health_info = api_request(
                    endpoint="/system/health/detailed",
                    method="GET"
                )

                if not health_info:
                    st.error("无法获取系统状态信息")
                    return

            # Display CPU usage
            st.markdown("### CPU使用情况")
            cpu_usage = health_info.get("system", {}).get("cpu_usage", 0)
            st.progress(min(int(cpu_usage), 100), text=f"CPU使用率: {cpu_usage:.1f}%")

            # Display memory usage
            st.markdown("### 内存使用情况")
            memory_usage = health_info.get("system", {}).get("memory_usage", 0)
            st.progress(min(int(memory_usage), 100), text=f"内存使用率: {memory_usage:.1f}%")

            # Display GPU usage if available
            gpu_health = health_info.get("gpu_health", {})
            if gpu_health:
                st.markdown("### GPU使用情况")

                for gpu_id, gpu_info in gpu_health.items():
                    # Extract GPU info
                    device_name = gpu_info.get("device_name", gpu_id)
                    is_healthy = gpu_info.get("is_healthy", False)
                    total_memory_gb = gpu_info.get("total_memory_gb", 0)
                    allocated_memory_gb = gpu_info.get("allocated_memory_gb", 0)
                    free_memory_gb = gpu_info.get("free_memory_gb", 0)
                    free_percentage = gpu_info.get("free_percentage", 0)

                    # Display GPU card with status and memory
                    with st.expander(f"{device_name} - {'健康' if is_healthy else '异常'}", expanded=True):
                        # Memory usage bar
                        memory_usage = 100 - free_percentage
                        st.progress(min(int(memory_usage), 100), text=f"显存使用率: {memory_usage:.1f}%")

                        # Memory details
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("总显存", f"{total_memory_gb:.1f} GB")
                        with col2:
                            st.metric("已使用", f"{allocated_memory_gb:.1f} GB")
                        with col3:
                            st.metric("可用", f"{free_memory_gb:.1f} GB")

                        # Health message if not healthy
                        if not is_healthy:
                            st.error(f"GPU状态异常: {gpu_info.get('health_message', '')}")

                        # Add memory cleanup button
                        if st.button(f"清理{device_name}显存", key=f"cleanup_{gpu_id}"):
                            # This button would trigger a cache cleanup action
                            cleanup_response = api_request(
                                endpoint="/system/clear-gpu-cache",
                                method="POST",
                                data={"gpu_id": gpu_id}
                            )
                            st.success("已发送显存清理指令")
            else:
                st.info("未检测到GPU设备")

            # Display disk usage
            st.markdown("### 磁盘使用情况")
            disk_info = api_request(
                endpoint="/system/disk-usage",
                method="GET"
            )

            if disk_info:
                # Display disk partitions
                for partition, usage in disk_info.get("partitions", {}).items():
                    st.markdown(f"**{partition}:**")

                    # Extract usage information
                    total = usage.get("total", 0)
                    used = usage.get("used", 0)
                    free = usage.get("free", 0)
                    percent = usage.get("percent", 0)

                    # Display usage bar
                    st.progress(min(int(percent), 100), text=f"使用率: {percent:.1f}%")

                    # Display details
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("总容量", format_bytes(total))
                    with col2:
                        st.metric("已使用", format_bytes(used))
                    with col3:
                        st.metric("可用", format_bytes(free))

                # Display data directories
                st.markdown("### 数据目录")
                for dir_name, dir_info in disk_info.get("data_dirs", {}).items():
                    st.markdown(f"**{dir_name}:** {format_bytes(dir_info.get('size', 0))}")
            else:
                st.warning("无法获取磁盘使用信息")

            # Auto-refresh option
            auto_refresh = st.checkbox("自动刷新 (每30秒)", value=False)
            if auto_refresh:
                st.text("下次刷新倒计时...")
                for i in range(30, 0, -1):
                    # Update countdown every second
                    time_placeholder = st.empty()
                    time_placeholder.text(f"{i}秒后刷新...")
                    time.sleep(1)

                # Refresh after countdown
                st.rerun()

        with tab3:
            st.subheader("任务队列管理")

            # Get priority queue status
            priority_queue_status = api_request(
                endpoint="/query/queue-status",
                method="GET"
            )

            if priority_queue_status:
                # Render queue visualization using the interactive component
                render_interactive_queue_visualization(priority_queue_status)

                # Show active task details
                active_task = priority_queue_status.get("active_task")
                if active_task:
                    st.markdown("### 活动任务详情")

                    # Get job details for active task
                    job_id = active_task.get("job_id")
                    if job_id:
                        job_details = api_request(
                            endpoint=f"/ingest/jobs/{job_id}",
                            method="GET"
                        )

                        if job_details:
                            # Display basic job info
                            job_type = job_details.get("job_type", "unknown")
                            status = job_details.get("status", "unknown")
                            created_at = time.strftime("%Y-%m-%d %H:%M:%S",
                                                       time.localtime(job_details.get("created_at", 0)))

                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.markdown(f"**作业ID:** {job_id}")
                            with col2:
                                st.markdown(f"**类型:** {job_type}")
                            with col3:
                                st.markdown(f"**创建时间:** {created_at}")

                            # Display job metadata
                            metadata = job_details.get("metadata", {})
                            if metadata:
                                with st.expander("作业元数据", expanded=False):
                                    # Filter out sensitive information or huge content
                                    for key, value in metadata.items():
                                        if key != "content" and not isinstance(value, (dict, list)):
                                            st.markdown(f"**{key}:** {value}")

                            # Add option to terminate task if it seems stalled
                            task_age = time.time() - active_task.get("registered_at", time.time())
                            if task_age > 600:  # 10 minutes
                                st.warning(f"此任务已运行超过 {task_age / 60:.1f} 分钟，可能已经停滞")
                                if st.button("终止此任务", key="terminate_task"):
                                    terminate_response = api_request(
                                        endpoint="/system/terminate-task",
                                        method="POST",
                                        data={"task_id": active_task.get("task_id")}
                                    )
                                    if terminate_response:
                                        st.success("已发送终止信号")
                                    else:
                                        st.error("发送终止信号失败")
                        else:
                            st.warning(f"无法获取作业详情: {job_id}")
                    else:
                        st.info("活动任务未关联到作业ID")

                # Show queue tasks
                st.markdown("### 待处理任务")

                # Display tasks by queue
                tasks_by_queue = priority_queue_status.get("tasks_by_queue", {})
                for queue, count in tasks_by_queue.items():
                    if count > 0:
                        st.markdown(f"**{queue}:** {count} 个任务待处理")

                # Option to flush queues
                with st.expander("任务队列管理", expanded=False):
                    queue_options = list(tasks_by_queue.keys())
                    selected_queue = st.selectbox("选择队列", queue_options)

                    if st.button("清空所选队列", key="flush_queue"):
                        flush_response = api_request(
                            endpoint="/system/flush-queue",
                            method="POST",
                            data={"queue": selected_queue}
                        )
                        if flush_response:
                            st.success(f"已清空队列: {selected_queue}")
                        else:
                            st.error("清空队列失败")
            else:
                st.warning("无法获取队列状态信息")

            # Display dramatiq queue status
            st.subheader("Dramatiq队列状态")

            dramatiq_status = api_request(
                endpoint="/system/queue-stats",
                method="GET"
            )

            if dramatiq_status:
                queue_data = []

                for queue, stats in dramatiq_status.get("queues", {}).items():
                    queue_data.append({
                        "队列名称": queue,
                        "待处理消息": stats.get("messages", 0),
                        "已处理消息": stats.get("processed", 0),
                        "失败消息": stats.get("failed", 0),
                        "重试消息": stats.get("retried", 0)
                    })

                if queue_data:
                    queue_df = pd.DataFrame(queue_data)
                    st.dataframe(queue_df, hide_index=True, use_container_width=True)
                else:
                    st.info("没有队列数据")
            else:
                st.warning("无法获取Dramatiq队列信息")

            # Refresh button
            if st.button("刷新队列状态", key="refresh_queues"):
                st.rerun()

        with tab4:
            st.subheader("系统配置")

            # Get current system configuration
            config_info = api_request(
                endpoint="/system/config",
                method="GET"
            )

            if config_info:
                # Group configuration by category
                categories = {
                    "基本设置": ["host", "port", "api_auth_enabled"],
                    "模型设置": ["default_embedding_model", "default_colbert_model",
                                 "default_llm_model", "default_whisper_model"],
                    "GPU设置": ["device", "use_fp16", "batch_size", "llm_use_4bit", "llm_use_8bit"],
                    "检索设置": ["retriever_top_k", "reranker_top_k", "colbert_batch_size",
                                 "colbert_weight", "bge_weight"],
                    "分块设置": ["chunk_size", "chunk_overlap"],
                    "资源目录": ["data_dir", "models_dir", "upload_dir"]
                }

                # Display configuration by category with ability to edit
                for category, settings in categories.items():
                    with st.expander(category, expanded=True):
                        for setting in settings:
                            if setting in config_info:
                                value = config_info[setting]

                                # Different input types based on value type
                                if isinstance(value, bool):
                                    new_value = st.checkbox(setting, value)
                                elif isinstance(value, int):
                                    new_value = st.number_input(setting, value=value)
                                elif isinstance(value, float):
                                    new_value = st.number_input(setting, value=value, format="%.2f")
                                elif setting.endswith("_dir") or setting.endswith("_path"):
                                    new_value = st.text_input(setting, value)
                                else:
                                    new_value = st.text_input(setting, value)

                                # Track changes
                                if new_value != value:
                                    config_info[setting] = new_value

                # Add save button
                if st.button("保存配置更改", key="save_config"):
                    save_response = api_request(
                        endpoint="/system/update-config",
                        method="POST",
                        data=config_info
                    )

                    if save_response:
                        st.success("配置已更新")
                    else:
                        st.error("更新配置失败")
            else:
                st.warning("无法获取系统配置信息")

            # System maintenance tools
            st.subheader("系统维护")

            maintenance_option = st.selectbox(
                "选择维护操作",
                [
                    "清理过期任务",
                    "优化向量数据库",
                    "重置作业追踪器",
                    "清理临时文件",
                    "重新加载所有模型"
                ]
            )

            if st.button("执行", key="execute_maintenance"):
                endpoint = {
                    "清理过期任务": "/system/cleanup-old-jobs",
                    "优化向量数据库": "/system/optimize-vectorstore",
                    "重置作业追踪器": "/system/reset-job-tracker",
                    "清理临时文件": "/system/cleanup-temp-files",
                    "重新加载所有模型": "/system/reload-models"
                }.get(maintenance_option)

                if endpoint:
                    maintenance_response = api_request(
                        endpoint=endpoint,
                        method="POST"
                    )

                    if maintenance_response:
                        st.success(f"维护操作执行成功: {maintenance_option}")
                    else:
                        st.error("维护操作执行失败")

            # System logs
            st.subheader("系统日志")

            log_types = ["system", "worker", "api", "error"]
            selected_log = st.selectbox("选择日志类型", log_types)

            # Get log data
            logs_response = api_request(
                endpoint=f"/system/logs/{selected_log}",
                method="GET"
            )

            if logs_response and "content" in logs_response:
                st.text_area("日志内容", logs_response["content"], height=300)
            else:
                st.warning("无法获取日志内容")

        # Add a footer
        st.markdown("---")
        st.caption("系统管理控制台 - 仅供系统管理员使用")
        st.caption(f"当前时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    else:
        st.error("无法连接到API服务。请确保API服务正在运行。")
        st.info("您可以使用以下命令启动API服务：")
        st.code("docker-compose up -d api")


# Add this page to src/ui/pages/系统管理.py
render_system_dashboard()