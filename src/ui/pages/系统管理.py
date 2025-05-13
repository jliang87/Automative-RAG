"""
系统管理页面 - 使用增强的worker状态和通知系统
"""

import streamlit as st
import pandas as pd
import time
import os
import json
import datetime
from typing import Dict, List, Any, Optional

# 导入统一的 API 客户端
from src.ui.api_client import api_request
from src.ui.components import header

# 导入增强组件
from src.ui.enhanced_worker_status import enhanced_worker_status
from src.ui.system_notifications import display_notifications_sidebar
from src.ui.interactive_priority_queue_visualization import render_interactive_queue_visualization
from src.ui.enhanced_error_handling import robust_api_status_indicator


def format_bytes(size_bytes):
    """将字节格式化为人类可读格式。"""
    if size_bytes == 0:
        return "0B"
    size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
    i = 0
    while size_bytes >= 1024 and i < len(size_name) - 1:
        size_bytes /= 1024
        i += 1
    return f"{size_bytes:.2f} {size_name[i]}"


def render_system_dashboard():
    """渲染系统监控仪表板。"""
    header(
        "系统管理控制台",
        "监控系统状态、资源使用和任务队列。"
    )

    # 在侧边栏显示通知
    display_notifications_sidebar(st.session_state.api_url, st.session_state.api_key)

    # 使用增强的 API 状态指示器代替简单的状态检查
    with st.sidebar:
        api_available = robust_api_status_indicator(show_detail=True)

    # 仅在 API 可用时继续
    if api_available:

        # 创建不同监控视图的选项卡
        tab1, tab2, tab3, tab4 = st.tabs(["系统状态", "资源监控", "队列管理", "系统配置"])

        with tab1:
            st.subheader("系统状态概览")

            # 获取详细的健康信息
            health_info = api_request(
                endpoint="/system/health/detailed",
                method="GET"
            )

            if not health_info:
                st.error("无法获取系统状态信息")
                if st.button("重试"):
                    st.rerun()
                return

            # 显示系统信息
            system_info = health_info.get("system", {})

            # 创建带有度量的仪表板布局
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

            # 使用增强组件显示 worker 状态
            st.subheader("Worker 状态")
            enhanced_worker_status()

            # 显示模型状态
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

                # 显示为 DataFrame
                model_df = pd.DataFrame(model_data)
                st.dataframe(model_df, hide_index=True, use_container_width=True)
            else:
                st.warning("未获取到模型状态信息")

            # 显示刷新按钮
            if st.button("刷新状态", key="refresh_status"):
                st.rerun()

        with tab2:
            st.subheader("资源监控")

            # 如果尚未获取，则获取详细的健康信息
            if not health_info:
                health_info = api_request(
                    endpoint="/system/health/detailed",
                    method="GET"
                )

                if not health_info:
                    st.error("无法获取系统状态信息")
                    return

            # 显示 CPU 使用情况
            st.markdown("### CPU使用情况")
            cpu_usage = health_info.get("system", {}).get("cpu_usage", 0)
            st.progress(min(int(cpu_usage), 100), text=f"CPU使用率: {cpu_usage:.1f}%")

            # 显示内存使用情况
            st.markdown("### 内存使用情况")
            memory_usage = health_info.get("system", {}).get("memory_usage", 0)
            st.progress(min(int(memory_usage), 100), text=f"内存使用率: {memory_usage:.1f}%")

            # 如果可用，显示 GPU 使用情况
            gpu_health = health_info.get("gpu_health", {})
            if gpu_health:
                st.markdown("### GPU使用情况")

                for gpu_id, gpu_info in gpu_health.items():
                    # 提取 GPU 信息
                    device_name = gpu_info.get("device_name", gpu_id)
                    is_healthy = gpu_info.get("is_healthy", False)
                    total_memory_gb = gpu_info.get("total_memory_gb", 0)
                    allocated_memory_gb = gpu_info.get("allocated_memory_gb", 0)
                    free_memory_gb = gpu_info.get("free_memory_gb", 0)
                    free_percentage = gpu_info.get("free_percentage", 0)

                    # 显示带有状态和内存的 GPU 卡片
                    with st.expander(f"{device_name} - {'健康' if is_healthy else '异常'}", expanded=True):
                        # 内存使用栏
                        memory_usage = 100 - free_percentage
                        st.progress(min(int(memory_usage), 100), text=f"显存使用率: {memory_usage:.1f}%")

                        # 内存详情
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("总显存", f"{total_memory_gb:.1f} GB")
                        with col2:
                            st.metric("已使用", f"{allocated_memory_gb:.1f} GB")
                        with col3:
                            st.metric("可用", f"{free_memory_gb:.1f} GB")

                        # 如果不健康则显示健康消息
                        if not is_healthy:
                            st.error(f"GPU状态异常: {gpu_info.get('health_message', '')}")

                        # 添加内存清理按钮
                        if st.button(f"清理{device_name}显存", key=f"cleanup_{gpu_id}"):
                            # 此按钮将触发缓存清理操作
                            cleanup_response = api_request(
                                endpoint="/system/clear-gpu-cache",
                                method="POST",
                                data={"gpu_id": gpu_id}
                            )
                            st.success("已发送显存清理指令")
            else:
                st.info("未检测到GPU设备")

            # 显示磁盘使用情况
            st.markdown("### 磁盘使用情况")
            disk_info = api_request(
                endpoint="/system/disk-usage",
                method="GET"
            )

            if disk_info:
                # 显示磁盘分区
                for partition, usage in disk_info.get("partitions", {}).items():
                    st.markdown(f"**{partition}:**")

                    # 提取使用信息
                    total = usage.get("total", 0)
                    used = usage.get("used", 0)
                    free = usage.get("free", 0)
                    percent = usage.get("percent", 0)

                    # 显示使用栏
                    st.progress(min(int(percent), 100), text=f"使用率: {percent:.1f}%")

                    # 显示详情
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("总容量", format_bytes(total))
                    with col2:
                        st.metric("已使用", format_bytes(used))
                    with col3:
                        st.metric("可用", format_bytes(free))

                # 显示数据目录
                st.markdown("### 数据目录")
                for dir_name, dir_info in disk_info.get("data_dirs", {}).items():
                    st.markdown(f"**{dir_name}:** {format_bytes(dir_info.get('size', 0))}")
            else:
                st.warning("无法获取磁盘使用信息")

            # 自动刷新选项
            auto_refresh = st.checkbox("自动刷新 (每30秒)", value=False)
            if auto_refresh:
                st.text("下次刷新倒计时...")
                for i in range(30, 0, -1):
                    # 每秒更新倒计时
                    time_placeholder = st.empty()
                    time_placeholder.text(f"{i}秒后刷新...")
                    time.sleep(1)

                # 倒计时后刷新
                st.rerun()

        with tab3:
            st.subheader("任务队列管理")

            # 获取优先队列状态
            priority_queue_status = api_request(
                endpoint="/query/queue-status",
                method="GET"
            )

            if priority_queue_status:
                # 使用交互式组件渲染队列可视化
                render_interactive_queue_visualization(st.session_state.api_url, st.session_state.api_key)

                # 显示活动任务详情
                active_task = priority_queue_status.get("active_task")
                if active_task:
                    st.markdown("### 活动任务详情")

                    # 获取活动任务的作业详情
                    job_id = active_task.get("job_id")
                    if job_id:
                        job_details = api_request(
                            endpoint=f"/ingest/jobs/{job_id}",
                            method="GET"
                        )

                        if job_details:
                            # 显示基本作业信息
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

                            # 显示作业元数据
                            metadata = job_details.get("metadata", {})
                            if metadata:
                                with st.expander("作业元数据", expanded=False):
                                    # 过滤掉敏感信息或大内容
                                    for key, value in metadata.items():
                                        if key != "content" and not isinstance(value, (dict, list)):
                                            st.markdown(f"**{key}:** {value}")

                            # 如果任务似乎停滞，添加终止任务的选项
                            task_age = time.time() - active_task.get("registered_at", time.time())
                            if task_age > 600:  # 10 分钟
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

                # 显示队列任务
                st.markdown("### 待处理任务")

                # 按队列显示任务
                tasks_by_queue = priority_queue_status.get("tasks_by_queue", {})
                for queue, count in tasks_by_queue.items():
                    if count > 0:
                        st.markdown(f"**{queue}:** {count} 个任务待处理")

                # 清空队列的选项
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

            # 显示 dramatiq 队列状态
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

            # 刷新按钮
            if st.button("刷新队列状态", key="refresh_queues"):
                st.rerun()

        with tab4:
            st.subheader("系统配置")

            # 获取当前系统配置
            config_info = api_request(
                endpoint="/system/config",
                method="GET"
            )

            if config_info:
                # 按类别分组配置
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

                # 按类别显示配置，并能够编辑
                for category, settings in categories.items():
                    with st.expander(category, expanded=True):
                        for setting in settings:
                            if setting in config_info:
                                value = config_info[setting]

                                # 根据值类型使用不同的输入类型
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

                                # 跟踪更改
                                if new_value != value:
                                    config_info[setting] = new_value

                # 添加保存按钮
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

            # 系统维护工具
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

            # 系统日志
            st.subheader("系统日志")

            log_types = ["system", "worker", "api", "error"]
            selected_log = st.selectbox("选择日志类型", log_types)

            # 获取日志数据
            logs_response = api_request(
                endpoint=f"/system/logs/{selected_log}",
                method="GET"
            )

            if logs_response and "content" in logs_response:
                st.text_area("日志内容", logs_response["content"], height=300)
            else:
                st.warning("无法获取日志内容")

        # 添加页脚
        st.markdown("---")
        st.caption("系统管理控制台 - 仅供系统管理员使用")
        st.caption(f"当前时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    else:
        st.error("无法连接到API服务。请确保API服务正在运行。")
        st.info("您可以使用以下命令启动API服务：")
        st.code("docker-compose up -d api")


# 调用函数渲染系统仪表板
render_system_dashboard()