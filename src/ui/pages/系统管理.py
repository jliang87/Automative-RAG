"""
系统管理页面 - 更新为自触发作业链架构
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
from src.ui.enhanced_error_handling import robust_api_status_indicator
from src.ui.session_init import initialize_session_state

initialize_session_state()


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


def render_job_chain_queue_management():
    """渲染作业链队列管理 (替代优先队列可视化)"""
    st.subheader("作业链队列管理")

    # 获取作业链队列状态
    queue_status = api_request(
        endpoint="/query/queue-status",
        method="GET"
    )

    if not queue_status:
        st.warning("无法获取队列状态")
        return

    # 显示队列状态概览
    st.markdown("### 专用队列状态")

    queue_mapping = {
        "transcription_tasks": {"name": "🎵 语音转录队列", "worker": "GPU-Whisper"},
        "embedding_tasks": {"name": "🔢 向量嵌入队列", "worker": "GPU-嵌入"},
        "inference_tasks": {"name": "🧠 LLM推理队列", "worker": "GPU-推理"},
        "cpu_tasks": {"name": "💻 CPU处理队列", "worker": "CPU"}
    }

    queue_data = []
    for queue_name, queue_info in queue_status.get("queue_status", {}).items():
        if queue_name in queue_mapping:
            mapping = queue_mapping[queue_name]
            status = queue_info.get("status", "free")
            waiting = queue_info.get("waiting_tasks", 0)

            if status == "busy":
                current_job = queue_info.get("current_job", "unknown")
                current_task = queue_info.get("current_task", "unknown")
                busy_since = queue_info.get("busy_since", 0)

                if busy_since > 0:
                    elapsed = time.time() - busy_since
                    elapsed_str = f"{elapsed:.0f}秒" if elapsed < 60 else f"{elapsed / 60:.1f}分钟"
                else:
                    elapsed_str = "未知"

                status_display = f"🔄 处理中 ({elapsed_str})"
                details = f"作业: {current_job[:8]}... | 任务: {current_task}"
            else:
                status_display = "✅ 空闲"
                details = "-"

            queue_data.append({
                "队列": mapping["name"],
                "专用Worker": mapping["worker"],
                "状态": status_display,
                "等待任务": waiting,
                "详情": details
            })

    if queue_data:
        queue_df = pd.DataFrame(queue_data)
        st.dataframe(queue_df, hide_index=True, use_container_width=True)

        # 队列利用率可视化
        st.markdown("### 队列利用率")
        for row in queue_data:
            queue_name = row["队列"]
            is_busy = "处理中" in row["状态"]
            waiting_count = row["等待任务"]

            if is_busy:
                utilization = 100
                status_text = f"100% - {row['详情']}"
            elif waiting_count > 0:
                utilization = min(waiting_count * 10, 90)
                status_text = f"{waiting_count}个任务等待"
            else:
                utilization = 0
                status_text = "空闲"

            st.progress(utilization / 100, text=f"{queue_name}: {status_text}")

    # 活跃作业链管理
    st.markdown("### 活跃作业链")

    # 获取作业链概览
    job_chains_overview = api_request(
        endpoint="/job-chains",
        method="GET"
    )

    if job_chains_overview:
        active_chains = job_chains_overview.get("active_chains", [])

        if active_chains:
            chain_data = []
            for chain in active_chains:
                job_id = chain.get("job_id", "")
                job_type = chain.get("job_type", "")
                current_task = chain.get("current_task", "")
                progress = chain.get("progress_percentage", 0)
                started_at = chain.get("started_at", 0)

                # 计算运行时间
                if started_at > 0:
                    elapsed = time.time() - started_at
                    elapsed_str = f"{elapsed:.0f}秒" if elapsed < 60 else f"{elapsed / 60:.1f}分钟"
                else:
                    elapsed_str = "未知"

                chain_data.append({
                    "作业ID": job_id[:8] + "...",
                    "类型": job_type,
                    "当前任务": current_task,
                    "进度": f"{progress:.1f}%",
                    "运行时间": elapsed_str
                })

            chain_df = pd.DataFrame(chain_data)
            st.dataframe(chain_df, hide_index=True, use_container_width=True)

            # 作业链管理操作
            with st.expander("作业链管理操作", expanded=False):
                # 选择作业进行操作
                job_ids = [chain.get("job_id", "") for chain in active_chains]
                if job_ids:
                    selected_job = st.selectbox("选择作业链", job_ids, format_func=lambda x: x[:8] + "...")

                    col1, col2 = st.columns(2)

                    with col1:
                        if st.button("查看详细信息", key="view_chain_details"):
                            chain_details = api_request(
                                endpoint=f"/job-chains/{selected_job}",
                                method="GET"
                            )
                            if chain_details:
                                st.json(chain_details)

                    with col2:
                        if st.button("取消作业链", key="cancel_chain"):
                            cancel_response = api_request(
                                endpoint=f"/ingest/jobs/{selected_job}",
                                method="DELETE"
                            )
                            if cancel_response:
                                st.success("作业链取消请求已发送")
                            else:
                                st.error("取消作业链失败")
        else:
            st.info("当前没有活跃的作业链")

    # 队列清理操作
    st.markdown("### 队列管理操作")

    col1, col2 = st.columns(2)

    with col1:
        queue_names = list(queue_mapping.keys())
        selected_queue = st.selectbox("选择队列", queue_names,
                                     format_func=lambda x: queue_mapping[x]["name"])

    with col2:
        queue_action = st.selectbox("队列操作", ["查看状态", "清空等待任务", "重启对应Worker"])

    if st.button("执行队列操作", key="execute_queue_action"):
        if queue_action == "查看状态":
            st.json(queue_status.get("queue_status", {}).get(selected_queue, {}))

        elif queue_action == "清空等待任务":
            st.warning("⚠️ 这将清空队列中的所有等待任务")
            if st.button("确认清空", key="confirm_clear"):
                # 清空队列的逻辑需要后端API支持
                st.info("清空队列功能需要后端API支持")

        elif queue_action == "重启对应Worker":
            worker_type = {
                "transcription_tasks": "gpu-whisper",
                "embedding_tasks": "gpu-embedding",
                "inference_tasks": "gpu-inference",
                "cpu_tasks": "cpu"
            }.get(selected_queue)

            if worker_type:
                restart_response = api_request(
                    endpoint="/system/restart-workers",
                    method="POST",
                    data={"worker_type": worker_type}
                )
                if restart_response:
                    st.success(f"已发送重启信号到 {worker_type} workers")
                else:
                    st.error("重启信号发送失败")


def render_system_dashboard():
    """渲染系统监控仪表板。"""
    header(
        "系统管理控制台",
        "监控自触发作业链系统状态、专用Worker和GPU资源。"
    )

    # 在侧边栏显示通知
    display_notifications_sidebar(st.session_state.api_url, st.session_state.api_key)

    # 使用增强的 API 状态指示器
    with st.sidebar:
        api_available = robust_api_status_indicator(show_detail=True)

    # 仅在 API 可用时继续
    if api_available:

        # 创建不同监控视图的选项卡
        tab1, tab2, tab3, tab4 = st.tabs(["系统状态", "资源监控", "作业链管理", "系统配置"])

        with tab1:
            st.subheader("自触发作业链系统状态")

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

            # 使用增强组件显示专用 worker 状态
            st.subheader("专用Worker状态")
            enhanced_worker_status()

            # 显示作业链架构信息
            st.subheader("作业链架构状态")

            # 获取作业链统计
            job_stats = api_request(
                endpoint="/job-chains",
                method="GET"
            )

            if job_stats:
                job_statistics = job_stats.get("job_statistics", {})
                active_chains = job_stats.get("active_chains", [])

                metric_cols = st.columns(4)
                with metric_cols[0]:
                    st.metric("活跃作业链", len(active_chains))
                with metric_cols[1]:
                    st.metric("处理中任务", job_statistics.get("processing", 0))
                with metric_cols[2]:
                    st.metric("已完成任务", job_statistics.get("completed", 0))
                with metric_cols[3]:
                    st.metric("失败任务", job_statistics.get("failed", 0))

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
            st.progress(min(int(cpu_usage), 100) / 100, text=f"CPU使用率: {cpu_usage:.1f}%")

            # 显示内存使用情况
            st.markdown("### 内存使用情况")
            memory_usage = health_info.get("system", {}).get("memory_usage", 0)
            st.progress(min(int(memory_usage), 100) / 100, text=f"内存使用率: {memory_usage:.1f}%")

            # 显示专用Worker GPU 使用情况
            gpu_health = health_info.get("gpu_health", {})
            if gpu_health:
                st.markdown("### 专用Worker GPU使用情况")

                for gpu_id, gpu_info in gpu_health.items():
                    device_name = gpu_info.get("device_name", gpu_id)
                    is_healthy = gpu_info.get("is_healthy", False)
                    total_memory_gb = gpu_info.get("total_memory_gb", 0)
                    allocated_memory_gb = gpu_info.get("allocated_memory_gb", 0)
                    free_memory_gb = gpu_info.get("free_memory_gb", 0)
                    free_percentage = gpu_info.get("free_percentage", 0)

                    with st.expander(f"{device_name} - {'健康' if is_healthy else '异常'}", expanded=True):
                        # 内存使用栏
                        memory_usage = 100 - free_percentage
                        st.progress(min(int(memory_usage), 100) / 100, text=f"显存使用率: {memory_usage:.1f}%")

                        # 专用Worker分配显示
                        st.markdown("**专用Worker分配:**")
                        alloc_cols = st.columns(4)
                        with alloc_cols[0]:
                            st.metric("Whisper", "2GB", "语音转录")
                        with alloc_cols[1]:
                            st.metric("嵌入", "3GB", "向量计算")
                        with alloc_cols[2]:
                            st.metric("推理", "6GB", "LLM生成")
                        with alloc_cols[3]:
                            st.metric("可用", f"{free_memory_gb:.1f}GB", "剩余空间")

                        # 如果不健康则显示消息
                        if not is_healthy:
                            st.error(f"GPU状态异常: {gpu_info.get('health_message', '')}")

                        # 添加GPU缓存清理按钮
                        if st.button(f"清理{device_name}显存缓存", key=f"cleanup_{gpu_id}"):
                            cleanup_response = api_request(
                                endpoint="/system/clear-gpu-cache",
                                method="POST",
                                data={"gpu_id": gpu_id}
                            )
                            if cleanup_response:
                                st.success("已发送显存清理指令")
                            else:
                                st.error("清理指令发送失败")
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

                    total = usage.get("total", 0)
                    used = usage.get("used", 0)
                    free = usage.get("free", 0)
                    percent = usage.get("percent", 0)

                    st.progress(min(int(percent), 100) / 100, text=f"使用率: {percent:.1f}%")

                    disk_cols = st.columns(3)
                    with disk_cols[0]:
                        st.metric("总容量", format_bytes(total))
                    with disk_cols[1]:
                        st.metric("已使用", format_bytes(used))
                    with disk_cols[2]:
                        st.metric("可用", format_bytes(free))

                # 显示数据目录
                st.markdown("### 数据目录使用情况")
                for dir_name, dir_info in disk_info.get("data_dirs", {}).items():
                    st.markdown(f"**{dir_name}:** {format_bytes(dir_info.get('size', 0))}")
            else:
                st.warning("无法获取磁盘使用信息")

            # 自动刷新选项
            auto_refresh = st.checkbox("自动刷新 (每30秒)", value=False)
            if auto_refresh:
                time.sleep(30)
                st.rerun()

        with tab3:
            # 使用新的作业链队列管理组件
            render_job_chain_queue_management()

        with tab4:
            st.subheader("系统配置")

            # 获取当前系统配置
            config_info = api_request(
                endpoint="/system/config",
                method="GET"
            )

            if config_info:
                # 专门针对自触发架构的配置分类
                categories = {
                    "自触发架构设置": ["job_chain_enabled", "self_triggering_mode"],
                    "专用Worker设置": ["gpu_whisper_memory", "gpu_embedding_memory", "gpu_inference_memory"],
                    "基本设置": ["host", "port", "api_auth_enabled"],
                    "模型设置": ["default_embedding_model", "default_colbert_model",
                                 "default_llm_model", "default_whisper_model"],
                    "GPU设置": ["device", "use_fp16", "batch_size"],
                    "检索设置": ["retriever_top_k", "reranker_top_k"],
                    "分块设置": ["chunk_size", "chunk_overlap"]
                }

                # 按类别显示配置
                for category, settings in categories.items():
                    with st.expander(category, expanded=False):
                        for setting in settings:
                            if setting in config_info:
                                value = config_info[setting]

                                if isinstance(value, bool):
                                    new_value = st.checkbox(setting, value)
                                elif isinstance(value, int):
                                    new_value = st.number_input(setting, value=value)
                                elif isinstance(value, float):
                                    new_value = st.number_input(setting, value=value, format="%.2f")
                                else:
                                    new_value = st.text_input(setting, str(value))

                                if new_value != value:
                                    config_info[setting] = new_value

                # 保存配置更改
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

            # 自触发架构维护工具
            st.subheader("自触发架构维护")

            maintenance_option = st.selectbox(
                "选择维护操作",
                [
                    "清理已完成作业链",
                    "重置作业链系统",
                    "优化向量数据库",
                    "重启所有专用Workers",
                    "清理GPU内存缓存",
                    "检查作业链完整性"
                ]
            )

            if st.button("执行维护操作", key="execute_maintenance"):
                endpoint_mapping = {
                    "清理已完成作业链": "/system/cleanup-completed-chains",
                    "重置作业链系统": "/system/reset-job-chains",
                    "优化向量数据库": "/system/optimize-vectorstore",
                    "重启所有专用Workers": "/system/restart-workers",
                    "清理GPU内存缓存": "/system/clear-gpu-cache",
                    "检查作业链完整性": "/system/verify-job-chains"
                }

                endpoint = endpoint_mapping.get(maintenance_option)
                if endpoint:
                    maintenance_response = api_request(
                        endpoint=endpoint,
                        method="POST"
                    )

                    if maintenance_response:
                        st.success(f"维护操作执行成功: {maintenance_option}")
                    else:
                        st.error("维护操作执行失败")

            # 专用Worker控制
            st.subheader("专用Worker控制")

            worker_types = {
                "gpu-whisper": "🎵 语音转录Worker",
                "gpu-embedding": "🔢 向量嵌入Worker",
                "gpu-inference": "🧠 LLM推理Worker",
                "cpu": "💻 CPU处理Worker"
            }

            selected_worker_type = st.selectbox("选择Worker类型", list(worker_types.keys()),
                                               format_func=lambda x: worker_types[x])

            worker_actions = st.selectbox("Worker操作", ["重启", "查看日志", "检查状态"])

            if st.button("执行Worker操作", key="execute_worker_action"):
                if worker_actions == "重启":
                    restart_response = api_request(
                        endpoint="/system/restart-workers",
                        method="POST",
                        data={"worker_type": selected_worker_type}
                    )
                    if restart_response:
                        st.success(f"已发送重启信号到 {worker_types[selected_worker_type]}")
                    else:
                        st.error("重启信号发送失败")

                elif worker_actions == "查看日志":
                    logs_response = api_request(
                        endpoint=f"/system/logs/worker",
                        method="GET"
                    )
                    if logs_response and "content" in logs_response:
                        st.text_area("Worker日志", logs_response["content"], height=300)
                    else:
                        st.warning("无法获取Worker日志")

                elif worker_actions == "检查状态":
                    # 显示特定Worker类型的详细状态
                    if health_info:
                        workers = health_info.get("workers", {})
                        matching_workers = [w for w in workers.keys() if selected_worker_type in w]

                        if matching_workers:
                            st.subheader(f"{worker_types[selected_worker_type]} 详细状态")
                            for worker_id in matching_workers:
                                worker_info = workers[worker_id]
                                st.json({worker_id: worker_info})
                        else:
                            st.warning(f"未找到 {worker_types[selected_worker_type]} 实例")

        # 添加页脚
        st.markdown("---")
        st.caption("自触发作业链系统管理控制台 - 专用Worker架构")
        st.caption(f"当前时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    else:
        st.error("无法连接到API服务。请确保自触发作业链系统正在运行。")
        st.info("您可以使用以下命令启动系统：")
        st.code("""
# 启动基础服务
docker-compose up -d redis qdrant api

# 启动专用Workers
docker-compose up -d worker-gpu-whisper worker-gpu-embedding worker-gpu-inference worker-cpu
        """)


# 调用函数渲染系统仪表板
render_system_dashboard()