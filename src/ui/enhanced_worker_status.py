import streamlit as st
import time
import pandas as pd
from typing import Dict, List, Optional, Any

# 导入统一的 API 客户端
from src.ui.api_client import api_request

def enhanced_worker_status():
    """
    在侧边栏显示增强的 worker 状态和健康信息。
    """
    try:
        # 从 API 获取详细的健康信息
        response = api_request(
            endpoint="/system/health/detailed",
            method="GET",
            timeout=3.0  # 较短的超时以避免 UI 阻塞
        )

        if not response:
            st.sidebar.warning("⚠️ 无法获取系统健康信息")
            return

        # 显示整体系统状态
        status = response.get("status", "unknown")
        if status == "healthy":
            st.sidebar.success("✅ 系统正常")
        else:
            st.sidebar.warning("⚠️ 系统状态: " + status)

        # 显示活动 workers
        workers = response.get("workers", {})
        if workers:
            with st.sidebar.expander("Worker 状态", expanded=True):
                worker_data = []

                # 按类型分组 workers
                worker_types = {}
                for worker_id, info in workers.items():
                    worker_type = info.get("type", "unknown")
                    if worker_type not in worker_types:
                        worker_types[worker_type] = []
                    worker_types[worker_type].append((worker_id, info))

                # 按类型显示带状态指示器的 workers
                for worker_type, workers_of_type in worker_types.items():
                    healthy_count = sum(1 for _, info in workers_of_type if info.get("status") == "healthy")
                    total_count = len(workers_of_type)

                    # 创建一个漂亮的显示名称
                    display_name = {
                        "gpu-inference": "LLM 和重排序",
                        "gpu-embedding": "向量嵌入",
                        "gpu-whisper": "语音转录",
                        "cpu": "文本处理",
                        "system": "系统管理"
                    }.get(worker_type, worker_type)

                    # 显示带颜色的状态
                    if healthy_count == total_count:
                        st.success(f"✅ {display_name}: {healthy_count}/{total_count} 个 worker 正常")
                    elif healthy_count > 0:
                        st.warning(f"⚠️ {display_name}: {healthy_count}/{total_count} 个 worker 正常")
                    else:
                        st.error(f"❌ {display_name}: 0/{total_count} 个 worker 正常")

                    # 为离线 workers 添加重启按钮
                    if healthy_count < total_count and worker_type != "system":
                        if st.button(f"重启 {display_name} workers", key=f"restart_{worker_type}"):
                            restart_response = api_request(
                                endpoint=f"/system/restart-worker/{worker_type}",
                                method="POST"
                            )
                            if restart_response:
                                st.success(f"已发送重启信号到 {display_name} workers")

                # 如果有，显示队列信息
                queue_stats = response.get("queues", {})
                if queue_stats:
                    st.subheader("任务队列")
                    queue_data = []
                    for queue, count in queue_stats.items():
                        display_name = {
                            "inference_tasks": "LLM 生成",
                            "embedding_tasks": "嵌入",
                            "transcription_tasks": "转录",
                            "cpu_tasks": "文本处理",
                            "system_tasks": "系统任务"
                        }.get(queue, queue)

                        queue_data.append({
                            "队列": display_name,
                            "等待任务": count
                        })

                    # 创建 DataFrame 并显示
                    if queue_data:
                        queue_df = pd.DataFrame(queue_data)
                        st.dataframe(queue_df, hide_index=True)
        else:
            st.sidebar.warning("⚠️ 未检测到活动 workers")
            st.sidebar.info("使用 docker-compose 启动所需的 worker 服务")

        # 如果可用，显示 GPU 状态
        gpu_health = response.get("gpu_health", {})
        if gpu_health:
            with st.sidebar.expander("GPU 状态", expanded=False):
                for gpu_id, gpu_info in gpu_health.items():
                    # 显示名称和健康状况
                    if gpu_info.get("is_healthy", False):
                        st.success(f"✅ {gpu_info.get('device_name', gpu_id)}")
                    else:
                        st.error(
                            f"❌ {gpu_info.get('device_name', gpu_id)}: {gpu_info.get('health_message', '不健康')}")

                    # 内存使用情况
                    free_pct = gpu_info.get("free_percentage", 0)
                    memory_color = "normal"
                    if free_pct < 10:
                        memory_color = "off"
                    elif free_pct < 30:
                        memory_color = "warning"

                    # 创建一个显示 GPU 内存使用情况的进度条
                    st.progress(100 - free_pct, text=f"内存: {100 - free_pct:.1f}% 已使用")

                    # 显示内存详细信息
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("已使用", f"{gpu_info.get('allocated_memory_gb', 0):.1f} GB")
                    with col2:
                        st.metric("空闲", f"{gpu_info.get('free_memory_gb', 0):.1f} GB")

        # 显示模型加载状态
        model_status = response.get("model_status", {})
        if model_status:
            with st.sidebar.expander("模型状态", expanded=False):
                for model_type, status in model_status.items():
                    display_name = {
                        "embedding": "嵌入模型",
                        "llm": "语言模型",
                        "colbert": "重排序模型",
                        "whisper": "语音识别"
                    }.get(model_type, model_type)

                    if status.get("loaded", False):
                        st.success(f"✅ {display_name} 已加载")
                        if status.get("loading_time", 0) > 0:
                            st.caption(f"加载用时 {status.get('loading_time', 0):.1f}秒")
                    else:
                        st.warning(f"⚠️ {display_name} 未加载")

        # 简单的刷新按钮
        if st.sidebar.button("刷新状态", key="refresh_worker_status"):
            st.rerun()

    except Exception as e:
        st.sidebar.warning(f"⚠️ 检查 worker 状态时出错: {str(e)}")

        # 添加重试按钮
        if st.sidebar.button("重试", key="try_worker_status_again"):
            st.rerun()