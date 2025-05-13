"""
UI 组件的集中式错误处理。

此模块提供统一的错误处理函数和组件，以确保 UI 中的错误管理一致性。
"""

import streamlit as st
import time
from typing import Dict, Any, List, Optional, Callable, Tuple, Union

# 导入统一的 API 客户端
from src.ui.api_client import api_request, check_worker_availability


def robust_api_status_indicator(show_detail: bool = False) -> bool:
    """
    显示 API 连接状态，具有针对 worker 故障的强大错误处理功能。

    参数:
        show_detail: 是否显示详细的错误信息

    返回:
        如果 API 完全可用则为 True，否则为 False
    """
    try:
        with st.spinner("检查 API 连接..."):
            # 首先尝试基本的健康检查
            basic_health = api_request(
                endpoint="/health",
                method="GET",
                timeout=2.0,  # API 检查的短超时
                silent=True  # 不显示错误消息
            )

            if not basic_health:
                st.sidebar.error("❌ API 服务不可用")
                if show_detail:
                    st.sidebar.info("请确保 API 服务正在运行。您可以使用以下命令启动：")
                    st.sidebar.code("docker-compose up -d api")
                return False

            # 如果基本健康检查通过，尝试获取 worker 状态
            worker_status = api_request(
                endpoint="/query/llm-info",
                method="GET",
                timeout=2.0,
                silent=True
            )

            # 检查是否有活动的 workers
            active_workers = worker_status.get("active_workers", {}) if worker_status else {}

            if not worker_status or not active_workers:
                st.sidebar.warning("⚠️ API 可用，但没有活动 Worker")

                # 提供有关启动 worker 的有用信息
                if show_detail:
                    st.sidebar.info("Worker 服务未运行。您可以使用以下命令启动所需的 Worker：")
                    st.sidebar.code(
                        "docker-compose up -d worker-gpu-inference worker-gpu-embedding worker-gpu-whisper worker-cpu system-worker")

                return False

            # 一切正常
            st.sidebar.success("✅ API 连接正常")
            return True

    except Exception as e:
        st.sidebar.error(f"❌ 连接错误: {str(e)}")
        return False


def handle_worker_dependency(operation_type: str) -> bool:
    """
    处理不同操作的 worker 依赖关系，并显示适当的消息。

    参数:
        operation_type: 操作类型 (query, video, pdf, text, transcription)

    返回:
        如果所需的 worker 可用则为 True，否则为 False
    """
    # 将操作映射到所需的 worker 类型
    operation_workers = {
        "query": ["gpu-inference"],
        "video": ["gpu-whisper", "gpu-embedding"],
        "pdf": ["cpu", "gpu-embedding"],
        "text": ["cpu", "gpu-embedding"],
        "transcription": ["gpu-whisper"]
    }

    # 获取所需的 workers
    required_workers = operation_workers.get(operation_type, [])

    # 检查每个所需的 worker
    for worker_type in required_workers:
        if not check_worker_availability(worker_type):
            graceful_worker_failure(worker_type, operation_type)
            return False

    return True


def graceful_worker_failure(worker_type: str, operation: str) -> None:
    """
    当需要特定的 worker 类型但不可用时，显示用户友好的错误。

    参数:
        worker_type: 不可用的 worker 类型 (gpu-inference, gpu-embedding 等)
        operation: 需要此 worker 的操作
    """
    worker_names = {
        "gpu-inference": "推理 (LLM) Worker",
        "gpu-embedding": "向量嵌入 Worker",
        "gpu-whisper": "语音转录 Worker",
        "cpu": "CPU 处理 Worker",
        "system": "系统 Worker"
    }

    operation_names = {
        "query": "查询处理",
        "video": "视频处理",
        "pdf": "PDF处理",
        "text": "文本处理",
        "transcription": "语音转录"
    }

    # 创建用户友好的错误消息
    worker_name = worker_names.get(worker_type, worker_type)
    operation_name = operation_names.get(operation, operation)

    st.error(f"⚠️ {operation_name}需要{worker_name}，但该服务当前不可用")

    # 向用户显示选项
    st.info("您可以：")
    st.markdown("""
    1. 等待系统管理员启动所需的服务
    2. 如果您有权限，使用以下命令启动所需的服务：
    """)

    # 显示启动特定 worker 的命令
    st.code(f"docker-compose up -d {worker_type}")

    # 添加重新检查按钮
    if st.button(f"重新检查{worker_name}状态"):
        st.rerun()


def handle_error(error_message: str, show_retry: bool = True) -> None:
    """
    显示用户友好的错误消息，可选择性地显示重试按钮。

    参数:
        error_message: 要显示的错误消息
        show_retry: 是否显示重试按钮
    """
    st.error(error_message)

    if show_retry:
        if st.button("重试"):
            st.rerun()


def show_api_unavailable_message() -> None:
    """
    当 API 不可用时显示标准消息。
    """
    st.error("无法连接到API服务或所需的Worker未运行。")
    st.info("请确保API服务和相关Worker正在运行，然后刷新页面。")

    if st.button("刷新"):
        st.rerun()


def validation_error(message: str) -> None:
    """
    显示验证错误消息。

    参数:
        message: 验证错误消息
    """
    st.warning(message)