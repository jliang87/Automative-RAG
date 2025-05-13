"""
Enhanced error handling component for the UI
"""

import streamlit as st
import time
from typing import Dict, Any, List, Optional, Callable


def robust_api_status_indicator(show_detail: bool = False):
    """
    Display API connection status with robust error handling for when workers are down.

    Args:
        show_detail: Whether to show detailed error information
    """
    try:
        with st.spinner("Checking API connection..."):
            # First try a basic health check
            basic_health = api_request(
                endpoint="/health",
                method="GET",
                timeout=2.0,  # Short timeout for API check
                silent=True  # Don't show error messages
            )

            if not basic_health:
                st.sidebar.error("❌ API 服务不可用")
                if show_detail:
                    st.sidebar.info("请确保API服务正在运行。您可以使用以下命令启动：")
                    st.sidebar.code("docker-compose up -d api")
                return False

            # If basic health check passes, try to get worker status
            worker_status = api_request(
                endpoint="/query/llm-info",
                method="GET",
                timeout=2.0,
                silent=True
            )

            # Check if there are active workers
            active_workers = worker_status.get("active_workers", {}) if worker_status else {}

            if not worker_status or not active_workers:
                st.sidebar.warning("⚠️ API 可用，但没有活动 Worker")

                # Provide helpful information about starting workers
                if show_detail:
                    st.sidebar.info("Worker 服务未运行。您可以使用以下命令启动所需的 Worker：")
                    st.sidebar.code(
                        "docker-compose up -d worker-gpu-inference worker-gpu-embedding worker-gpu-whisper worker-cpu system-worker")

                return False

            # All is good
            st.sidebar.success("✅ API 连接正常")
            return True

    except Exception as e:
        st.sidebar.error(f"❌ 连接错误: {str(e)}")
        return False


def graceful_worker_failure(worker_type: str, operation: str):
    """
    Display a user-friendly error when a specific worker type is needed but unavailable.

    Args:
        worker_type: Type of worker that's unavailable (gpu-inference, gpu-embedding, etc.)
        operation: Operation that requires this worker
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

    # Create user-friendly error message
    worker_name = worker_names.get(worker_type, worker_type)
    operation_name = operation_names.get(operation, operation)

    st.error(f"⚠️ {operation_name}需要{worker_name}，但该服务当前不可用")

    # Show options for the user
    st.info("您可以：")
    st.markdown("""
    1. 等待系统管理员启动所需的服务
    2. 如果您有权限，使用以下命令启动所需的服务：
    """)

    # Show command to start the specific worker
    st.code(f"docker-compose up -d {worker_type}")

    # Add button to check again
    if st.button(f"重新检查{worker_name}状态"):
        st.rerun()


def api_request(
        endpoint: str,
        method: str = "GET",
        data: Optional[Dict] = None,
        files: Optional[Dict] = None,
        params: Optional[Dict] = None,
        timeout: float = 10.0,
        retries: int = 1,
        silent: bool = False,
        handle_error: Optional[Callable] = None,
) -> Optional[Dict]:
    """
    Send API request with enhanced error handling and retry logic.

    Args:
        endpoint: API endpoint path
        method: HTTP method (GET, POST, DELETE)
        data: Data to send in request body
        files: Files to upload
        params: Query parameters
        timeout: Request timeout in seconds
        retries: Number of retries on failure
        silent: Whether to suppress error messages
        handle_error: Custom error handler function

    Returns:
        Response data or None on failure
    """
    import httpx

    headers = {"x-token": st.session_state.api_key}
    url = f"{st.session_state.api_url}{endpoint}"

    for attempt in range(retries + 1):
        try:
            with httpx.Client(timeout=timeout) as client:
                if method == "GET":
                    response = client.get(url, headers=headers, params=params)
                elif method == "POST":
                    if files:
                        response = client.post(url, headers=headers, data=data, files=files)
                    else:
                        response = client.post(url, headers=headers, json=data)
                elif method == "DELETE":
                    response = client.delete(url, headers=headers)
                else:
                    if not silent:
                        st.error(f"不支持的请求方法: {method}")
                    return None

                # Handle HTTP errors (4xx, 5xx)
                if response.status_code >= 400:
                    error_msg = f"API 错误 ({response.status_code}): {response.text}"

                    # Check if we should retry
                    should_retry = attempt < retries and (
                        # Retry server errors (5xx)
                            response.status_code >= 500 or
                            # Retry rate limit errors
                            response.status_code == 429
                    )

                    if should_retry:
                        # Exponential backoff
                        wait_time = (2 ** attempt) * 0.5
                        time.sleep(wait_time)
                        continue

                    # Handle the error
                    if handle_error:
                        return handle_error(error_msg)
                    elif not silent:
                        st.error(error_msg)
                    return None

                # Success
                return response.json()

        except httpx.TimeoutException:
            error_msg = f"请求超时 ({timeout}秒)"

            # Check if we should retry
            if attempt < retries:
                wait_time = (2 ** attempt) * 0.5
                time.sleep(wait_time)
                continue

            # Handle the error
            if handle_error:
                return handle_error(error_msg)
            elif not silent:
                st.error(error_msg)
            return None

        except httpx.ConnectError:
            error_msg = "无法连接到API服务器"

            # Check if we should retry
            if attempt < retries:
                wait_time = (2 ** attempt) * 0.5
                time.sleep(wait_time)
                continue

            # Handle the error
            if handle_error:
                return handle_error(error_msg)
            elif not silent:
                st.error(error_msg)
            return None

        except Exception as e:
            error_msg = f"连接错误: {str(e)}"

            # Check if we should retry for certain errors
            if attempt < retries:
                wait_time = (2 ** attempt) * 0.5
                time.sleep(wait_time)
                continue

            # Handle the error
            if handle_error:
                return handle_error(error_msg)
            elif not silent:
                st.error(error_msg)
            return None

    # If we get here, all retries failed
    return None


def check_worker_availability(worker_type: str) -> bool:
    """
    Check if a specific worker type is available.

    Args:
        worker_type: Type of worker to check (gpu-inference, gpu-embedding, etc.)

    Returns:
        True if worker is available, False otherwise
    """
    # Try to get detailed health info
    health_info = api_request(
        endpoint="/system/health/detailed",
        method="GET",
        silent=True,
        timeout=2.0
    )

    if not health_info:
        return False

    # Check workers
    workers = health_info.get("workers", {})
    for worker_id, info in workers.items():
        if worker_type in worker_id and info.get("status") == "healthy":
            return True

    return False


def handle_worker_dependency(operation_type: str) -> bool:
    """
    Handle worker dependencies for different operations and show appropriate messages.

    Args:
        operation_type: Type of operation (query, video, pdf, text, transcription)

    Returns:
        True if required workers are available, False otherwise
    """
    # Map operations to required worker types
    operation_workers = {
        "query": ["gpu-inference"],
        "video": ["gpu-whisper", "gpu-embedding"],
        "pdf": ["cpu", "gpu-embedding"],
        "text": ["cpu", "gpu-embedding"],
        "transcription": ["gpu-whisper"]
    }

    # Get required workers
    required_workers = operation_workers.get(operation_type, [])

    # Check each required worker
    for worker_type in required_workers:
        if not check_worker_availability(worker_type):
            graceful_worker_failure(worker_type, operation_type)
            return False

    return True