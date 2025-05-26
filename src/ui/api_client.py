"""
Enhanced API client with specific support for the new job chain architecture.
This updates src/ui/api_client.py with job chain specific methods.
"""

import streamlit as st
import httpx
import time
from typing import Dict, List, Any, Optional, Union, Callable


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
    Enhanced API request function with improved error handling for job chains.

    Args:
        endpoint: API endpoint path
        method: HTTP method (GET, POST, DELETE)
        data: Request body data
        files: Files to upload
        params: Query parameters
        timeout: Request timeout in seconds
        retries: Number of retries on failure
        silent: Whether to suppress error messages
        handle_error: Custom error handling function

    Returns:
        Response data or None on failure
    """
    if "api_url" not in st.session_state or "api_key" not in st.session_state:
        if not silent:
            st.error("API 配置未初始化")
        return None

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

                    # Handle error
                    if handle_error:
                        return handle_error(error_msg)
                    elif not silent:
                        st.error(error_msg)
                    return None

                # Success
                return response.json()

        except httpx.TimeoutException:
            error_msg = f"请求超时 ({timeout} 秒)"

            # Check if we should retry
            if attempt < retries:
                wait_time = (2 ** attempt) * 0.5
                time.sleep(wait_time)
                continue

            # Handle error
            if handle_error:
                return handle_error(error_msg)
            elif not silent:
                st.error(error_msg)
            return None

        except httpx.ConnectError:
            error_msg = "无法连接到 API 服务器"

            # Check if we should retry
            if attempt < retries:
                wait_time = (2 ** attempt) * 0.5
                time.sleep(wait_time)
                continue

            # Handle error
            if handle_error:
                return handle_error(error_msg)
            elif not silent:
                st.error(error_msg)
            return None

        except Exception as e:
            error_msg = f"连接错误: {str(e)}"

            # Check if we should retry for some errors
            if attempt < retries:
                wait_time = (2 ** attempt) * 0.5
                time.sleep(wait_time)
                continue

            # Handle error
            if handle_error:
                return handle_error(error_msg)
            elif not silent:
                st.error(error_msg)
            return None

    # If we get here, all retries failed
    return None


def check_worker_availability(worker_type: str) -> bool:
    """
    Check if a specific type of worker is available in the new architecture.

    Args:
        worker_type: Worker type to check (gpu-inference, gpu-embedding, gpu-whisper, cpu)

    Returns:
        True if worker is available, False otherwise
    """
    # Get detailed health information
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


def check_job_chain_status(job_id: str) -> Optional[Dict[str, Any]]:
    """
    Check the status of a specific job chain.

    Args:
        job_id: Job chain identifier

    Returns:
        Job chain status data or None if not found
    """
    return api_request(
        endpoint=f"/job-chains/{job_id}",
        method="GET",
        silent=True
    )


def get_queue_status() -> Optional[Dict[str, Any]]:
    """
    Get the status of all job chain queues.

    Returns:
        Queue status data or None on error
    """
    return api_request(
        endpoint="/query/queue-status",
        method="GET",
        silent=True
    )


def get_system_overview() -> Optional[Dict[str, Any]]:
    """
    Get system overview including job chains and workers.

    Returns:
        System overview data or None on error
    """
    return api_request(
        endpoint="/job-chains",
        method="GET",
        silent=True
    )


def submit_job_chain(job_type: str, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Submit a new job to the job chain system.

    Args:
        job_type: Type of job (video, pdf, text, query)
        data: Job data

    Returns:
        Job submission response or None on error
    """
    endpoint_mapping = {
        "video": "/ingest/video",
        "pdf": "/ingest/pdf",
        "text": "/ingest/text",
        "query": "/query"
    }

    endpoint = endpoint_mapping.get(job_type)
    if not endpoint:
        st.error(f"不支持的作业类型: {job_type}")
        return None

    return api_request(
        endpoint=endpoint,
        method="POST",
        data=data
    )


def cancel_job_chain(job_id: str) -> bool:
    """
    Cancel a running job chain.

    Args:
        job_id: Job chain identifier

    Returns:
        True if cancellation was successful
    """
    response = api_request(
        endpoint=f"/ingest/jobs/{job_id}",
        method="DELETE"
    )

    return response is not None


def get_worker_performance_metrics() -> Optional[Dict[str, Any]]:
    """
    Get performance metrics for dedicated workers.

    Returns:
        Performance metrics or None on error
    """
    return api_request(
        endpoint="/system/worker-metrics",
        method="GET",
        silent=True
    )


def restart_worker(worker_type: str) -> bool:
    """
    Restart a specific type of worker.

    Args:
        worker_type: Type of worker to restart

    Returns:
        True if restart signal was sent successfully
    """
    response = api_request(
        endpoint="/system/restart-workers",
        method="POST",
        data={"worker_type": worker_type}
    )

    return response is not None


def get_gpu_allocation_status() -> Optional[Dict[str, Any]]:
    """
    Get GPU memory allocation status for dedicated workers.

    Returns:
        GPU allocation data or None on error
    """
    health_data = api_request(
        endpoint="/system/health/detailed",
        method="GET",
        silent=True
    )

    if not health_data:
        return None

    gpu_health = health_data.get("gpu_health", {})
    workers = health_data.get("workers", {})

    # Combine GPU and worker information
    allocation_data = {
        "gpu_health": gpu_health,
        "workers": workers,
        "allocation_map": {
            "gpu-whisper": {"memory_gb": 2, "queue": "transcription_tasks"},
            "gpu-embedding": {"memory_gb": 3, "queue": "embedding_tasks"},
            "gpu-inference": {"memory_gb": 6, "queue": "inference_tasks"},
            "cpu": {"memory_gb": 0, "queue": "cpu_tasks"}
        }
    }

    return allocation_data


def poll_job_completion(job_id: str, max_polls: int = 120, poll_interval: float = 1.0) -> Optional[Dict[str, Any]]:
    """
    Poll for job completion with built-in timeout.

    Args:
        job_id: Job identifier to poll
        max_polls: Maximum number of polls before timeout
        poll_interval: Time between polls in seconds

    Returns:
        Final job status or None on timeout
    """
    for i in range(max_polls):
        job_status = api_request(
            endpoint=f"/ingest/jobs/{job_id}",
            method="GET",
            silent=True
        )

        if not job_status:
            return None

        status = job_status.get("status", "")

        if status in ["completed", "failed", "timeout"]:
            return job_status

        time.sleep(poll_interval)

    # Timeout reached
    return None


def check_architecture_health() -> Dict[str, Any]:
    """
    Comprehensive health check for the new job chain architecture.

    Returns:
        Health check results
    """
    health_results = {
        "overall_healthy": True,
        "issues": [],
        "worker_status": {},
        "queue_status": {},
        "gpu_status": {}
    }

    # Check workers
    health_data = api_request(
        endpoint="/system/health/detailed",
        method="GET",
        silent=True
    )

    if health_data:
        workers = health_data.get("workers", {})

        # Required worker types for the new architecture
        required_workers = ["gpu-whisper", "gpu-embedding", "gpu-inference", "cpu"]

        for worker_type in required_workers:
            found_workers = [w for w in workers.keys() if worker_type in w]
            healthy_workers = [w for w in found_workers if workers[w].get("status") == "healthy"]

            health_results["worker_status"][worker_type] = {
                "found": len(found_workers),
                "healthy": len(healthy_workers),
                "is_available": len(healthy_workers) > 0
            }

            if len(healthy_workers) == 0:
                health_results["overall_healthy"] = False
                health_results["issues"].append(f"没有健康的{worker_type}工作器")

        # Check GPU health
        gpu_health = health_data.get("gpu_health", {})
        health_results["gpu_status"] = gpu_health

        for gpu_id, gpu_info in gpu_health.items():
            if not gpu_info.get("is_healthy", True):
                health_results["overall_healthy"] = False
                health_results["issues"].append(f"GPU {gpu_id} 不健康")
    else:
        health_results["overall_healthy"] = False
        health_results["issues"].append("无法获取系统健康数据")

    # Check queue status
    queue_data = get_queue_status()
    if queue_data:
        health_results["queue_status"] = queue_data.get("queue_status", {})
    else:
        health_results["issues"].append("无法获取队列状态")

    return health_results