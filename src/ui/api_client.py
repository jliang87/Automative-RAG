"""
Simplified API client - src/ui/api_client.py
Clean, focused functions for the two-page architecture
"""

import streamlit as st
import httpx
import time
from typing import Dict, Optional, Any, List

def api_request(
    endpoint: str,
    method: str = "GET",
    data: Optional[Dict] = None,
    files: Optional[Dict] = None,
    params: Optional[Dict] = None,
    timeout: float = 10.0,
    silent: bool = False,
) -> Optional[Dict]:
    """Core API request function"""

    if "api_url" not in st.session_state or "api_key" not in st.session_state:
        if not silent:
            st.error("系统配置错误")
        return None

    headers = {"x-token": st.session_state.api_key}
    url = f"{st.session_state.api_url}{endpoint}"

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

            if response.status_code >= 400:
                if not silent:
                    if response.status_code == 503:
                        st.error("系统暂时不可用，请稍后再试")
                    else:
                        st.error("请求失败，请重试")
                return None

            return response.json()

    except httpx.TimeoutException:
        if not silent:
            st.error("请求超时，请重试")
        return None
    except httpx.ConnectError:
        if not silent:
            st.error("无法连接到服务器")
        return None
    except Exception as e:
        if not silent:
            st.error("网络错误，请检查连接")
        return None

    return None


# === SYSTEM HEALTH FUNCTIONS (for 系统信息.py) ===

def get_system_health() -> Dict[str, Any]:
    """Get system health data for 系统信息.py"""
    try:
        health = api_request("/system/health/detailed", silent=True, timeout=5.0)
        if health:
            return {
                "status": health.get("status", "unknown"),
                "workers": health.get("workers", {}),
                "gpu_health": health.get("gpu_health", {}),
                "system_info": health.get("system", {})
            }
    except:
        pass

    return {
        "status": "unknown",
        "workers": {},
        "gpu_health": {},
        "system_info": {}
    }


def get_worker_summary() -> Dict[str, Any]:
    """Get worker status summary for 系统信息.py"""
    health_data = get_system_health()
    workers = health_data.get("workers", {})

    # Count workers by type
    worker_counts = {
        "gpu-inference": 0,
        "gpu-embedding": 0,
        "gpu-whisper": 0,
        "cpu": 0
    }

    total_healthy = 0
    for worker_id, info in workers.items():
        if info.get("status") == "healthy":
            total_healthy += 1
            for worker_type in worker_counts.keys():
                if worker_type in worker_id:
                    worker_counts[worker_type] += 1
                    break

    return {
        "total_workers": len(workers),
        "healthy_workers": total_healthy,
        "worker_counts": worker_counts,
        "all_services_available": total_healthy > 0 and all(count > 0 for count in worker_counts.values())
    }


def get_gpu_status() -> Dict[str, Any]:
    """Get GPU status for 系统信息.py"""
    health_data = get_system_health()
    gpu_health = health_data.get("gpu_health", {})

    if not gpu_health:
        return {"available": False}

    total_memory = 0
    used_memory = 0
    gpu_details = []

    for gpu_id, gpu_info in gpu_health.items():
        if isinstance(gpu_info, dict):
            total_gb = gpu_info.get("total_memory_gb", 0)
            allocated_gb = gpu_info.get("allocated_memory_gb", 0)

            if isinstance(total_gb, (int, float)) and isinstance(allocated_gb, (int, float)):
                total_memory += total_gb
                used_memory += allocated_gb

                gpu_details.append({
                    "id": gpu_id,
                    "name": gpu_info.get("device_name", f"GPU {gpu_id}"),
                    "total_gb": total_gb,
                    "used_gb": allocated_gb,
                    "usage_percent": (allocated_gb / total_gb * 100) if total_gb > 0 else 0
                })

    overall_usage = (used_memory / total_memory * 100) if total_memory > 0 else 0

    return {
        "available": len(gpu_details) > 0,
        "total_memory_gb": total_memory,
        "used_memory_gb": used_memory,
        "usage_percent": overall_usage,
        "gpu_details": gpu_details
    }


def get_system_status_summary() -> Dict[str, str]:
    """Get simple system status for display"""
    worker_summary = get_worker_summary()

    if worker_summary["all_services_available"]:
        return {
            "status": "healthy",
            "message": "✅ 系统运行正常",
            "color": "success"
        }
    elif worker_summary["healthy_workers"] > 0:
        return {
            "status": "partial",
            "message": "⚠️ 部分服务可用",
            "color": "warning"
        }
    else:
        return {
            "status": "down",
            "message": "❌ 系统暂时不可用",
            "color": "error"
        }


# === JOB MANAGEMENT FUNCTIONS (for 后台任务.py) ===

def get_jobs_list(limit: int = 20) -> List[Dict[str, Any]]:
    """Get jobs list for 后台任务.py"""
    try:
        return api_request("/ingest/jobs", method="GET", params={"limit": limit}, silent=True) or []
    except:
        return []


def get_job_details(job_id: str) -> Optional[Dict[str, Any]]:
    """Get detailed job information for 后台任务.py"""
    try:
        return api_request(f"/ingest/jobs/{job_id}", method="GET", silent=True)
    except:
        return None


def get_job_statistics() -> Dict[str, int]:
    """Get job statistics for 后台任务.py"""
    try:
        overview = api_request("/job-chains", method="GET", silent=True)
        if overview:
            return overview.get("job_statistics", {})
    except:
        pass

    return {"completed": 0, "processing": 0, "pending": 0, "failed": 0}


# === SIMPLE UTILITIES ===

def simple_health_check() -> bool:
    """Basic health check - returns True if system is responsive"""
    try:
        response = api_request("/health", method="GET", silent=True, timeout=3.0)
        return response is not None
    except:
        return False