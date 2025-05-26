"""
简化的错误处理组件，专为自触发作业链架构优化
"""

import streamlit as st
import time
from typing import Dict, List, Optional, Any

# 导入统一的 API 客户端
from src.ui.api_client import api_request

def robust_api_status_indicator(show_detail: bool = False) -> bool:
    """
    强化的API状态指示器，专为自触发架构优化

    Args:
        show_detail: 是否显示详细信息

    Returns:
        True if API is available and workers are healthy
    """
    try:
        # 基本API连接检查
        health_response = api_request(
            endpoint="/health",
            method="GET",
            timeout=3.0,
            silent=True
        )

        if not health_response:
            if show_detail:
                st.error("❌ API服务不可用")
            return False

        # 检查自触发架构状态
        detailed_health = api_request(
            endpoint="/system/health/detailed",
            method="GET",
            timeout=5.0,
            silent=True
        )

        if not detailed_health:
            if show_detail:
                st.warning("⚠️ 无法获取系统详细状态")
            return True  # API可用，但无详细信息

        # 检查专用Worker状态
        workers = detailed_health.get("workers", {})
        required_workers = ["gpu-whisper", "gpu-embedding", "gpu-inference", "cpu"]

        healthy_workers = {}
        for worker_type in required_workers:
            matching_workers = [w for w in workers.keys() if worker_type in w]
            healthy_count = sum(1 for w in matching_workers if workers[w].get("status") == "healthy")
            healthy_workers[worker_type] = {
                "healthy": healthy_count,
                "total": len(matching_workers),
                "available": healthy_count > 0
            }

        # 显示详细状态
        if show_detail:
            st.success("✅ API服务连接正常")

            # 显示专用Worker状态
            with st.expander("专用Worker状态", expanded=False):
                worker_names = {
                    "gpu-whisper": "🎵 语音转录Worker",
                    "gpu-embedding": "🔢 向量嵌入Worker",
                    "gpu-inference": "🧠 LLM推理Worker",
                    "cpu": "💻 CPU处理Worker"
                }

                for worker_type, status in healthy_workers.items():
                    display_name = worker_names.get(worker_type, worker_type)
                    if status["available"]:
                        st.success(f"✅ {display_name} ({status['healthy']}/{status['total']})")
                    else:
                        st.error(f"❌ {display_name} (不可用)")

            # 显示作业链状态
            job_chains = api_request(
                endpoint="/job-chains",
                method="GET",
                timeout=3.0,
                silent=True
            )

            if job_chains:
                active_chains = len(job_chains.get("active_chains", []))
                queue_status = job_chains.get("queue_status", {})
                busy_queues = sum(1 for q in queue_status.values() if q.get("status") == "busy")

                st.info(f"🔄 活跃作业链: {active_chains} | 忙碌队列: {busy_queues}")

        # 返回整体健康状态
        all_workers_available = all(status["available"] for status in healthy_workers.values())
        return all_workers_available

    except Exception as e:
        if show_detail:
            st.error(f"❌ 系统状态检查失败: {str(e)}")
        return False


def handle_worker_dependency(task_type: str) -> bool:
    """
    检查特定任务类型所需的Worker依赖

    Args:
        task_type: 任务类型 ("video", "pdf", "text", "query")

    Returns:
        True if required workers are available
    """
    # 任务类型到Worker的映射
    task_worker_mapping = {
        "video": ["cpu", "gpu-whisper", "gpu-embedding"],
        "pdf": ["cpu", "gpu-embedding"],
        "text": ["cpu", "gpu-embedding"],
        "query": ["gpu-embedding", "gpu-inference"]
    }

    required_workers = task_worker_mapping.get(task_type, [])

    if not required_workers:
        st.warning(f"⚠️ 未知任务类型: {task_type}")
        return False

    # 检查Worker可用性
    health_data = api_request(
        endpoint="/system/health/detailed",
        method="GET",
        timeout=5.0,
        silent=True
    )

    if not health_data:
        st.error("❌ 无法检查Worker状态")
        return False

    workers = health_data.get("workers", {})

    # 检查每个必需的Worker类型
    missing_workers = []
    for worker_type in required_workers:
        matching_workers = [w for w in workers.keys() if worker_type in w]
        healthy_count = sum(1 for w in matching_workers if workers[w].get("status") == "healthy")

        if healthy_count == 0:
            worker_names = {
                "cpu": "CPU处理Worker",
                "gpu-whisper": "语音转录Worker",
                "gpu-embedding": "向量嵌入Worker",
                "gpu-inference": "LLM推理Worker"
            }
            missing_workers.append(worker_names.get(worker_type, worker_type))

    if missing_workers:
        st.error(f"❌ 缺少必需的Worker: {', '.join(missing_workers)}")

        # 提供启动建议
        worker_commands = {
            "CPU处理Worker": "docker-compose up -d worker-cpu",
            "语音转录Worker": "docker-compose up -d worker-gpu-whisper",
            "向量嵌入Worker": "docker-compose up -d worker-gpu-embedding",
            "LLM推理Worker": "docker-compose up -d worker-gpu-inference"
        }

        st.info("请启动以下Worker服务:")
        for worker in missing_workers:
            if worker in worker_commands:
                st.code(worker_commands[worker])

        return False

    return True


def display_worker_allocation_chart():
    """
    显示专用Worker分配图表 (简化版)
    """
    st.subheader("专用Worker GPU分配")

    # 获取GPU状态
    health_data = api_request(
        endpoint="/system/health/detailed",
        method="GET",
        silent=True
    )

    if not health_data:
        st.warning("无法获取GPU分配数据")
        return

    gpu_health = health_data.get("gpu_health", {})

    # 显示分配策略
    allocation_data = [
        {"Worker类型": "🎵 Whisper转录", "GPU分配": "2GB", "队列": "transcription_tasks"},
        {"Worker类型": "🔢 向量嵌入", "GPU分配": "3GB", "队列": "embedding_tasks"},
        {"Worker类型": "🧠 LLM推理", "GPU分配": "6GB", "队列": "inference_tasks"},
        {"Worker类型": "💻 CPU处理", "GPU分配": "0GB", "队列": "cpu_tasks"}
    ]

    import pandas as pd
    df = pd.DataFrame(allocation_data)
    st.dataframe(df, hide_index=True, use_container_width=True)

    # 显示实际GPU使用情况
    if gpu_health:
        for gpu_id, gpu_info in gpu_health.items():
            device_name = gpu_info.get("device_name", gpu_id)
            total_memory = gpu_info.get("total_memory_gb", 0)
            allocated_memory = gpu_info.get("allocated_memory_gb", 0)

            if total_memory > 0:
                usage_pct = (allocated_memory / total_memory) * 100

                st.markdown(f"**{device_name}**")
                st.progress(
                    usage_pct / 100,
                    text=f"使用率: {usage_pct:.1f}% ({allocated_memory:.1f}GB/{total_memory:.1f}GB)"
                )

                # 显示分配详情
                st.caption("• Whisper: 2GB | 嵌入: 3GB | 推理: 6GB | 预留: 5GB")


def check_system_readiness(task_type: Optional[str] = None) -> Dict[str, Any]:
    """
    检查系统就绪状态

    Args:
        task_type: 可选的特定任务类型检查

    Returns:
        系统就绪状态字典
    """
    readiness = {
        "api_available": False,
        "workers_healthy": False,
        "gpu_available": False,
        "task_ready": False,
        "issues": []
    }

    try:
        # 检查API
        health_response = api_request("/health", "GET", timeout=3.0, silent=True)
        readiness["api_available"] = bool(health_response)

        if not readiness["api_available"]:
            readiness["issues"].append("API服务不可用")
            return readiness

        # 检查详细状态
        detailed_health = api_request("/system/health/detailed", "GET", timeout=5.0, silent=True)

        if detailed_health:
            # 检查Worker
            workers = detailed_health.get("workers", {})
            required_workers = ["gpu-whisper", "gpu-embedding", "gpu-inference", "cpu"]

            healthy_workers = []
            for worker_type in required_workers:
                matching = [w for w in workers.keys() if worker_type in w]
                healthy = sum(1 for w in matching if workers[w].get("status") == "healthy")
                if healthy > 0:
                    healthy_workers.append(worker_type)

            readiness["workers_healthy"] = len(healthy_workers) >= 3  # 至少3种Worker

            if len(healthy_workers) < 4:
                missing = set(required_workers) - set(healthy_workers)
                readiness["issues"].extend([f"缺少{w}Worker" for w in missing])

            # 检查GPU
            gpu_health = detailed_health.get("gpu_health", {})
            healthy_gpus = sum(1 for gpu in gpu_health.values() if gpu.get("is_healthy", False))
            readiness["gpu_available"] = healthy_gpus > 0

            if healthy_gpus == 0:
                readiness["issues"].append("无可用GPU")

        # 检查特定任务就绪性
        if task_type:
            readiness["task_ready"] = handle_worker_dependency(task_type)
        else:
            readiness["task_ready"] = readiness["workers_healthy"]

    except Exception as e:
        readiness["issues"].append(f"系统检查失败: {str(e)}")

    return readiness