"""
Enhanced worker status component for dedicated GPU workers architecture.
This replaces src/ui/enhanced_worker_status.py with better support for the new worker types.
"""

import streamlit as st
import time
import pandas as pd
from typing import Dict, List, Optional, Any

# Import unified API client
from src.ui.api_client import api_request

def enhanced_worker_status():
    """
    Display enhanced worker status and health information in the sidebar,
    optimized for the new dedicated GPU worker architecture.
    """
    try:
        # Get detailed health information from API
        response = api_request(
            endpoint="/system/health/detailed",
            method="GET",
            timeout=3.0
        )

        if not response:
            st.sidebar.warning("⚠️ 无法获取系统健康信息")
            return

        # Display overall system status
        status = response.get("status", "unknown")
        if status == "healthy":
            st.sidebar.success("✅ 系统正常")
        else:
            st.sidebar.warning("⚠️ 系统状态: " + status)

        # Display active workers with dedicated GPU worker focus
        workers = response.get("workers", {})
        if workers:
            with st.sidebar.expander("专用GPU Worker状态", expanded=True):
                # Group workers by type for the new architecture
                worker_types = {
                    "gpu-whisper": {"name": "语音转录", "color": "blue", "workers": []},
                    "gpu-embedding": {"name": "向量嵌入", "color": "green", "workers": []},
                    "gpu-inference": {"name": "LLM推理", "color": "purple", "workers": []},
                    "cpu": {"name": "CPU处理", "color": "orange", "workers": []},
                    "api": {"name": "API服务", "color": "gray", "workers": []}
                }

                # Categorize workers
                for worker_id, info in workers.items():
                    worker_type = info.get("type", "unknown")
                    if worker_type in worker_types:
                        worker_types[worker_type]["workers"].append((worker_id, info))

                # Display each worker type with GPU memory allocation info
                for worker_type, type_info in worker_types.items():
                    workers_of_type = type_info["workers"]
                    if not workers_of_type:
                        continue

                    healthy_count = sum(1 for _, info in workers_of_type if info.get("status") == "healthy")
                    total_count = len(workers_of_type)

                    display_name = type_info["name"]

                    # Show status with memory allocation info for GPU workers
                    if worker_type.startswith("gpu-"):
                        memory_allocation = {
                            "gpu-whisper": "2GB",
                            "gpu-embedding": "3GB",
                            "gpu-inference": "6GB"
                        }.get(worker_type, "未知")

                        if healthy_count == total_count:
                            st.success(f"✅ {display_name} ({memory_allocation}): {healthy_count}/{total_count} 正常")
                        elif healthy_count > 0:
                            st.warning(f"⚠️ {display_name} ({memory_allocation}): {healthy_count}/{total_count} 正常")
                        else:
                            st.error(f"❌ {display_name} ({memory_allocation}): 0/{total_count} 正常")
                    else:
                        # Non-GPU workers
                        if healthy_count == total_count:
                            st.success(f"✅ {display_name}: {healthy_count}/{total_count} 正常")
                        elif healthy_count > 0:
                            st.warning(f"⚠️ {display_name}: {healthy_count}/{total_count} 正常")
                        else:
                            st.error(f"❌ {display_name}: 0/{total_count} 正常")

                    # Add restart button for problematic workers
                    if healthy_count < total_count and worker_type != "api":
                        if st.button(f"重启{display_name}Workers", key=f"restart_{worker_type}"):
                            restart_response = api_request(
                                endpoint="/system/restart-workers",
                                method="POST",
                                data={"worker_type": worker_type}
                            )
                            if restart_response:
                                st.success(f"已发送重启信号到{display_name}workers")

        # Display job chain queue information
        queue_stats = response.get("job_chains", {})
        if queue_stats:
            with st.sidebar.expander("任务队列状态", expanded=False):
                # Get queue status from the job chain system
                queue_response = api_request(
                    endpoint="/query/queue-status",
                    method="GET",
                    timeout=2.0
                )

                if queue_response:
                    queue_data = []
                    queue_mapping = {
                        "transcription_tasks": "语音转录",
                        "embedding_tasks": "向量嵌入",
                        "inference_tasks": "LLM推理",
                        "cpu_tasks": "CPU处理"
                    }

                    for queue, display_name in queue_mapping.items():
                        # Check if queue is busy
                        queue_info = queue_response.get("queue_status", {}).get(queue, {})
                        status = queue_info.get("status", "free")
                        waiting_tasks = queue_info.get("waiting_tasks", 0)

                        if status == "busy":
                            current_job = queue_info.get("current_job", "unknown")
                            st.info(f"🔄 {display_name}: 处理中 (作业: {current_job[:8]}...)")
                        elif waiting_tasks > 0:
                            st.warning(f"⏳ {display_name}: {waiting_tasks}个任务等待")
                        else:
                            st.success(f"✅ {display_name}: 空闲")

        # Display GPU health if available
        gpu_health = response.get("gpu_health", {})
        if gpu_health:
            with st.sidebar.expander("GPU状态", expanded=False):
                for gpu_id, gpu_info in gpu_health.items():
                    device_name = gpu_info.get("device_name", gpu_id)
                    is_healthy = gpu_info.get("is_healthy", False)

                    if is_healthy:
                        st.success(f"✅ {device_name}")
                    else:
                        st.error(f"❌ {device_name}: {gpu_info.get('health_message', '不健康')}")

                    # Memory usage with dedicated worker allocation context
                    free_pct = gpu_info.get("free_percentage", 0)
                    allocated_gb = gpu_info.get("allocated_memory_gb", 0)
                    total_gb = gpu_info.get("total_memory_gb", 0)

                    # Show memory bar
                    st.progress(min(100 - free_pct, 100) / 100,
                              text=f"显存: {100 - free_pct:.1f}% ({allocated_gb:.1f}GB/{total_gb:.1f}GB)")

                    # Show which workers are using this GPU
                    st.caption("专用Worker分配:")
                    st.caption("• Whisper: 2GB • 嵌入: 3GB • 推理: 6GB")

        # Simple refresh button
        if st.sidebar.button("刷新状态", key="refresh_worker_status"):
            st.rerun()

    except Exception as e:
        st.sidebar.warning(f"⚠️ 检查worker状态时出错: {str(e)}")
        if st.sidebar.button("重试", key="try_worker_status_again"):
            st.rerun()


def display_worker_allocation_chart():
    """
    Display a visual chart showing GPU memory allocation across dedicated workers.
    """
    st.subheader("GPU内存分配策略")

    # Create allocation data
    allocation_data = [
        {"Worker类型": "Whisper转录", "分配内存(GB)": 2, "用途": "faster-whisper模型", "队列": "transcription_tasks"},
        {"Worker类型": "向量嵌入", "分配内存(GB)": 3, "用途": "BGE-M3嵌入模型", "队列": "embedding_tasks"},
        {"Worker类型": "LLM推理", "分配内存(GB)": 6, "用途": "DeepSeek-R1 + ColBERT", "队列": "inference_tasks"},
        {"Worker类型": "预留空间", "分配内存(GB)": 5, "用途": "系统开销和缓冲", "队列": "N/A"}
    ]

    df = pd.DataFrame(allocation_data)

    # Display as a table
    st.dataframe(df, hide_index=True, use_container_width=True)

    # Show total allocation
    total_allocated = sum(row["分配内存(GB)"] for row in allocation_data if row["Worker类型"] != "预留空间")
    st.metric("总GPU内存分配", f"{total_allocated}GB / 16GB", f"{(total_allocated/16)*100:.1f}%")

    # Benefits explanation
    with st.expander("专用Worker架构优势", expanded=False):
        st.markdown("""
        **消除模型颠簸(Model Thrashing):**
        - 每个GPU worker只加载和使用特定模型
        - 避免频繁的模型加载/卸载操作
        - 显著减少GPU内存碎片

        **并行任务处理:**
        - 同时进行视频转录、文档嵌入和查询推理
        - 每个任务类型有专门的处理队列
        - 提高整体系统吞吐量

        **资源优化:**
        - 精确的内存分配避免OOM错误
        - 更好的GPU利用率
        - 降低系统延迟
        """)