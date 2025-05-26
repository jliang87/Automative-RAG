"""
Enhanced job chain visualization component for the new self-triggering architecture.
This should be added as src/ui/job_chain_visualization.py
"""

import streamlit as st
import time
import json
from typing import Dict, List, Any, Optional
import pandas as pd

from src.ui.api_client import api_request


def display_job_chain_progress(job_id: str, job_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Display detailed job chain progress with self-triggering workflow visualization.

    Args:
        job_id: Job identifier
        job_data: Job data from the tracker

    Returns:
        Dictionary with user actions (retry, cancel, etc.)
    """
    # Get job chain status
    chain_status = api_request(
        endpoint=f"/job-chains/{job_id}",
        method="GET"
    )

    if not chain_status:
        st.warning("无法获取作业链状态")
        return {"action": None}

    # Extract job chain information
    job_chain_data = chain_status.get("job_chain", {})
    combined_view = chain_status.get("combined_view", {})

    # Display job chain header
    st.subheader(f"自触发作业链: {job_id}")

    # Job chain overview
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("作业状态", combined_view.get("status", "unknown"))

    with col2:
        progress = combined_view.get("progress", 0)
        st.metric("完成进度", f"{progress:.1f}%")

    with col3:
        current_task = combined_view.get("current_task", "无")
        st.metric("当前任务", current_task)

    with col4:
        total_steps = combined_view.get("total_steps", 0)
        current_step = job_chain_data.get("current_step", 0)
        st.metric("步骤进度", f"{current_step}/{total_steps}")

    # Workflow visualization
    workflow = job_chain_data.get("workflow", [])
    if workflow:
        st.subheader("作业链工作流")

        # Create workflow steps visualization
        workflow_data = []
        step_timings = combined_view.get("step_timings", {})
        current_step_idx = job_chain_data.get("current_step", 0)

        for i, (task_name, queue_name) in enumerate(workflow):
            # Determine step status
            if i < current_step_idx:
                status = "✅ 已完成"
                color = "green"
            elif i == current_step_idx:
                status = "🔄 进行中"
                color = "blue"
            else:
                status = "⏳ 待处理"
                color = "gray"

            # Get timing information
            timing_info = step_timings.get(task_name, {})
            duration = timing_info.get("duration", 0)

            if duration > 0:
                if duration < 60:
                    duration_str = f"{duration:.1f}秒"
                else:
                    duration_str = f"{duration / 60:.1f}分钟"
            else:
                duration_str = "-"

            # Map queue to worker type
            worker_mapping = {
                "cpu_tasks": "CPU处理器",
                "transcription_tasks": "GPU-Whisper",
                "embedding_tasks": "GPU-嵌入",
                "inference_tasks": "GPU-推理"
            }

            worker_type = worker_mapping.get(queue_name, queue_name)

            workflow_data.append({
                "步骤": f"{i + 1}. {task_name}",
                "Worker类型": worker_type,
                "队列": queue_name,
                "状态": status,
                "用时": duration_str
            })

        # Display workflow table
        workflow_df = pd.DataFrame(workflow_data)
        st.dataframe(workflow_df, hide_index=True, use_container_width=True)

        # Progress bar for overall completion
        if total_steps > 0:
            progress_value = min(current_step_idx / total_steps, 1.0)
            st.progress(progress_value, text=f"作业链进度: {progress_value * 100:.1f}%")

    # Self-triggering information
    with st.expander("自触发机制详情", expanded=False):
        st.markdown("""
        **自触发作业链特点:**
        - 无需轮询，任务完成自动触发下一步
        - 事件驱动架构，降低系统开销
        - 每个任务完成后立即传递到下一个Worker
        - 支持任务失败时的自动重试和恢复

        **Worker专用化:**
        - 每种任务类型分配到专用GPU Worker
        - 消除模型加载/卸载开销
        - 支持真正的并行处理
        """)

        # Show detailed step timings if available
        if step_timings:
            st.subheader("详细计时信息")
            timing_data = []

            for task_name, timing in step_timings.items():
                started_at = timing.get("started_at")
                completed_at = timing.get("completed_at")
                duration = timing.get("duration", 0)

                if started_at:
                    start_time = time.strftime("%H:%M:%S", time.localtime(started_at))
                else:
                    start_time = "-"

                if completed_at:
                    end_time = time.strftime("%H:%M:%S", time.localtime(completed_at))
                else:
                    end_time = "-"

                timing_data.append({
                    "任务": task_name,
                    "开始时间": start_time,
                    "结束时间": end_time,
                    "持续时间": f"{duration:.2f}秒" if duration > 0 else "-"
                })

            if timing_data:
                timing_df = pd.DataFrame(timing_data)
                st.dataframe(timing_df, hide_index=True, use_container_width=True)

    # Action buttons based on job status
    status = combined_view.get("status", "")

    if status == "failed":
        st.error("作业链执行失败")
        if st.button("重试作业链", key=f"retry_chain_{job_id}"):
            return {"action": "retry", "job_id": job_id}

    elif status == "processing":
        if st.button("取消作业链", key=f"cancel_chain_{job_id}"):
            return {"action": "cancel", "job_id": job_id}

    elif status == "completed":
        st.success("✅ 作业链执行完成")

        # Show final results
        final_result = job_data.get("result", {})
        if isinstance(final_result, dict):
            with st.expander("最终结果", expanded=True):
                # Format results based on job type
                job_type = job_data.get("job_type", "")

                if "document_count" in final_result:
                    st.metric("生成文档数", final_result["document_count"])

                if "embedding_completed_at" in final_result:
                    st.success("✅ 向量嵌入已完成")

                if "total_duration" in final_result:
                    duration = final_result["total_duration"]
                    if duration < 60:
                        duration_str = f"{duration:.1f}秒"
                    else:
                        duration_str = f"{duration / 60:.1f}分钟"
                    st.metric("总处理时间", duration_str)

    return {"action": None}


def display_queue_worker_mapping():
    """
    Display the mapping between queues and dedicated workers.
    """
    st.subheader("队列到Worker映射")

    mapping_data = [
        {
            "队列名称": "transcription_tasks",
            "专用Worker": "gpu-whisper",
            "模型": "Whisper Medium",
            "GPU分配": "2GB",
            "处理类型": "视频音频转录"
        },
        {
            "队列名称": "embedding_tasks",
            "专用Worker": "gpu-embedding",
            "模型": "BGE-M3",
            "GPU分配": "3GB",
            "处理类型": "文档向量嵌入"
        },
        {
            "队列名称": "inference_tasks",
            "专用Worker": "gpu-inference",
            "模型": "DeepSeek-R1 + ColBERT",
            "GPU分配": "6GB",
            "处理类型": "LLM推理和重排序"
        },
        {
            "队列名称": "cpu_tasks",
            "专用Worker": "cpu",
            "模型": "N/A",
            "GPU分配": "0GB",
            "处理类型": "PDF解析和文本处理"
        }
    ]

    mapping_df = pd.DataFrame(mapping_data)
    st.dataframe(mapping_df, hide_index=True, use_container_width=True)

    # Current queue status
    queue_status = api_request(
        endpoint="/query/queue-status",
        method="GET"
    )

    if queue_status:
        st.subheader("实时队列状态")

        current_status = []
        for queue_data in mapping_data:
            queue_name = queue_data["队列名称"]
            queue_info = queue_status.get("queue_status", {}).get(queue_name, {})

            status = queue_info.get("status", "free")
            waiting_tasks = queue_info.get("waiting_tasks", 0)

            if status == "busy":
                current_job = queue_info.get("current_job", "unknown")
                status_display = f"🔄 忙碌 (作业: {current_job[:8]}...)"
            elif waiting_tasks > 0:
                status_display = f"⏳ {waiting_tasks}个任务等待"
            else:
                status_display = "✅ 空闲"

            current_status.append({
                "队列": queue_name,
                "Worker类型": queue_data["专用Worker"],
                "当前状态": status_display
            })

        status_df = pd.DataFrame(current_status)
        st.dataframe(status_df, hide_index=True, use_container_width=True)


def display_job_chain_overview():
    """
    Display system-wide job chain overview.
    """
    # Get job chains overview
    overview = api_request(
        endpoint="/job-chains",
        method="GET"
    )

    if not overview:
        st.warning("无法获取作业链概览")
        return

    st.subheader("系统作业链概览")

    # Overall statistics
    col1, col2, col3 = st.columns(3)

    with col1:
        active_chains = overview.get("active_chains", [])
        st.metric("活跃作业链", len(active_chains))

    with col2:
        queue_status = overview.get("queue_status", {})
        busy_queues = sum(1 for q in queue_status.values() if q.get("status") == "busy")
        st.metric("忙碌队列", busy_queues)

    with col3:
        total_waiting = sum(q.get("waiting_tasks", 0) for q in queue_status.values())
        st.metric("等待任务总数", total_waiting)

    # Active job chains
    if active_chains:
        st.subheader("活跃作业链")

        chain_data = []
        for chain in active_chains:
            job_id = chain.get("job_id", "unknown")
            job_type = chain.get("job_type", "unknown")
            current_task = chain.get("current_task", "无")
            progress = chain.get("progress_percentage", 0)

            chain_data.append({
                "作业ID": job_id[:8] + "...",
                "类型": job_type,
                "当前任务": current_task,
                "进度": f"{progress:.1f}%"
            })

        chain_df = pd.DataFrame(chain_data)
        st.dataframe(chain_df, hide_index=True, use_container_width=True)
    else:
        st.info("当前没有活跃的作业链")

    # System performance metrics
    with st.expander("系统性能指标", expanded=False):
        st.markdown("""
        **自触发架构优势:**
        - 零轮询开销，事件驱动执行
        - 毫秒级任务切换延迟
        - 专用Worker无模型颠簸
        - 支持真正的任务并行性

        **内存优化:**
        - Whisper Worker: 固定2GB分配
        - 嵌入Worker: 固定3GB分配  
        - 推理Worker: 固定6GB分配
        - 避免动态内存分配导致的碎片化
        """)