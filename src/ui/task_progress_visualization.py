"""
任务进度可视化组件，具有进度条和估计完成时间。
"""

import streamlit as st
import pandas as pd
import time
from typing import Dict, Any, List, Optional

# 导入统一的 API 客户端
from src.ui.api_client import api_request


def display_task_progress(job_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    显示带有进度条和估计完成时间的任务。

    参数:
        job_data: 任务数据字典

    返回:
        包含用户操作的字典 (例如 retry, cancel)
    """
    # 提取关键信息
    job_id = job_data.get("job_id", "")
    status = job_data.get("status", "")
    job_type = job_data.get("job_type", "")

    # 获取进度信息
    progress_info = job_data.get("progress_info", {})
    progress = progress_info.get("progress", 0)
    progress_message = progress_info.get("message", "")

    # 获取时间信息
    created_at = job_data.get("created_at", 0)
    updated_at = job_data.get("updated_at", 0)
    elapsed_time = time.time() - created_at
    estimated_remaining = job_data.get("estimated_remaining_seconds")

    # 显示基本任务信息
    st.subheader(f"任务: {job_id}")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"**类型:** {job_type}")
        st.markdown(f"**状态:** {status}")

        # 格式化已用时间
        if elapsed_time < 60:
            elapsed_str = f"{elapsed_time:.1f} 秒"
        elif elapsed_time < 3600:
            elapsed_str = f"{elapsed_time / 60:.1f} 分钟"
        else:
            elapsed_str = f"{elapsed_time / 3600:.1f} 小时"

        st.markdown(f"**已用时间:** {elapsed_str}")

    with col2:
        # 显示创建和更新时间
        created_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(created_at))
        updated_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(updated_at))

        st.markdown(f"**创建时间:** {created_str}")
        st.markdown(f"**最后更新:** {updated_str}")

        # 如果可用，显示估计完成时间
        if estimated_remaining and status == "processing":
            if estimated_remaining < 60:
                remaining_str = f"{estimated_remaining:.1f} 秒"
            elif estimated_remaining < 3600:
                remaining_str = f"{estimated_remaining / 60:.1f} 分钟"
            else:
                remaining_str = f"{estimated_remaining / 3600:.1f} 小时"

            st.markdown(f"**预计剩余时间:** {remaining_str}")

    # 显示进度条
    if status == "processing":
        st.progress(int(progress), text=f"{int(progress)}% - {progress_message}")
    elif status == "completed":
        st.progress(100, text="100% - 任务完成")
    elif status in ["failed", "timeout"]:
        st.error("任务失败")
    else:
        st.progress(0, text="0% - 任务等待中")

    # 如果可用，显示当前阶段
    current_stage = job_data.get("current_stage")
    if current_stage:
        st.info(f"当前阶段: {current_stage}")

    # 对于已完成的任务，显示结果摘要
    if status == "completed":
        st.subheader("结果摘要")
        result = job_data.get("result", {})

        if isinstance(result, dict):
            # 根据任务类型显示不同的结果
            if job_type == "llm_inference":
                if "answer" in result:
                    st.markdown("### 回答")
                    st.markdown(result.get("answer", ""))

                if "documents" in result:
                    st.markdown("### 引用文档")
                    st.markdown(f"找到 {len(result.get('documents', []))} 个相关文档")

                if "execution_time" in result:
                    st.markdown(f"查询处理用时 {result.get('execution_time', 0):.2f} 秒")

            elif job_type in ["video_processing", "batch_video_processing"]:
                if "transcript" in result:
                    st.markdown("### 转录摘要")
                    transcript = result.get("transcript", "")
                    if len(transcript) > 500:
                        st.markdown(f"{transcript[:500]}...")
                        if st.button("显示完整转录"):
                            st.text_area("完整转录", transcript, height=300)
                    else:
                        st.markdown(transcript)

                if "document_count" in result:
                    st.markdown(f"生成了 {result.get('document_count', 0)} 个文档块")

            elif job_type == "pdf_processing":
                if "chunk_count" in result:
                    st.markdown(f"从 PDF 中提取了 {result.get('chunk_count', 0)} 个文本块")

                if "processing_time" in result:
                    st.markdown(f"处理完成，用时 {result.get('processing_time', 0):.2f} 秒")

        elif isinstance(result, str):
            st.markdown(result)

    # 对于失败的任务，显示错误
    elif status in ["failed", "timeout"]:
        st.subheader("错误详情")
        error = job_data.get("error", "未知错误")
        st.error(error)

        # 添加重试按钮
        if st.button("重试任务", key=f"retry_{job_id}"):
            return {"action": "retry", "job_id": job_id}

    # 对于处理中的任务，显示取消按钮
    elif status == "processing":
        if st.button("取消任务", key=f"cancel_{job_id}"):
            return {"action": "cancel", "job_id": job_id}

    # 添加分隔线
    st.divider()

    return {"action": None}


def display_stage_timeline(job_data: Dict[str, Any]):
    """
    显示任务阶段的时间线可视化。

    参数:
        job_data: 任务数据字典
    """
    stage_history = job_data.get("stage_history", [])

    if not stage_history:
        st.info("没有可用的阶段信息")
        return

    # 创建时间线数据
    timeline_data = []
    created_at = job_data.get("created_at", 0)

    for i, stage in enumerate(stage_history):
        stage_name = stage.get("stage", "unknown")
        start_time = stage.get("started_at", 0)

        # 计算结束时间（下一个阶段的开始或当前时间）
        if i < len(stage_history) - 1:
            end_time = stage_history[i + 1].get("started_at", time.time())
        else:
            end_time = time.time()

        # 计算时间线的相对位置和宽度
        start_pos = (start_time - created_at) / (time.time() - created_at) * 100
        width = (end_time - start_time) / (time.time() - created_at) * 100

        # 格式化时间
        start_str = time.strftime("%H:%M:%S", time.localtime(start_time))
        duration = end_time - start_time

        if duration < 60:
            duration_str = f"{duration:.1f}秒"
        elif duration < 3600:
            duration_str = f"{duration / 60:.1f}分钟"
        else:
            duration_str = f"{duration / 3600:.1f}小时"

        timeline_data.append({
            "stage": stage_name,
            "start_time": start_time,
            "end_time": end_time,
            "start_str": start_str,
            "duration": duration,
            "duration_str": duration_str,
            "start_pos": start_pos,
            "width": width
        })

    # 显示时间线
    st.subheader("处理时间线")

    # 创建自定义时间线的容器
    timeline_container = st.container()

    # 以可视化方式显示时间线
    total_duration = time.time() - created_at
    if total_duration > 0:
        # 以可读格式计算总持续时间
        if total_duration < 60:
            total_duration_str = f"{total_duration:.1f} 秒"
        elif total_duration < 3600:
            total_duration_str = f"{total_duration / 60:.1f} 分钟"
        else:
            total_duration_str = f"{total_duration / 3600:.1f} 小时"

        st.caption(f"总处理时间: {total_duration_str}")

        # 将每个阶段显示为带有自定义颜色的进度条
        colors = ["#1E88E5", "#FFC107", "#43A047", "#E53935", "#8E24AA"]

        for i, stage in enumerate(timeline_data):
            col1, col2 = st.columns([stage["start_pos"] / 100, 1 - stage["start_pos"] / 100]) if stage["start_pos"] > 0 else (
            None, st.columns([1])[0])

            if col1 is not None:
                col1.write("")  # 留空以进行定位

            # 计算颜色索引
            color_idx = i % len(colors)
            color = colors[color_idx]

            # 显示带有适当宽度的阶段条
            if stage["width"] > 0:
                col2.progress(min(100, stage["width"]), text=f"{stage['stage']} ({stage['duration_str']})")

    # 同时显示为表格
    timeline_table = pd.DataFrame([
        {
            "阶段": t["stage"],
            "开始时间": t["start_str"],
            "持续时间": t["duration_str"],
        } for t in timeline_data
    ])

    st.dataframe(timeline_table, hide_index=True, use_container_width=True)


def render_priority_queue_visualization(queue_data: Dict[str, Any]):
    """
    渲染优先队列系统的可视化。

    参数:
        queue_data: 优先队列数据字典
    """
    if not queue_data:
        st.warning("没有可用的优先队列数据")
        return

    st.subheader("优先队列状态")

    # 显示活动任务
    active_task = queue_data.get("active_task")
    if active_task:
        st.markdown("### 活动 GPU 任务")

        active_task_info = {
            "队列": active_task.get("queue_name", "未知"),
            "优先级": active_task.get("priority", "-"),
            "任务 ID": active_task.get("task_id", "未知"),
            "作业 ID": active_task.get("job_id", "未知"),
        }

        # 计算任务活动时间
        registered_at = active_task.get("registered_at")
        if registered_at:
            active_duration = time.time() - registered_at
            if active_duration < 60:
                duration_str = f"{active_duration:.1f} 秒"
            elif active_duration < 3600:
                duration_str = f"{active_duration / 60:.1f} 分钟"
            else:
                duration_str = f"{active_duration / 3600:.1f} 小时"

            active_task_info["活动时间"] = duration_str

        # 显示为 DataFrame
        active_df = pd.DataFrame([active_task_info])
        st.dataframe(active_df, hide_index=True, use_container_width=True)
    else:
        st.info("当前没有正在运行的活动 GPU 任务")

    # 显示等待中的任务
    st.markdown("### 等待中的任务")

    tasks_by_queue = queue_data.get("tasks_by_queue", {})
    if tasks_by_queue:
        # 准备可视化数据
        queue_data_list = []

        for queue, count in tasks_by_queue.items():
            if count > 0:
                # 获取此队列的优先级
                priority_level = queue_data.get("priority_levels", {}).get(queue, "-")

                queue_data_list.append({
                    "队列": queue,
                    "优先级": priority_level,
                    "等待任务": count
                })

        if queue_data_list:
            # 按优先级排序（数字越小 = 优先级越高）
            queue_data_list.sort(key=lambda x: x["优先级"])

            # 显示为 DataFrame
            queue_df = pd.DataFrame(queue_data_list)
            st.dataframe(queue_df, hide_index=True, use_container_width=True)

            # 以水平条形图可视化
            for queue_info in queue_data_list:
                st.text(f"{queue_info['队列']} (优先级 {queue_info['优先级']})")
                st.progress(min(100, queue_info["等待任务"] * 10), text=f"{queue_info['等待任务']} 个任务")
        else:
            st.info("队列中没有等待的任务")
    else:
        st.info("队列中没有等待的任务")