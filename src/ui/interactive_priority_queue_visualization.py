"""
交互式优先队列可视化组件
"""

import streamlit as st
import pandas as pd
import time
import datetime
from typing import Dict, List, Any, Optional

# 导入统一的 API 客户端
from src.ui.api_client import api_request


def render_interactive_queue_visualization(api_url: str, api_key: str):
    """
    渲染优先队列系统的交互式可视化。

    参数:
        api_url: API URL
        api_key: API 认证密钥
    """
    st.subheader("任务队列监控")

    # 获取优先队列状态
    queue_status = api_request(
        endpoint="/query/queue-status",
        method="GET"
    )

    if not queue_status:
        st.warning("无法获取队列状态")
        if st.button("重试"):
            st.rerun()
        return

    # 显示交互式视图
    tab1, tab2, tab3 = st.tabs(["队列概览", "活动任务", "任务详情"])

    with tab1:
        # 显示队列统计信息
        priority_tasks = queue_status.get("priority_tasks", [])
        active_task = queue_status.get("active_task")

        # 创建队列统计指标
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("活动任务", "1" if active_task else "0")

        with col2:
            st.metric("等待任务", len(priority_tasks))

        with col3:
            # 计算队列中任务的平均等待时间
            if priority_tasks:
                wait_times = [time.time() - task.get("registered_at", time.time()) for task in priority_tasks]
                avg_wait = sum(wait_times) / len(wait_times)

                if avg_wait < 60:
                    wait_str = f"{avg_wait:.1f}秒"
                elif avg_wait < 3600:
                    wait_str = f"{avg_wait / 60:.1f}分钟"
                else:
                    wait_str = f"{avg_wait / 3600:.1f}小时"

                st.metric("平均等待时间", wait_str)
            else:
                st.metric("平均等待时间", "0秒")

        # 显示队列可视化
        st.subheader("队列分布")

        # 按队列获取任务
        tasks_by_queue = queue_status.get("tasks_by_queue", {})

        if tasks_by_queue:
            # 为每个队列创建条形图
            for queue_name, count in tasks_by_queue.items():
                if count > 0:
                    # 获取优美的名称
                    pretty_name = {
                        "inference_tasks": "LLM 生成队列",
                        "embedding_tasks": "向量嵌入队列",
                        "transcription_tasks": "语音转录队列",
                        "cpu_tasks": "CPU 处理队列",
                        "system_tasks": "系统任务队列"
                    }.get(queue_name, queue_name)

                    # 获取优先级
                    priority_level = queue_status.get("priority_levels", {}).get(queue_name, "-")

                    # 创建进度条
                    st.text(f"{pretty_name} (优先级: {priority_level})")

                    # 缩放进度条（最大 100%）
                    max_display = 100
                    display_value = min(count * 5, max_display)  # 将计数乘以 5 以提高可见性

                    st.progress(display_value, text=f"{count} 任务")
        else:
            st.info("队列中没有等待的任务")

        # 显示优先级配置
        with st.expander("队列优先级配置", expanded=False):
            priority_levels = queue_status.get("priority_levels", {})

            if priority_levels:
                priority_data = [
                    {"队列": queue, "优先级": priority}
                    for queue, priority in priority_levels.items()
                ]

                priority_df = pd.DataFrame(priority_data)
                st.dataframe(priority_df, hide_index=True, use_container_width=True)

                st.info("优先级数字越低，优先级越高。1是最高优先级。")
            else:
                st.info("没有找到优先级配置")

    with tab2:
        # 显示活动任务详情
        if active_task:
            st.subheader("当前活动任务")

            # 提取任务信息
            task_id = active_task.get("task_id", "unknown")
            queue_name = active_task.get("queue_name", "unknown")
            priority = active_task.get("priority", "-")
            job_id = active_task.get("job_id", "unknown")
            registered_at = active_task.get("registered_at", time.time())

            # 计算任务活动时间
            active_time = time.time() - registered_at
            if active_time < 60:
                active_str = f"{active_time:.1f}秒"
            elif active_time < 3600:
                active_str = f"{active_time / 60:.1f}分钟"
            else:
                active_str = f"{active_time / 3600:.1f}小时"

            # 布局任务信息
            col1, col2 = st.columns(2)

            with col1:
                st.markdown(f"**任务ID:** {task_id}")
                st.markdown(f"**作业ID:** {job_id}")
                st.markdown(f"**活动时间:** {active_str}")

            with col2:
                st.markdown(f"**队列:** {queue_name}")
                st.markdown(f"**优先级:** {priority}")
                registered_time = datetime.datetime.fromtimestamp(registered_at).strftime("%H:%M:%S")
                st.markdown(f"**开始时间:** {registered_time}")

            # 获取此任务的作业详情
            job_details = api_request(
                endpoint=f"/ingest/jobs/{job_id}",
                method="GET"
            )

            if job_details:
                # 显示作业类型和状态
                job_type = job_details.get("job_type", "unknown")
                status = job_details.get("status", "unknown")

                st.markdown(f"**作业类型:** {job_type}")
                st.markdown(f"**状态:** {status}")

                # 如果可用，显示进度
                progress_info = job_details.get("progress_info", {})
                progress = progress_info.get("progress", 0)
                progress_message = progress_info.get("message", "")

                st.progress(progress, text=f"{progress}% - {progress_message}")

                # 提供查看完整作业详情的选项
                if st.button("查看完整作业详情", key="view_active_job"):
                    st.session_state.selected_job_id = job_id
                    st.rerun()

            # 如果运行时间过长，添加终止任务的选项
            if active_time > 600:  # 10 分钟
                st.warning("此任务运行时间较长")
                if st.button("终止此任务", key="terminate_active"):
                    # 调用 API 终止任务
                    terminate_response = api_request(
                        endpoint="/system/terminate-task",
                        method="POST",
                        data={"task_id": task_id}
                    )

                    if terminate_response:
                        st.success("已发送终止信号")
                    else:
                        st.error("发送终止信号失败")
        else:
            st.info("当前没有活动任务")

    with tab3:
        # 显示队列中等待的任务
        st.subheader("等待中的任务")

        if priority_tasks:
            # 为任务创建数据框
            task_data = []

            for task in priority_tasks:
                task_id = task.get("task_id", "unknown")
                queue_name = task.get("queue_name", "unknown")
                priority = task.get("priority", "-")
                job_id = task.get("job_id", "unknown")
                registered_at = task.get("registered_at", time.time())

                # 计算等待时间
                wait_time = time.time() - registered_at
                if wait_time < 60:
                    wait_str = f"{wait_time:.1f}秒"
                elif wait_time < 3600:
                    wait_str = f"{wait_time / 60:.1f}分钟"
                else:
                    wait_str = f"{wait_time / 3600:.1f}小时"

                task_data.append({
                    "任务ID": task_id,
                    "队列": queue_name,
                    "优先级": priority,
                    "作业ID": job_id,
                    "等待时间": wait_str,
                    "_wait_seconds": wait_time  # 用于排序
                })

            # 按等待时间排序（最长的优先）
            task_data.sort(key=lambda x: x["_wait_seconds"], reverse=True)

            # 移除 _wait_seconds 字段
            for task in task_data:
                task.pop("_wait_seconds")

            # 显示为交互式表格
            task_df = pd.DataFrame(task_data)
            selected_rows = st.dataframe(
                task_df,
                hide_index=True,
                use_container_width=True,
                column_config={
                    "任务ID": st.column_config.TextColumn("任务ID", width="medium"),
                    "队列": st.column_config.TextColumn("队列", width="medium"),
                    "优先级": st.column_config.NumberColumn("优先级", width="small"),
                    "作业ID": st.column_config.TextColumn("作业ID", width="medium"),
                    "等待时间": st.column_config.TextColumn("等待时间", width="medium")
                }
            )

            # 检查是否单击了某行
            if selected_rows is not None and len(selected_rows) > 0:
                # 获取所选作业 ID
                selected_job_id = task_df.iloc[selected_rows[0]]["作业ID"]

                # 设置会话状态并重新运行以显示作业详情
                st.session_state.selected_job_id = selected_job_id
                st.rerun()

            # 显示管理选项
            with st.expander("任务队列管理", expanded=False):
                # 优先处理作业的选项
                job_ids = [task["作业ID"] for task in task_data]
                selected_job = st.selectbox("选择作业", job_ids)

                col1, col2 = st.columns(2)

                with col1:
                    if st.button("优先处理所选作业", key="prioritize_job"):
                        # 调用 API 优先处理作业
                        prioritize_response = api_request(
                            endpoint="/system/prioritize-job",
                            method="POST",
                            data={"job_id": selected_job}
                        )
                        if prioritize_response:
                            st.success("已提高作业优先级")
                        else:
                            st.error("提高优先级失败")

                with col2:
                    if st.button("取消所选作业", key="cancel_job"):
                        # 调用 API 取消作业
                        cancel_response = api_request(
                            endpoint="/ingest/jobs/cancel",
                            method="POST",
                            data={"job_id": selected_job}
                        )
                        if cancel_response:
                            st.success("已取消作业")
                        else:
                            st.error("取消作业失败")
        else:
            st.info("队列中没有等待的任务")

    # 添加刷新按钮
    if st.button("刷新队列状态", key="refresh_queue_status"):
        st.rerun()