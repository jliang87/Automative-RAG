"""
系统通知和警报组件，用于管理员。

此模块提供功能来跟踪和显示系统警报和通知，
用于重要事件，如 worker 故障、任务超时和资源问题。
"""

import streamlit as st
import time
from typing import Dict, List, Any, Optional
import pandas as pd
import datetime

# 导入统一的 API 客户端
from src.ui.api_client import api_request


class SystemNotifications:
    """
    系统通知和警报管理器。

    此类处理跟踪、显示和管理系统通知，
    专注于关键事件，如 worker 故障、资源耗尽和任务超时。
    """

    def __init__(self, api_url: str, api_key: str):
        """
        初始化通知管理器。

        参数:
            api_url: API URL
            api_key: API 认证密钥
        """
        self.api_url = api_url
        self.api_key = api_key

        # 如果需要，初始化通知的会话状态
        if "system_notifications" not in st.session_state:
            st.session_state.system_notifications = []

        if "last_notification_check" not in st.session_state:
            st.session_state.last_notification_check = time.time()

    def check_for_new_alerts(self) -> List[Dict[str, Any]]:
        """
        从各种来源检查新的系统警报。

        返回:
            新警报字典列表
        """
        new_alerts = []

        # 检查 worker 健康状况
        worker_alerts = self._check_worker_health()
        if worker_alerts:
            new_alerts.extend(worker_alerts)

        # 检查 GPU 内存
        gpu_alerts = self._check_gpu_memory()
        if gpu_alerts:
            new_alerts.extend(gpu_alerts)

        # 检查停滞的任务
        task_alerts = self._check_stalled_tasks()
        if task_alerts:
            new_alerts.extend(task_alerts)

        # 将新警报保存到会话状态
        if new_alerts:
            for alert in new_alerts:
                # 如果不存在则添加时间戳
                if "timestamp" not in alert:
                    alert["timestamp"] = time.time()

                # 添加到列表的开头，以便按时间倒序排列
                st.session_state.system_notifications.insert(0, alert)

            # 限制为最近的 100 个通知
            if len(st.session_state.system_notifications) > 100:
                st.session_state.system_notifications = st.session_state.system_notifications[:100]

        # 更新上次检查时间
        st.session_state.last_notification_check = time.time()

        return new_alerts

    def _check_worker_health(self) -> List[Dict[str, Any]]:
        """
        检查所有 worker 进程的健康状况。

        返回:
            与 worker 相关的警报列表
        """
        alerts = []

        try:
            # 使用统一的 API 客户端获取健康数据
            health_data = api_request(
                endpoint="/system/health/detailed",
                method="GET",
                silent=True
            )

            if not health_data:
                return []

            workers = health_data.get("workers", {})

            for worker_id, info in workers.items():
                worker_type = info.get("type", "unknown")
                status = info.get("status", "unknown")
                heartbeat_age = info.get("last_heartbeat_seconds_ago", 0)

                # 如果 worker 不健康则发出警报
                if status != "healthy":
                    alerts.append({
                        "level": "error",
                        "category": "worker",
                        "title": f"Worker 不健康: {worker_type}",
                        "message": f"Worker {worker_id} 处于 {status} 状态",
                        "timestamp": time.time(),
                        "details": {
                            "worker_id": worker_id,
                            "worker_type": worker_type,
                            "status": status
                        }
                    })

                # 如果心跳太旧（超过 2 分钟）则发出警报
                elif heartbeat_age > 120:
                    alerts.append({
                        "level": "warning",
                        "category": "worker",
                        "title": f"Worker 心跳延迟: {worker_type}",
                        "message": f"Worker {worker_id} 最后心跳是 {heartbeat_age:.1f} 秒前",
                        "timestamp": time.time(),
                        "details": {
                            "worker_id": worker_id,
                            "worker_type": worker_type,
                            "heartbeat_age": heartbeat_age
                        }
                    })

            # 检查是否缺少任何必需的 worker 类型
            required_workers = ["gpu-inference", "gpu-embedding", "gpu-whisper", "cpu"]
            active_worker_types = set()

            for worker_id, info in workers.items():
                worker_type = info.get("type", "unknown")
                if worker_type in required_workers:
                    active_worker_types.add(worker_type)

            # 对缺少的 worker 类型发出警报
            for worker_type in required_workers:
                if worker_type not in active_worker_types:
                    alerts.append({
                        "level": "error",
                        "category": "worker",
                        "title": f"缺少 Worker: {worker_type}",
                        "message": f"未检测到 {worker_type} 类型的活动 worker",
                        "timestamp": time.time(),
                        "details": {
                            "missing_worker_type": worker_type
                        }
                    })

            return alerts
        except Exception as e:
            # 记录错误，但不创建无限循环的错误警报
            print(f"检查 worker 健康状况时出错: {str(e)}")
            return []

    def _check_gpu_memory(self) -> List[Dict[str, Any]]:
        """
        检查 GPU 内存使用情况，查找潜在问题。

        返回:
            与 GPU 相关的警报列表
        """
        alerts = []

        try:
            health_data = api_request(
                endpoint="/system/health/detailed",
                method="GET",
                silent=True
            )

            if not health_data:
                return []

            gpu_health = health_data.get("gpu_health", {})

            for gpu_id, info in gpu_health.items():
                # 检查 GPU 是否被报告为不健康
                if not info.get("is_healthy", True):
                    alerts.append({
                        "level": "error",
                        "category": "gpu",
                        "title": f"GPU 不健康: {gpu_id}",
                        "message": info.get("health_message", "GPU 健康检查失败"),
                        "timestamp": time.time(),
                        "details": {
                            "gpu_id": gpu_id,
                            "device_name": info.get("device_name", "未知"),
                            "health_message": info.get("health_message", "未知问题")
                        }
                    })

                # 检查 GPU 内存是否严重不足（小于 10% 可用）
                free_percentage = info.get("free_percentage", 100)
                if free_percentage < 10:
                    alerts.append({
                        "level": "warning",
                        "category": "gpu",
                        "title": f"GPU 内存不足: {gpu_id}",
                        "message": f"GPU {gpu_id} 只有 {free_percentage:.1f}% 可用内存",
                        "timestamp": time.time(),
                        "details": {
                            "gpu_id": gpu_id,
                            "device_name": info.get("device_name", "未知"),
                            "free_percentage": free_percentage,
                            "free_memory_gb": info.get("free_memory_gb", 0),
                            "total_memory_gb": info.get("total_memory_gb", 0)
                        }
                    })

            return alerts
        except Exception as e:
            print(f"检查 GPU 内存时出错: {str(e)}")
            return []

    def _check_stalled_tasks(self) -> List[Dict[str, Any]]:
        """
        检查停滞或超时的任务。

        返回:
            与任务相关的警报列表
        """
        alerts = []

        try:
            # 检查优先队列中的活动任务
            queue_status = api_request(
                endpoint="/query/queue-status",
                method="GET",
                silent=True
            )

            if not queue_status:
                return []

            active_task = queue_status.get("active_task")

            if active_task:
                task_id = active_task.get("task_id")
                job_id = active_task.get("job_id")
                registered_at = active_task.get("registered_at", 0)

                # 计算任务的年龄
                task_age = time.time() - registered_at

                # 如果任务运行时间过长（超过 30 分钟）则发出警报
                if task_age > 1800:  # 30 分钟
                    alerts.append({
                        "level": "warning",
                        "category": "task",
                        "title": "潜在停滞任务",
                        "message": f"任务 {task_id} (作业 {job_id}) 已活动 {task_age / 60:.1f} 分钟",
                        "timestamp": time.time(),
                        "details": {
                            "task_id": task_id,
                            "job_id": job_id,
                            "age_minutes": task_age / 60,
                            "queue": active_task.get("queue_name")
                        }
                    })

            # 检查超时的作业
            jobs = api_request(
                endpoint="/ingest/jobs",
                method="GET",
                params={"limit": 20},
                silent=True
            )

            if jobs:
                for job in jobs:
                    if job.get("status") == "timeout":
                        job_id = job.get("job_id")
                        job_type = job.get("job_type")

                        alerts.append({
                            "level": "error",
                            "category": "task",
                            "title": "作业超时",
                            "message": f"作业 {job_id} ({job_type}) 已超时",
                            "timestamp": time.time(),
                            "details": {
                                "job_id": job_id,
                                "job_type": job_type,
                                "created_at": job.get("created_at")
                            }
                        })

            return alerts
        except Exception as e:
            print(f"检查停滞任务时出错: {str(e)}")
            return []

    def display_notification_center(self, expanded: bool = False):
        """
        显示通知中心 UI 组件。

        参数:
            expanded: 是否默认展开通知中心
        """
        # 首先检查新通知
        self.check_for_new_alerts()

        # 获取所有通知
        notifications = st.session_state.system_notifications

        # 按严重程度计数通知
        error_count = sum(1 for n in notifications if n.get("level") == "error")
        warning_count = sum(1 for n in notifications if n.get("level") == "warning")

        # 显示带计数的通知图标
        total_count = error_count + warning_count

        if total_count > 0:
            # 创建带有警报计数的标题
            title = f"🔔 系统通知 ({total_count})"
            if error_count > 0:
                title += f" | ❌ {error_count} 错误"
            if warning_count > 0:
                title += f" | ⚠️ {warning_count} 警告"
        else:
            title = "🔔 系统通知"

        # 为通知创建可展开部分
        with st.expander(title, expanded=(expanded or error_count > 0)):
            # 如果没有通知
            if not notifications:
                st.info("系统运行正常，无通知")
                return

            # 筛选选项
            col1, col2 = st.columns(2)

            with col1:
                filter_level = st.selectbox("筛选级别", ["全部", "错误", "警告"])

            with col2:
                filter_category = st.selectbox("筛选类别", ["全部", "工作器", "GPU", "任务"])

            # 应用筛选
            filtered_notifications = notifications

            if filter_level == "错误":
                filtered_notifications = [n for n in notifications if n.get("level") == "error"]
            elif filter_level == "警告":
                filtered_notifications = [n for n in notifications if n.get("level") == "warning"]

            if filter_category == "工作器":
                filtered_notifications = [n for n in filtered_notifications if n.get("category") == "worker"]
            elif filter_category == "GPU":
                filtered_notifications = [n for n in filtered_notifications if n.get("category") == "gpu"]
            elif filter_category == "任务":
                filtered_notifications = [n for n in filtered_notifications if n.get("category") == "task"]

            # 显示通知
            if not filtered_notifications:
                st.info("没有符合筛选条件的通知")
                return

            for i, notification in enumerate(filtered_notifications[:10]):  # 显示前 10 个通知
                self._render_notification(notification, i)

            if len(filtered_notifications) > 10:
                st.caption(f"还有 {len(filtered_notifications) - 10} 个通知未显示")

            # 添加清除所有按钮
            if st.button("清除所有通知", key="clear_all_notifications"):
                st.session_state.system_notifications = []
                st.success("已清除所有通知")
                time.sleep(1)
                st.rerun()

    def _render_notification(self, notification: Dict[str, Any], index: int):
        """
        Render a single notification card.

        Args:
            notification: Notification data dictionary
            index: Index for unique key
        """
        level = notification.get("level", "info")
        title = notification.get("title", "系统通知")
        message = notification.get("message", "")
        timestamp = notification.get("timestamp", time.time())
        details = notification.get("details", {})

        # Format timestamp
        time_str = datetime.datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")

        # Choose color and icon based on level
        if level == "error":
            color = "#F44336"  # Red
            icon = "❌"
        elif level == "warning":
            color = "#FF9800"  # Orange/amber
            icon = "⚠️"
        else:
            color = "#2196F3"  # Blue
            icon = "ℹ️"

        # Create notification card
        with st.container():
            # Add colored title bar
            st.markdown(
                f"<div style='padding: 5px 10px; background-color: {color}; color: white; border-radius: 5px 5px 0 0;'>"
                f"<b>{icon} {title}</b>"
                f"</div>",
                unsafe_allow_html=True
            )

            # Add message and timestamp
            st.markdown(
                f"<div style='padding: 10px; border: 1px solid {color}; border-top: none; border-radius: 0 0 5px 5px;'>"
                f"{message}<br/><small>{time_str}</small>"
                f"</div>",
                unsafe_allow_html=True)

            # Display details if present - BUT NOT AS AN EXPANDER
            # This avoids nesting expanders which is causing the error
            if details:
                # Use a collapsible container with button instead
                detail_key = f"show_details_{index}"
                if detail_key not in st.session_state:
                    st.session_state[detail_key] = False

                # Add button to toggle details visibility
                if st.button("显示详细信息" if not st.session_state[detail_key] else "隐藏详细信息",
                             key=f"toggle_details_{index}"):
                    st.session_state[detail_key] = not st.session_state[detail_key]

                # Show details if toggled on
                if st.session_state[detail_key]:
                    with st.container():
                        st.markdown("**详细信息:**")
                        for key, value in details.items():
                            st.text(f"{key}: {value}")

            # Add dismiss button
            if st.button("忽略", key=f"dismiss_{index}"):
                # Remove this notification
                st.session_state.system_notifications.remove(notification)
                st.rerun()

            # Add space between notifications
            st.markdown("<br/>", unsafe_allow_html=True)


def display_notifications_sidebar(api_url: str, api_key: str, check_interval: int = 60):
    """
    在侧边栏显示通知，并定期检查。

    参数:
        api_url: API URL
        api_key: API 认证密钥
        check_interval: 检查之间的时间间隔（秒）
    """
    # 初始化通知管理器
    notifications = SystemNotifications(api_url, api_key)

    # 检查是否是时候刷新通知
    current_time = time.time()
    last_check = st.session_state.get("last_notification_check", 0)

    if current_time - last_check >= check_interval:
        # 检查新通知
        new_alerts = notifications.check_for_new_alerts()

        # 更新上次检查时间
        st.session_state.last_notification_check = current_time

        # 显示新通知的警报
        if new_alerts:
            st.sidebar.warning(f"⚠️ {len(new_alerts)} 个新系统通知")

    # 在侧边栏显示通知中心
    notifications.display_notification_center(False)


def main_notification_dashboard(api_url: str, api_key: str):
    """
    渲染完整的通知仪表板页面。

    参数:
        api_url: API URL
        api_key: API 认证密钥
    """
    st.title("系统通知中心")
    st.markdown("查看所有系统通知、警告和错误")

    # 初始化通知管理器
    notifications = SystemNotifications(api_url, api_key)

    # 添加刷新按钮
    if st.button("检查新通知", key="refresh_notifications"):
        with st.spinner("检查中..."):
            new_alerts = notifications.check_for_new_alerts()
            if new_alerts:
                st.success(f"收到 {len(new_alerts)} 个新通知")
            else:
                st.info("没有新通知")

    # 显示通知中心（默认展开）
    notifications.display_notification_center(True)

    # 添加统计部分
    st.subheader("通知统计")

    # 获取所有通知
    all_notifications = st.session_state.get("system_notifications", [])

    # 按类别和级别计数
    category_counts = {}
    level_counts = {}

    for notification in all_notifications:
        category = notification.get("category", "other")
        level = notification.get("level", "info")

        category_counts[category] = category_counts.get(category, 0) + 1
        level_counts[level] = level_counts.get(level, 0) + 1

    # 显示统计
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**通知类别分布**")
        if category_counts:
            category_df = pd.DataFrame({
                "类别": ["工作器" if c == "worker" else ("GPU" if c == "gpu" else ("任务" if c == "task" else c))
                        for c in category_counts.keys()],
                "数量": list(category_counts.values())
            })
            st.dataframe(category_df, hide_index=True)
        else:
            st.info("没有通知数据")

    with col2:
        st.markdown("**通知级别分布**")
        if level_counts:
            level_df = pd.DataFrame({
                "级别": ["错误" if l == "error" else ("警告" if l == "warning" else l) for l in level_counts.keys()],
                "数量": list(level_counts.values())
            })
            st.dataframe(level_df, hide_index=True)
        else:
            st.info("没有通知数据")