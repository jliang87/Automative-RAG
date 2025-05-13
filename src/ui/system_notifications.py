"""
System notifications and alerts component for administrators.

This module provides functionality to track and display system alerts and notifications
for important events like worker failures, task timeouts, and resource issues.
"""

import streamlit as st
import time
from typing import Dict, List, Any, Optional
import pandas as pd
import httpx
import datetime


class SystemNotifications:
    """
    Manager for system notifications and alerts.

    This class handles tracking, displaying, and managing system notifications
    for administrators, focusing on critical events like worker failures,
    resource exhaustion, and task timeouts.
    """

    def __init__(self, api_url: str, api_key: str):
        """
        Initialize the notifications manager.

        Args:
            api_url: API URL
            api_key: API authentication key
        """
        self.api_url = api_url
        self.api_key = api_key

        # Initialize session state for notifications if needed
        if "system_notifications" not in st.session_state:
            st.session_state.system_notifications = []

        if "last_notification_check" not in st.session_state:
            st.session_state.last_notification_check = time.time()

    def check_for_new_alerts(self) -> List[Dict[str, Any]]:
        """
        Check for new system alerts from various sources.

        Returns:
            List of new alert dictionaries
        """
        new_alerts = []

        # Check worker health
        worker_alerts = self._check_worker_health()
        if worker_alerts:
            new_alerts.extend(worker_alerts)

        # Check GPU memory
        gpu_alerts = self._check_gpu_memory()
        if gpu_alerts:
            new_alerts.extend(gpu_alerts)

        # Check for stalled tasks
        task_alerts = self._check_stalled_tasks()
        if task_alerts:
            new_alerts.extend(task_alerts)

        # Save new alerts to session state
        if new_alerts:
            for alert in new_alerts:
                # Add timestamp if not present
                if "timestamp" not in alert:
                    alert["timestamp"] = time.time()

                # Add to the start of the list for reverse chronological order
                st.session_state.system_notifications.insert(0, alert)

            # Limit to most recent 100 notifications
            if len(st.session_state.system_notifications) > 100:
                st.session_state.system_notifications = st.session_state.system_notifications[:100]

        # Update last check time
        st.session_state.last_notification_check = time.time()

        return new_alerts

    def _check_worker_health(self) -> List[Dict[str, Any]]:
        """
        Check health of all worker processes.

        Returns:
            List of worker-related alerts
        """
        alerts = []

        try:
            health_data = self._api_request("/system/health/detailed")

            if not health_data:
                return []

            workers = health_data.get("workers", {})

            for worker_id, info in workers.items():
                worker_type = info.get("type", "unknown")
                status = info.get("status", "unknown")
                heartbeat_age = info.get("last_heartbeat_seconds_ago", 0)

                # Alert if worker is unhealthy
                if status != "healthy":
                    alerts.append({
                        "level": "error",
                        "category": "worker",
                        "title": f"Worker Unhealthy: {worker_type}",
                        "message": f"Worker {worker_id} is in {status} state",
                        "timestamp": time.time(),
                        "details": {
                            "worker_id": worker_id,
                            "worker_type": worker_type,
                            "status": status
                        }
                    })

                # Alert if heartbeat is too old (more than 2 minutes)
                elif heartbeat_age > 120:
                    alerts.append({
                        "level": "warning",
                        "category": "worker",
                        "title": f"Worker Heartbeat Delayed: {worker_type}",
                        "message": f"Worker {worker_id} last heartbeat was {heartbeat_age:.1f} seconds ago",
                        "timestamp": time.time(),
                        "details": {
                            "worker_id": worker_id,
                            "worker_type": worker_type,
                            "heartbeat_age": heartbeat_age
                        }
                    })

            # Check if any required worker types are missing
            required_workers = ["gpu-inference", "gpu-embedding", "gpu-whisper", "cpu"]
            active_worker_types = set()

            for worker_id, info in workers.items():
                worker_type = info.get("type", "unknown")
                if worker_type in required_workers:
                    active_worker_types.add(worker_type)

            # Alert for missing worker types
            for worker_type in required_workers:
                if worker_type not in active_worker_types:
                    alerts.append({
                        "level": "error",
                        "category": "worker",
                        "title": f"Missing Worker: {worker_type}",
                        "message": f"No active workers of type {worker_type} detected",
                        "timestamp": time.time(),
                        "details": {
                            "missing_worker_type": worker_type
                        }
                    })

            return alerts
        except Exception as e:
            # Log error but don't create an infinite loop of error alerts
            print(f"Error checking worker health: {str(e)}")
            return []

    def _check_gpu_memory(self) -> List[Dict[str, Any]]:
        """
        Check GPU memory usage for potential issues.

        Returns:
            List of GPU-related alerts
        """
        alerts = []

        try:
            health_data = self._api_request("/system/health/detailed")

            if not health_data:
                return []

            gpu_health = health_data.get("gpu_health", {})

            for gpu_id, info in gpu_health.items():
                # Check if GPU is reported as unhealthy
                if not info.get("is_healthy", True):
                    alerts.append({
                        "level": "error",
                        "category": "gpu",
                        "title": f"GPU Unhealthy: {gpu_id}",
                        "message": info.get("health_message", "GPU health check failed"),
                        "timestamp": time.time(),
                        "details": {
                            "gpu_id": gpu_id,
                            "device_name": info.get("device_name", "Unknown"),
                            "health_message": info.get("health_message", "Unknown issue")
                        }
                    })

                # Check if GPU memory is critically low (less than 10% free)
                free_percentage = info.get("free_percentage", 100)
                if free_percentage < 10:
                    alerts.append({
                        "level": "warning",
                        "category": "gpu",
                        "title": f"Low GPU Memory: {gpu_id}",
                        "message": f"GPU {gpu_id} has only {free_percentage:.1f}% free memory",
                        "timestamp": time.time(),
                        "details": {
                            "gpu_id": gpu_id,
                            "device_name": info.get("device_name", "Unknown"),
                            "free_percentage": free_percentage,
                            "free_memory_gb": info.get("free_memory_gb", 0),
                            "total_memory_gb": info.get("total_memory_gb", 0)
                        }
                    })

            return alerts
        except Exception as e:
            print(f"Error checking GPU memory: {str(e)}")
            return []

    def _check_stalled_tasks(self) -> List[Dict[str, Any]]:
        """
        Check for stalled or timed-out tasks.

        Returns:
            List of task-related alerts
        """
        alerts = []

        try:
            # Check active task in the priority queue
            queue_status = self._api_request("/query/queue-status")

            if not queue_status:
                return []

            active_task = queue_status.get("active_task")

            if active_task:
                task_id = active_task.get("task_id")
                job_id = active_task.get("job_id")
                registered_at = active_task.get("registered_at", 0)

                # Calculate age of task
                task_age = time.time() - registered_at

                # Alert if task has been running for too long (more than 30 minutes)
                if task_age > 1800:  # 30 minutes
                    alerts.append({
                        "level": "warning",
                        "category": "task",
                        "title": "Potential Stalled Task",
                        "message": f"Task {task_id} for job {job_id} has been active for {task_age / 60:.1f} minutes",
                        "timestamp": time.time(),
                        "details": {
                            "task_id": task_id,
                            "job_id": job_id,
                            "age_minutes": task_age / 60,
                            "queue": active_task.get("queue_name")
                        }
                    })

            # Check for timed out jobs
            jobs = self._api_request("/ingest/jobs", params={"limit": 20})

            if jobs:
                for job in jobs:
                    if job.get("status") == "timeout":
                        job_id = job.get("job_id")
                        job_type = job.get("job_type")

                        alerts.append({
                            "level": "error",
                            "category": "task",
                            "title": "Job Timeout",
                            "message": f"Job {job_id} ({job_type}) has timed out",
                            "timestamp": time.time(),
                            "details": {
                                "job_id": job_id,
                                "job_type": job_type,
                                "created_at": job.get("created_at")
                            }
                        })

            return alerts
        except Exception as e:
            print(f"Error checking stalled tasks: {str(e)}")
            return []

    def display_notification_center(self, expanded: bool = False):
        """
        Display the notification center UI component.

        Args:
            expanded: Whether to show notification center expanded by default
        """
        # First check for new notifications
        self.check_for_new_alerts()

        # Get all notifications
        notifications = st.session_state.system_notifications

        # Count notifications by severity
        error_count = sum(1 for n in notifications if n.get("level") == "error")
        warning_count = sum(1 for n in notifications if n.get("level") == "warning")

        # Display notification icon with count
        total_count = error_count + warning_count

        if total_count > 0:
            # Create a title with alert counts
            title = f"🔔 系统通知 ({total_count})"
            if error_count > 0:
                title += f" | ❌ {error_count} 错误"
            if warning_count > 0:
                title += f" | ⚠️ {warning_count} 警告"
        else:
            title = "🔔 系统通知"

        # Create expandable section for notifications
        with st.expander(title, expanded=(expanded or error_count > 0)):
            # If no notifications
            if not notifications:
                st.info("系统运行正常，无通知")
                return

            # Filter options
            col1, col2 = st.columns(2)

            with col1:
                filter_level = st.selectbox("筛选级别", ["全部", "错误", "警告"])

            with col2:
                filter_category = st.selectbox("筛选类别", ["全部", "工作器", "GPU", "任务"])

            # Apply filters
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

            # Display notifications
            if not filtered_notifications:
                st.info("没有符合筛选条件的通知")
                return

            for i, notification in enumerate(filtered_notifications[:10]):  # Show first 10 notifications
                self._render_notification(notification, i)

            if len(filtered_notifications) > 10:
                st.caption(f"还有 {len(filtered_notifications) - 10} 个通知未显示")

            # Add clear all button
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
            index: Index for unique keys
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

            # Show details if available
            if details:
                with st.expander("详细信息", expanded=False):
                    for key, value in details.items():
                        st.text(f"{key}: {value}")

            # Add dismiss button
            if st.button("忽略", key=f"dismiss_{index}"):
                # Remove this notification
                st.session_state.system_notifications.remove(notification)
                st.rerun()

            # Add a small space between notifications
            st.markdown("<br/>", unsafe_allow_html=True)

    def _api_request(self, endpoint: str, method: str = "GET", data: Optional[Dict] = None,
                     params: Optional[Dict] = None) -> Optional[Any]:
        """
        Make an API request with error handling.

        Args:
            endpoint: API endpoint path
            method: HTTP method
            data: Request data
            params: Query parameters

        Returns:
            Response data or None on failure
        """
        try:
            headers = {"x-token": self.api_key}
            url = f"{self.api_url}{endpoint}"

            with httpx.Client(timeout=5.0) as client:  # Short timeout for notifications
                if method == "GET":
                    response = client.get(url, headers=headers, params=params)
                elif method == "POST":
                    response = client.post(url, headers=headers, json=data)
                else:
                    print(f"Unsupported method: {method}")
                    return None

                if response.status_code >= 400:
                    print(f"API error ({response.status_code}): {response.text}")
                    return None

                return response.json()
        except Exception as e:
            print(f"API request error: {str(e)}")
            return None


def display_notifications_sidebar(api_url: str, api_key: str, check_interval: int = 60):
    """
    Display notifications in the sidebar with periodic checks.

    Args:
        api_url: API URL
        api_key: API authentication key
        check_interval: Time between checks in seconds
    """
    # Initialize notification manager
    notifications = SystemNotifications(api_url, api_key)

    # Check if it's time to refresh notifications
    current_time = time.time()
    last_check = st.session_state.get("last_notification_check", 0)

    if current_time - last_check >= check_interval:
        # Check for new notifications
        new_alerts = notifications.check_for_new_alerts()

        # Update last check time
        st.session_state.last_notification_check = current_time

        # Show alert for new notifications
        if new_alerts:
            st.sidebar.warning(f"⚠️ {len(new_alerts)} 个新系统通知")

    # Display notification center in sidebar
    notifications.display_notification_center(False)


def main_notification_dashboard(api_url: str, api_key: str):
    """
    Render a full notification dashboard page.

    Args:
        api_url: API URL
        api_key: API authentication key
    """
    st.title("系统通知中心")
    st.markdown("查看所有系统通知、警告和错误")

    # Initialize notification manager
    notifications = SystemNotifications(api_url, api_key)

    # Add refresh button
    if st.button("检查新通知", key="refresh_notifications"):
        with st.spinner("检查中..."):
            new_alerts = notifications.check_for_new_alerts()
            if new_alerts:
                st.success(f"收到 {len(new_alerts)} 个新通知")
            else:
                st.info("没有新通知")

    # Display notification center (expanded by default)
    notifications.display_notification_center(True)

    # Add statistics section
    st.subheader("通知统计")

    # Get all notifications
    all_notifications = st.session_state.get("system_notifications", [])

    # Count by category and level
    category_counts = {}
    level_counts = {}

    for notification in all_notifications:
        category = notification.get("category", "other")
        level = notification.get("level", "info")

        category_counts[category] = category_counts.get(category, 0) + 1
        level_counts[level] = level_counts.get(level, 0) + 1

    # Display statistics
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