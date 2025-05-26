"""
简化的系统通知组件，专为自触发作业链架构优化

此模块提供功能来跟踪和显示系统警报和通知，
专注于作业链、专用Worker和GPU状态监控。
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
    系统通知和警报管理器 - 自触发架构优化版
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

        # 初始化通知的会话状态
        if "system_notifications" not in st.session_state:
            st.session_state.system_notifications = []

        if "last_notification_check" not in st.session_state:
            st.session_state.last_notification_check = time.time()

    def check_for_new_alerts(self) -> List[Dict[str, Any]]:
        """
        检查新的系统警报 - 专注于作业链和专用Worker
        """
        new_alerts = []

        # 检查专用Worker健康状况
        worker_alerts = self._check_dedicated_workers()
        if worker_alerts:
            new_alerts.extend(worker_alerts)

        # 检查作业链状态
        job_chain_alerts = self._check_job_chains()
        if job_chain_alerts:
            new_alerts.extend(job_chain_alerts)

        # 检查GPU内存 (简化版)
        gpu_alerts = self._check_gpu_status()
        if gpu_alerts:
            new_alerts.extend(gpu_alerts)

        # 保存新警报到会话状态
        if new_alerts:
            for alert in new_alerts:
                if "timestamp" not in alert:
                    alert["timestamp"] = time.time()
                st.session_state.system_notifications.insert(0, alert)

            # 限制为最近的 50 个通知 (减少内存使用)
            if len(st.session_state.system_notifications) > 50:
                st.session_state.system_notifications = st.session_state.system_notifications[:50]

        st.session_state.last_notification_check = time.time()
        return new_alerts

    def _check_dedicated_workers(self) -> List[Dict[str, Any]]:
        """检查专用Worker状态"""
        alerts = []

        try:
            health_data = api_request(
                endpoint="/system/health/detailed",
                method="GET",
                silent=True
            )

            if not health_data:
                return []

            workers = health_data.get("workers", {})

            # 检查必需的专用Worker类型
            required_workers = {
                "gpu-whisper": "语音转录Worker",
                "gpu-embedding": "向量嵌入Worker",
                "gpu-inference": "LLM推理Worker",
                "cpu": "CPU处理Worker"
            }

            active_worker_types = set()

            # 检查每个Worker的健康状态
            for worker_id, info in workers.items():
                worker_type = info.get("type", "unknown")
                status = info.get("status", "unknown")
                heartbeat_age = info.get("last_heartbeat_seconds_ago", 0)

                if worker_type in required_workers:
                    active_worker_types.add(worker_type)

                    # Worker不健康警报
                    if status != "healthy":
                        alerts.append({
                            "level": "error",
                            "category": "worker",
                            "title": f"{required_workers[worker_type]}不健康",
                            "message": f"Worker {worker_id} 状态: {status}",
                            "timestamp": time.time(),
                            "details": {"worker_id": worker_id, "worker_type": worker_type}
                        })

                    # 心跳延迟警报 (1分钟)
                    elif heartbeat_age > 60:
                        alerts.append({
                            "level": "warning",
                            "category": "worker",
                            "title": f"{required_workers[worker_type]}心跳延迟",
                            "message": f"最后心跳: {heartbeat_age:.1f}秒前",
                            "timestamp": time.time(),
                            "details": {"worker_id": worker_id, "heartbeat_age": heartbeat_age}
                        })

            # 检查缺失的Worker类型
            for worker_type, display_name in required_workers.items():
                if worker_type not in active_worker_types:
                    alerts.append({
                        "level": "error",
                        "category": "worker",
                        "title": f"缺少{display_name}",
                        "message": f"未检测到{display_name}实例",
                        "timestamp": time.time(),
                        "details": {"missing_worker_type": worker_type}
                    })

            return alerts
        except Exception as e:
            return []

    def _check_job_chains(self) -> List[Dict[str, Any]]:
        """检查作业链状态"""
        alerts = []

        try:
            # 获取作业链概览
            overview = api_request(
                endpoint="/job-chains",
                method="GET",
                silent=True
            )

            if not overview:
                return []

            # 检查活跃作业链
            active_chains = overview.get("active_chains", [])

            for chain in active_chains:
                job_id = chain.get("job_id", "")
                started_at = chain.get("started_at", 0)
                current_task = chain.get("current_task", "")

                # 检查长时间运行的作业链 (30分钟)
                if started_at > 0:
                    chain_age = time.time() - started_at
                    if chain_age > 1800:  # 30分钟
                        alerts.append({
                            "level": "warning",
                            "category": "job_chain",
                            "title": "作业链运行时间过长",
                            "message": f"作业链 {job_id[:8]}... 已运行 {chain_age/60:.1f} 分钟",
                            "timestamp": time.time(),
                            "details": {
                                "job_id": job_id,
                                "current_task": current_task,
                                "age_minutes": chain_age/60
                            }
                        })

            # 检查队列状态
            queue_status = overview.get("queue_status", {})

            for queue_name, queue_info in queue_status.items():
                waiting_tasks = queue_info.get("waiting_tasks", 0)

                # 队列积压警报 (超过10个任务)
                if waiting_tasks > 10:
                    alerts.append({
                        "level": "warning",
                        "category": "queue",
                        "title": f"队列积压: {queue_name}",
                        "message": f"{waiting_tasks} 个任务等待处理",
                        "timestamp": time.time(),
                        "details": {"queue_name": queue_name, "waiting_tasks": waiting_tasks}
                    })

            return alerts
        except Exception as e:
            return []

    def _check_gpu_status(self) -> List[Dict[str, Any]]:
        """检查GPU状态 (简化版)"""
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
                # 检查GPU健康状态
                if not info.get("is_healthy", True):
                    alerts.append({
                        "level": "error",
                        "category": "gpu",
                        "title": f"GPU异常: {gpu_id}",
                        "message": info.get("health_message", "GPU健康检查失败"),
                        "timestamp": time.time(),
                        "details": {"gpu_id": gpu_id}
                    })

                # 检查GPU内存 (阈值提高到5%)
                free_percentage = info.get("free_percentage", 100)
                if free_percentage < 5:
                    alerts.append({
                        "level": "warning",
                        "category": "gpu",
                        "title": f"GPU内存严重不足: {gpu_id}",
                        "message": f"仅剩 {free_percentage:.1f}% 可用内存",
                        "timestamp": time.time(),
                        "details": {"gpu_id": gpu_id, "free_percentage": free_percentage}
                    })

            return alerts
        except Exception as e:
            return []

    def display_notification_center(self, expanded: bool = False):
        """显示通知中心 UI 组件"""
        # 检查新通知
        self.check_for_new_alerts()

        # 获取通知
        notifications = st.session_state.system_notifications

        # 计数通知
        error_count = sum(1 for n in notifications if n.get("level") == "error")
        warning_count = sum(1 for n in notifications if n.get("level") == "warning")
        total_count = error_count + warning_count

        # 显示通知标题
        if total_count > 0:
            title = f"🔔 系统通知 ({total_count})"
            if error_count > 0:
                title += f" | ❌ {error_count}"
            if warning_count > 0:
                title += f" | ⚠️ {warning_count}"
        else:
            title = "🔔 系统通知"

        # 通知中心
        with st.expander(title, expanded=(expanded or error_count > 0)):
            if not notifications:
                st.info("🟢 自触发作业链系统运行正常，无通知")
                return

            # 简化的筛选选项
            filter_level = st.selectbox("筛选级别", ["全部", "错误", "警告"], key="notif_filter")

            # 应用筛选
            filtered_notifications = notifications
            if filter_level == "错误":
                filtered_notifications = [n for n in notifications if n.get("level") == "error"]
            elif filter_level == "警告":
                filtered_notifications = [n for n in notifications if n.get("level") == "warning"]

            # 显示通知 (限制为前5个)
            if not filtered_notifications:
                st.info("没有符合筛选条件的通知")
                return

            for i, notification in enumerate(filtered_notifications[:5]):
                self._render_notification(notification, i)

            if len(filtered_notifications) > 5:
                st.caption(f"还有 {len(filtered_notifications) - 5} 个通知")

            # 清除按钮
            if st.button("清除所有通知", key="clear_notifications"):
                st.session_state.system_notifications = []
                st.success("已清除所有通知")
                st.rerun()

    def _render_notification(self, notification: Dict[str, Any], index: int):
        """渲染单个通知"""
        level = notification.get("level", "info")
        title = notification.get("title", "系统通知")
        message = notification.get("message", "")
        timestamp = notification.get("timestamp", time.time())

        # 格式化时间
        time_str = datetime.datetime.fromtimestamp(timestamp).strftime("%H:%M:%S")

        # 选择颜色和图标
        if level == "error":
            color = "#F44336"
            icon = "❌"
        elif level == "warning":
            color = "#FF9800"
            icon = "⚠️"
        else:
            color = "#2196F3"
            icon = "ℹ️"

        # 创建通知卡片
        with st.container():
            st.markdown(
                f"<div style='padding: 8px; border-left: 4px solid {color}; background-color: rgba(128,128,128,0.1); margin: 4px 0;'>"
                f"<b>{icon} {title}</b><br/>"
                f"{message}<br/>"
                f"<small>{time_str}</small>"
                f"</div>",
                unsafe_allow_html=True
            )

            # 忽略按钮
            if st.button("忽略", key=f"dismiss_{index}"):
                st.session_state.system_notifications.remove(notification)
                st.rerun()


def display_notifications_sidebar(api_url: str, api_key: str, check_interval: int = 120):
    """
    在侧边栏显示通知 (简化版)
    """
    notifications = SystemNotifications(api_url, api_key)

    # 减少检查频率 (2分钟)
    current_time = time.time()
    last_check = st.session_state.get("last_notification_check", 0)

    if current_time - last_check >= check_interval:
        new_alerts = notifications.check_for_new_alerts()
        if new_alerts:
            st.sidebar.warning(f"⚠️ {len(new_alerts)} 个新通知")

    # 侧边栏通知中心
    notifications.display_notification_center(False)


def main_notification_dashboard(api_url: str, api_key: str):
    """
    完整通知仪表板页面 (简化版)
    """
    st.title("系统通知中心")
    st.markdown("自触发作业链和专用Worker监控")

    notifications = SystemNotifications(api_url, api_key)

    # 刷新按钮
    if st.button("检查新通知"):
        with st.spinner("检查中..."):
            new_alerts = notifications.check_for_new_alerts()
            if new_alerts:
                st.success(f"收到 {len(new_alerts)} 个新通知")
            else:
                st.info("没有新通知")

    # 显示通知中心
    notifications.display_notification_center(True)

    # 简化的统计
    all_notifications = st.session_state.get("system_notifications", [])
    if all_notifications:
        st.subheader("通知统计")

        category_counts = {}
        for notification in all_notifications:
            category = notification.get("category", "other")
            category_counts[category] = category_counts.get(category, 0) + 1

        if category_counts:
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Worker通知", category_counts.get("worker", 0))
                st.metric("作业链通知", category_counts.get("job_chain", 0))
            with col2:
                st.metric("GPU通知", category_counts.get("gpu", 0))
                st.metric("队列通知", category_counts.get("queue", 0))