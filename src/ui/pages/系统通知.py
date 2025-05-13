"""
系统通知页面 - 显示所有系统警告和错误
"""

import streamlit as st
import os
from src.ui.components import header
from src.ui.enhanced_worker_status import enhanced_worker_status
from src.ui.enhanced_error_handling import robust_api_status_indicator

# API 配置
API_URL = os.environ.get("API_URL", "http://localhost:8000")
API_KEY = os.environ.get("API_KEY", "default-api-key")

# 会话状态初始化
if "api_url" not in st.session_state:
    st.session_state.api_url = API_URL
if "api_key" not in st.session_state:
    st.session_state.api_key = API_KEY
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

# 导入通知模块
from src.ui.system_notifications import main_notification_dashboard

# 渲染通知中心页面
# Check API status in sidebar
with st.sidebar:
    api_available = robust_api_status_indicator(show_detail=True)

# Only proceed with main content if API is available
if api_available:
    header(
        "系统通知中心",
        "查看和管理重要的系统警告和错误"
    )

    # 显示通知仪表板
    main_notification_dashboard(st.session_state.api_url, st.session_state.api_key)

    # Add enhanced worker status display as well
    st.subheader("Worker Status")
    enhanced_worker_status()
else:
    header(
        "系统通知中心",
        "查看和管理重要的系统警告和错误"
    )

    st.error("无法连接到API服务。请确保API服务正在运行。")
    st.info("您可以使用以下命令启动API服务：")
    st.code("docker-compose up -d api")