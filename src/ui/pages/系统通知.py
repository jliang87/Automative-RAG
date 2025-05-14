"""
系统通知页面 - 显示所有系统警告和错误
"""

import streamlit as st
import os
from src.ui.components import header
from src.ui.enhanced_error_handling import robust_api_status_indicator

# 导入通知模块
from src.ui.system_notifications import main_notification_dashboard
from src.ui.session_init import initialize_session_state

initialize_session_state()


# 在侧边栏检查 API 状态
with st.sidebar:
    api_available = robust_api_status_indicator(show_detail=True)

# 仅在 API 可用时继续主要内容
if api_available:
    header(
        "系统通知中心",
        "查看和管理重要的系统警告和错误"
    )

    # 显示通知仪表板
    main_notification_dashboard(st.session_state.api_url, st.session_state.api_key)

    # 添加查看系统状态的链接
    st.markdown("---")
    st.info("要查看详细的系统组件状态，请访问[系统管理页面](/系统管理)")

else:
    header(
        "系统通知中心",
        "查看和管理重要的系统警告和错误"
    )

    st.error("无法连接到API服务。请确保API服务正在运行。")
    st.info("您可以使用以下命令启动API服务：")
    st.code("docker-compose up -d api")