"""
Simple system info page (optional) - src/ui/pages/系统信息.py
"""

import streamlit as st
from src.ui.api_client import api_request
from src.ui.session_init import initialize_session_state

initialize_session_state()

st.title("📊 系统信息")

# Simple system status
try:
    health = api_request("/health", silent=True, timeout=3.0)
    if health:
        st.success("✅ 系统运行正常")
    else:
        st.error("⚠️ 系统状态异常")
except:
    st.error("⚠️ 无法获取系统状态")

# Simple usage stats
try:
    overview = api_request("/job-chains", silent=True)
    if overview:
        stats = overview.get("job_statistics", {})

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("已完成任务", stats.get("completed", 0))
        with col2:
            st.metric("处理中任务", stats.get("processing", 0))
        with col3:
            st.metric("失败任务", stats.get("failed", 0))
except:
    st.info("暂无统计数据")

# Simple info
st.markdown("---")
st.subheader("使用说明")
st.markdown("""
- **查询**: 搜索汽车规格信息
- **上传**: 添加新的汽车资料
- **状态**: 查看处理进度
""")

st.markdown("---")
st.caption("如有问题，请联系系统管理员")

# That's it - no technical details, no backend management options