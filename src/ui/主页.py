"""
Clean, user-focused main page - src/ui/主页.py
"""

import streamlit as st
import time
from src.ui.api_client import api_request
from src.ui.session_init import initialize_session_state

# Initialize session
initialize_session_state()

# Page config
st.set_page_config(
    page_title="汽车规格查询系统",
    page_icon="🚗",
    layout="wide"
)

def check_system_status():
    """Simple system health check"""
    try:
        response = api_request("/health", method="GET", silent=True, timeout=3.0)
        return response is not None
    except:
        return False

def get_recent_stats():
    """Get basic usage statistics"""
    try:
        overview = api_request("/job-chains", method="GET", silent=True)
        if overview:
            stats = overview.get("job_statistics", {})
            return {
                "completed_today": stats.get("completed", 0),
                "processing": stats.get("processing", 0)
            }
    except:
        pass
    return {"completed_today": 0, "processing": 0}

# Main interface
st.title("🚗 汽车规格查询系统")
st.markdown("### 智能汽车规格信息检索平台")

# Simple status check
system_ok = check_system_status()

if not system_ok:
    st.error("⚠️ 系统暂时不可用，请稍后再试")
    st.stop()

# Quick stats
stats = get_recent_stats()
if stats["completed_today"] > 0 or stats["processing"] > 0:
    col1, col2 = st.columns(2)
    with col1:
        st.metric("今日处理完成", stats["completed_today"])
    with col2:
        st.metric("正在处理", stats["processing"])

st.markdown("---")

# Main actions
st.subheader("🔍 开始使用")

action_cols = st.columns(3)

with action_cols[0]:
    st.markdown("#### 查询汽车信息")
    st.markdown("搜索汽车规格、配置和技术参数")
    if st.button("开始查询", use_container_width=True, type="primary"):
        st.switch_page("pages/查询.py")

with action_cols[1]:
    st.markdown("#### 上传资料")
    st.markdown("上传PDF手册或视频资料")
    if st.button("上传资料", use_container_width=True):
        st.switch_page("pages/数据摄取.py")

with action_cols[2]:
    st.markdown("#### 查看处理状态")
    st.markdown("查看文件处理进度")
    if st.button("查看状态", use_container_width=True):
        st.switch_page("pages/后台任务.py")

st.markdown("---")

# Simple feature overview
st.subheader("📋 系统功能")

features = [
    {"icon": "🎬", "title": "视频处理", "desc": "自动提取YouTube、Bilibili等视频中的汽车信息"},
    {"icon": "📄", "title": "文档解析", "desc": "智能分析PDF手册和技术文档"},
    {"icon": "🔍", "title": "智能搜索", "desc": "基于AI的精准信息检索"},
    {"icon": "💬", "title": "自然语言查询", "desc": "用自然语言提问，获得准确答案"}
]

feature_cols = st.columns(2)
for i, feature in enumerate(features):
    col = feature_cols[i % 2]
    with col:
        st.markdown(f"**{feature['icon']} {feature['title']}**")
        st.markdown(feature['desc'])
        st.markdown("")

st.markdown("---")

# Simple help section
with st.expander("💡 使用说明"):
    st.markdown("""
    **查询汽车信息：**
    - 输入车型名称、年份或具体问题
    - 例如："2023年宝马X5的发动机参数"
    
    **上传资料：**
    - 支持PDF文档和视频链接
    - 系统会自动提取其中的汽车信息
    
    **查看状态：**
    - 查看上传文件的处理进度
    - 管理历史查询记录
    """)

# Footer
st.markdown("---")
st.caption("汽车规格查询系统 - 让汽车信息触手可及")