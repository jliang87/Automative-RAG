"""
Simplified main page - src/ui/主页.py
Clean, focused main page without unnecessary complexity
"""

import streamlit as st
from src.ui.api_client import (
    api_request,
    simple_health_check,
    get_job_statistics
)
from src.ui.session_init import initialize_session_state

# Initialize session
initialize_session_state()

# Page config
st.set_page_config(
    page_title="汽车规格查询系统",
    page_icon="🚗",
    layout="wide"
)

# Main interface
st.title("🚗 汽车规格查询系统")
st.markdown("### 智能汽车规格信息检索平台")

# Simple system status check
system_ok = simple_health_check()

if not system_ok:
    st.error("⚠️ 系统暂时不可用，请稍后再试")
    if st.button("🔄 重试连接"):
        st.rerun()
    st.stop()
else:
    st.success("✅ 系统运行正常")

st.markdown("---")

# Quick stats
stats = get_job_statistics()
if any(stats.values()):
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("已完成任务", stats.get("completed", 0))
    with col2:
        st.metric("处理中任务", stats.get("processing", 0))
    with col3:
        st.metric("等待任务", stats.get("pending", 0))

st.markdown("---")

# Main actions - Always available (jobs will queue automatically)
st.subheader("🔍 开始使用")

action_cols = st.columns(4)  # Changed from 3 to 4 columns

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
    st.markdown("#### 浏览文档")  # NEW
    st.markdown("查看所有已存储的文档")
    if st.button("浏览文档", use_container_width=True):
        st.switch_page("pages/文档浏览.py")

with action_cols[3]:
    st.markdown("#### 查看处理状态")
    st.markdown("跟踪任务处理进度")
    if st.button("查看状态", use_container_width=True):
        st.switch_page("pages/后台任务.py")

st.markdown("---")

# Feature overview
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

# Quick examples
st.subheader("💡 查询示例")

example_cols = st.columns(2)

with example_cols[0]:
    st.markdown("**汽车规格查询:**")
    examples = [
        "2023年宝马X5的发动机参数",
        "特斯拉Model 3的续航里程",
        "奔驰E级的安全配置"
    ]

    for example in examples:
        if st.button(example, key=f"example_{example[:10]}", use_container_width=True):
            st.session_state.quick_query = example
            st.switch_page("pages/查询.py")

with example_cols[1]:
    st.markdown("**系统信息:**")
    info_buttons = [
        ("📊 查看系统状态", "pages/系统信息.py"),
        ("📋 查看任务进度", "pages/后台任务.py"),
        ("📤 上传新资料", "pages/数据摄取.py")
    ]

    for button_text, page_path in info_buttons:
        if st.button(button_text, key=f"info_{button_text[:2]}", use_container_width=True):
            st.switch_page(page_path)

st.markdown("---")

# Usage instructions
with st.expander("📖 使用说明"):
    st.markdown("""
    **查询汽车信息：**
    - 输入车型名称、年份或具体问题
    - 例如："2023年宝马X5的发动机参数"
    
    **上传资料：**
    - 支持YouTube、Bilibili视频链接
    - 支持PDF文档（自动OCR识别）
    - 支持直接输入文字内容
    
    **查看状态：**
    - 跟踪上传文件的处理进度
    - 查看历史查询记录
    - 管理任务状态
    
    **系统信息：**
    - 查看系统健康状况
    - 监控资源使用情况
    - 了解服务可用性
    """)

# Footer
st.markdown("---")
st.caption("汽车规格查询系统 - 让汽车信息触手可及")