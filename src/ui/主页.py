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
    page_title="智能汽车问答系统",
    page_icon="🚗",
    layout="wide"
)

# Main interface
st.title("🚗 智能汽车问答系统")
st.markdown("### 统一智能查询平台")

# Simple system status check
system_ok = simple_health_check()

if not system_ok:
    st.error("⚠️ 系统暂时不可用，请稍后再试")
    if st.button("🔄 重试连接"):
        st.rerun()
    st.stop()

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

# Main action
st.subheader("🔍 开始查询")

col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    st.markdown("### 🧠 智能查询")
    st.markdown("包含所有查询功能：信息总览、智能分析、多角度评估")

    if st.button("🚀 开始查询", use_container_width=True, type="primary"):
        st.switch_page("pages/智能查询.py")

st.markdown("---")

# Secondary actions
st.subheader("📚 数据管理")

action_cols = st.columns(3)

with action_cols[0]:
    st.markdown("#### 📤 上传资料")
    st.markdown("上传PDF手册或视频资料")
    if st.button("上传资料", use_container_width=True):
        st.switch_page("pages/数据摄取.py")

with action_cols[1]:
    st.markdown("#### 📚 浏览文档")
    st.markdown("查看所有已存储的文档")
    if st.button("浏览文档", use_container_width=True):
        st.switch_page("pages/文档浏览.py")

with action_cols[2]:
    st.markdown("#### 📋 查看状态")
    st.markdown("跟踪任务处理进度")
    if st.button("查看状态", use_container_width=True):
        st.switch_page("pages/后台任务.py")

st.markdown("---")

# Feature showcase
st.subheader("✨ 查询模式")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**基础模式**")
    basic_modes = [
        "📌 车辆信息总览 - 查询车辆的各类信息和参数",
        "💡 功能建议 - 评估新功能价值",
        "⚖️ 权衡分析 - 深度利弊对比"
    ]

    for mode in basic_modes:
        st.markdown(f"• {mode}")

with col2:
    st.markdown("**高级模式**")
    advanced_modes = [
        "🧩 场景分析 - 实际使用体验评估",
        "🗣️ 多角色讨论 - 专业观点对话",
        "🔍 用户评论 - 真实反馈摘录"
    ]

    for mode in advanced_modes:
        st.markdown(f"• {mode}")

st.markdown("---")

# Quick examples
st.subheader("💡 查询示例")

example_tabs = st.tabs(["📌 信息总览", "💡 功能建议", "⚖️ 权衡分析"])

with example_tabs[0]:
    facts_examples = [
        "2023年宝马X5的后备箱容积是多少？",
        "特斯拉Model 3的刹车性能怎么样？",
        "奔驰E级有哪些安全配置？"
    ]

    for i, example in enumerate(facts_examples):
        if st.button(example, key=f"facts_example_{i}", use_container_width=True):
            st.session_state.smart_query = example
            st.session_state.smart_mode = "facts"
            st.switch_page("pages/智能查询.py")

with example_tabs[1]:
    features_examples = [
        "是否应该为电动车增加氛围灯功能？",
        "AR抬头显示器值得投资吗？",
        "增加模拟引擎声音对用户体验的影响"
    ]

    for i, example in enumerate(features_examples):
        if st.button(example, key=f"features_example_{i}", use_container_width=True):
            st.session_state.smart_query = example
            st.session_state.smart_mode = "features"
            st.switch_page("pages/智能查询.py")

with example_tabs[2]:
    tradeoffs_examples = [
        "大屏幕中控 vs 传统按键的利弊分析",
        "使用模拟声音 vs 自然静音的权衡",
        "移除物理按键的优缺点分析"
    ]

    for i, example in enumerate(tradeoffs_examples):
        if st.button(example, key=f"tradeoffs_example_{i}", use_container_width=True):
            st.session_state.smart_query = example
            st.session_state.smart_mode = "tradeoffs"
            st.switch_page("pages/智能查询.py")

st.markdown("---")

# Quick access buttons
st.subheader("🚀 快速访问")

quick_cols = st.columns(4)

with quick_cols[0]:
    if st.button("📊 系统状态", use_container_width=True):
        st.switch_page("pages/系统信息.py")

with quick_cols[1]:
    if st.button("📋 后台任务", use_container_width=True):
        st.switch_page("pages/后台任务.py")

with quick_cols[2]:
    if st.button("📚 文档浏览", use_container_width=True):
        st.switch_page("pages/文档浏览.py")

with quick_cols[3]:
    if st.button("📤 上传资料", use_container_width=True):
        st.switch_page("pages/数据摄取.py")

# Footer
st.markdown("---")
st.caption("智能汽车问答系统 - 统一智能查询平台")