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
st.markdown("### 统一智能查询平台 v2.0")

# System upgrade banner
st.success("🔄 **系统已升级** - 统一查询架构，智能分析包含所有查询功能")

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

# Main action - Unified query system
st.subheader("🔍 开始查询")

# Single main action for unified system
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    st.markdown("### 🧠 统一智能查询")
    st.markdown("**包含所有查询功能：Facts验证、智能分析、多角度评估**")
    st.markdown("📌 Facts模式为默认，快速验证规格参数")
    st.markdown("🧠 多种分析模式，满足不同深度需求")

    if st.button("🚀 开始查询", use_container_width=True, type="primary",
                 help="进入统一查询系统，支持Facts验证和智能分析"):
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

# Feature showcase - Unified system
st.subheader("✨ 统一查询系统特色")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**🔍 多模式查询**")
    query_modes = [
        "📌 Facts模式 - 基于文档的规格验证 (默认)",
        "💡 功能建议 - 评估新功能价值",
        "⚖️ 权衡分析 - 深度利弊对比",
        "🧩 场景分析 - 实际使用体验评估",
        "🗣️ 多角色讨论 - 专业观点对话",
        "🔍 用户评论 - 真实反馈摘录"
    ]

    for mode in query_modes:
        st.markdown(f"• {mode}")

with col2:
    st.markdown("**🚀 系统功能**")
    system_features = [
        "🎬 视频处理 - 自动提取视频中的汽车信息",
        "📄 文档解析 - 智能分析PDF手册和技术文档",
        "🔍 精准检索 - 基于向量相似度的智能搜索",
        "💬 自然语言 - 支持中文自然语言查询",
        "🔄 统一架构 - 一致的处理流程和响应格式",
        "⚡ 智能优化 - 根据查询类型自动优化参数"
    ]

    for feature in system_features:
        st.markdown(f"• {feature}")

st.markdown("---")

# Quick examples - All in one place
st.subheader("💡 查询示例")

example_tabs = st.tabs(["📌 Facts验证", "💡 功能建议", "⚖️ 权衡分析", "🗣️ 多角色讨论"])

with example_tabs[0]:
    st.markdown("**快速验证规格信息：**")
    facts_examples = [
        "2023年宝马X5的后备箱容积是多少？",
        "特斯拉Model 3的充电速度参数",
        "奔驰E级有哪些安全配置？"
    ]

    for i, example in enumerate(facts_examples):
        if st.button(example, key=f"facts_example_{i}", use_container_width=True):
            st.session_state.smart_query = example
            st.session_state.smart_mode = "facts"
            st.switch_page("pages/智能查询.py")

with example_tabs[1]:
    st.markdown("**功能价值评估：**")
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
    st.markdown("**深度利弊分析：**")
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

with example_tabs[3]:
    st.markdown("**多角色专业讨论：**")
    debate_examples = [
        "不同角色如何看待自动驾驶技术？",
        "产品团队对电池技术路线的观点",
        "关于车内空间设计的专业讨论"
    ]

    for i, example in enumerate(debate_examples):
        if st.button(example, key=f"debate_example_{i}", use_container_width=True):
            st.session_state.smart_query = example
            st.session_state.smart_mode = "debate"
            st.switch_page("pages/智能查询.py")

st.markdown("---")

# Mode guidance
with st.expander("🤔 如何选择查询模式？"):
    st.markdown("""
    **📌 Facts模式 (默认推荐) - 适合：**
    - ✅ 验证具体的车辆规格参数
    - ✅ 查询确切的技术数据和配置信息
    - ✅ 基于文档的事实验证
    - ✅ 日常使用的快速查询需求
    - ⏱️ 响应时间：~10秒

    **💡 功能建议模式 - 适合：**
    - 🎯 评估是否应该添加某项功能
    - 🎯 产品决策支持
    - 🎯 功能价值分析
    - ⏱️ 响应时间：~30秒

    **⚖️ 权衡分析模式 - 适合：**
    - 🔍 深度利弊对比分析
    - 🔍 设计选择评估
    - 🔍 技术方案对比
    - ⏱️ 响应时间：~45秒

    **🧩 场景分析模式 - 适合：**
    - 🎭 实际使用场景评估
    - 🎭 用户体验分析
    - 🎭 功能在特定场景下的表现
    - ⏱️ 响应时间：~40秒

    **🗣️ 多角色讨论模式 - 适合：**
    - 👥 需要多个专业角度的观点
    - 👥 复杂决策的全面评估
    - 👥 产品、技术、用户多维度分析
    - ⏱️ 响应时间：~50秒

    **🔍 用户评论模式 - 适合：**
    - 💬 获取真实用户反馈
    - 💬 市场研究和用户洞察
    - 💬 了解实际使用体验
    - ⏱️ 响应时间：~20秒

    **💡 建议：** 新用户推荐从Facts模式开始，熟悉后可尝试其他高级分析模式。
    """)

# Usage instructions
with st.expander("📖 系统使用说明"):
    st.markdown("""
    **统一查询系统使用指南：**

    **🚀 开始查询：**
    1. 点击"开始查询"进入统一查询界面
    2. 选择合适的分析模式（Facts为默认推荐）
    3. 输入您的问题
    4. 可选择筛选条件（品牌、年份等）
    5. 点击开始分析

    **📝 查询技巧：**
    - 使用具体的车型名称和年份
    - 描述清楚您想了解的方面
    - Facts模式适合大部分日常查询
    - 复杂决策建议使用高级分析模式

    **📤 上传资料：**
    - 🎬 支持YouTube、Bilibili视频链接
    - 📄 支持PDF文档（自动OCR识别）
    - ✍️ 支持直接输入文字内容

    **📊 查看状态：**
    - 📋 跟踪任务处理进度
    - 📈 查看系统健康状况
    - 🔧 管理历史记录
    """)

# Quick access buttons
st.markdown("---")
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
st.caption("汽车规格查询系统 v2.0 - 统一智能查询，让汽车信息触手可及")
st.caption("🧠 智能查询包含所有功能：从快速Facts验证到深度专业分析")