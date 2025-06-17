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

# Main actions - Enhanced with smart query
st.subheader("🔍 开始使用")

# Two rows of actions for better organization
query_cols = st.columns(2)

with query_cols[0]:
    st.markdown("#### 🔍 基础查询")
    st.markdown("简单快速的汽车信息查询")
    if st.button("开始基础查询", use_container_width=True, type="primary"):
        st.switch_page("pages/查询.py")

with query_cols[1]:
    st.markdown("#### 🧠 智能分析")  # NEW
    st.markdown("多角度深度分析，适合专业用户")
    if st.button("开始智能分析", use_container_width=True, type="primary"):
        st.switch_page("pages/智能查询.py")

st.markdown("---")

# Secondary actions
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

# Feature showcase - Enhanced with smart features
st.subheader("✨ 系统特色")

# Basic features
col1, col2 = st.columns(2)

with col1:
    st.markdown("**📋 基础功能**")
    basic_features = [
        "🎬 视频处理 - 自动提取视频中的汽车信息",
        "📄 文档解析 - 智能分析PDF手册和技术文档",
        "🔍 快速搜索 - 基于向量相似度的精准检索",
        "💬 自然语言 - 用自然语言提问获得答案"
    ]

    for feature in basic_features:
        st.markdown(f"• {feature}")

with col2:
    st.markdown("**🧠 智能分析模式**")  # NEW
    smart_features = [
        "📌 事实验证 - 严格基于文档的规格查询",
        "💡 功能建议 - 评估新功能的可行性和价值",
        "⚖️ 权衡分析 - 深度分析设计选择的利弊",
        "🗣️ 多角色讨论 - 模拟不同角色的专业观点"
    ]

    for feature in smart_features:
        st.markdown(f"• {feature}")

st.markdown("---")

# Quick examples - Enhanced with mode examples
st.subheader("💡 使用示例")

example_tabs = st.tabs(["🔍 基础查询", "🧠 智能分析"])

with example_tabs[0]:
    st.markdown("**适合快速获取信息的查询：**")
    basic_examples = [
        "2023年宝马X5的发动机参数",
        "特斯拉Model 3的续航里程",
        "奔驰E级的安全配置有哪些"
    ]

    example_col1, example_col2 = st.columns(2)

    for i, example in enumerate(basic_examples):
        col = example_col1 if i % 2 == 0 else example_col2
        with col:
            if st.button(example, key=f"basic_example_{i}", use_container_width=True):
                st.session_state.quick_query = example
                st.switch_page("pages/查询.py")

with example_tabs[1]:
    st.markdown("**适合深度分析的查询：**")
    smart_examples = [
        "📌 2023年奔驰E级的后备箱容积规格验证",
        "💡 是否应该为电动车增加氛围灯功能？",
        "⚖️ 大屏幕中控 vs 传统按键的利弊分析",
        "🗣️ 不同角色如何看待自动驾驶技术？"
    ]

    smart_col1, smart_col2 = st.columns(2)

    for i, example in enumerate(smart_examples):
        col = smart_col1 if i % 2 == 0 else smart_col2
        with col:
            if st.button(example, key=f"smart_example_{i}", use_container_width=True):
                # Extract mode from icon
                mode_map = {"📌": "facts", "💡": "features", "⚖️": "tradeoffs", "🗣️": "debate"}
                icon = example[:2]
                mode = mode_map.get(icon, "facts")

                st.session_state.smart_query = example[3:]  # Remove icon
                st.session_state.smart_mode = mode
                st.switch_page("pages/智能查询.py")

st.markdown("---")

# Mode comparison section
with st.expander("🤔 如何选择查询模式？"):
    st.markdown("""
    **🔍 基础查询 - 适合以下情况：**
    - 需要快速获取车辆信息
    - 查询具体的技术参数
    - 简单的对比和说明
    - 日常使用的基本需求
    
    **🧠 智能分析 - 适合以下情况：**
    - 需要深度分析和专业见解
    - 产品决策和技术选型
    - 多角度评估和权衡
    - 用户体验和场景分析
    - 需要结构化的专业报告
    
    **📊 各模式特点对比：**
    
    | 特性 | 基础查询 | 智能分析 |
    |------|----------|----------|
    | 响应速度 | 快速 | 较慢（更深入） |
    | 分析深度 | 基础 | 专业深度 |
    | 输出结构 | 简单回答 | 结构化分析 |
    | 适用人群 | 一般用户 | 专业用户 |
    | 使用场景 | 日常查询 | 决策支持 |
    """)

# Usage instructions
with st.expander("📖 系统使用说明"):
    st.markdown("""
    **查询汽车信息：**
    - 🔍 基础查询：输入车型名称、年份或具体问题
    - 🧠 智能分析：选择分析模式，进行深度专业分析
    - 📋 支持筛选条件：品牌、年份、车型等
    
    **上传资料：**
    - 🎬 支持YouTube、Bilibili视频链接
    - 📄 支持PDF文档（自动OCR识别）
    - ✍️ 支持直接输入文字内容
    
    **查看状态：**
    - 📋 跟踪上传文件的处理进度
    - 📊 查看历史查询记录
    - 🔧 管理任务状态
    
    **系统信息：**
    - 📈 查看系统健康状况
    - 💻 监控资源使用情况
    - ⚙️ 了解服务可用性
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
st.caption("汽车规格查询系统 - 让汽车信息触手可及 | 现在支持智能分析模式")
st.caption("💡 提示：专业用户推荐使用智能分析模式获得更深入的见解")