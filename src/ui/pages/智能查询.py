import streamlit as st
import time
import json
from typing import Dict, Any, Optional
from src.ui.api_client import api_request
from src.ui.session_init import initialize_session_state

# UPDATED: Import unified validation display instead of separate components
from src.ui.components.validation_display import (
    render_unified_validation_display,
    render_quick_validation_badge,
    render_validation_help,
    render_real_time_validation_feedback
)
from src.ui.components.metadata_display import (
    render_embedded_metadata_display,
    render_metadata_summary_card,
    add_metadata_display_to_sources,
    EmbeddedMetadataExtractor
)

initialize_session_state()

# Query mode configurations (simplified)
QUERY_MODES = {
    "facts": {
        "icon": "📌",
        "name": "车辆信息总览",
        "description": "查询车辆的各类信息和参数",
        "two_layer": True,
        "is_default": True,
        "examples": [
            "2023年宝马X5的后备箱容积是多少？",
            "特斯拉Model 3的刹车性能怎么样？",
            "奔驰E级有哪些安全配置？"
        ],
        "validation_priority": "high"  # High priority for validation display
    },
    "features": {
        "icon": "💡",
        "name": "功能建议",
        "description": "评估是否应该添加某项功能",
        "two_layer": True,
        "examples": [
            "是否应该为电动车增加氛围灯功能？",
            "增加模拟引擎声音对用户体验的影响",
            "AR抬头显示器值得投资吗？"
        ],
        "validation_priority": "medium"
    },
    "tradeoffs": {
        "icon": "⚖️",
        "name": "权衡分析",
        "description": "分析设计选择的优缺点",
        "two_layer": True,
        "examples": [
            "使用模拟声音 vs 自然静音的利弊",
            "移除物理按键的优缺点分析",
            "大屏幕 vs 传统仪表盘的对比"
        ],
        "validation_priority": "medium"
    },
    "scenarios": {
        "icon": "🧩",
        "name": "场景分析",
        "description": "评估功能在实际使用场景中的表现",
        "two_layer": True,
        "examples": [
            "长途旅行时这个功能如何表现？",
            "家庭用户在日常通勤中的体验如何？",
            "寒冷气候下的性能表现分析"
        ],
        "validation_priority": "medium"
    },
    "debate": {
        "icon": "🗣️",
        "name": "多角色讨论",
        "description": "模拟不同角色的观点和讨论",
        "two_layer": False,
        "examples": [
            "产品经理、工程师和用户代表如何看待自动驾驶功能？",
            "不同团队对电池技术路线的观点",
            "关于车内空间设计的多方讨论"
        ],
        "validation_priority": "low"
    },
    "quotes": {
        "icon": "🔍",
        "name": "用户评论",
        "description": "提取相关的用户评论和反馈",
        "two_layer": False,
        "examples": [
            "用户对续航里程的真实评价",
            "关于内饰质量的用户反馈",
            "充电体验的用户评论摘录"
        ],
        "validation_priority": "low"
    }
}


def submit_unified_query(query_text: str, mode: str, filters: Optional[Dict] = None) -> Optional[str]:
    """Submit unified query with enhanced error handling."""
    try:
        unified_data = {
            "query": query_text,
            "metadata_filter": filters,
            "top_k": 8,
            "query_mode": mode,
            "prompt_template": None
        }

        result = api_request(
            endpoint="/query",
            method="POST",
            data=unified_data
        )

        if result and "job_id" in result:
            return result["job_id"]
        else:
            st.error("查询提交失败：服务器未返回有效响应")
            return None

    except Exception as e:
        st.error(f"查询提交时发生错误: {str(e)}")
        return None


def get_query_result(job_id: str) -> Optional[Dict]:
    """Get unified query results with enhanced error handling."""
    try:
        return api_request(f"/query/results/{job_id}", method="GET")
    except Exception as e:
        st.error(f"获取查询结果时发生错误: {str(e)}")
        return None


def display_enhanced_results(result: Dict[str, Any], mode: str):
    """Display results with unified validation system."""

    answer = result.get("answer", "")
    if not answer:
        st.warning("未获得查询结果")
        return

    # Main answer display
    st.markdown("### 📋 分析结果")

    # UPDATED: Quick validation badge at the top using unified system
    validation_badge = render_quick_validation_badge(result)
    st.markdown(f"**验证状态**: {validation_badge}")

    # Display the answer based on mode
    mode_info = QUERY_MODES.get(mode, {})

    if mode_info.get("two_layer"):
        display_two_layer_result(result, mode)
    elif mode == "debate":
        display_debate_result(answer)
    elif mode == "quotes":
        display_quotes_result(answer)
    else:
        st.markdown(answer)

    validation_priority = mode_info.get("validation_priority", "medium")

    if validation_priority in ["high", "medium"]:
        st.markdown("---")
        # Check if we're about to enter an expander context
        # We'll modify the validation display to be aware of the sources display
        render_unified_validation_display(result)

        # Enhanced sources display - pass context that we might be in expander
        display_enhanced_sources(result, in_expander=False)
    else:
        # For low priority modes, just show sources normally
        display_enhanced_sources(result, in_expander=False)


def display_two_layer_result(result: Dict[str, Any], mode: str):
    """Display results with two-layer structure for enhanced modes."""
    answer = result.get("answer", "")
    analysis_structure = result.get("analysis_structure")

    if analysis_structure and isinstance(analysis_structure, dict):
        if mode == "facts":
            if "【实证分析】" in analysis_structure:
                st.subheader("📊 基于文档的实证分析")
                with st.container():
                    st.info(analysis_structure["【实证分析】"])

            if "【策略推理】" in analysis_structure:
                st.subheader("🧠 专业推理补充")
                with st.container():
                    st.warning(analysis_structure["【策略推理】"])
                    st.caption("⚠️ 此部分为AI推理，请结合实证分析参考")

        elif mode == "features":
            if "【实证分析】" in analysis_structure:
                st.subheader("📊 文档实证分析")
                st.info(analysis_structure["【实证分析】"])

            if "【策略推理】" in analysis_structure:
                st.subheader("💡 功能策略推理")
                st.success(analysis_structure["【策略推理】"])

        elif mode == "tradeoffs":
            if "【文档支撑】" in analysis_structure:
                st.subheader("📋 文档支撑信息")
                st.info(analysis_structure["【文档支撑】"])

            if "【利弊分析】" in analysis_structure:
                st.subheader("⚖️ 权衡利弊分析")
                st.warning(analysis_structure["【利弊分析】"])

        elif mode == "scenarios":
            if "【文档场景】" in analysis_structure:
                st.subheader("📖 文档场景信息")
                st.info(analysis_structure["【文档场景】"])

            if "【场景推理】" in analysis_structure:
                st.subheader("🎯 场景应用推理")
                st.success(analysis_structure["【场景推理】"])
        else:
            st.markdown(answer)
    else:
        st.markdown("**📋 分析结果:**")
        st.info(answer)


def display_debate_result(answer: str):
    """Display debate-style results with multiple perspectives."""
    st.subheader("🗣️ 多角色讨论")

    roles = ["产品经理观点", "工程师观点", "用户代表观点"]
    sections = {}

    current_role = None
    current_content = []

    lines = answer.split('\n')

    for line in lines:
        line = line.strip()

        found_role = None
        for role in roles:
            if role in line or f"**{role}" in line:
                found_role = role
                break

        if found_role:
            if current_role and current_content:
                sections[current_role] = '\n'.join(current_content).strip()

            current_role = found_role
            current_content = []
        elif current_role and line:
            current_content.append(line)

    if current_role and current_content:
        sections[current_role] = '\n'.join(current_content).strip()

    role_icons = {
        "产品经理观点": "👔",
        "工程师观点": "🔧",
        "用户代表观点": "👥"
    }

    if sections:
        for role, content in sections.items():
            if content:
                icon = role_icons.get(role, "💬")
                st.markdown(f"### {icon} {role}")
                st.markdown(content)
                st.markdown("---")
    else:
        st.markdown(answer)


def display_quotes_result(answer: str):
    """Display user quotes in a structured format."""
    st.subheader("💬 用户评论摘录")

    import re
    quote_pattern = r'【来源\d+】[：:]"([^"]+)"'
    quotes = re.findall(quote_pattern, answer)

    if quotes:
        for i, quote in enumerate(quotes, 1):
            with st.container():
                st.markdown(f"**来源 {i}:**")
                st.quote(quote)
                st.markdown("")
    else:
        st.markdown(answer)


def display_enhanced_sources(result: Dict[str, Any], in_expander: bool = False):
    """Display sources with enhanced validation and metadata analysis."""

    documents = result.get("documents", [])
    if not documents:
        return

    st.markdown("---")
    st.subheader(f"📚 参考来源 ({len(documents)} 个)")

    # Add metadata quality overview
    st.markdown("#### 🔍 元数据质量概览")
    quality_col1, quality_col2, quality_col3, quality_col4 = st.columns(4)

    # Analyze metadata quality across all documents
    extractor = EmbeddedMetadataExtractor()
    total_docs = len(documents)
    docs_with_embedded = 0
    docs_with_vehicle = 0
    avg_metadata_injection = 0

    for doc in documents:
        content = doc.get("content", "")
        metadata = doc.get("metadata", {})

        embedded_metadata, _ = extractor.extract_embedded_metadata(content)
        if embedded_metadata:
            docs_with_embedded += 1

        if metadata.get('has_vehicle_info'):
            docs_with_vehicle += 1

        if metadata.get('metadata_injected'):
            avg_metadata_injection += 1

    with quality_col1:
        st.metric("含嵌入元数据", f"{docs_with_embedded}/{total_docs}")

    with quality_col2:
        st.metric("车辆信息检测", f"{docs_with_vehicle}/{total_docs}")

    with quality_col3:
        injection_rate = (avg_metadata_injection / total_docs * 100) if total_docs > 0 else 0
        st.metric("注入成功率", f"{injection_rate:.0f}%")

    with quality_col4:
        if docs_with_embedded > total_docs * 0.8:
            st.success("质量优秀")
        elif docs_with_embedded > total_docs * 0.5:
            st.warning("质量良好")
        else:
            st.error("质量待改进")

    # Only create expander if we're not already inside one
    if not in_expander:
        with st.expander("查看所有来源及元数据", expanded=False):
            _render_sources_content_with_metadata(documents)
    else:
        _render_sources_content_with_metadata(documents)


def _render_sources_content_with_metadata(documents):
    """Render the actual sources content with metadata display."""

    for i, doc in enumerate(documents):
        metadata = doc.get("metadata", {})
        relevance = doc.get("relevance_score", 0)

        # Enhanced source display with validation status
        title = metadata.get("title", f"文档 {i + 1}")
        source_type = metadata.get("source", "unknown")

        # Source quality indicator with metadata awareness
        extractor = EmbeddedMetadataExtractor()
        embedded_metadata, _ = extractor.extract_embedded_metadata(doc.get("content", ""))
        has_good_metadata = len(embedded_metadata) > 2  # Has substantial metadata

        validation_status = metadata.get("validation_status", "unknown")
        automotive_warnings = metadata.get("automotive_warnings", [])

        # Enhanced quality assessment
        if validation_status == "validated" and relevance > 0.8 and has_good_metadata:
            st.success(f"**来源 {i + 1}** 🟢: {title[:60]}...")
            st.caption("✅ 高质量来源，已通过验证，元数据完整")
        elif validation_status == "has_warnings" or automotive_warnings:
            st.warning(f"**来源 {i + 1}** 🟡: {title[:60]}...")
            st.caption("⚠️ 包含需注意信息，请参考验证详情")
        elif relevance > 0.6 and has_good_metadata:
            st.info(f"**来源 {i + 1}** 🟡: {title[:60]}...")
            st.caption("📋 中等质量来源，元数据较好")
        elif has_good_metadata:
            st.info(f"**来源 {i + 1}** 🔵: {title[:60]}...")
            st.caption("📊 元数据丰富，但相关度一般")
        else:
            st.error(f"**来源 {i + 1}** 🔴: {title[:60]}...")
            st.caption("❗ 低质量来源，元数据缺失")

        # Basic source details
        col1, col2 = st.columns([1, 1])
        with col1:
            st.caption(f"**来源类型**: {source_type}")
            st.caption(f"**相关度**: {relevance:.1%}")
        with col2:
            if metadata.get("author"):
                st.caption(f"**作者**: {metadata['author']}")
            if metadata.get("published_date"):
                st.caption(f"**发布**: {metadata['published_date']}")

        # Show validation warnings
        if automotive_warnings:
            st.caption("⚠️ **验证提醒**:")
            for warning in automotive_warnings[:2]:
                st.caption(f"  • {warning}")
            if len(automotive_warnings) > 2:
                st.caption(f"  • 还有 {len(automotive_warnings) - 2} 项提醒...")

            # NEW: Add metadata summary card
            st.markdown("**🏷️ 元数据摘要:**")
            # FIXED: Add unique_id to prevent key conflicts
            unique_id = f"query_source_{i}"
            render_metadata_summary_card(doc, compact=True, unique_id=unique_id)

        # Content and detailed metadata display
        if doc.get("content"):
            button_key = f"btn_show_content_{i}"
            metadata_key = f"btn_show_metadata_{i}"
            state_key = f"content_visible_{i}"
            metadata_state_key = f"metadata_visible_{i}"

            # Buttons for content and metadata
            btn_col1, btn_col2 = st.columns(2)

            with btn_col1:
                if st.button(f"查看来源 {i + 1} 内容", key=button_key):
                    current_state = st.session_state.get(state_key, False)
                    st.session_state[state_key] = not current_state
                    st.rerun()

            with btn_col2:
                if st.button(f"查看来源 {i + 1} 详细元数据", key=metadata_key):
                    current_state = st.session_state.get(metadata_state_key, False)
                    st.session_state[metadata_state_key] = not current_state
                    st.rerun()

            # Show content if state is True
            if st.session_state.get(state_key, False):
                content_preview = doc['content'][:300] + "..." if len(doc['content']) > 300 else doc['content']
                st.text_area(
                    f"来源 {i + 1} 内容预览",
                    content_preview,
                    height=100,
                    disabled=True,
                    key=f"content_display_{i}"
                )

                # Show detailed metadata if state is True
                if st.session_state.get(metadata_state_key, False):
                    with st.container():
                        # FIXED: Add unique_id to prevent key conflicts
                        unique_id = f"query_metadata_{i}"
                        render_embedded_metadata_display(doc, show_full_content=False, unique_id=unique_id)

        st.markdown("---")
        

# Main interface
st.title("🧠 智能查询")
st.markdown("带有专业验证的统一查询平台")

# System status check
try:
    health_response = api_request("/health", silent=True, timeout=3.0)
    if not health_response or health_response.get("status") != "healthy":
        st.warning("⚠️ 系统状态异常，查询结果可能不准确")
except:
    st.error("❌ 无法连接到服务器，请稍后重试")
    st.stop()

# Mode selection
st.subheader("📋 选择查询模式")

if 'selected_mode' not in st.session_state:
    st.session_state.selected_mode = "facts"

if hasattr(st.session_state, 'smart_mode'):
    st.session_state.selected_mode = st.session_state.smart_mode
    del st.session_state.smart_mode

# Display modes in a grid with validation indicators
mode_cols = st.columns(3)

for i, (mode_key, mode_info) in enumerate(QUERY_MODES.items()):
    col = mode_cols[i % 3]

    with col:
        is_selected = st.session_state.get('selected_mode') == mode_key
        button_type = "primary" if is_selected else "secondary"

        # UPDATED: Add validation priority indicator
        validation_indicator = ""
        if mode_info.get("validation_priority") == "high":
            validation_indicator = " 🛡️"
        elif mode_info.get("validation_priority") == "medium":
            validation_indicator = " 🔍"

        button_text = f"{mode_info['icon']} {mode_info['name']}{validation_indicator}"
        if is_selected:
            button_text = f"✅ {button_text}"

        if st.button(
                button_text,
                key=f"mode_{mode_key}",
                use_container_width=True,
                help=mode_info['description'],
                type=button_type
        ):
            st.session_state.selected_mode = mode_key
            st.rerun()

# Show selected mode info
if st.session_state.get('selected_mode'):
    mode = st.session_state.selected_mode
    mode_info = QUERY_MODES[mode]

    st.markdown("---")

    # Mode description with validation info
    selected_mode_info = QUERY_MODES[mode]  # Get the actual selected mode info
    validation_priority = selected_mode_info.get("validation_priority",
                                                 "medium")  # Use selected_mode_info instead of mode_info

    if validation_priority == "high":
        st.info(f"🛡️ **{selected_mode_info['name']}** - 此模式包含高级专业验证功能")
    elif validation_priority == "medium":
        st.info(f"🔍 **{selected_mode_info['name']}** - 此模式包含分析验证功能")
    else:
        st.info(f"📝 **{selected_mode_info['name']}** - {selected_mode_info['description']}")

    # Query input section
    st.subheader("💭 输入您的问题")

    # Show examples for the selected mode
    with st.expander(f"💡 {mode_info['name']} 示例"):
        for example in mode_info['examples']:
            if st.button(example, key=f"example_{example[:20]}", use_container_width=True):
                st.session_state.example_query = example

    # Handle pre-filled queries
    default_query = ""
    if hasattr(st.session_state, 'smart_query'):
        default_query = st.session_state.smart_query
        del st.session_state.smart_query
    elif hasattr(st.session_state, 'example_query'):
        default_query = st.session_state.example_query
        del st.session_state.example_query

    query = st.text_area(
        "请输入您的问题",
        value=default_query,
        placeholder=f"例如：{mode_info['examples'][0]}",
        height=100
    )

    # UPDATED: Real-time validation feedback for high-priority modes
    if validation_priority == "high" and query.strip():
        render_real_time_validation_feedback(query)

    # Filters
    with st.expander("🔧 筛选条件（可选）"):
        filter_col1, filter_col2 = st.columns(2)

        with filter_col1:
            manufacturer = st.selectbox(
                "品牌",
                ["", "宝马", "奔驰", "奥迪", "丰田", "本田", "特斯拉", "大众"],
                key=f"manufacturer_{mode}"
            )

        with filter_col2:
            year = st.selectbox(
                "年份",
                [""] + [str(y) for y in range(2024, 2018, -1)],
                key=f"year_{mode}"
            )

    # Build filters
    filters = {}
    if manufacturer:
        filters["manufacturer"] = manufacturer
    if year:
        filters["year"] = int(year)

    # Submit button
    submit_text = f"🚀 开始{mode_info['name']}"

    if st.button(
            submit_text,
            type="primary",
            disabled=not query.strip(),
            use_container_width=True
    ):
        if query.strip():
            with st.spinner(f"正在进行{mode_info['name']}..."):
                job_id = submit_unified_query(query.strip(), mode, filters if filters else None)

                if job_id:
                    st.session_state.current_job_id = job_id
                    st.session_state.query_text = query.strip()
                    st.session_state.query_mode = mode
                    st.session_state.query_submitted_at = time.time()
                    st.success(f"✅ {mode_info['name']}已提交，任务ID: {job_id[:8]}...")
                    st.rerun()

else:
    st.info("👆 请选择一个查询模式开始分析")

    # Quick start recommendations with validation info
    st.markdown("### 🚀 快速开始")
    rec_col1, rec_col2 = st.columns(2)

    with rec_col1:
        st.markdown("**新用户推荐：**")
        if st.button("📌 开始信息总览 🛡️", type="primary", use_container_width=True,
                     help="包含高级专业验证功能"):
            st.session_state.selected_mode = "facts"
            st.rerun()

    with rec_col2:
        st.markdown("**专业用户推荐：**")
        if st.button("💡 开始功能建议 ✅", use_container_width=True,
                     help="包含基础专业验证功能"):
            st.session_state.selected_mode = "features"
            st.rerun()

# Enhanced results section
if hasattr(st.session_state, 'current_job_id') and st.session_state.current_job_id:
    job_id = st.session_state.current_job_id
    query_mode = getattr(st.session_state, 'query_mode', 'facts')
    mode_info = QUERY_MODES[query_mode]

    st.markdown("---")

    # Rate-limited result checking
    if 'last_result_check' not in st.session_state:
        st.session_state.last_result_check = 0

    current_time = time.time()
    if current_time - st.session_state.last_result_check > 3:
        result = get_query_result(job_id)
        st.session_state.last_query_result = result
        st.session_state.last_result_check = current_time
    else:
        result = st.session_state.get('last_query_result')

    if result:
        status = result.get("status", "")

        if status == "completed":
            st.success("✅ 分析完成！")

            # UPDATED: Display results with unified validation system
            display_enhanced_results(result, query_mode)

            # Enhanced action buttons
            st.markdown("---")
            action_col1, action_col2, action_col3, action_col4 = st.columns(4)

            with action_col1:
                if st.button("🔄 新的查询", key="new_analysis"):
                    for key in ['current_job_id', 'query_text', 'last_query_result']:
                        if key in st.session_state:
                            del st.session_state[key]
                    st.rerun()

            with action_col2:
                if st.button("🔀 切换模式", key="switch_mode"):
                    st.session_state.example_query = st.session_state.get('query_text', '')
                    for key in ['current_job_id', 'last_query_result']:
                        if key in st.session_state:
                            del st.session_state[key]
                    st.rerun()

            with action_col3:
                if st.button("📋 查看详情", key="view_job_details"):
                    st.session_state.selected_job_id = job_id
                    st.switch_page("pages/后台任务.py")

            with action_col4:
                if st.button("🛡️ 验证说明", key="validation_help"):
                    st.session_state.show_validation_help = True
                    st.rerun()

        elif status == "failed":
            st.error("❌ 分析失败")
            error_msg = result.get("answer", "未知错误")
            st.error(f"错误信息: {error_msg}")

        else:
            st.info("⏳ 正在分析中...")
            progress_msg = result.get("answer", "正在处理您的查询...")
            st.info(progress_msg)

            if st.button("🔄 刷新状态", key="refresh_status"):
                st.session_state.last_result_check = 0
                st.rerun()
    else:
        st.error("❌ 无法获取分析状态")

# UPDATED: Validation help modal using unified system
if st.session_state.get('show_validation_help', False):
    render_validation_help()

    if st.button("关闭说明", key="close_help"):
        st.session_state.show_validation_help = False
        st.rerun()

# Navigation
st.markdown("---")
nav_cols = st.columns(4)

with nav_cols[0]:
    if st.button("📤 上传资料", use_container_width=True):
        st.switch_page("pages/数据摄取.py")

with nav_cols[1]:
    if st.button("📚 浏览文档", use_container_width=True):
        st.switch_page("pages/文档浏览.py")

with nav_cols[2]:
    if st.button("📋 查看任务", use_container_width=True):
        st.switch_page("pages/后台任务.py")

with nav_cols[3]:
    if st.button("🏠 返回主页", use_container_width=True):
        st.switch_page("src/ui/主页.py")

# Footer with validation info
st.markdown("---")
st.caption("🛡️ 此查询系统配备汽车领域专业验证功能，帮助确保信息准确性")
st.caption("⚠️ 对于重要决策，建议结合多个权威来源进行验证")