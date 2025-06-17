import streamlit as st
import time
import json
from typing import Dict, Any, Optional
from src.ui.api_client import api_request
from src.ui.session_init import initialize_session_state

initialize_session_state()

# Query mode configurations (updated for unified system)
QUERY_MODES = {
    "facts": {
        "icon": "📌",
        "name": "车辆规格查询",
        "description": "验证具体的车辆规格参数",
        "use_case": "查询确切的技术规格、配置信息",
        "two_layer": True,
        "is_default": True,
        "complexity": "简单",
        "time_estimate": "~10秒",
        "examples": [
            "2023年宝马X5的后备箱容积是多少？",
            "特斯拉Model 3的充电速度参数",
            "奔驰E级使用什么轮胎型号？"
        ]
    },
    "features": {
        "icon": "💡",
        "name": "新功能建议",
        "description": "评估是否应该添加某项功能",
        "use_case": "产品决策，功能规划",
        "two_layer": True,
        "complexity": "中等",
        "time_estimate": "~30秒",
        "examples": [
            "是否应该为电动车增加氛围灯功能？",
            "增加模拟引擎声音对用户体验的影响",
            "AR抬头显示器值得投资吗？"
        ]
    },
    "tradeoffs": {
        "icon": "🧾",
        "name": "权衡利弊分析",
        "description": "分析设计选择的优缺点",
        "use_case": "设计决策，技术选型",
        "two_layer": True,
        "complexity": "复杂",
        "time_estimate": "~45秒",
        "examples": [
            "使用模拟声音 vs 自然静音的利弊",
            "移除物理按键的优缺点分析",
            "大屏幕 vs 传统仪表盘的对比"
        ]
    },
    "scenarios": {
        "icon": "🧩",
        "name": "用户场景分析",
        "description": "评估功能在实际使用场景中的表现",
        "use_case": "用户体验设计，产品规划",
        "two_layer": True,
        "complexity": "复杂",
        "time_estimate": "~40秒",
        "examples": [
            "长途旅行时这个功能如何表现？",
            "家庭用户在日常通勤中的体验如何？",
            "寒冷气候下的性能表现分析"
        ]
    },
    "debate": {
        "icon": "🗣️",
        "name": "多角色讨论",
        "description": "模拟不同角色的观点和讨论",
        "use_case": "决策支持，全面评估",
        "two_layer": False,
        "complexity": "复杂",
        "time_estimate": "~50秒",
        "examples": [
            "产品经理、工程师和用户代表如何看待自动驾驶功能？",
            "不同团队对电池技术路线的观点",
            "关于车内空间设计的多方讨论"
        ]
    },
    "quotes": {
        "icon": "🔍",
        "name": "原始用户评论",
        "description": "提取相关的用户评论和反馈",
        "use_case": "市场研究，用户洞察",
        "two_layer": False,
        "complexity": "简单",
        "time_estimate": "~20秒",
        "examples": [
            "用户对续航里程的真实评价",
            "关于内饰质量的用户反馈",
            "充电体验的用户评论摘录"
        ]
    }
}


def submit_unified_query(query_text: str, mode: str, filters: Optional[Dict] = None) -> Optional[str]:
    """Submit unified query (all queries now use the same endpoint)"""
    try:
        # UNIFIED: All queries use the same endpoint with mode specification
        unified_data = {
            "query": query_text,
            "metadata_filter": filters,
            "top_k": 8,  # Get more documents for better analysis
            "query_mode": mode,  # Specify the analysis mode
            "prompt_template": None  # Use default mode template
        }

        result = api_request(
            endpoint="/query",  # UNIFIED: Single endpoint for all queries
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
    """Get unified query results"""
    try:
        return api_request(f"/query/results/{job_id}", method="GET")
    except Exception as e:
        st.error(f"获取查询结果时发生错误: {str(e)}")
        return None


def display_two_layer_result(result: Dict[str, Any], mode: str):
    """Display results with two-layer structure for enhanced modes"""
    answer = result.get("answer", "")
    analysis_structure = result.get("analysis_structure")

    if not answer:
        st.warning("未获得查询结果")
        return

    # Check if we have structured analysis
    if analysis_structure and isinstance(analysis_structure, dict):
        # Display structured sections
        if mode == "facts":
            if "【实证分析】" in analysis_structure:
                st.subheader("📋 基于文档的实证分析")
                st.info(analysis_structure["【实证分析】"])

            if "【策略推理】" in analysis_structure:
                st.subheader("🧠 专业推理补充")
                st.warning(analysis_structure["【策略推理】"])

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
            # Fallback: show full answer
            st.markdown(answer)
    else:
        # No structured analysis, show full answer
        st.markdown("**📋 分析结果:**")
        st.info(answer)


def display_debate_result(answer: str):
    """Display debate-style results with multiple perspectives"""
    st.subheader("🗣️ 多角色讨论")

    # Look for role-based sections
    roles = ["产品经理观点", "工程师观点", "用户代表观点"]
    sections = {}

    current_role = None
    current_content = []

    lines = answer.split('\n')

    for line in lines:
        line = line.strip()

        # Check for role headers
        found_role = None
        for role in roles:
            if role in line or f"**{role}" in line:
                found_role = role
                break

        if found_role:
            # Save previous role content
            if current_role and current_content:
                sections[current_role] = '\n'.join(current_content).strip()

            current_role = found_role
            current_content = []
        elif current_role and line:
            current_content.append(line)

    # Save last role
    if current_role and current_content:
        sections[current_role] = '\n'.join(current_content).strip()

    # Display each perspective
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
        # Fallback: display full answer
        st.markdown(answer)


def display_quotes_result(answer: str):
    """Display user quotes in a structured format"""
    st.subheader("💬 用户评论摘录")

    # Look for quote patterns like 【来源1】："..."
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
        # Fallback: display full answer
        st.markdown(answer)


# Main interface
st.title("🧠 汽车智能查询系统")
st.markdown("统一查询平台 - 从快速Facts验证到深度专业分析")

# System info banner
st.info("🔄 **统一查询系统** - 一个平台包含所有查询功能，Facts模式为默认推荐")

# Mode selection
st.subheader("📋 选择查询模式")

# Initialize selected mode in session state
if 'selected_mode' not in st.session_state:
    st.session_state.selected_mode = "facts"  # Default to facts mode

# Handle pre-selected mode from homepage
if hasattr(st.session_state, 'smart_mode'):
    st.session_state.selected_mode = st.session_state.smart_mode
    del st.session_state.smart_mode

# Display modes in a grid with enhanced information
mode_cols = st.columns(3)

for i, (mode_key, mode_info) in enumerate(QUERY_MODES.items()):
    col = mode_cols[i % 3]

    with col:
        # Check if this mode is selected
        is_selected = st.session_state.get('selected_mode') == mode_key

        # Set button style based on selection
        button_type = "primary" if is_selected else "secondary"

        # Add visual indicator to button text
        button_text = f"{mode_info['icon']} {mode_info['name']}"
        if is_selected:
            button_text = f"✅ {button_text}"

        # Add default indicator for facts mode
        if mode_info.get('is_default'):
            button_text += " (推荐)"

        if st.button(
                button_text,
                key=f"mode_{mode_key}",
                use_container_width=True,
                help=f"{mode_info['description']} | 复杂度: {mode_info['complexity']} | 时间: {mode_info['time_estimate']}",
                type=button_type
        ):
            st.session_state.selected_mode = mode_key
            st.rerun()

        # Show mode info below button for selected mode
        if is_selected:
            st.caption(f"⏱️ {mode_info['time_estimate']} | 🎯 {mode_info['use_case']}")

# Show selected mode info
if st.session_state.get('selected_mode'):
    mode = st.session_state.selected_mode
    mode_info = QUERY_MODES[mode]

    st.markdown("---")

    # Query input section
    st.subheader("💭 输入您的问题")

    # Show examples for the selected mode
    with st.expander(f"💡 {mode_info['name']} 示例"):
        st.markdown(f"**适合{mode_info['name']}的问题类型：**")
        for example in mode_info['examples']:
            if st.button(example, key=f"example_{example[:20]}", use_container_width=True):
                st.session_state.example_query = example

    # Handle pre-filled queries from other pages
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
        height=100,
        help=f"当前模式：{mode_info['description']} | 预计耗时：{mode_info['time_estimate']}"
    )

    # Filters (simplified for UX)
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

    # Submit button with enhanced information
    submit_text = f"🚀 开始{mode_info['name']}"
    if mode_info.get('is_default'):
        submit_text += " (推荐)"

    if st.button(
            submit_text,
            type="primary",
            disabled=not query.strip(),
            use_container_width=True,
            help=f"预计处理时间：{mode_info['time_estimate']}"
    ):
        if query.strip():
            with st.spinner(f"正在进行{mode_info['name']}，预计{mode_info['time_estimate']}..."):
                job_id = submit_unified_query(query.strip(), mode, filters if filters else None)

                if job_id:
                    st.session_state.current_job_id = job_id
                    st.session_state.query_text = query.strip()
                    st.session_state.query_mode = mode
                    st.session_state.query_submitted_at = time.time()
                    st.success(f"✅ {mode_info['name']}已提交，任务ID: {job_id[:8]}...")
                    st.rerun()

else:
    # Initial instructions when no mode is selected
    st.info("👆 请选择一个查询模式开始分析")

    # Mode comparison table
    st.subheader("📊 查询模式对比")

    comparison_data = []
    for mode_key, mode_info in QUERY_MODES.items():
        mode_name = f"{mode_info['icon']} {mode_info['name']}"
        if mode_info.get('is_default'):
            mode_name += " (推荐)"

        comparison_data.append({
            "模式": mode_name,
            "复杂度": mode_info['complexity'],
            "预计时间": mode_info['time_estimate'],
            "适用场景": mode_info['use_case'],
            "输出结构": "双层结构" if mode_info['two_layer'] else "单层输出"
        })

    # Display as table
    import pandas as pd

    df = pd.DataFrame(comparison_data)
    st.dataframe(df, use_container_width=True, hide_index=True)

    # Quick start recommendations
    st.markdown("### 🚀 快速开始建议")
    rec_col1, rec_col2 = st.columns(2)

    with rec_col1:
        st.markdown("**🆕 新用户推荐：**")
        if st.button("📌 开始Facts查询", type="primary", use_container_width=True):
            st.session_state.selected_mode = "facts"
            st.rerun()
        st.caption("简单快速，验证具体规格参数")

    with rec_col2:
        st.markdown("**🏆 专业用户推荐：**")
        if st.button("💡 开始功能建议", use_container_width=True):
            st.session_state.selected_mode = "features"
            st.rerun()
        st.caption("深度分析，产品决策支持")

# Results section
if hasattr(st.session_state, 'current_job_id') and st.session_state.current_job_id:
    job_id = st.session_state.current_job_id
    query_mode = getattr(st.session_state, 'query_mode', 'facts')
    mode_info = QUERY_MODES[query_mode]

    st.markdown("---")
    st.subheader(f"📋 {mode_info['name']} 结果")

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
        result_query_mode = result.get("query_mode", query_mode)

        if status == "completed":
            st.success("✅ 分析完成！")

            # Display unified system metadata
            mode_metadata = result.get("mode_metadata", {})
            if mode_metadata.get("unified_system"):
                execution_time = result.get("execution_time", 0)
                complexity = mode_metadata.get("complexity_level", "unknown")
                st.caption(f"⏱️ 实际耗时: {execution_time:.1f}秒 | 复杂度: {complexity} | 系统: 统一查询v2.0")

            # Display results based on mode
            if mode_info['two_layer']:
                display_two_layer_result(result, query_mode)
            elif query_mode == "debate":
                display_debate_result(result.get("answer", ""))
            elif query_mode == "quotes":
                display_quotes_result(result.get("answer", ""))
            else:
                st.markdown(result.get("answer", ""))

            # Show sources
            documents = result.get("documents", [])
            if documents:
                with st.expander(f"📚 参考来源 ({len(documents)} 个)"):
                    for i, doc in enumerate(documents[:5]):
                        st.markdown(f"**来源 {i + 1}:** {doc.get('metadata', {}).get('title', '文档')}")
                        if doc.get("content"):
                            st.caption(doc['content'][:200] + "...")
                        st.markdown("---")

            # Actions for completed queries
            action_col1, action_col2, action_col3 = st.columns(3)
            with action_col1:
                if st.button("🔄 新的查询", key="new_analysis"):
                    # Clear session state
                    for key in ['current_job_id', 'query_text', 'last_query_result']:
                        if key in st.session_state:
                            del st.session_state[key]
                    st.rerun()

            with action_col2:
                if st.button("🔀 切换模式", key="switch_mode"):
                    # Keep query but allow mode change
                    st.session_state.example_query = st.session_state.get('query_text', '')
                    for key in ['current_job_id', 'last_query_result']:
                        if key in st.session_state:
                            del st.session_state[key]
                    st.rerun()

            with action_col3:
                if st.button("📋 查看详情", key="view_job_details"):
                    st.session_state.selected_job_id = job_id
                    st.switch_page("pages/后台任务.py")

        elif status == "failed":
            st.error("❌ 分析失败")
            error_msg = result.get("answer", "未知错误")
            st.error(f"错误信息: {error_msg}")

        else:
            # Still processing
            st.info("⏳ 正在分析中...")
            progress_msg = result.get("answer", "正在处理您的查询...")
            st.info(progress_msg)

            # Show elapsed time and estimated remaining
            if hasattr(st.session_state, 'query_submitted_at'):
                elapsed = time.time() - st.session_state.query_submitted_at
                estimated_total = int(mode_info['time_estimate'].replace('~', '').replace('秒', ''))
                remaining = max(0, estimated_total - elapsed)

                st.caption(f"已运行 {elapsed:.0f}秒 | 预计还需 {remaining:.0f}秒")

            # Manual refresh option
            if st.button("🔄 刷新状态", key="refresh_status"):
                st.session_state.last_result_check = 0
                st.rerun()
    else:
        st.error("❌ 无法获取分析状态")

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

st.caption("智能查询系统 - 统一平台，从快速验证到深度分析")