import streamlit as st
import time
import json
from typing import Dict, Any, Optional
from src.ui.api_client import api_request
from src.ui.session_init import initialize_session_state

initialize_session_state()

# Query mode configurations
QUERY_MODES = {
    "facts": {
        "icon": "📌",
        "name": "车辆规格查询",
        "description": "验证具体的车辆规格参数",
        "use_case": "查询确切的技术规格、配置信息",
        "two_layer": True,
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
        "examples": [
            "用户对续航里程的真实评价",
            "关于内饰质量的用户反馈",
            "充电体验的用户评论摘录"
        ]
    }
}

# Prompt templates for each mode
PROMPT_TEMPLATES = {
    "facts": """你是专业的汽车技术规格验证专家。请严格按照以下格式回答：

【实证答案】
基于提供的文档内容，回答用户的具体问题。如果文档中没有相关信息，必须明确说明"根据提供的文档，未提及相关信息"。

【推理补充】
如果有需要，可以基于汽车行业常识进行合理推测，但必须清楚标注这是推理而非文档事实。

用户查询：{query}
文档内容：{context}

请确保回答精确、可验证，并明确区分事实和推理。""",

    "features": """你是汽车产品策略专家。请按照以下格式分析是否应该添加某项功能：

【实证分析】
基于提供的文档中关于类似功能或相关技术的信息进行分析。

【策略推理】
基于产品思维和用户需求，分析这个功能的潜在价值：
- 用户受益分析
- 技术可行性
- 市场竞争优势
- 成本效益评估

用户询问功能：{query}
参考文档：{context}

请提供平衡的评估意见。""",

    "tradeoffs": """你是汽车设计决策分析师。请按照以下格式分析设计选择的利弊：

【文档支撑】
基于提供文档中的相关信息和数据。

【利弊分析】
**优点：**
- [基于文档的优点]
- [推理得出的优点]

**缺点：**
- [基于文档的缺点] 
- [推理得出的缺点]

**总结建议：**
综合评估和建议

设计决策：{query}
参考资料：{context}

请确保分析客观全面。""",

    "scenarios": """你是用户体验分析专家。请按照以下格式分析功能在不同场景下的表现：

【文档场景】
提取文档中提到的使用场景和用户反馈。

【场景推理】
基于产品思维和用户同理心，分析在以下场景中的表现：
- 谁会受益（目标用户群）
- 什么时候有用（使用时机）
- 什么条件下效果最好（最佳使用条件）
- 可能的问题和限制

分析主题：{query}
参考信息：{context}

请提供具体、实用的场景分析。""",

    "debate": """你是汽车行业圆桌讨论主持人。请模拟以下角色的讨论：

**产品经理观点：**
从商业价值和用户需求角度分析

**工程师观点：** 
从技术实现和成本角度分析

**用户代表观点：**
从实际使用体验和需求角度分析

讨论话题：{query}
参考信息：{context}

请让每个角色提出不同的观点，最后总结共识和分歧点。""",

    "quotes": """你是汽车市场研究分析师。请从提供的文档中提取用户的原始评论和反馈：

请按以下格式提供用户评论：

【来源1】："用户原始评论内容..."
【来源2】："用户原始评论内容..."
【来源3】："用户原始评论内容..."

查询主题：{query}
文档来源：{context}

只提取真实的用户评论，不要编造内容。如果没有找到相关的用户评论，请明确说明。"""
}


def submit_enhanced_query(query_text: str, mode: str, filters: Optional[Dict] = None) -> Optional[str]:
    """Submit enhanced query with specific prompt template"""
    try:
        # Get the prompt template for the selected mode
        template = PROMPT_TEMPLATES.get(mode, PROMPT_TEMPLATES["facts"])

        # Enhanced query data
        enhanced_data = {
            "query": query_text,
            "metadata_filter": filters,
            "top_k": 8,  # Get more documents for better analysis
            "query_mode": mode,
            "prompt_template": template
        }

        result = api_request(
            endpoint="/query/",
            method="POST",
            data=enhanced_data
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
    """Get query results"""
    try:
        return api_request(f"/query/results/{job_id}", method="GET")
    except Exception as e:
        st.error(f"获取查询结果时发生错误: {str(e)}")
        return None


def display_two_layer_result(result: Dict[str, Any], mode: str):
    """Display results with two-layer structure"""
    answer = result.get("answer", "")

    if not answer:
        st.warning("未获得查询结果")
        return

    # Parse two-layer structure based on mode
    if mode == "facts":
        # Look for 【实证答案】 and 【推理补充】
        sections = parse_structured_answer(answer, ["【实证答案】", "【推理补充】"])

        if sections.get("【实证答案】"):
            st.subheader("📋 实证答案")
            st.info(sections["【实证答案】"])

        if sections.get("【推理补充】"):
            st.subheader("🧠 推理补充")
            st.warning(sections["【推理补充】"])

    elif mode == "features":
        sections = parse_structured_answer(answer, ["【实证分析】", "【策略推理】"])

        if sections.get("【实证分析】"):
            st.subheader("📊 实证分析")
            st.info(sections["【实证分析】"])

        if sections.get("【策略推理】"):
            st.subheader("💡 策略推理")
            st.success(sections["【策略推理】"])

    elif mode == "tradeoffs":
        sections = parse_structured_answer(answer, ["【文档支撑】", "【利弊分析】"])

        if sections.get("【文档支撑】"):
            st.subheader("📋 文档支撑")
            st.info(sections["【文档支撑】"])

        if sections.get("【利弊分析】"):
            st.subheader("⚖️ 利弊分析")
            st.warning(sections["【利弊分析】"])

    elif mode == "scenarios":
        sections = parse_structured_answer(answer, ["【文档场景】", "【场景推理】"])

        if sections.get("【文档场景】"):
            st.subheader("📖 文档场景")
            st.info(sections["【文档场景】"])

        if sections.get("【场景推理】"):
            st.subheader("🎯 场景推理")
            st.success(sections["【场景推理】"])
    else:
        # Fallback: show full answer
        st.markdown(answer)


def parse_structured_answer(answer: str, section_headers: list) -> Dict[str, str]:
    """Parse structured answer into sections"""
    sections = {}
    current_section = None
    current_content = []

    lines = answer.split('\n')

    for line in lines:
        line = line.strip()

        # Check if this line is a section header
        found_header = None
        for header in section_headers:
            if header in line:
                found_header = header
                break

        if found_header:
            # Save previous section if exists
            if current_section and current_content:
                sections[current_section] = '\n'.join(current_content).strip()

            # Start new section
            current_section = found_header
            current_content = []
        elif current_section:
            # Add content to current section
            if line:  # Only add non-empty lines
                current_content.append(line)

    # Save the last section
    if current_section and current_content:
        sections[current_section] = '\n'.join(current_content).strip()

    return sections


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
st.title("🔍 智能汽车查询")
st.markdown("多角度分析模式，适合不同的查询需求")

# Mode selection
st.subheader("📋 选择查询模式")

# Display modes in a grid
mode_cols = st.columns(3)
selected_mode = None

for i, (mode_key, mode_info) in enumerate(QUERY_MODES.items()):
    col = mode_cols[i % 3]

    with col:
        if st.button(
                f"{mode_info['icon']} {mode_info['name']}",
                key=f"mode_{mode_key}",
                use_container_width=True,
                help=mode_info['description']
        ):
            selected_mode = mode_key

# Show selected mode info
if selected_mode:
    st.session_state.selected_mode = selected_mode

if 'selected_mode' in st.session_state:
    mode = st.session_state.selected_mode
    mode_info = QUERY_MODES[mode]

    st.success(f"已选择：{mode_info['icon']} {mode_info['name']}")
    st.markdown(f"**适用场景：** {mode_info['use_case']}")

    # Query input
    st.subheader("💭 输入您的问题")

    # Show examples for the selected mode
    with st.expander(f"💡 {mode_info['name']} 示例"):
        for example in mode_info['examples']:
            if st.button(example, key=f"example_{example[:20]}", use_container_width=True):
                st.session_state.example_query = example

    # Use example query if selected
    default_query = st.session_state.get('example_query', '')
    if default_query:
        del st.session_state.example_query

    query = st.text_area(
        "请输入您的问题",
        value=default_query,
        placeholder=f"例如：{mode_info['examples'][0]}",
        height=100,
        help=f"当前模式：{mode_info['description']}"
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

    # Submit button
    if st.button(
            f"🚀 开始{mode_info['name']}",
            type="primary",
            disabled=not query.strip(),
            use_container_width=True
    ):
        if query.strip():
            with st.spinner(f"正在进行{mode_info['name']}..."):
                job_id = submit_enhanced_query(query.strip(), mode, filters if filters else None)

                if job_id:
                    st.session_state.current_job_id = job_id
                    st.session_state.query_text = query.strip()
                    st.session_state.query_mode = mode
                    st.session_state.query_submitted_at = time.time()
                    st.success(f"✅ {mode_info['name']}已提交，任务ID: {job_id[:8]}...")
                    st.rerun()

# Results section
if hasattr(st.session_state, 'current_job_id') and st.session_state.current_job_id:
    job_id = st.session_state.current_job_id
    query_mode = getattr(st.session_state, 'query_mode', 'facts')
    mode_info = QUERY_MODES[query_mode]

    st.markdown("---")
    st.subheader(f"📋 {mode_info['name']} 结果")

    # Get results with minimal API calls
    if 'last_result_check' not in st.session_state:
        st.session_state.last_result_check = 0

    current_time = time.time()
    if current_time - st.session_state.last_result_check > 3:  # Check every 3 seconds
        result = get_query_result(job_id)
        st.session_state.last_query_result = result
        st.session_state.last_result_check = current_time
    else:
        result = st.session_state.get('last_query_result')

    if result:
        status = result.get("status", "")

        if status == "completed":
            st.success("✅ 分析完成！")

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
            action_col1, action_col2 = st.columns(2)
            with action_col1:
                if st.button("🔄 新的分析", key="new_analysis"):
                    # Clear session state
                    for key in ['current_job_id', 'query_text', 'last_query_result', 'selected_mode']:
                        if key in st.session_state:
                            del st.session_state[key]
                    st.rerun()

            with action_col2:
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

            # Manual refresh option
            if st.button("🔄 刷新状态", key="refresh_status"):
                st.session_state.last_result_check = 0
                st.rerun()
    else:
        st.error("❌ 无法获取分析状态")

else:
    # Initial instructions
    st.info("👆 请选择一个查询模式开始分析")

    # Mode comparison table
    st.subheader("📊 模式对比")

    comparison_data = []
    for mode_key, mode_info in QUERY_MODES.items():
        comparison_data.append({
            "模式": f"{mode_info['icon']} {mode_info['name']}",
            "描述": mode_info['description'],
            "适用场景": mode_info['use_case'],
            "输出结构": "双层结构" if mode_info['two_layer'] else "单层输出"
        })

    # Display as table
    import pandas as pd

    df = pd.DataFrame(comparison_data)
    st.dataframe(df, use_container_width=True, hide_index=True)

# Navigation
st.markdown("---")
nav_cols = st.columns(4)

with nav_cols[0]:
    if st.button("🔍 基础查询", use_container_width=True):
        st.switch_page("pages/查询.py")

with nav_cols[1]:
    if st.button("📤 上传资料", use_container_width=True):
        st.switch_page("pages/数据摄取.py")

with nav_cols[2]:
    if st.button("📋 查看任务", use_container_width=True):
        st.switch_page("pages/后台任务.py")

with nav_cols[3]:
    if st.button("🏠 返回主页", use_container_width=True):
        st.switch_page("src/ui/主页.py")

st.caption("智能查询 - 多角度深度分析，适合专业用户的不同需求")