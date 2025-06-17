import streamlit as st
from typing import Optional, Tuple

class QueryTemplateSystem:
    """Provides query templates and shortcuts for common scenarios."""

    def __init__(self):
        self.templates = {
            "车型对比": {
                "icon": "⚖️",
                "template": "{车型A} 与 {车型B} 的 {对比方面} 对比分析",
                "suggested_mode": "tradeoffs",
                "examples": [
                    "宝马X5与奔驰GLE的安全配置对比分析",
                    "特斯拉Model 3与蔚来ET5的充电便利性对比分析"
                ]
            },
            "功能评估": {
                "icon": "🎯",
                "template": "是否应该选择带有{功能名称}的{车型}？",
                "suggested_mode": "features",
                "examples": [
                    "是否应该选择带有自动驾驶功能的特斯拉？",
                    "是否应该选择带有空气悬挂的奔驰S级？"
                ]
            },
            "使用场景": {
                "icon": "🎭",
                "template": "{车型}在{使用场景}下的表现如何？",
                "suggested_mode": "scenarios",
                "examples": [
                    "宝马iX3在长途高速驾驶下的表现如何？",
                    "丰田汉兰达在城市拥堵路况下的表现如何？"
                ]
            },
            "规格查询": {
                "icon": "📏",
                "template": "{年份}年{车型}的{具体参数}是多少？",
                "suggested_mode": "facts",
                "examples": [
                    "2023年奔驰E级的后备箱容积是多少？",
                    "2024年宝马3系的百公里加速时间是多少？"
                ]
            }
        }

    def render_template_selector(self) -> Optional[Tuple[str, str]]:
        """Render template selector and return selected template and mode."""
        st.markdown("### 🎯 快速模板")

        cols = st.columns(2)
        for i, (category, config) in enumerate(self.templates.items()):
            col = cols[i % 2]

            with col:
                if st.button(f"{config['icon']} {category}", key=f"template_{category}"):
                    return config["template"], config["suggested_mode"]

        return None, None


def render_query_templates():
    """Render the complete query template system."""
    template_system = QueryTemplateSystem()

    selected = template_system.render_template_selector()
    if selected[0]:
        template, mode = selected
        st.session_state.selected_template = template
        st.session_state.suggested_mode = mode
        st.rerun()