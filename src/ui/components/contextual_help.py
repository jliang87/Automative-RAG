import streamlit as st
from typing import Dict

class ContextualHelpSystem:
    """Provides contextual help based on user actions and current state."""

    def __init__(self):
        self.help_content = {
            "mode_selection": {
                "title": "如何选择查询模式？",
                "content": """
                **📌 事实查询**: 当您需要具体的技术参数时
                - 例如：后备箱容积、发动机功率、油耗数据

                **💡 功能建议**: 当您评估是否需要某项功能时
                - 例如：是否值得购买自动驾驶套件

                **🧾 权衡分析**: 当您在两个选择之间犹豫时
                - 例如：燃油车 vs 电动车的优缺点

                **🧩 场景分析**: 当您想了解实际使用体验时
                - 例如：长途旅行时的充电便利性

                **🗣️ 多角色讨论**: 当您需要全面的专业观点时
                - 例如：不同专家如何看待某项技术

                **🔍 用户评论**: 当您想了解真实用户反馈时
                - 例如：车主对某个功能的真实评价
                """
            },
            "query_writing": {
                "title": "如何写出好的查询？",
                "content": """
                **✅ 好的查询示例:**
                - "2023年宝马X5的后备箱容积是多少？"
                - "特斯拉Model 3与奔驰C级的安全配置对比"
                - "家庭用户购买7座SUV的注意事项"

                **❌ 避免的查询方式:**
                - "汽车怎么样？" (太宽泛)
                - "这个好不好？" (缺乏具体信息)
                - "ECU参数配置" (过于技术化)

                **💡 提升查询质量的技巧:**
                1. 包含具体的年份和车型
                2. 明确您想了解的方面
                3. 使用日常语言而非技术术语
                4. 提供使用场景或需求背景
                """
            },
            "result_interpretation": {
                "title": "如何理解分析结果？",
                "content": """
                **双层结构解读:**
                - **实证部分**: 基于文档的确定信息
                - **推理部分**: AI基于知识的合理推测

                **相关度分数含义:**
                - 0.8+: 高度相关，结果可信度高
                - 0.6-0.8: 中等相关，结果有参考价值
                - 0.4-0.6: 低相关，建议调整查询

                **来源多样性:**
                - 多个来源支持的信息更可靠
                - 注意查看信息的时效性
                - 优先参考官方和权威来源
                """
            }
        }

    def get_contextual_help(self, context: str) -> Dict[str, str]:
        """Get help content for specific context."""
        return self.help_content.get(context, {
            "title": "帮助信息",
            "content": "暂无相关帮助信息。"
        })


def render_contextual_help(context: str):
    """Render contextual help widget."""
    help_system = ContextualHelpSystem()
    help_content = help_system.get_contextual_help(context)

    with st.expander(f"❓ {help_content['title']}"):
        st.markdown(help_content['content'])
