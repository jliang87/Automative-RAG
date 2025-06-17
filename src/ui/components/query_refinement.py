import streamlit as st
from typing import Dict

class QueryRefinementAssistant:
    """Helps users refine their queries for better results."""

    def __init__(self):
        self.refinement_patterns = {
            "too_broad": {
                "indicators": ["汽车", "车辆", "所有", "全部", "任何"],
                "suggestions": [
                    "尝试指定具体的车型和年份",
                    "添加品牌名称会获得更精确的结果",
                    "考虑限制到特定的车辆类别"
                ]
            },
            "missing_context": {
                "indicators": ["这个", "那个", "它", "该"],
                "suggestions": [
                    "请明确指出具体的车型或功能",
                    "添加更多背景信息",
                    "指定您关心的具体方面"
                ]
            },
            "too_technical": {
                "indicators": ["ECU", "CAN总线", "OBD", "涡轮增压器"],
                "suggestions": [
                    "考虑使用更通用的术语",
                    "添加功能描述而非技术名称",
                    "说明您想了解的具体用途"
                ]
            },
            "perfect": {
                "indicators": ["2023年", "宝马X5", "特斯拉Model 3"],
                "message": "查询表述很好！包含了具体的年份和车型信息。"
            }
        }

    def analyze_query_quality(self, query: str) -> Dict[str, any]:
        """Analyze query quality and provide improvement suggestions."""
        analysis = {
            "quality_score": 0.5,
            "issues": [],
            "suggestions": [],
            "strengths": []
        }

        query_lower = query.lower()

        # Check for specificity
        if any(year in query for year in ["2020", "2021", "2022", "2023", "2024"]):
            analysis["quality_score"] += 0.2
            analysis["strengths"].append("包含具体年份")

        # Check for brand/model
        brands = ["宝马", "奔驰", "奥迪", "丰田", "本田", "特斯拉", "大众", "福特"]
        if any(brand in query for brand in brands):
            analysis["quality_score"] += 0.2
            analysis["strengths"].append("包含具体品牌")

        # Check for issues
        for issue_type, config in self.refinement_patterns.items():
            if issue_type == "perfect":
                continue

            if any(indicator in query_lower for indicator in config["indicators"]):
                analysis["issues"].append(issue_type)
                analysis["suggestions"].extend(config["suggestions"])
                analysis["quality_score"] -= 0.1

        return analysis


def render_query_refinement(query: str) -> str:
    """Render query refinement suggestions."""
    if not query.strip():
        return query

    assistant = QueryRefinementAssistant()
    analysis = assistant.analyze_query_quality(query)

    if analysis["quality_score"] < 0.7 and analysis["suggestions"]:
        with st.expander("✨ 查询优化建议"):
            st.markdown("**改进建议:**")
            for suggestion in analysis["suggestions"][:3]:  # Limit to top 3
                st.write(f"• {suggestion}")

            # Quick refinement templates
            st.markdown("**快速模板:**")
            templates = [
                "2023年{品牌}{车型}的{具体参数}",
                "{品牌}{车型}与{其他车型}的对比",
                "{具体功能}在{使用场景}下的表现"
            ]

            for template in templates:
                if st.button(template, key=f"template_{template[:10]}"):
                    return template

    elif analysis["quality_score"] >= 0.8:
        st.success("🎯 查询表述很好！期待为您提供精确的分析。")

    return query