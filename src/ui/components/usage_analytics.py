import streamlit as st
from datetime import datetime
from typing import Dict

class UsageAnalytics:
    """Tracks and analyzes system usage patterns."""

    def __init__(self):
        self.analytics_key = "system_analytics"
        if self.analytics_key not in st.session_state:
            st.session_state[self.analytics_key] = {
                "mode_usage": {},
                "query_categories": {},
                "satisfaction_scores": [],
                "session_start": datetime.now().isoformat()
            }

    def track_mode_usage(self, mode: str):
        """Track usage of different modes."""
        analytics = st.session_state[self.analytics_key]
        analytics["mode_usage"][mode] = analytics["mode_usage"].get(mode, 0) + 1

    def track_query_category(self, category: str):
        """Track query categories."""
        analytics = st.session_state[self.analytics_key]
        analytics["query_categories"][category] = analytics["query_categories"].get(category, 0) + 1

    def add_satisfaction_score(self, score: int):
        """Add user satisfaction score (1-5)."""
        analytics = st.session_state[self.analytics_key]
        analytics["satisfaction_scores"].append(score)

        # Keep only last 20 scores
        if len(analytics["satisfaction_scores"]) > 20:
            analytics["satisfaction_scores"] = analytics["satisfaction_scores"][-20:]

    def get_usage_insights(self) -> Dict[str, any]:
        """Get usage insights and recommendations."""
        analytics = st.session_state[self.analytics_key]

        insights = {
            "most_used_mode": None,
            "avg_satisfaction": 0,
            "recommendations": [],
            "session_summary": {}
        }

        # Most used mode
        if analytics["mode_usage"]:
            insights["most_used_mode"] = max(analytics["mode_usage"], key=analytics["mode_usage"].get)

        # Average satisfaction
        if analytics["satisfaction_scores"]:
            insights["avg_satisfaction"] = sum(analytics["satisfaction_scores"]) / len(analytics["satisfaction_scores"])

        # Generate recommendations
        if insights["avg_satisfaction"] < 3:
            insights["recommendations"].append("尝试使用不同的查询模式以获得更好的结果")

        if analytics["mode_usage"].get("facts", 0) > 5:
            insights["recommendations"].append("您经常使用事实查询，可以尝试更高级的分析模式")

        return insights


def render_satisfaction_feedback():
    """Render satisfaction feedback widget."""
    st.markdown("### 💝 您对这次查询结果满意吗？")

    satisfaction_cols = st.columns(5)
    analytics = UsageAnalytics()

    for i, (score, emoji) in enumerate([(1, "😞"), (2, "😐"), (3, "🙂"), (4, "😊"), (5, "🤩")]):
        with satisfaction_cols[i]:
            if st.button(f"{emoji}\n{score}分", key=f"satisfaction_{score}"):
                analytics.add_satisfaction_score(score)
                if score >= 4:
                    st.success("感谢您的积极反馈！")
                elif score <= 2:
                    st.info("我们会努力改进。您可以尝试调整查询方式或使用不同的分析模式。")
                else:
                    st.info("谢谢反馈！")