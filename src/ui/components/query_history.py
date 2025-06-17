import streamlit as st
from datetime import datetime
from typing import List, Dict


class QueryHistoryManager:
    """Manages personalized query history and learning."""

    def __init__(self):
        self.session_key = "query_history"
        if self.session_key not in st.session_state:
            st.session_state[self.session_key] = []

    def add_query(self, query: str, mode: str, quality_score: float):
        """Add a query to history."""
        history_entry = {
            "query": query,
            "mode": mode,
            "timestamp": datetime.now().isoformat(),
            "quality_score": quality_score
        }

        history = st.session_state[self.session_key]
        history.append(history_entry)

        # Keep only last 50 queries
        if len(history) > 50:
            st.session_state[self.session_key] = history[-50:]

    def get_query_patterns(self) -> Dict[str, any]:
        """Analyze user query patterns."""
        history = st.session_state[self.session_key]

        if not history:
            return {"patterns": [], "recommendations": []}

        # Mode preferences
        mode_counts = {}
        for entry in history[-20:]:  # Last 20 queries
            mode = entry["mode"]
            mode_counts[mode] = mode_counts.get(mode, 0) + 1

        preferred_modes = sorted(mode_counts.items(), key=lambda x: x[1], reverse=True)

        # Quality trends
        recent_quality = [entry["quality_score"] for entry in history[-10:]]
        avg_quality = sum(recent_quality) / len(recent_quality) if recent_quality else 0

        # Generate recommendations
        recommendations = []
        if avg_quality < 0.6:
            recommendations.append("尝试使用更具体的车型和年份信息")
            recommendations.append("考虑使用智能分析模式获得更深入的回答")

        if preferred_modes and preferred_modes[0][0] == "facts":
            recommendations.append("您经常使用事实查询，可以尝试功能建议或场景分析模式")

        return {
            "preferred_modes": preferred_modes,
            "avg_quality": avg_quality,
            "recommendations": recommendations,
            "total_queries": len(history)
        }

    def get_similar_queries(self, current_query: str) -> List[Dict]:
        """Find similar historical queries."""
        history = st.session_state[self.session_key]
        current_words = set(current_query.lower().split())

        similar = []
        for entry in history:
            entry_words = set(entry["query"].lower().split())
            similarity = len(current_words & entry_words) / len(current_words | entry_words)

            if similarity > 0.3:  # 30% similarity threshold
                similar.append({**entry, "similarity": similarity})

        return sorted(similar, key=lambda x: x["similarity"], reverse=True)[:5]


def render_query_history_insights():
    """Render query history insights."""
    manager = QueryHistoryManager()
    patterns = manager.get_query_patterns()

    if patterns["total_queries"] > 0:
        with st.expander(f"📈 查询洞察 (共{patterns['total_queries']}次查询)"):

            # Preferred modes
            if patterns["preferred_modes"]:
                st.markdown("**您的偏好模式:**")
                for mode, count in patterns["preferred_modes"][:3]:
                    percentage = count / sum(dict(patterns["preferred_modes"]).values()) * 100
                    st.write(f"• {mode}: {count}次 ({percentage:.1f}%)")

            # Quality trend
            avg_quality = patterns["avg_quality"]
            if avg_quality > 0:
                st.metric("平均查询质量", f"{avg_quality:.1%}")

            # Personalized recommendations
            if patterns["recommendations"]:
                st.markdown("**个性化建议:**")
                for rec in patterns["recommendations"]:
                    st.write(f"• {rec}")
