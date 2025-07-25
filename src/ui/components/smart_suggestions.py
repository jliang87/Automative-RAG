import streamlit as st
import re
from typing import List, Dict, Optional, Tuple
from src.models import QueryMode


class SmartQuerySuggester:
    """Intelligent query suggestion system that analyzes user input and suggests optimal modes."""

    def __init__(self):
        self.mode_patterns = {
            QueryMode.FACTS: {
                "keywords": ["什么", "多少", "哪个", "规格", "参数", "尺寸", "重量", "容量", "功率", "扭矩"],
                "patterns": [r"\d+年.*?的.*?是多少", r".*?有什么.*?配置", r".*?的.*?参数"],
                "confidence_boost": 0.3
            },
            QueryMode.FEATURES: {
                "keywords": ["应该", "值得", "建议", "增加", "功能", "是否", "要不要", "考虑"],
                "patterns": [r"是否应该.*?", r".*?值得.*?吗", r"建议.*?功能"],
                "confidence_boost": 0.4
            },
            QueryMode.TRADEOFFS: {
                "keywords": ["vs", "对比", "优缺点", "利弊", "比较", "选择", "差异", "区别"],
                "patterns": [r".*?vs.*?", r".*?和.*?的.*?对比", r".*?优缺点"],
                "confidence_boost": 0.5
            },
            QueryMode.SCENARIOS: {
                "keywords": ["场景", "情况下", "使用时", "体验", "用户", "日常", "长途", "城市", "家庭"],
                "patterns": [r".*?场景下", r".*?使用体验", r".*?用户.*?如何"],
                "confidence_boost": 0.4
            },
            QueryMode.DEBATE: {
                "keywords": ["观点", "看法", "认为", "角度", "团队", "讨论", "专家", "分析师"],
                "patterns": [r".*?如何看待.*?", r".*?观点.*?", r"专家.*?认为"],
                "confidence_boost": 0.6
            },
            QueryMode.QUOTES: {
                "keywords": ["评价", "反馈", "评论", "用户说", "真实", "体验", "口碑", "评测"],
                "patterns": [r"用户.*?评价", r".*?真实.*?体验", r".*?怎么说"],
                "confidence_boost": 0.5
            }
        }

    def analyze_query(self, query: str) -> Dict[QueryMode, float]:
        """Analyze query and return confidence scores for each mode."""
        scores = {mode: 0.0 for mode in QueryMode}
        query_lower = query.lower()

        for mode, config in self.mode_patterns.items():
            # Keyword matching
            keyword_matches = sum(1 for kw in config["keywords"] if kw in query_lower)
            keyword_score = min(keyword_matches / len(config["keywords"]), 1.0)

            # Pattern matching
            pattern_matches = sum(1 for pattern in config["patterns"] if re.search(pattern, query))
            pattern_score = min(pattern_matches / len(config["patterns"]), 1.0)

            # Combined score with confidence boost
            base_score = (keyword_score * 0.6 + pattern_score * 0.4)
            scores[mode] = base_score + (base_score * config["confidence_boost"])

        return scores

    def suggest_mode(self, query: str) -> Tuple[QueryMode, float, List[str]]:
        """Suggest the best mode with confidence and reasoning."""
        scores = self.analyze_query(query)
        best_mode = max(scores, key=scores.get)
        confidence = scores[best_mode]

        # Generate reasoning
        reasoning = []
        if confidence > 0.7:
            reasoning.append(f"强烈推荐使用{best_mode.value}模式")
        elif confidence > 0.4:
            reasoning.append(f"建议使用{best_mode.value}模式")
        else:
            reasoning.append("查询较为通用，可使用基础查询模式")

        # Add specific reasoning based on detected patterns
        query_lower = query.lower()
        if "vs" in query_lower or "对比" in query_lower:
            reasoning.append("检测到对比关键词，适合权衡分析")
        if any(kw in query_lower for kw in ["应该", "建议", "值得"]):
            reasoning.append("检测到评估性问题，适合功能建议模式")
        if any(kw in query_lower for kw in ["什么", "多少", "参数"]):
            reasoning.append("检测到具体问题，适合事实查询模式")

        return best_mode, confidence, reasoning


def render_smart_suggestions(query: str) -> Optional[QueryMode]:
    """Render smart suggestions component in Streamlit."""
    if not query.strip():
        return None

    suggester = SmartQuerySuggester()
    suggested_mode, confidence, reasoning = suggester.suggest_mode(query)

    if confidence > 0.3:
        with st.container():
            st.markdown("### 🤖 智能建议")

            # Mode suggestion with confidence indicator
            if confidence > 0.7:
                st.success(f"🎯 **强烈推荐**: {suggested_mode.value}模式 (置信度: {confidence:.1%})")
            elif confidence > 0.5:
                st.info(f"💡 **建议**: {suggested_mode.value}模式 (置信度: {confidence:.1%})")
            else:
                st.warning(f"🤔 **可考虑**: {suggested_mode.value}模式 (置信度: {confidence:.1%})")

            # Show reasoning
            with st.expander("🧠 分析原因"):
                for reason in reasoning:
                    st.write(f"• {reason}")

            # Quick action buttons
            col1, col2 = st.columns(2)
            with col1:
                if st.button(f"✅ 使用{suggested_mode.value}模式", key="accept_suggestion"):
                    return suggested_mode
            with col2:
                if st.button("🔄 查看其他模式", key="show_all_modes"):
                    return "show_all"

    return None
