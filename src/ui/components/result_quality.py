import streamlit as st
from typing import Dict

class ResultQualityIndicator:
    """Provides quality indicators for query results."""

    def calculate_result_quality(self, result: Dict[str, any]) -> Dict[str, any]:
        """Calculate quality metrics for query results."""
        quality = {
            "overall_score": 0.0,
            "document_relevance": 0.0,
            "answer_completeness": 0.0,
            "source_diversity": 0.0,
            "confidence_level": "medium"
        }

        documents = result.get("documents", [])
        answer = result.get("answer", "")

        # Document relevance (average of relevance scores)
        if documents:
            avg_relevance = sum(doc.get("relevance_score", 0) for doc in documents) / len(documents)
            quality["document_relevance"] = min(avg_relevance, 1.0)

        # Answer completeness (based on length and structure)
        if answer:
            word_count = len(answer.split())
            if word_count > 50:
                quality["answer_completeness"] = min(word_count / 100, 1.0)
            else:
                quality["answer_completeness"] = word_count / 50

        # Source diversity (unique sources)
        if documents:
            sources = set(doc.get("metadata", {}).get("source", "") for doc in documents)
            quality["source_diversity"] = min(len(sources) / 3, 1.0)

        # Overall score
        quality["overall_score"] = (
                quality["document_relevance"] * 0.4 +
                quality["answer_completeness"] * 0.4 +
                quality["source_diversity"] * 0.2
        )

        # Confidence level
        if quality["overall_score"] > 0.8:
            quality["confidence_level"] = "high"
        elif quality["overall_score"] > 0.6:
            quality["confidence_level"] = "medium"
        else:
            quality["confidence_level"] = "low"

        return quality


def render_result_quality_indicator(result: Dict[str, any]):
    """Render result quality indicators."""
    indicator = ResultQualityIndicator()
    quality = indicator.calculate_result_quality(result)

    # Quality badge
    score = quality["overall_score"]
    if score > 0.8:
        st.success(f"🌟 高质量结果 (质量分: {score:.1%})")
    elif score > 0.6:
        st.info(f"✅ 良好结果 (质量分: {score:.1%})")
    else:
        st.warning(f"⚠️ 一般结果 (质量分: {score:.1%})")

    # Detailed metrics
    with st.expander("📊 结果质量详情"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("文档相关性", f"{quality['document_relevance']:.1%}")
        with col2:
            st.metric("回答完整性", f"{quality['answer_completeness']:.1%}")
        with col3:
            st.metric("来源多样性", f"{quality['source_diversity']:.1%}")

        # Improvement suggestions
        if score < 0.7:
            st.markdown("**改进建议:**")
            suggestions = []
            if quality["document_relevance"] < 0.6:
                suggestions.append("尝试使用更具体的关键词")
            if quality["answer_completeness"] < 0.6:
                suggestions.append("考虑使用复杂分析模式获得更详细的回答")
            if quality["source_diversity"] < 0.5:
                suggestions.append("扩大搜索范围或减少筛选条件")

            for suggestion in suggestions:
                st.write(f"• {suggestion}")