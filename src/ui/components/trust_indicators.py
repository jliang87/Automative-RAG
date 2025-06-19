import streamlit as st
import re
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime


class TrustIndicatorSystem:
    """Comprehensive trust indicators for automotive AI responses."""

    def __init__(self):
        self.trust_levels = {
            "high": {"color": "🟢", "label": "高可信", "threshold": 0.8},
            "medium": {"color": "🟡", "label": "中等可信", "threshold": 0.6},
            "low": {"color": "🔴", "label": "需谨慎", "threshold": 0.4},
            "uncertain": {"color": "⚪", "label": "不确定", "threshold": 0.0}
        }

        # Chinese automotive spec validation patterns
        self.spec_patterns = {
            "acceleration": r"(\d+\.?\d*)\s*秒.*?(百公里|0-100)",
            "power": r"(\d+)\s*(?:马力|HP|千瓦|kW)",
            "torque": r"(\d+)\s*(?:牛米|Nm)",
            "consumption": r"(\d+\.?\d*)\s*(?:升|L).*?(?:百公里|100)",
            "speed": r"(\d+)\s*(?:公里|km).*?(?:时速|小时)",
            "capacity": r"(\d+)\s*(?:升|L).*?(?:后备箱|容积|空间)"
        }

        # Realistic ranges for validation
        self.realistic_ranges = {
            "acceleration": (2.0, 20.0),  # 0-100 km/h in seconds
            "power": (50, 2000),  # HP
            "torque": (50, 2000),  # Nm
            "consumption": (3.0, 25.0),  # L/100km
            "speed": (120, 400),  # km/h top speed
            "capacity": (100, 2000)  # trunk capacity in liters
        }

    def calculate_overall_trust_score(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate comprehensive trust score based on multiple factors."""

        trust_factors = {
            "automotive_validation": 0.0,
            "source_quality": 0.0,
            "numerical_accuracy": 0.0,
            "citation_completeness": 0.0,
            "content_coherence": 0.0
        }

        # Factor 1: Automotive domain validation (40% weight)
        automotive_validation = result.get("automotive_validation", {})
        if automotive_validation:
            confidence = automotive_validation.get("confidence_level", "unknown")
            confidence_scores = {"high": 1.0, "medium": 0.7, "low": 0.3, "unknown": 0.1}
            trust_factors["automotive_validation"] = confidence_scores.get(confidence, 0.1)

            # Reduce score if there are warnings
            if automotive_validation.get("has_warnings", False):
                trust_factors["automotive_validation"] *= 0.8

        # Factor 2: Source quality (25% weight)
        documents = result.get("documents", [])
        if documents:
            total_relevance = sum(doc.get("relevance_score", 0) for doc in documents)
            avg_relevance = total_relevance / len(documents) if documents else 0
            trust_factors["source_quality"] = min(avg_relevance, 1.0)

            # Check for source diversity
            sources = set(doc.get("metadata", {}).get("source", "") for doc in documents)
            if len(sources) > 1:
                trust_factors["source_quality"] *= 1.1  # Bonus for diversity

        # Factor 3: Numerical accuracy (20% weight)
        answer = result.get("answer", "")
        trust_factors["numerical_accuracy"] = self._validate_numerical_claims(answer)

        # Factor 4: Citation completeness (10% weight)
        trust_factors["citation_completeness"] = self._check_citation_quality(answer)

        # Factor 5: Content coherence (5% weight)
        trust_factors["content_coherence"] = self._assess_content_coherence(answer)

        # Calculate weighted score
        weights = {
            "automotive_validation": 0.40,
            "source_quality": 0.25,
            "numerical_accuracy": 0.20,
            "citation_completeness": 0.10,
            "content_coherence": 0.05
        }

        overall_score = sum(
            trust_factors[factor] * weights[factor]
            for factor in trust_factors
        )

        # Determine trust level
        trust_level = "uncertain"
        for level, config in self.trust_levels.items():
            if overall_score >= config["threshold"]:
                trust_level = level
                break

        return {
            "overall_score": overall_score,
            "trust_level": trust_level,
            "factors": trust_factors,
            "breakdown": self._generate_trust_breakdown(trust_factors, weights)
        }

    def _validate_numerical_claims(self, text: str) -> float:
        """Validate numerical claims against realistic automotive ranges."""

        if not text:
            return 0.5

        valid_claims = 0
        total_claims = 0

        for spec_type, pattern in self.spec_patterns.items():
            matches = re.findall(pattern, text, re.IGNORECASE)

            for match in matches:
                total_claims += 1
                try:
                    if isinstance(match, tuple):
                        value = float(match[0])
                    else:
                        value = float(match)

                    min_val, max_val = self.realistic_ranges[spec_type]
                    if min_val <= value <= max_val:
                        valid_claims += 1

                except (ValueError, KeyError):
                    continue

        if total_claims == 0:
            return 0.7  # Neutral score if no numerical claims

        return valid_claims / total_claims

    def _check_citation_quality(self, text: str) -> float:
        """Check quality and completeness of citations."""

        if not text:
            return 0.0

        # Count citation patterns
        citation_patterns = [
            r'【来源[：:]?DOC_\d+】',
            r'【来源[：:]?文档\d+】',
            r'【来源[：:]?\d+】'
        ]

        citations = 0
        for pattern in citation_patterns:
            citations += len(re.findall(pattern, text))

        # Count factual sentences (rough estimation)
        sentences = re.split(r'[。！？]', text)
        factual_sentences = 0

        for sentence in sentences:
            # Look for sentences with specific claims
            if any(keyword in sentence for keyword in
                   ['是', '为', '达到', '具有', '配备', '搭载', '采用']):
                factual_sentences += 1

        if factual_sentences == 0:
            return 0.5

        citation_ratio = citations / factual_sentences
        return min(citation_ratio, 1.0)

    def _assess_content_coherence(self, text: str) -> float:
        """Assess logical coherence and consistency of content."""

        if not text:
            return 0.0

        # Basic coherence indicators
        coherence_score = 0.5  # Start with neutral

        # Check for contradictions (simple heuristic)
        contradiction_patterns = [
            (r'不.*?有', r'有.*?'),  # "没有" vs "有"
            (r'无法.*?', r'可以.*?'),  # "无法" vs "可以"
            (r'不支持', r'支持')  # "不支持" vs "支持"
        ]

        contradictions = 0
        for neg_pattern, pos_pattern in contradiction_patterns:
            if re.search(neg_pattern, text) and re.search(pos_pattern, text):
                contradictions += 1

        if contradictions > 0:
            coherence_score *= 0.7

        # Check for proper structure
        if '【' in text and '】' in text:  # Has citations
            coherence_score += 0.2

        if len(text.split('。')) > 2:  # Multiple sentences
            coherence_score += 0.1

        return min(coherence_score, 1.0)

    def _generate_trust_breakdown(self, factors: Dict[str, float], weights: Dict[str, float]) -> List[Dict[str, Any]]:
        """Generate detailed breakdown of trust factors."""

        breakdown = []

        factor_names = {
            "automotive_validation": "汽车领域验证",
            "source_quality": "来源质量",
            "numerical_accuracy": "数值准确性",
            "citation_completeness": "引用完整性",
            "content_coherence": "内容连贯性"
        }

        for factor, score in factors.items():
            weight = weights[factor]
            contribution = score * weight

            breakdown.append({
                "name": factor_names[factor],
                "score": score,
                "weight": weight,
                "contribution": contribution,
                "status": "良好" if score > 0.7 else "一般" if score > 0.4 else "需改进"
            })

        return sorted(breakdown, key=lambda x: x["contribution"], reverse=True)

    def display_trust_indicator(self, result: Dict[str, Any]) -> None:
        """Display main trust indicator with clear visual feedback."""

        trust_analysis = self.calculate_overall_trust_score(result)
        trust_level = trust_analysis["trust_level"]
        overall_score = trust_analysis["overall_score"]

        config = self.trust_levels[trust_level]

        # Main trust indicator
        st.markdown("### 🛡️ 信息可信度评估")

        col1, col2 = st.columns([1, 3])

        with col1:
            st.markdown(f"## {config['color']}")

        with col2:
            st.markdown(f"**{config['label']}** ({overall_score:.1%})")

            if trust_level == "high":
                st.success("此回答经过多项验证，可信度较高")
            elif trust_level == "medium":
                st.info("此回答具有一定可信度，建议参考多个来源")
            elif trust_level == "low":
                st.warning("此回答存在不确定因素，请谨慎使用")
            else:
                st.error("无法充分验证此回答的可信度")

        # Progress bar
        st.progress(overall_score)

        # Detailed breakdown
        self._display_trust_breakdown(trust_analysis["breakdown"])

    def _display_trust_breakdown(self, breakdown: List[Dict[str, Any]]) -> None:
        """Display detailed trust factor breakdown."""

        with st.expander("🔍 查看详细评估"):
            st.markdown("**各项指标评估:**")

            for factor in breakdown:
                col1, col2, col3 = st.columns([2, 1, 1])

                with col1:
                    st.write(f"**{factor['name']}**")

                with col2:
                    score_color = "🟢" if factor['score'] > 0.7 else "🟡" if factor['score'] > 0.4 else "🔴"
                    st.write(f"{score_color} {factor['score']:.1%}")

                with col3:
                    st.write(factor['status'])

                # Progress bar for each factor
                st.progress(factor['score'])
                st.caption(f"权重: {factor['weight']:.0%} | 贡献: {factor['contribution']:.1%}")
                st.markdown("---")

    def display_trust_tips(self, trust_level: str) -> None:
        """Display context-appropriate tips for users."""

        tips = {
            "high": [
                "✅ 此信息已通过多项验证，可以作为可靠参考",
                "💡 如需最终决策，建议确认具体车型和配置",
                "🔗 可查看引用的来源文档获取更多详情"
            ],
            "medium": [
                "⚠️ 建议交叉验证关键信息",
                "📚 参考官方说明书或经销商确认具体参数",
                "🤔 注意信息的时效性，规格可能因年份而异"
            ],
            "low": [
                "❗ 请务必通过权威渠道验证重要信息",
                "📞 建议直接咨询汽车经销商或制造商",
                "⏰ 信息可能不够准确或已过时"
            ],
            "uncertain": [
                "🚫 无法确认信息准确性，请勿作为决策依据",
                "🔍 建议重新查询或使用更具体的问题",
                "📖 参考官方文档和权威汽车媒体"
            ]
        }

        if trust_level in tips:
            st.markdown("**💡 使用建议:**")
            for tip in tips[trust_level]:
                st.markdown(f"• {tip}")

    def display_improvement_suggestions(self, result: Dict[str, Any]) -> None:
        """Suggest ways to improve query for better trust scores."""

        trust_analysis = self.calculate_overall_trust_score(result)
        factors = trust_analysis["factors"]

        suggestions = []

        # Low source quality
        if factors["source_quality"] < 0.6:
            suggestions.append("🎯 尝试使用更具体的车型和年份信息")

        # Low numerical accuracy
        if factors["numerical_accuracy"] < 0.6:
            suggestions.append("📊 查询具体的技术参数而非模糊描述")

        # Low citation completeness
        if factors["citation_completeness"] < 0.5:
            suggestions.append("🔍 询问更具体的问题以获得详细引用")

        if suggestions:
            with st.expander("💡 提升查询质量的建议"):
                for suggestion in suggestions:
                    st.markdown(f"• {suggestion}")


def render_trust_indicators(result: Dict[str, Any]) -> None:
    """Main function to render complete trust indicator system."""

    trust_system = TrustIndicatorSystem()

    # Main trust indicator
    trust_system.display_trust_indicator(result)

    # Trust tips
    trust_analysis = trust_system.calculate_overall_trust_score(result)
    trust_system.display_trust_tips(trust_analysis["trust_level"])

    # Improvement suggestions
    trust_system.display_improvement_suggestions(result)


def render_quick_trust_badge(result: Dict[str, Any]) -> str:
    """Render a quick trust badge for inline display."""

    trust_system = TrustIndicatorSystem()
    trust_analysis = trust_system.calculate_overall_trust_score(result)

    trust_level = trust_analysis["trust_level"]
    config = trust_system.trust_levels[trust_level]
    score = trust_analysis["overall_score"]

    return f"{config['color']} **{config['label']}** ({score:.0%})"
