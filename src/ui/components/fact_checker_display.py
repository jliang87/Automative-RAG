import streamlit as st
import re
from typing import Dict, List, Any, Optional


class FactCheckingDisplay:
    """Enhanced display component for automotive fact-checking results."""

    def __init__(self):
        self.confidence_colors = {
            "high": "🟢",
            "medium": "🟡",
            "low": "🔴",
            "unknown": "⚪",
            "error": "❌"
        }

        self.warning_icons = {
            "acceleration": "🏁",
            "spec": "📊",
            "ocr": "👁️",
            "general": "⚠️"
        }

    def display_fact_check_summary(self, result: Dict[str, Any]) -> None:
        """Display main fact-checking summary with clear visual indicators."""

        # Get automotive validation data
        automotive_validation = result.get("automotive_validation", {})
        if not automotive_validation:
            return

        confidence = automotive_validation.get("confidence_level", "unknown")
        has_warnings = automotive_validation.get("has_warnings", False)

        # Main confidence indicator
        confidence_icon = self.confidence_colors.get(confidence, "⚪")
        confidence_text = {
            "high": "高可信度",
            "medium": "中等可信度",
            "low": "低可信度",
            "unknown": "置信度未知",
            "error": "验证错误"
        }.get(confidence, "未知")

        # Display main indicator
        if confidence == "high" and not has_warnings:
            st.success(f"{confidence_icon} **汽车领域验证**: {confidence_text}")
        elif confidence == "medium" or (confidence == "high" and has_warnings):
            st.info(f"{confidence_icon} **汽车领域验证**: {confidence_text}")
        else:
            st.warning(f"{confidence_icon} **汽车领域验证**: {confidence_text}")

        # Show detailed validation info if available
        if has_warnings:
            self._display_detailed_warnings(automotive_validation)

        # Show trust features
        trust_features = automotive_validation.get("user_trust_features", {})
        if trust_features.get("validation_footnotes_added"):
            st.caption("✅ 答案已添加验证说明以提高透明度")

    def _display_detailed_warnings(self, validation_data: Dict[str, Any]) -> None:
        """Display detailed warnings in an expandable section."""

        answer_validation = validation_data.get("answer_validation", {})
        answer_warnings = answer_validation.get("answer_warnings", [])
        source_warnings = answer_validation.get("source_warnings", [])

        if not answer_warnings and not source_warnings:
            return

        with st.expander("🔍 查看验证详情", expanded=False):

            # Answer warnings
            if answer_warnings:
                st.markdown("**📝 答案验证提醒:**")
                for warning in answer_warnings[:3]:  # Limit to 3 most important
                    warning_type = self._categorize_warning(warning)
                    icon = self.warning_icons.get(warning_type, "⚠️")
                    st.markdown(f"{icon} {warning}")

                if len(answer_warnings) > 3:
                    st.caption(f"还有 {len(answer_warnings) - 3} 项其他提醒...")

            # Source warnings
            if source_warnings:
                st.markdown("**📚 来源验证提醒:**")
                unique_warnings = list(set(source_warnings))[:3]
                for warning in unique_warnings:
                    warning_type = self._categorize_warning(warning)
                    icon = self.warning_icons.get(warning_type, "⚠️")
                    st.markdown(f"{icon} {warning}")

                if len(source_warnings) > 3:
                    st.caption(f"检测到 {len(set(source_warnings))} 种不同的来源提醒")

            # Fact-check summary
            fact_summary = answer_validation.get("fact_check_summary", {})
            if fact_summary:
                self._display_fact_check_metrics(fact_summary)

    def _display_fact_check_metrics(self, summary: Dict[str, Any]) -> None:
        """Display fact-checking metrics in a clean format."""

        st.markdown("**🔬 验证指标:**")

        col1, col2 = st.columns(2)

        with col1:
            if summary.get("has_numerical_data"):
                st.markdown("✅ 包含数值数据")
            else:
                st.markdown("➖ 无具体数值")

            phrases_count = summary.get("automotive_phrases_count", 0)
            if phrases_count > 0:
                st.markdown(f"✅ {phrases_count} 个汽车专业术语")

        with col2:
            terms_count = summary.get("automotive_terms_count", 0)
            if terms_count > 0:
                st.markdown(f"✅ {terms_count} 个汽车关键词")

            sources_info = summary.get("sources_with_warnings", "0/0")
            if "/" in sources_info:
                warned, total = sources_info.split("/")
                if int(warned) == 0:
                    st.markdown(f"✅ 来源验证通过 ({total}个)")
                else:
                    st.markdown(f"⚠️ 来源提醒 ({warned}/{total})")

    def _categorize_warning(self, warning: str) -> str:
        """Categorize warning type for appropriate icon selection."""
        warning_lower = warning.lower()

        if "加速" in warning or "acceleration" in warning_lower:
            return "acceleration"
        elif "规格" in warning or "数据" in warning or "spec" in warning_lower:
            return "spec"
        elif "ocr" in warning_lower or "识别" in warning:
            return "ocr"
        else:
            return "general"

    def display_document_validation(self, documents: List[Dict[str, Any]]) -> None:
        """Display validation status for source documents."""

        if not documents:
            return

        # Count documents with warnings
        docs_with_warnings = 0
        total_warnings = 0

        for doc in documents:
            metadata = doc.get("metadata", {})
            if metadata.get("automotive_warnings"):
                docs_with_warnings += 1
                total_warnings += len(metadata["automotive_warnings"])

        if docs_with_warnings == 0:
            st.success(f"✅ 所有 {len(documents)} 个来源文档通过汽车领域验证")
            return

        # Show warning summary
        warning_rate = docs_with_warnings / len(documents)

        if warning_rate < 0.3:
            st.info(f"📋 {docs_with_warnings}/{len(documents)} 个文档包含验证提醒")
        else:
            st.warning(f"⚠️ {docs_with_warnings}/{len(documents)} 个文档包含验证提醒")

        # Show detailed document validation in expander
        with st.expander("📚 查看文档验证详情"):
            for i, doc in enumerate(documents):
                metadata = doc.get("metadata", {})
                warnings = metadata.get("automotive_warnings", [])

                if warnings:
                    st.markdown(f"**文档 {i + 1}**: {metadata.get('title', '未知标题')[:50]}...")
                    for warning in warnings[:2]:  # Max 2 per document
                        st.caption(f"⚠️ {warning}")
                    if len(warnings) > 2:
                        st.caption(f"还有 {len(warnings) - 2} 项提醒...")
                else:
                    st.markdown(f"**文档 {i + 1}**: ✅ 验证通过")

    def display_confidence_explanation(self, confidence: str) -> None:
        """Display explanation of confidence levels to educate users."""

        explanations = {
            "high": {
                "meaning": "答案基于可靠来源，包含具体数据，未发现明显错误",
                "recommendation": "可以较为放心地参考这个答案",
                "color": "success"
            },
            "medium": {
                "meaning": "答案有一定依据，但可能存在部分不确定信息",
                "recommendation": "建议结合其他来源进行验证",
                "color": "info"
            },
            "low": {
                "meaning": "答案缺乏充分验证，或检测到潜在问题",
                "recommendation": "请谨慎使用，建议查阅权威资料确认",
                "color": "warning"
            }
        }

        if confidence in explanations:
            exp = explanations[confidence]

            with st.expander("❓ 置信度说明"):
                if exp["color"] == "success":
                    st.success(f"**{confidence.upper()}置信度含义**: {exp['meaning']}")
                elif exp["color"] == "info":
                    st.info(f"**{confidence.upper()}置信度含义**: {exp['meaning']}")
                else:
                    st.warning(f"**{confidence.upper()}置信度含义**: {exp['meaning']}")

                st.markdown(f"**建议**: {exp['recommendation']}")

                st.markdown("""
                **验证依据包括:**
                - 🔢 是否包含具体数值和技术参数
                - 🏭 是否使用汽车行业专业术语
                - 📚 来源文档的质量和一致性
                - ⚠️ 是否检测到可疑或不合理的数据
                """)


def render_fact_checking_display(result: Dict[str, Any]) -> None:
    """Main function to render the complete fact-checking display."""

    checker = FactCheckingDisplay()

    # Main fact-check summary
    checker.display_fact_check_summary(result)

    # Document validation
    documents = result.get("documents", [])
    if documents:
        checker.display_document_validation(documents)

    # Confidence explanation
    automotive_validation = result.get("automotive_validation", {})
    if automotive_validation:
        confidence = automotive_validation.get("confidence_level", "unknown")
        checker.display_confidence_explanation(confidence)


def render_real_time_validation_feedback(query: str) -> None:
    """Show real-time feedback about query quality for fact-checking."""

    if not query.strip():
        return

    # Analyze query for fact-checkability
    has_specific_model = bool(re.search(r'(宝马|奔驰|奥迪|丰田|特斯拉|X5|Model\s*\d)', query))
    has_year = bool(re.search(r'20\d{2}年?|2\d{3}', query))
    has_spec_terms = bool(re.search(r'(容积|加速|油耗|功率|马力|后备箱|百公里)', query))

    score = sum([has_specific_model, has_year, has_spec_terms])

    if score >= 2:
        st.info("✅ 您的查询包含具体信息，有利于获得高质量的验证结果")
    elif score == 1:
        st.warning("💡 建议添加具体的车型、年份或技术参数，以获得更准确的验证")
    else:
        st.warning("❓ 查询较为宽泛，可能影响事实验证的准确性")
