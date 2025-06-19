import streamlit as st
from typing import Dict, List, Any, Optional


class ValidationDisplaySystem:
    def __init__(self):
        self.confidence_config = {
            "high": {
                "color": "🟢",
                "label": "高可信度",
                "ui_type": "success",
                "description": "通过多项汽车领域验证"
            },
            "medium": {
                "color": "🟡",
                "label": "中等可信度",
                "ui_type": "info",
                "description": "基本通过验证，建议交叉确认"
            },
            "low": {
                "color": "🔴",
                "label": "需要谨慎",
                "ui_type": "warning",
                "description": "验证发现潜在问题"
            },
            "unknown": {
                "color": "⚪",
                "label": "无法验证",
                "ui_type": "error",
                "description": "验证过程异常"
            }
        }

    def render_main_validation_summary(self, result: Dict[str, Any]) -> None:
        """
        Section 1: Main validation summary - what users see first.
        Uses ONLY backend data, no frontend calculations.
        """
        automotive_validation = result.get("automotive_validation", {})

        if not automotive_validation:
            st.info("📋 此查询未进行专业验证")
            return

        # Get backend calculations
        confidence_level = automotive_validation.get("confidence_level", "unknown")
        has_warnings = automotive_validation.get("has_warnings", False)
        simple_confidence = result.get("simple_confidence", 0)

        config = self.confidence_config.get(confidence_level, self.confidence_config["unknown"])

        # Main validation header
        st.markdown("### 🛡️ 汽车领域验证结果")

        # Primary indicator
        col1, col2 = st.columns([1, 4])

        with col1:
            st.markdown(f"## {config['color']}")

        with col2:
            if simple_confidence > 0:
                confidence_text = f"**{config['label']}** ({simple_confidence:.0f}%)"
            else:
                confidence_text = f"**{config['label']}**"

            st.markdown(confidence_text)
            st.caption(config['description'])

        # Progress bar (if we have numeric confidence)
        if simple_confidence > 0:
            st.progress(simple_confidence / 100.0)

        # Status message
        if config["ui_type"] == "success" and not has_warnings:
            st.success("✅ 此回答已通过汽车领域专业验证，可作为可靠参考")
        elif config["ui_type"] == "info":
            st.info("📋 此回答具有一定可信度，建议结合其他来源验证")
        elif config["ui_type"] == "warning":
            st.warning("⚠️ 此回答存在需注意的信息，请谨慎参考")
        else:
            st.error("❌ 无法充分验证此回答，请通过权威渠道确认")

        # Show if backend added validation footnotes
        trust_features = automotive_validation.get("user_trust_features", {})
        if trust_features.get("validation_footnotes_added"):
            st.caption("📝 答案中已包含详细验证说明")

    def render_detailed_validation_breakdown(self, result: Dict[str, Any]) -> None:
        """
        Section 2: Detailed validation breakdown - for users who want specifics.
        Shows all backend validation details in organized manner.
        """
        automotive_validation = result.get("automotive_validation", {})

        if not automotive_validation:
            return

        with st.expander("🔍 查看详细验证报告", expanded=False):

            # Answer-level validation
            answer_validation = automotive_validation.get("answer_validation", {})
            if answer_validation:
                self._render_answer_validation_details(answer_validation)

            # Document-level validation
            doc_validation = automotive_validation.get("document_validation_summary", {})
            if doc_validation:
                self._render_document_validation_summary(doc_validation)

            # Source document details
            documents = result.get("documents", [])
            if documents:
                self._render_individual_document_status(documents)

    def _render_answer_validation_details(self, answer_validation: Dict[str, Any]) -> None:
        """Render answer-specific validation details from backend."""
        st.markdown("#### 📝 答案验证详情")

        # Validation metrics (calculated by backend)
        fact_summary = answer_validation.get("fact_check_summary", {})
        if fact_summary:
            col1, col2 = st.columns(2)

            with col1:
                if fact_summary.get("has_numerical_data"):
                    st.markdown("✅ 包含具体数值数据")
                else:
                    st.markdown("➖ 无具体数值数据")

                phrases_count = fact_summary.get("automotive_phrases_count", 0)
                st.markdown(f"🔧 {phrases_count} 个汽车专业术语")

            with col2:
                terms_count = fact_summary.get("automotive_terms_count", 0)
                st.markdown(f"🏷️ {terms_count} 个汽车关键词")

                sources_info = fact_summary.get("sources_with_warnings", "0/0")
                st.markdown(f"📚 来源验证: {sources_info}")

        # Specific warnings (from backend validation)
        answer_warnings = answer_validation.get("answer_warnings", [])
        if answer_warnings:
            st.markdown("**⚠️ 答案验证提醒:**")
            for warning in answer_warnings[:3]:  # Show top 3
                st.markdown(f"• {warning}")
            if len(answer_warnings) > 3:
                st.caption(f"还有 {len(answer_warnings) - 3} 项其他提醒...")

        source_warnings = answer_validation.get("source_warnings", [])
        if source_warnings:
            st.markdown("**📄 来源验证提醒:**")
            unique_warnings = list(set(source_warnings))[:3]
            for warning in unique_warnings:
                st.markdown(f"• {warning}")
            if len(source_warnings) > 3:
                st.caption(f"检测到 {len(set(source_warnings))} 种不同的来源提醒")

    def _render_document_validation_summary(self, doc_validation: Dict[str, Any]) -> None:
        """Render document validation summary from backend."""
        st.markdown("#### 📚 文档验证汇总")

        total_docs = doc_validation.get("total_documents", 0)
        docs_with_warnings = doc_validation.get("documents_with_warnings", 0)
        validation_quality = doc_validation.get("validation_quality", "unknown")

        # Overall document quality
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("总文档数", total_docs)
        with col2:
            st.metric("包含提醒", docs_with_warnings)
        with col3:
            warning_rate = (docs_with_warnings / total_docs * 100) if total_docs > 0 else 0
            st.metric("提醒率", f"{warning_rate:.0f}%")

        # Quality assessment
        if validation_quality == "high":
            st.success("🟢 文档质量评估: 优秀")
        elif validation_quality == "medium":
            st.info("🟡 文档质量评估: 良好")
        else:
            st.warning("🔴 文档质量评估: 需注意")

    def _render_individual_document_status(self, documents: List[Dict[str, Any]]) -> None:
        """Render individual document validation status."""
        st.markdown("#### 📄 各文档验证状态")

        for i, doc in enumerate(documents):
            metadata = doc.get("metadata", {})
            warnings = metadata.get("automotive_warnings", [])  # Backend-set
            relevance = doc.get("relevance_score", 0)
            title = metadata.get("title", f"文档 {i + 1}")

            # Document status line
            if warnings:
                status_icon = "⚠️"
                status_text = f"{len(warnings)} 项提醒"
            else:
                status_icon = "✅"
                status_text = "验证通过"

            relevance_icon = "🔥" if relevance > 0.8 else "⭐" if relevance > 0.6 else "📄"

            st.markdown(
                f"**{i + 1}.** {status_icon} {title[:40]}... | {relevance_icon} {relevance:.0%} | {status_text}")

            # Show warnings if any
            if warnings:
                for warning in warnings[:2]:  # Max 2 per document
                    st.caption(f"  • {warning}")
                if len(warnings) > 2:
                    st.caption(f"  • 还有 {len(warnings) - 2} 项...")

    def render_user_guidance(self, result: Dict[str, Any]) -> None:
        """
        Section 3: User guidance - actionable advice based on validation results.
        """
        automotive_validation = result.get("automotive_validation", {})
        confidence_level = automotive_validation.get("confidence_level", "unknown")

        guidance_map = {
            "high": {
                "title": "💡 使用建议 - 高可信度",
                "tips": [
                    "✅ 此信息已通过汽车领域专业验证，可作为可靠参考",
                    "🔗 建议查看引用来源获取更多详细信息",
                    "📋 如需最终决策，可结合具体车型配置确认"
                ]
            },
            "medium": {
                "title": "⚠️ 使用建议 - 中等可信度",
                "tips": [
                    "📚 建议通过多个来源交叉验证关键信息",
                    "🏪 重要参数可咨询官方经销商确认",
                    "📅 注意信息时效性，规格可能因年份而异"
                ]
            },
            "low": {
                "title": "🚨 使用建议 - 需要谨慎",
                "tips": [
                    "❗ 请务必通过权威渠道验证重要信息",
                    "📞 建议直接联系汽车制造商或授权经销商",
                    "⏰ 此信息可能存在准确性或时效性问题"
                ]
            },
            "unknown": {
                "title": "❓ 使用建议 - 无法验证",
                "tips": [
                    "🚫 请勿将此信息用于重要决策",
                    "🔍 建议重新查询或使用更具体的问题",
                    "📖 参考官方文档和权威汽车媒体"
                ]
            }
        }

        guidance = guidance_map.get(confidence_level, guidance_map["unknown"])

        with st.expander(guidance["title"]):
            for tip in guidance["tips"]:
                st.markdown(f"• {tip}")

            # Additional context about validation system
            st.markdown("---")
            st.markdown("**🔬 关于验证系统:**")
            st.caption("• 本系统专门针对汽车领域进行事实验证")
            st.caption("• 验证内容包括技术规格、性能数据、配置信息等")
            st.caption("• 验证结果仅供参考，重要决策请咨询专业人士")


# MAIN RENDERING FUNCTIONS - Replace both old components

def render_unified_validation_display(result: Dict[str, Any]) -> None:
    """
    MAIN: Unified validation display replacing both fact-checker and trust components.

    This single function provides:
    - Main validation summary (replaces trust_indicators main display)
    - Detailed breakdown (replaces fact_checker detailed display)
    - User guidance (combines both systems' guidance)
    """
    validation_system = ValidationDisplaySystem()

    # Section 1: Main summary - always visible
    validation_system.render_main_validation_summary(result)

    # Section 2: Detailed breakdown - expandable
    validation_system.render_detailed_validation_breakdown(result)

    # Section 3: User guidance - expandable
    validation_system.render_user_guidance(result)


def render_quick_validation_badge(result: Dict[str, Any]) -> str:
    """
    Quick validation badge for inline display.
    Replaces both quick trust badge and fact-check badge functions.
    """
    automotive_validation = result.get("automotive_validation", {})

    if not automotive_validation:
        return "⚪ **验证状态未知**"

    confidence = automotive_validation.get("confidence_level", "unknown")
    simple_confidence = result.get("simple_confidence", 0)

    config = {
        "high": {"color": "🟢", "label": "高可信"},
        "medium": {"color": "🟡", "label": "中等可信"},
        "low": {"color": "🔴", "label": "需谨慎"},
        "unknown": {"color": "⚪", "label": "未验证"}
    }

    badge_config = config.get(confidence, config["unknown"])

    if simple_confidence > 0:
        return f"{badge_config['color']} **{badge_config['label']}** ({simple_confidence:.0f}%)"
    else:
        return f"{badge_config['color']} **{badge_config['label']}**"


def render_validation_help() -> None:
    """
    Help/explanation component for the validation system.
    Replaces multiple help components from both old systems.
    """
    with st.expander("❓ 验证系统说明", expanded=False):
        st.markdown("""
        ### 🛡️ 汽车领域专业验证系统

        **本系统如何工作:**

        #### 🔍 验证内容
        - **技术规格**: 功率、扭矩、加速时间、油耗等参数合理性
        - **配置信息**: 后备箱容积、尺寸数据、容量规格等
        - **性能数据**: 最高时速、制动距离、续航里程等
        - **来源质量**: 文档可信度、信息一致性、OCR质量等

        #### 📊 可信度级别
        - **🟢 高可信度**: 多项验证通过，数据合理，来源可靠
        - **🟡 中等可信度**: 基本验证通过，建议交叉确认
        - **🔴 需要谨慎**: 发现潜在问题，请通过权威渠道验证
        - **⚪ 无法验证**: 验证过程异常或信息不足

        #### 💡 使用建议
        - 本验证仅为辅助参考，不替代专业咨询
        - 重要决策前请通过官方渠道确认
        - 注意信息时效性，规格可能因年份和配置而异
        - 结合多个权威来源进行交叉验证

        #### 🔬 技术原理
        - 基于大量汽车数据训练的专业验证模型
        - 智能识别中文汽车术语和技术表达
        - 实时检测数值规格的合理性范围
        - 多层次验证：答案级别 + 文档级别 + 来源级别
        """)


def render_real_time_validation_feedback(query: str) -> None:
    """Show real-time feedback about query quality for validation."""
    if not query.strip():
        return

    # Analyze query for validation-friendly characteristics
    import re

    has_specific_model = bool(re.search(r'(宝马|奔驰|奥迪|丰田|特斯拉|X5|Model\s*\d)', query))
    has_year = bool(re.search(r'20\d{2}年?|2\d{3}', query))
    has_spec_terms = bool(re.search(r'(容积|加速|油耗|功率|马力|后备箱|百公里)', query))

    score = sum([has_specific_model, has_year, has_spec_terms])

    if score >= 2:
        st.info("✅ 您的查询包含具体信息，有利于获得高质量的验证结果")
    elif score == 1:
        st.warning("💡 建议添加具体的车型、年份或技术参数，以获得更准确的验证")
    else:
        st.warning("❓ 查询较为宽泛，可能影响验证的准确性")