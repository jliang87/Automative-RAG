import re
import streamlit as st
from typing import Dict, List, Optional, Any, Tuple


class EmbeddedMetadataExtractor:
    """Extract and parse embedded metadata from chunk content."""

    @staticmethod
    def extract_embedded_metadata(content: str) -> Tuple[Dict[str, str], str]:
        """
        Extract embedded metadata from chunk content.

        Args:
            content: Chunk content with embedded metadata

        Returns:
            Tuple of (extracted_metadata_dict, clean_content_without_metadata)
        """
        # Pattern to match embedded metadata: 【key:value】
        metadata_pattern = r'【([^:：]+)[：:]([^】]+)】'

        # Find all metadata
        matches = re.findall(metadata_pattern, content)
        extracted_metadata = {key.strip(): value.strip() for key, value in matches}

        # Remove metadata from content to get clean text
        clean_content = re.sub(metadata_pattern, '', content).strip()

        return extracted_metadata, clean_content

    @staticmethod
    def parse_metadata_prefix(content: str) -> Tuple[str, str]:
        """
        Extract just the metadata prefix and return prefix + remaining content.

        Args:
            content: Content with potential metadata prefix

        Returns:
            Tuple of (metadata_prefix, remaining_content)
        """
        # Find the end of metadata prefix (first non-metadata content)
        metadata_pattern = r'^((?:【[^】]+】)*)(.*)'
        match = re.match(metadata_pattern, content, re.DOTALL)

        if match:
            prefix = match.group(1)
            remaining = match.group(2).strip()
            return prefix, remaining

        return "", content


def render_embedded_metadata_display(document: Dict[str, Any], show_full_content: bool = False) -> None:
    """
    Render embedded metadata display for a document/chunk.

    Args:
        document: Document dictionary with 'content' and 'metadata' keys
        show_full_content: Whether to show full content or just preview
    """
    content = document.get('content', '')
    structured_metadata = document.get('metadata', {})

    # Extract embedded metadata
    extractor = EmbeddedMetadataExtractor()
    embedded_metadata, clean_content = extractor.extract_embedded_metadata(content)
    metadata_prefix, _ = extractor.parse_metadata_prefix(content)

    # Create tabs for different views
    tab1, tab2, tab3 = st.tabs(["📊 元数据概览", "🏷️ 嵌入元数据", "📝 内容预览"])

    with tab1:
        render_metadata_overview(structured_metadata, embedded_metadata)

    with tab2:
        render_embedded_metadata_details(embedded_metadata, metadata_prefix, content)

    with tab3:
        render_content_preview(clean_content, content, show_full_content)


def render_metadata_overview(structured_metadata: Dict[str, Any], embedded_metadata: Dict[str, str]) -> None:
    """Render metadata overview comparing structured vs embedded metadata."""

    st.subheader("📊 元数据对比")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**🗂️ 结构化元数据 (后端)**")
        key_fields = ['model', 'manufacturer', 'year', 'category', 'title', 'author', 'source']

        overview_data = {}
        for field in key_fields:
            value = structured_metadata.get(field)
            if value:
                overview_data[field] = value

        if overview_data:
            for key, value in overview_data.items():
                # Truncate long values
                display_value = str(value)[:50] + "..." if len(str(value)) > 50 else str(value)
                st.text(f"{key}: {display_value}")
        else:
            st.info("无结构化元数据")

    with col2:
        st.markdown("**🏷️ 嵌入元数据 (文本中)**")
        if embedded_metadata:
            for key, value in embedded_metadata.items():
                # Truncate long values
                display_value = value[:50] + "..." if len(value) > 50 else value
                st.text(f"{key}: {display_value}")
        else:
            st.info("无嵌入元数据")

    # Metadata quality indicators
    st.markdown("---")
    render_metadata_quality_indicators(structured_metadata, embedded_metadata)


def render_metadata_quality_indicators(structured_metadata: Dict[str, Any], embedded_metadata: Dict[str, str]) -> None:
    """Render quality indicators for metadata."""

    st.markdown("**🔍 元数据质量检查**")

    col1, col2, col3 = st.columns(3)

    with col1:
        # Vehicle detection quality
        vehicle_confidence = structured_metadata.get('vehicle_confidence', 0)
        has_vehicle_info = structured_metadata.get('has_vehicle_info', False)

        if has_vehicle_info and vehicle_confidence > 0.7:
            st.success("🚗 车辆识别: 高质量")
        elif has_vehicle_info and vehicle_confidence > 0.4:
            st.warning("🚗 车辆识别: 中等质量")
        elif has_vehicle_info:
            st.error("🚗 车辆识别: 低质量")
        else:
            st.info("🚗 车辆识别: 未检测到")

    with col2:
        # Metadata injection status
        metadata_injected = structured_metadata.get('metadata_injected', False)
        if metadata_injected and embedded_metadata:
            st.success("💉 元数据注入: 成功")
        elif metadata_injected:
            st.warning("💉 元数据注入: 部分成功")
        else:
            st.error("💉 元数据注入: 失败")

    with col3:
        # Content enhancement
        original_length = structured_metadata.get('original_chunk_length', 0)
        enhanced_length = structured_metadata.get('enhanced_chunk_length', 0)
        enhancement_gain = enhanced_length - original_length

        if enhancement_gain > 50:
            st.success(f"📈 内容增强: +{enhancement_gain} 字符")
        elif enhancement_gain > 0:
            st.info(f"📈 内容增强: +{enhancement_gain} 字符")
        else:
            st.warning("📈 内容增强: 无增强")


def render_embedded_metadata_details(embedded_metadata: Dict[str, str], metadata_prefix: str,
                                     full_content: str) -> None:
    """Render detailed embedded metadata information."""

    if not embedded_metadata:
        st.warning("⚠️ 此块未检测到嵌入元数据")
        st.info("这可能表示元数据注入失败或内容不包含可提取的车辆信息")
        return

    st.subheader("🏷️ 嵌入元数据详情")

    # Show metadata prefix
    if metadata_prefix:
        st.markdown("**元数据前缀:**")
        st.code(metadata_prefix, language="text")

    # Show extracted metadata in a table
    st.markdown("**提取的元数据:**")

    metadata_rows = []
    for key, value in embedded_metadata.items():
        # Determine metadata type
        metadata_type = _determine_metadata_type(key)
        metadata_rows.append({
            "字段": key,
            "值": value,
            "类型": metadata_type,
            "长度": len(value)
        })

    if metadata_rows:
        st.dataframe(metadata_rows, use_container_width=True)

    # Show embedding pattern analysis
    st.markdown("**🔍 嵌入模式分析:**")
    embedding_analysis = _analyze_embedding_pattern(full_content, embedded_metadata)

    for analysis_key, analysis_value in embedding_analysis.items():
        if analysis_key == "pattern_count":
            st.metric("元数据块数量", analysis_value)
        elif analysis_key == "prefix_length":
            st.metric("前缀长度", f"{analysis_value} 字符")
        elif analysis_key == "embedding_ratio":
            st.metric("元数据占比", f"{analysis_value:.1%}")


def render_content_preview(clean_content: str, full_content: str, show_full: bool = False) -> None:
    """Render content preview with metadata highlighting."""

    st.subheader("📝 内容预览")

    # Show content comparison
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**🧹 清理后内容 (无元数据):**")
        preview_clean = clean_content[:300] + "..." if len(clean_content) > 300 and not show_full else clean_content
        st.text_area("清理后内容", preview_clean, height=150, disabled=True, key="clean_content")

    with col2:
        st.markdown("**📄 原始内容 (含元数据):**")
        preview_full = full_content[:300] + "..." if len(full_content) > 300 and not show_full else full_content
        st.text_area("原始内容", preview_full, height=150, disabled=True, key="full_content")

    # Content statistics
    st.markdown("**📊 内容统计:**")
    stats_col1, stats_col2, stats_col3 = st.columns(3)

    with stats_col1:
        st.metric("清理后长度", f"{len(clean_content)} 字符")

    with stats_col2:
        st.metric("原始长度", f"{len(full_content)} 字符")

    with stats_col3:
        metadata_overhead = len(full_content) - len(clean_content)
        st.metric("元数据开销", f"{metadata_overhead} 字符")


def render_metadata_summary_card(document: Dict[str, Any], compact: bool = True) -> None:
    """
    Render a compact metadata summary card for lists/grids.

    Args:
        document: Document with content and metadata
        compact: Whether to show compact view
    """
    structured_metadata = document.get('metadata', {})
    content = document.get('content', '')

    # Extract embedded metadata
    extractor = EmbeddedMetadataExtractor()
    embedded_metadata, _ = extractor.extract_embedded_metadata(content)

    if compact:
        # Compact card view
        with st.container():
            col1, col2, col3 = st.columns([2, 2, 1])

            with col1:
                # Vehicle info
                model = embedded_metadata.get('model') or structured_metadata.get('model')
                manufacturer = structured_metadata.get('manufacturer', '')
                if model:
                    st.markdown(f"**🚗 {manufacturer} {model}**")
                else:
                    st.markdown("**📄 通用文档**")

            with col2:
                # Source info
                title = embedded_metadata.get('title') or structured_metadata.get('title', '')
                if title:
                    display_title = title[:30] + "..." if len(title) > 30 else title
                    st.markdown(f"📺 {display_title}")

            with col3:
                # Quality indicator
                metadata_injected = structured_metadata.get('metadata_injected', False)
                if metadata_injected and embedded_metadata:
                    st.markdown("✅")
                elif embedded_metadata:
                    st.markdown("⚠️")
                else:
                    st.markdown("❌")

    else:
        # Full card view
        render_embedded_metadata_display(document, show_full_content=False)


def _determine_metadata_type(key: str) -> str:
    """Determine the type of metadata based on key."""
    vehicle_keys = ['model', 'manufacturer', 'year', 'category']
    source_keys = ['title', 'author', 'source']
    technical_keys = ['chunk_id', 'duration', 'view_count']

    if key in vehicle_keys:
        return "🚗 车辆信息"
    elif key in source_keys:
        return "📺 来源信息"
    elif key in technical_keys:
        return "🔧 技术信息"
    else:
        return "📋 其他"


def _analyze_embedding_pattern(content: str, embedded_metadata: Dict[str, str]) -> Dict[str, Any]:
    """Analyze the embedding pattern in content."""
    # Count metadata patterns
    pattern_count = len(re.findall(r'【[^】]+】', content))

    # Calculate prefix length
    prefix_match = re.match(r'^((?:【[^】]+】)*)', content)
    prefix_length = len(prefix_match.group(1)) if prefix_match else 0

    # Calculate embedding ratio
    total_length = len(content)
    embedding_ratio = prefix_length / total_length if total_length > 0 else 0

    return {
        "pattern_count": pattern_count,
        "prefix_length": prefix_length,
        "embedding_ratio": embedding_ratio,
        "has_embedded_metadata": len(embedded_metadata) > 0
    }


# Integration function for existing pages
def add_metadata_display_to_sources(documents: List[Dict[str, Any]]) -> None:
    """
    Add metadata display to existing source displays.
    This function can be called from existing source display functions.
    """
    if not documents:
        return

    st.markdown("### 🔍 元数据检查")

    with st.expander("查看文档元数据详情", expanded=False):
        for i, doc in enumerate(documents[:3]):  # Show first 3 docs to avoid overwhelming
            with st.container():
                st.markdown(f"#### 📄 文档 {i + 1}")
                render_embedded_metadata_display(doc, show_full_content=False)
                if i < len(documents[:3]) - 1:
                    st.markdown("---")


# Example usage function
def example_usage():
    """Example of how to use the metadata display components."""

    # Sample document with embedded metadata
    sample_document = {
        'content': '【model:星越L】【title:2023款吉利星越L深度评测】【year:2023】【category:SUV】今天我们来测试一下2023款吉利星越L的性能表现。这款SUV在动力方面搭载了2.0T涡轮增压发动机...',
        'metadata': {
            'model': '星越L',
            'manufacturer': 'Geely',
            'year': 2023,
            'category': 'SUV',
            'title': '2023款吉利星越L深度评测：性能、配置、油耗全面解析',
            'vehicle_confidence': 0.95,
            'has_vehicle_info': True,
            'metadata_injected': True,
            'original_chunk_length': 150,
            'enhanced_chunk_length': 200
        }
    }

    # Display the metadata
    render_embedded_metadata_display(sample_document)


if __name__ == "__main__":
    example_usage()