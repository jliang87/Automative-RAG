"""
Enhanced 文档浏览.py with metadata display integration
"""

import streamlit as st
import time
from typing import Dict, Any, List
from src.ui.api_client import api_request
from src.ui.session_init import initialize_session_state

# Import enhanced metadata display components
from src.ui.components.metadata_display import (
    render_metadata_summary_card,
    render_embedded_metadata_display,
    EmbeddedMetadataExtractor
)

initialize_session_state()

st.title("📚 文档浏览")
st.markdown("浏览向量数据库中的所有文档，包含详细元数据分析")


def get_vector_store_documents(limit: int = 100, source_filter: str = None) -> List[Dict]:
    """Get documents from vector store with optional filtering"""
    try:
        # Use the debug endpoint to get documents
        endpoint = "/query/debug-retrieval"

        # If we have a source filter, use it as a query to find matching docs
        if source_filter and source_filter != "全部":
            query = source_filter
        else:
            query = "document"  # Generic query to get all docs

        result = api_request(
            endpoint=endpoint,
            method="POST",
            data={"query": query, "limit": str(limit)}
        )

        if result and "documents" in result:
            docs = result["documents"]

            # Apply source filtering if specified
            if source_filter and source_filter != "全部":
                filtered_docs = []
                for doc in docs:
                    doc_source = doc.get("metadata", {}).get("source", "")
                    if source_filter.lower() in doc_source.lower():
                        filtered_docs.append(doc)
                return filtered_docs[:limit]

            return docs[:limit]

        return []
    except Exception as e:
        st.error(f"获取文档时出错: {str(e)}")
        return []


def format_document_source(metadata: Dict) -> str:
    """Format document source for display"""
    source = metadata.get("source", "unknown")
    source_icons = {
        "youtube": "🎬 YouTube",
        "bilibili": "📺 Bilibili",
        "pdf": "📄 PDF",
        "manual": "✍️ 手动输入",
        "video": "🎥 视频"
    }
    return source_icons.get(source, f"📄 {source}")


def display_enhanced_document_card(doc: Dict, index: int):
    """Display a document card with enhanced metadata display"""
    metadata = doc.get("metadata", {})
    content = doc.get("content", "")

    # Create expandable card
    with st.container():
        # Header with basic info
        col1, col2, col3 = st.columns([2, 1, 1])

        with col1:
            title = metadata.get("title", f"文档 {index + 1}")
            st.markdown(f"**{format_document_source(metadata)}**")
            st.markdown(f"📄 {title}")

        with col2:
            if metadata.get("author"):
                st.caption(f"👤 {metadata['author']}")
            if metadata.get("language"):
                st.caption(f"🌐 {metadata['language']}")

        with col3:
            # Document stats
            word_count = len(content.split()) if content else 0
            st.caption(f"📊 {word_count} 词")

            chunk_info = ""
            if metadata.get("chunk_index") is not None and metadata.get("total_chunks"):
                chunk_info = f"片段 {metadata['chunk_index'] + 1}/{metadata['total_chunks']}"
            elif metadata.get("chunk_index") is not None:
                chunk_info = f"片段 {metadata['chunk_index'] + 1}"

            if chunk_info:
                st.caption(f"🔢 {chunk_info}")

        # Enhanced metadata summary card
        st.markdown("**🏷️ 元数据摘要:**")
        # FIXED: Add unique_id to prevent key conflicts
        unique_id = f"doc_browser_{index}"
        render_metadata_summary_card(doc, compact=True, unique_id=unique_id)

        # Control buttons
        expand_key = f"expand_doc_{index}"
        metadata_key = f"show_metadata_{index}"

        col_btn1, col_btn2, col_btn3 = st.columns(3)

        with col_btn1:
            if st.button(f"📋 基本信息", key=f"expand_btn_{index}"):
                current_state = st.session_state.get(expand_key, False)
                st.session_state[expand_key] = not current_state
                # Close metadata view if opening basic info
                if not current_state:
                    st.session_state[metadata_key] = False
                st.rerun()

        with col_btn2:
            if st.button(f"🏷️ 详细元数据", key=f"metadata_btn_{index}"):
                current_state = st.session_state.get(metadata_key, False)
                st.session_state[metadata_key] = not current_state
                # Close basic info if opening metadata
                if not current_state:
                    st.session_state[expand_key] = False
                st.rerun()

        with col_btn3:
            # Quick metadata quality indicator
            extractor = EmbeddedMetadataExtractor()
            embedded_metadata, _ = extractor.extract_embedded_metadata(content)
            metadata_injected = metadata.get('metadata_injected', False)

            if metadata_injected and embedded_metadata:
                st.button("✅ 元数据优秀", key=f"quality_btn_{index}", disabled=True)
            elif embedded_metadata:
                st.button("⚠️ 元数据良好", key=f"quality_btn_{index}", disabled=True)
            else:
                st.button("❌ 元数据缺失", key=f"quality_btn_{index}", disabled=True)

        # Show basic details if expanded
        if st.session_state.get(expand_key, False):
            st.markdown("---")
            st.markdown("**📋 基本信息:**")

            # Basic info
            meta_col1, meta_col2 = st.columns(2)
            with meta_col1:
                st.write(f"**来源类型:** {format_document_source(metadata)}")
                if metadata.get("source_id"):
                    st.write(f"**来源ID:** {metadata['source_id']}")
                if metadata.get("url"):
                    st.write(f"**链接:** [查看原文]({metadata['url']})")
                if metadata.get("published_date"):
                    st.write(f"**发布日期:** {metadata['published_date']}")

            with meta_col2:
                if metadata.get("manufacturer"):
                    st.write(f"**制造商:** {metadata['manufacturer']}")
                if metadata.get("model"):
                    st.write(f"**车型:** {metadata['model']}")
                if metadata.get("year"):
                    st.write(f"**年份:** {metadata['year']}")
                if metadata.get("category"):
                    st.write(f"**类别:** {metadata['category']}")

            # Technical metadata
            if any(metadata.get(key) for key in ["duration", "view_count", "transcription_method"]):
                st.markdown("**🔧 技术信息:**")
                tech_col1, tech_col2 = st.columns(2)
                with tech_col1:
                    if metadata.get("duration"):
                        duration_mins = metadata['duration'] // 60
                        duration_secs = metadata['duration'] % 60
                        st.write(f"**视频时长:** {duration_mins}分{duration_secs}秒")
                    if metadata.get("transcription_method"):
                        st.write(f"**转录方法:** {metadata['transcription_method']}")

                with tech_col2:
                    if metadata.get("view_count"):
                        st.write(f"**观看次数:** {metadata['view_count']:,}")
                    if metadata.get("vehicle_confidence"):
                        confidence = metadata['vehicle_confidence']
                        st.write(f"**车辆识别置信度:** {confidence:.1%}")

            # Content section
            st.markdown("**📄 文档内容:**")
            if content:
                if len(content) > 500:
                    st.text_area("内容预览", content[:500] + "...", height=150, disabled=True, key=f"preview_content_{index}")
                    st.caption("💡 查看详细元数据可以看到完整内容分析")
                else:
                    st.text_area("文档内容", content, height=200, disabled=True, key=f"content_{index}")
            else:
                st.warning("此文档没有内容")

        # Show detailed metadata analysis if expanded
        if st.session_state.get(metadata_key, False):
            st.markdown("---")
            st.markdown("**🔍 详细元数据分析:**")
            # FIXED: Add unique_id to prevent key conflicts
            unique_id = f"doc_detailed_{index}"
            render_embedded_metadata_display(doc, show_full_content=True, unique_id=unique_id)

        st.divider()


def display_enhanced_statistics(documents: List[Dict]):
    """Display enhanced statistics with metadata analysis"""
    st.subheader(f"📊 文档统计 (共 {len(documents)} 个)")

    # Calculate enhanced statistics
    source_counts = {}
    total_words = 0
    docs_with_embedded_metadata = 0
    docs_with_vehicle_info = 0
    unique_vehicles = set()
    metadata_injection_successes = 0
    total_metadata_fields = 0

    extractor = EmbeddedMetadataExtractor()

    for doc in documents:
        metadata = doc.get("metadata", {})
        content = doc.get("content", "")
        source = metadata.get("source", "unknown")

        source_counts[source] = source_counts.get(source, 0) + 1

        if content:
            total_words += len(content.split())

            # Check for embedded metadata
            embedded_metadata, _ = extractor.extract_embedded_metadata(content)
            if embedded_metadata:
                docs_with_embedded_metadata += 1
                total_metadata_fields += len(embedded_metadata)

        # Check metadata injection status
        if metadata.get('metadata_injected'):
            metadata_injection_successes += 1

        # Check for vehicle info
        if metadata.get('has_vehicle_info') or metadata.get('model'):
            docs_with_vehicle_info += 1
            if metadata.get('model'):
                manufacturer = metadata.get('manufacturer', '')
                vehicle_name = f"{manufacturer} {metadata['model']}".strip()
                unique_vehicles.add(vehicle_name)

    # Display basic stats
    stats_cols = st.columns(len(source_counts) + 2)

    for i, (source, count) in enumerate(source_counts.items()):
        with stats_cols[i]:
            st.metric(format_document_source({"source": source}), count)

    with stats_cols[-2]:
        st.metric("📝 总词数", f"{total_words:,}")

    with stats_cols[-1]:
        st.metric("🚗 检测到车型", len(unique_vehicles))

    # Enhanced metadata quality metrics
    st.markdown("#### 🔍 元数据质量分析")
    quality_cols = st.columns(4)

    with quality_cols[0]:
        embedded_rate = (docs_with_embedded_metadata / len(documents) * 100) if documents else 0
        st.metric("🏷️ 嵌入元数据率", f"{embedded_rate:.0f}%")

    with quality_cols[1]:
        injection_rate = (metadata_injection_successes / len(documents) * 100) if documents else 0
        st.metric("💉 注入成功率", f"{injection_rate:.0f}%")

    with quality_cols[2]:
        vehicle_rate = (docs_with_vehicle_info / len(documents) * 100) if documents else 0
        st.metric("🚗 车辆检测率", f"{vehicle_rate:.0f}%")

    with quality_cols[3]:
        avg_fields = total_metadata_fields / max(docs_with_embedded_metadata, 1)
        st.metric("📊 平均元数据字段", f"{avg_fields:.1f}")

    # Quality assessment
    if embedded_rate > 80:
        st.success("🎯 文档质量优秀：大部分文档包含丰富的元数据")
    elif embedded_rate > 50:
        st.warning("⚡ 文档质量良好：部分文档缺少元数据")
    else:
        st.error("🔧 文档质量需要改进：元数据覆盖率较低")

    # Show detected vehicles if any
    if unique_vehicles:
        st.markdown("**🚗 检测到的车型:**")
        vehicles_list = list(unique_vehicles)[:10]  # Show first 10
        vehicles_display = ", ".join(vehicles_list)
        st.info(vehicles_display)
        if len(unique_vehicles) > 10:
            st.caption(f"... 还有 {len(unique_vehicles) - 10} 个车型")

    st.markdown("---")


# === FILTERS AND CONTROLS ===
st.subheader("🔍 筛选选项")

filter_col1, filter_col2, filter_col3, filter_col4 = st.columns(4)

with filter_col1:
    source_filter = st.selectbox(
        "文档来源",
        ["全部", "youtube", "bilibili", "pdf", "manual"],
        help="按文档来源类型筛选"
    )

with filter_col2:
    limit = st.selectbox(
        "显示数量",
        [50, 100, 200, 500],
        index=1,
        help="限制显示的文档数量"
    )

with filter_col3:
    sort_by = st.selectbox(
        "排序方式",
        ["默认", "按标题", "按来源", "按日期", "按元数据质量"],
        help="文档排序方式"
    )

with filter_col4:
    quality_filter = st.selectbox(
        "元数据质量",
        ["全部", "优秀", "良好", "缺失"],
        help="按元数据质量筛选"
    )

# Control buttons
control_col1, control_col2 = st.columns(2)

with control_col1:
    if st.button("🔄 刷新文档列表", use_container_width=True):
        st.rerun()

with control_col2:
    if st.button("📊 显示详细统计", use_container_width=True):
        st.session_state['show_detailed_stats'] = not st.session_state.get('show_detailed_stats', False)
        st.rerun()

# === DOCUMENT LOADING AND PROCESSING ===
with st.spinner("正在加载文档..."):
    documents = get_vector_store_documents(
        limit=limit,
        source_filter=source_filter if source_filter != "全部" else None
    )

if documents:
    # Apply quality filtering
    if quality_filter != "全部":
        extractor = EmbeddedMetadataExtractor()
        filtered_docs = []

        for doc in documents:
            content = doc.get("content", "")
            metadata = doc.get("metadata", {})
            embedded_metadata, _ = extractor.extract_embedded_metadata(content)

            if quality_filter == "优秀":
                if metadata.get('metadata_injected') and len(embedded_metadata) > 2:
                    filtered_docs.append(doc)
            elif quality_filter == "良好":
                if len(embedded_metadata) > 0:
                    filtered_docs.append(doc)
            elif quality_filter == "缺失":
                if len(embedded_metadata) == 0:
                    filtered_docs.append(doc)

        documents = filtered_docs

    # Display enhanced statistics
    display_enhanced_statistics(documents)

    # Show detailed statistics if requested
    if st.session_state.get('show_detailed_stats', False):
        st.subheader("📈 详细统计分析")

        with st.expander("查看处理质量分布", expanded=True):
            # Create processing quality distribution
            quality_distribution = {"优秀": 0, "良好": 0, "一般": 0, "缺失": 0}
            extractor = EmbeddedMetadataExtractor()

            for doc in documents:
                content = doc.get("content", "")
                metadata = doc.get("metadata", {})
                embedded_metadata, _ = extractor.extract_embedded_metadata(content)

                if metadata.get('metadata_injected') and len(embedded_metadata) > 2:
                    quality_distribution["优秀"] += 1
                elif len(embedded_metadata) > 1:
                    quality_distribution["良好"] += 1
                elif len(embedded_metadata) > 0:
                    quality_distribution["一般"] += 1
                else:
                    quality_distribution["缺失"] += 1

            # Display distribution
            dist_cols = st.columns(4)
            for i, (quality, count) in enumerate(quality_distribution.items()):
                with dist_cols[i]:
                    percentage = (count / len(documents) * 100) if documents else 0
                    st.metric(f"{quality}质量", f"{count} ({percentage:.1f}%)")

    # Sort documents
    if sort_by == "按标题":
        documents.sort(key=lambda x: x.get("metadata", {}).get("title", ""))
    elif sort_by == "按来源":
        documents.sort(key=lambda x: x.get("metadata", {}).get("source", ""))
    elif sort_by == "按日期":
        documents.sort(key=lambda x: x.get("metadata", {}).get("published_date", ""), reverse=True)
    elif sort_by == "按元数据质量":
        # Sort by metadata quality (embedded metadata count)
        extractor = EmbeddedMetadataExtractor()
        def get_metadata_score(doc):
            content = doc.get("content", "")
            metadata = doc.get("metadata", {})
            embedded_metadata, _ = extractor.extract_embedded_metadata(content)
            injection_bonus = 1 if metadata.get('metadata_injected') else 0
            return len(embedded_metadata) + injection_bonus

        documents.sort(key=get_metadata_score, reverse=True)

    # Display documents
    st.subheader("📚 文档列表")

    for i, doc in enumerate(documents):
        display_enhanced_document_card(doc, i)

else:
    st.info("📭 未找到文档")
    st.markdown("""
    **可能的原因:**
    - 向量数据库中暂无文档
    - 筛选条件过于严格
    - 服务暂时不可用

    **建议操作:**
    - 尝试上传一些文档
    - 调整筛选条件
    - 检查系统状态
    """)

# === NAVIGATION ===
st.markdown("---")
nav_cols = st.columns(4)

with nav_cols[0]:
    if st.button("📤 上传文档", use_container_width=True):
        st.switch_page("pages/数据摄取.py")

with nav_cols[1]:
    if st.button("🧠 智能查询", use_container_width=True):
        st.switch_page("pages/智能查询.py")

with nav_cols[2]:
    if st.button("📋 查看任务", use_container_width=True):
        st.switch_page("pages/后台任务.py")

with nav_cols[3]:
    if st.button("🏠 返回主页", use_container_width=True):
        st.switch_page("src/ui/主页.py")

st.caption("🔍 文档浏览 - 查看向量数据库中的所有文档，包含详细元数据分析")