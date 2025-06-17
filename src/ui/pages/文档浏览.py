"""
New page: src/ui/pages/文档浏览.py
Browse all documents in the vector store with filtering and search
"""

import streamlit as st
import time
from typing import Dict, Any, List
from src.ui.api_client import api_request
from src.ui.session_init import initialize_session_state

initialize_session_state()

st.title("📚 文档浏览")
st.markdown("浏览向量数据库中的所有文档")


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
            data={"query": query}
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


def display_document_card(doc: Dict, index: int):
    """Display a document card with all metadata"""
    metadata = doc.get("metadata", {})
    content = doc.get("content", "")

    # Create expandable card
    with st.container():
        # Header
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
            if metadata.get("chunk_id") is not None and metadata.get("total_chunks"):
                chunk_info = f"片段 {metadata['chunk_id'] + 1}/{metadata['total_chunks']}"
            elif metadata.get("chunk_id") is not None:
                chunk_info = f"片段 {metadata['chunk_id'] + 1}"

            if chunk_info:
                st.caption(f"🔢 {chunk_info}")

        # Expandable details
        expand_key = f"expand_doc_{index}"
        if st.button(f"📋 查看详情", key=f"expand_btn_{index}"):
            if expand_key not in st.session_state:
                st.session_state[expand_key] = False
            st.session_state[expand_key] = not st.session_state[expand_key]
            st.rerun()

        # Show details if expanded
        if st.session_state.get(expand_key, False):
            st.markdown("---")

            # Metadata section
            st.markdown("**📋 元数据信息:**")

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
            if any(metadata.get(key) for key in ["engine_type", "transmission", "length", "views"]):
                st.markdown("**🔧 技术信息:**")
                tech_col1, tech_col2 = st.columns(2)
                with tech_col1:
                    if metadata.get("engine_type"):
                        st.write(f"**发动机类型:** {metadata['engine_type']}")
                    if metadata.get("transmission"):
                        st.write(f"**变速箱:** {metadata['transmission']}")

                with tech_col2:
                    if metadata.get("length"):
                        duration_mins = metadata['length'] // 60
                        duration_secs = metadata['length'] % 60
                        st.write(f"**视频时长:** {duration_mins}分{duration_secs}秒")
                    if metadata.get("views"):
                        st.write(f"**观看次数:** {metadata['views']:,}")

            # Custom metadata
            custom_meta = metadata.get("custom_metadata", {})
            if custom_meta and isinstance(custom_meta, dict):
                st.markdown("**⚙️ 自定义元数据:**")
                for key, value in custom_meta.items():
                    if value:
                        st.write(f"**{key}:** {value}")

            # Content section
            st.markdown("**📄 文档内容:**")
            if content:
                # Show content with option to expand
                if len(content) > 1000:
                    content_key = f"show_full_content_{index}"
                    if content_key not in st.session_state:
                        st.session_state[content_key] = False

                    if st.session_state[content_key]:
                        st.text_area("完整内容", content, height=300, disabled=True, key=f"full_content_{index}")
                        if st.button("收起内容", key=f"hide_content_{index}"):
                            st.session_state[content_key] = False
                            st.rerun()
                    else:
                        st.text_area("内容预览", content[:1000] + "...", height=150, disabled=True,
                                     key=f"preview_content_{index}")
                        if st.button("显示完整内容", key=f"show_content_{index}"):
                            st.session_state[content_key] = True
                            st.rerun()
                else:
                    st.text_area("文档内容", content, height=200, disabled=True, key=f"content_{index}")
            else:
                st.warning("此文档没有内容")

        st.divider()


# === FILTERS AND CONTROLS ===
st.subheader("🔍 筛选选项")

filter_col1, filter_col2, filter_col3 = st.columns(3)

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
        ["默认", "按标题", "按来源", "按日期"],
        help="文档排序方式"
    )

# === DOCUMENT DISPLAY ===
if st.button("🔄 刷新文档列表", use_container_width=True):
    st.rerun()

# Get and display documents
with st.spinner("正在加载文档..."):
    documents = get_vector_store_documents(
        limit=limit,
        source_filter=source_filter if source_filter != "全部" else None
    )

if documents:
    # Statistics
    st.subheader(f"📊 文档统计 (共 {len(documents)} 个)")

    # Count by source
    source_counts = {}
    total_words = 0

    for doc in documents:
        source = doc.get("metadata", {}).get("source", "unknown")
        source_counts[source] = source_counts.get(source, 0) + 1

        content = doc.get("content", "")
        if content:
            total_words += len(content.split())

    # Display stats
    stats_cols = st.columns(len(source_counts) + 1)

    for i, (source, count) in enumerate(source_counts.items()):
        with stats_cols[i]:
            st.metric(format_document_source({"source": source}), count)

    with stats_cols[-1]:
        st.metric("📝 总词数", f"{total_words:,}")

    st.markdown("---")

    # Sort documents
    if sort_by == "按标题":
        documents.sort(key=lambda x: x.get("metadata", {}).get("title", ""))
    elif sort_by == "按来源":
        documents.sort(key=lambda x: x.get("metadata", {}).get("source", ""))
    elif sort_by == "按日期":
        documents.sort(key=lambda x: x.get("metadata", {}).get("published_date", ""), reverse=True)

    # Display documents
    st.subheader("📚 文档列表")

    for i, doc in enumerate(documents):
        display_document_card(doc, i)

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

st.caption("文档浏览 - 查看向量数据库中的所有文档")