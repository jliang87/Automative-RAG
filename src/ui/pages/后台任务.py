import streamlit as st
import time
import json
from typing import Dict, Any
import logging
from src.ui.api_client import (
    get_jobs_list,
    get_job_details,
    get_job_statistics,
    api_request
)
from src.ui.components.metadata_display import (
    add_metadata_display_to_sources,
    render_embedded_metadata_display,
    EmbeddedMetadataExtractor
)
from src.ui.session_init import initialize_session_state

# UPDATED: Import unified validation display
from src.ui.components.validation_display import (
    render_unified_validation_display,
    render_quick_validation_badge
)

logger = logging.getLogger(__name__)

initialize_session_state()

# Add simple modal CSS - just basic styling, no complex positioning
st.markdown("""
<style>
/* Simple modal overlay - bottom positioned */
.bottom-modal {
    position: fixed;
    bottom: 0;
    left: 0;
    right: 0;
    background-color: var(--background-color);
    border-top: 2px solid var(--border-color);
    box-shadow: 0 -4px 20px rgba(0, 0, 0, 0.15);
    z-index: 1000;
    max-height: 70vh;
    overflow-y: auto;
    animation: slideInUp 0.3s ease-in-out;
}

@keyframes slideInUp {
    from { transform: translateY(100%); }
    to { transform: translateY(0); }
}

/* Simple backdrop */
.simple-backdrop {
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background-color: rgba(0, 0, 0, 0.3);
    z-index: 999;
}
</style>
""", unsafe_allow_html=True)

st.title("📋 后台任务")
st.markdown("查看和管理您的处理任务，包括验证结果")

# === JOB STATISTICS OVERVIEW ===
job_stats = get_job_statistics()

if any(job_stats.values()):
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("已完成", job_stats.get("completed", 0))
    with col2:
        processing_count = job_stats.get("processing", 0)
        st.metric("处理中", processing_count)
    with col3:
        st.metric("等待中", job_stats.get("pending", 0))
    with col4:
        st.metric("失败", job_stats.get("failed", 0))

    st.markdown("---")

# === PAGINATION CONTROLS ===
col1, col2, col3 = st.columns(3)

with col1:
    jobs_limit = st.selectbox("获取任务数量", [50, 100, 200], index=1, help="限制获取的任务数量")

with col2:
    jobs_per_page = st.selectbox("每页显示", [10, 20, 30, 50], index=1, help="每页显示的任务数量")

with col3:
    if st.button("🔄 刷新任务列表", use_container_width=True):
        st.rerun()

# === FETCH JOBS ===
jobs = get_jobs_list(limit=jobs_limit)

if not jobs:
    st.info("📭 暂无处理任务")
    if st.button("🔄 刷新", use_container_width=True):
        st.rerun()
    st.stop()


# === PAGINATION HELPER FUNCTIONS ===
def paginate_jobs(jobs_list, page_num, per_page):
    """Paginate jobs list"""
    start_idx = (page_num - 1) * per_page
    end_idx = start_idx + per_page
    return jobs_list[start_idx:end_idx]


def render_pagination(total_jobs, current_page, per_page, tab_name):
    """Render elegant pagination controls"""
    total_pages = (total_jobs + per_page - 1) // per_page

    if total_pages <= 1:
        return current_page

    # Elegant inline pagination - everything on one clean line
    col1, col2, col3, col4, col5, col6 = st.columns([2, 0.7, 0.7, 1.5, 0.7, 0.7])

    new_page = current_page

    with col1:
        # Clean, normal-sized text
        st.write(f"第 {current_page} / {total_pages} 页 (共 {total_jobs} 个任务)")

    with col2:
        if current_page > 1:
            if st.button("⏮️", key=f"first_{tab_name}", help="首页", use_container_width=True):
                return 1

    with col3:
        if current_page > 1:
            if st.button("◀️", key=f"prev_{tab_name}", help="上一页", use_container_width=True):
                return current_page - 1

    with col4:
        # Put "跳转到" and selectbox on the same line
        page_options = list(range(1, total_pages + 1))
        selected_page = st.selectbox(
            "跳转到:",
            page_options,
            index=current_page - 1,
            key=f"page_select_{tab_name}",
            label_visibility="collapsed"  # Hide the label to save space
        )
        if selected_page != current_page:
            return selected_page

    with col5:
        if current_page < total_pages:
            if st.button("下一页", key=f"next_{tab_name}", use_container_width=True):
                return current_page + 1

    with col6:
        if current_page < total_pages:
            if st.button("末页", key=f"last_{tab_name}", use_container_width=True):
                return total_pages

    return current_page


# === INITIALIZE PAGINATION STATE ===
if "processing_page" not in st.session_state:
    st.session_state.processing_page = 1
if "completed_page" not in st.session_state:
    st.session_state.completed_page = 1
if "all_jobs_page" not in st.session_state:
    st.session_state.all_jobs_page = 1

# === BLADE MODAL STATE ===
if "modal_job_id" not in st.session_state:
    st.session_state.modal_job_id = None


def format_job_type(job_type: str) -> str:
    """Format job type for display - CLEANED UP VERSION"""
    type_names = {
        "video_processing": "视频处理",
        "pdf_processing": "PDF处理",
        "text_processing": "文字处理",
        "llm_inference": "查询处理",
        "batch_video_processing": "批量视频"
    }
    return type_names.get(job_type, job_type)


def format_time(timestamp: float) -> str:
    """Format timestamp for display"""
    if not timestamp:
        return "未知时间"
    try:
        return time.strftime("%m-%d %H:%M", time.localtime(timestamp))
    except:
        return "时间格式错误"


def has_validation_data(result: Dict[str, Any]) -> bool:
    """Check if job result contains validation data"""
    if not result or not isinstance(result, dict):
        return False

    # Check for automotive validation data
    automotive_validation = result.get("automotive_validation", {})
    if automotive_validation and isinstance(automotive_validation, dict):
        return True

    # Check for simple confidence score
    if result.get("simple_confidence", 0) > 0:
        return True

    # Check for documents with validation metadata
    documents = result.get("documents", [])
    if documents and isinstance(documents, list):
        for doc in documents:
            if isinstance(doc, dict):
                metadata = doc.get("metadata", {})
                if metadata.get("automotive_warnings") or metadata.get("validation_status"):
                    return True

    return False


def display_enhanced_job_metadata_analysis(job_details: Dict[str, Any]):
    """Display enhanced metadata analysis for completed jobs."""

    # Check if this is a video processing job with documents
    if job_details.get('result') and job_details.get('status') == 'completed':
        result = job_details['result']

        # Parse result if it's a string
        if isinstance(result, str):
            try:
                result = json.loads(result)
            except:
                result = {}

        if 'documents' in result or 'processed_documents' in result:
            documents = result.get('documents', result.get('processed_documents', []))

            if documents:
                st.markdown("---")
                st.subheader("📊 处理结果元数据分析")

                # Metadata quality analysis
                extractor = EmbeddedMetadataExtractor()
                total_docs = len(documents)
                metadata_stats = {
                    'docs_with_embedded': 0,
                    'docs_with_vehicle': 0,
                    'total_metadata_fields': 0,
                    'unique_vehicles': set(),
                    'unique_sources': set()
                }

                for doc in documents:
                    if isinstance(doc, dict):
                        content = doc.get('content', doc.get('page_content', ''))
                        metadata = doc.get('metadata', {})

                        # Extract embedded metadata
                        embedded_metadata, _ = extractor.extract_embedded_metadata(content)
                        if embedded_metadata:
                            metadata_stats['docs_with_embedded'] += 1
                            metadata_stats['total_metadata_fields'] += len(embedded_metadata)

                            # Track unique vehicles and sources
                            if 'model' in embedded_metadata:
                                vehicle_name = embedded_metadata['model']
                                manufacturer = metadata.get('manufacturer', '')
                                full_vehicle = f"{manufacturer} {vehicle_name}".strip()
                                metadata_stats['unique_vehicles'].add(full_vehicle)

                        if metadata.get('has_vehicle_info'):
                            metadata_stats['docs_with_vehicle'] += 1

                        if metadata.get('source'):
                            metadata_stats['unique_sources'].add(metadata['source'])

                # Display metadata statistics
                stats_col1, stats_col2, stats_col3, stats_col4 = st.columns(4)

                with stats_col1:
                    embedded_rate = (metadata_stats['docs_with_embedded'] / total_docs * 100) if total_docs > 0 else 0
                    st.metric("元数据注入率", f"{embedded_rate:.1f}%")

                with stats_col2:
                    vehicle_rate = (metadata_stats['docs_with_vehicle'] / total_docs * 100) if total_docs > 0 else 0
                    st.metric("车辆检测率", f"{vehicle_rate:.1f}%")

                with stats_col3:
                    avg_fields = (
                            metadata_stats['total_metadata_fields'] / max(metadata_stats['docs_with_embedded'], 1))
                    st.metric("平均元数据字段", f"{avg_fields:.1f}")

                with stats_col4:
                    st.metric("检测到的车型", len(metadata_stats['unique_vehicles']))

                # Show detected vehicles and sources
                if metadata_stats['unique_vehicles']:
                    st.markdown("**🚗 检测到的车型:**")
                    vehicles_list = list(metadata_stats['unique_vehicles'])[:5]  # Show first 5
                    st.write(", ".join(vehicles_list))
                    if len(metadata_stats['unique_vehicles']) > 5:
                        st.caption(f"... 还有 {len(metadata_stats['unique_vehicles']) - 5} 个车型")

                if metadata_stats['unique_sources']:
                    st.markdown("**📺 来源平台:**")
                    st.write(", ".join(metadata_stats['unique_sources']))

                    # Document sample with metadata details
                    st.markdown("---")
                    st.subheader("📄 文档样本及元数据")

                    # Show first few documents with detailed metadata
                    sample_docs = documents[:3] if len(documents) > 3 else documents

                    for i, doc in enumerate(sample_docs):
                        # FIXED: Add unique_id parameter to prevent key conflicts
                        job_id = job_details.get('job_id', 'unknown')
                        unique_id = f"job_{job_id[:8]}_doc_{i}"

                        with st.expander(f"查看文档 {i + 1} 元数据详情", expanded=False):
                            if isinstance(doc, dict):
                                # Convert to expected format
                                doc_format = {
                                    'content': doc.get('content', doc.get('page_content', '')),
                                    'metadata': doc.get('metadata', {})
                                }
                                # FIXED: Pass unique_id to prevent duplicate keys
                                render_embedded_metadata_display(doc_format, show_full_content=False,
                                                                 unique_id=unique_id)
                            else:
                                st.warning("文档格式不支持元数据显示")


def display_job_validation_summary(result: Dict[str, Any]) -> None:
    """Display a summary of validation results for completed jobs"""
    if not has_validation_data(result):
        st.caption("此任务无验证数据")
        return

    # UPDATED: Use unified validation badge
    validation_badge = render_quick_validation_badge(result)
    st.markdown(f"**验证状态**: {validation_badge}")

    # Quick summary of validation details
    automotive_validation = result.get("automotive_validation", {})
    if automotive_validation:
        confidence_level = automotive_validation.get("confidence_level", "unknown")
        has_warnings = automotive_validation.get("has_warnings", False)

        if confidence_level == "high" and not has_warnings:
            st.caption("✅ 高质量回答，已通过专业验证")
        elif confidence_level == "medium":
            st.caption("📋 中等质量回答，建议参考多个来源")
        elif confidence_level == "low" or has_warnings:
            st.caption("⚠️ 包含需注意信息，请查看验证详情")
        else:
            st.caption("❓ 验证状态未知")


def display_job_card(job: Dict[str, Any], context: str, index: int):
    """Display a job card with blade modal integration."""
    job_id = job.get("job_id", "")
    job_type = job.get("job_type", "")
    status = job.get("status", "")
    created_at = job.get("created_at", 0)

    if not job_id:
        st.error("无效任务数据")
        return

    # Status styling
    status_config = {
        "pending": {"icon": "⏳", "color": "#FFA500"},
        "processing": {"icon": "🔄", "color": "#1E90FF"},
        "completed": {"icon": "✅", "color": "#32CD32"},
        "failed": {"icon": "❌", "color": "#FF4500"}
    }

    config = status_config.get(status, {"icon": "❓", "color": "#808080"})

    # Extract key metadata to display directly
    def get_display_metadata(job_data):
        """Extract key metadata for direct display - CLEANED UP VERSION"""
        metadata = job_data.get("metadata", {})
        result = job_data.get("result", {})

        # Parse result if it's a string
        if isinstance(result, str):
            try:
                result = json.loads(result)
            except:
                result = {}

        if job_type == "llm_inference":
            # For queries, show the user's query
            query = metadata.get("query") or result.get("query", "")
            if query:
                return f"查询: {query[:80]}{'...' if len(query) > 80 else ''}"
            return "查询处理"

        elif job_type in ["video_processing", "batch_video_processing"]:
            # For videos, show title from video_metadata or URL
            video_metadata = result.get("video_metadata", {})
            title = video_metadata.get("title") or metadata.get("title", "")

            if title:
                return f"视频: {title[:60]}{'...' if len(title) > 60 else ''}"

            # Fallback to URL
            url = metadata.get("url", "")
            if url:
                return f"链接: {url[:50]}{'...' if len(url) > 50 else ''}"
            return "视频处理"

        elif job_type == "pdf_processing":
            # For PDFs, show filename or title
            filename = metadata.get("filename") or metadata.get("title", "")
            if filename:
                return f"文件: {filename[:50]}{'...' if len(filename) > 50 else ''}"

            filepath = metadata.get("filepath", "")
            if filepath:
                import os
                filename = os.path.basename(filepath)
                return f"文件: {filename[:50]}{'...' if len(filename) > 50 else ''}"
            return "PDF处理"

        elif job_type == "text_processing":
            # For text, show title or content preview
            title = metadata.get("title") or result.get("title", "")
            if title and title != "Manual Text Input":
                return f"标题: {title[:50]}{'...' if len(title) > 50 else ''}"

            # Fallback to content preview
            content = result.get("original_text", "")
            if content:
                preview = content.replace('\n', ' ')[:40]
                return f"内容: {preview}{'...' if len(content) > 40 else ''}"
            return "文字处理"

        return ""

    display_info = get_display_metadata(job)

    # Job card container
    with st.container():
        # Add CSS to prevent button text wrapping
        st.markdown("""
        <style>
        div[data-testid="column"] .stButton > button {
            white-space: nowrap !important;
            overflow: hidden !important;
            text-overflow: ellipsis !important;
            min-width: 80px !important;
            font-size: 0.875rem !important;
        }
        </style>
        """, unsafe_allow_html=True)

        # Header row - CLEANED UP
        col1, col2, col3, col4 = st.columns([1, 4, 2, 1.3])

        with col1:
            st.markdown(f"<span style='font-size: 2em'>{config['icon']}</span>",
                        unsafe_allow_html=True)

        with col2:
            st.markdown(f"**{format_job_type(job_type)}**")
            if display_info:
                # CLEANED UP: No more emoji spam, just clear text
                st.caption(display_info)
            else:
                st.caption(f"任务ID: {job_id[:12]}...")

        with col3:
            st.markdown(f"**状态: {status}**")
            st.caption(f"创建: {format_time(created_at)}")

        with col4:
            # Replace expansion with blade modal trigger
            if st.button("📄 详情", key=f"detail_{context}_{index}_{job_id[:8]}",
                         type="primary", use_container_width=True):
                st.session_state.blade_job_id = job_id
                st.rerun()

        # Progress bar for processing jobs
        if status == "processing":
            progress_info = job.get("progress_info", {})
            progress = progress_info.get("progress")
            message = progress_info.get("message", "")

            if progress is not None:
                st.progress(progress / 100.0)
                st.caption(f"进度: {progress}% - {message}")
            else:
                st.progress(0.0)
                st.caption("处理中...")


def render_bottom_modal():
    """Render the bottom modal for job details using pure Streamlit components."""
    if not st.session_state.modal_job_id:
        return

    job_id = st.session_state.modal_job_id
    job_detail = get_job_details(job_id)

    if not job_detail:
        st.error("无法获取任务详情")
        st.session_state.modal_job_id = None
        return

    # Create a visual separator
    st.markdown("---")

    # Modal header with background color
    st.markdown("""
    <div style="background-color: var(--secondary-background-color); 
                padding: 1rem; margin: -1rem -1rem 1rem -1rem; 
                border-left: 4px solid #007bff;">
        <h3 style="margin: 0; color: var(--text-color);">📋 任务详情面板</h3>
        <p style="margin: 0.5rem 0 0 0; color: var(--text-color); opacity: 0.8;">
            查看详细任务信息 | 点击"关闭详情"返回任务列表
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Close button prominently displayed at the top
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("❌ 关闭详情", key="close_modal", type="primary", use_container_width=True):
            st.session_state.modal_job_id = None
            st.rerun()

    st.markdown("---")

    # Create columns for organized layout
    basic_col, details_col = st.columns([1, 1])

    with basic_col:
        st.subheader("📋 基本信息")

        job_type = job_detail.get('job_type', '')
        status = job_detail.get('status', '')
        created = job_detail.get('created_at', 0)
        updated = job_detail.get('updated_at', 0)

        st.write(f"**任务ID:** `{job_id}`")
        st.write(f"**类型:** {format_job_type(job_type)}")
        st.write(f"**状态:** {status}")

        if created:
            st.write(f"**创建时间:** {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(created))}")
        if updated:
            st.write(f"**更新时间:** {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(updated))}")

        # Progress information
        progress_info = job_detail.get('progress_info', {})
        if progress_info:
            progress = progress_info.get('progress')
            message = progress_info.get('message', '')

            st.write("**当前进度:**")
            if progress is not None:
                st.progress(progress / 100.0)
                st.caption(f"{progress}% - {message}")
            else:
                st.caption(message or "处理中...")

    with details_col:
        st.subheader("🔧 任务元数据")

        metadata = job_detail.get('metadata', {})

        if metadata and isinstance(metadata, dict):
            if metadata.get('url'):
                st.write(f"**URL:** {metadata['url'][:60]}...")
                if st.button("🔗 打开链接", key=f"open_url_{job_id}"):
                    st.markdown(f"[打开原始链接]({metadata['url']})")

            if metadata.get('query'):
                st.write(f"**查询内容:**")
                st.info(metadata['query'])

            if metadata.get('platform'):
                st.write(f"**平台:** {metadata['platform']}")

            if metadata.get('query_mode'):
                st.write(f"**查询模式:** {metadata['query_mode']}")

            if metadata.get('mode_name'):
                st.write(f"**模式名称:** {metadata['mode_name']}")
        else:
            st.info("此任务暂无元数据信息")

    # Parse result properly
    result = job_detail.get('result', {})
    if isinstance(result, str):
        try:
            result = json.loads(result)
        except:
            result = {}

    # Results section - full width
    if job_detail.get('status') == 'completed' and result:
        st.markdown("---")
        st.subheader("📊 处理结果")

        # Create tabs for different result types
        result_tabs = []
        tab_contents = []

        # Video metadata tab
        video_metadata = result.get('video_metadata', {})
        if video_metadata and isinstance(video_metadata, dict):
            result_tabs.append("🎬 视频信息")
            tab_contents.append(('video', video_metadata))

        # Transcript tab
        transcript = result.get('transcript', '')
        if transcript:
            result_tabs.append("🎤 转录内容")
            tab_contents.append(('transcript', transcript))

        # Query answer tab
        if result.get('answer'):
            result_tabs.append("❓ 查询答案")
            tab_contents.append(('answer', result.get('answer')))

        # Document processing tab
        if result.get('document_count'):
            result_tabs.append("📄 文档处理")
            tab_contents.append(('documents', result.get('document_count')))

        # Validation tab
        if job_type == "llm_inference" and has_validation_data(result):
            result_tabs.append("🛡️ 验证结果")
            tab_contents.append(('validation', result))

        if result_tabs:
            tabs = st.tabs(result_tabs)

            for i, (tab_type, content) in enumerate(tab_contents):
                with tabs[i]:
                    if tab_type == 'video':
                        # Video metadata display
                        video_col1, video_col2 = st.columns(2)

                        with video_col1:
                            if content.get('title'):
                                st.write(f"**标题:** {content['title']}")
                            if content.get('author'):
                                st.write(f"**作者:** {content['author']}")
                            if content.get('published_date'):
                                pub_date = content['published_date']
                                if isinstance(pub_date, str) and len(pub_date) == 8:
                                    formatted_date = f"{pub_date[:4]}-{pub_date[4:6]}-{pub_date[6:8]}"
                                    st.write(f"**发布日期:** {formatted_date}")
                                else:
                                    st.write(f"**发布日期:** {pub_date}")

                        with video_col2:
                            if content.get('length'):
                                duration_mins = content['length'] // 60
                                duration_secs = content['length'] % 60
                                st.write(f"**时长:** {duration_mins}分{duration_secs}秒")
                            if content.get('views'):
                                st.write(f"**观看次数:** {content['views']:,}")
                            if content.get('video_id'):
                                st.write(f"**视频ID:** {content['video_id']}")

                        if content.get('url'):
                            st.markdown(f"[🔗 观看视频]({content['url']})")

                    elif tab_type == 'transcript':
                        # Transcript display
                        word_count = len(content.split())
                        char_count = len(content)

                        stats_col1, stats_col2, stats_col3 = st.columns(3)
                        with stats_col1:
                            st.metric("字数", f"{word_count:,}")
                        with stats_col2:
                            st.metric("字符数", f"{char_count:,}")
                        with stats_col3:
                            language = result.get('language', '未知')
                            lang_display = {"zh": "中文", "en": "英文"}.get(language, language)
                            st.metric("语言", lang_display)

                        st.text_area(
                            "完整转录内容",
                            content,
                            height=300,
                            disabled=True,
                            key=f"modal_transcript_{job_id}"
                        )

                    elif tab_type == 'answer':
                        # Query answer display
                        answer = content
                        # Clean up LLM thinking artifacts
                        if "</think>" in answer:
                            answer = answer.split("</think>")[-1].strip()
                        if answer.startswith("<think>"):
                            lines = answer.split('\n')
                            clean_lines = []
                            thinking_section = True
                            for line in lines:
                                if thinking_section and (not line.strip().startswith('<') and line.strip()):
                                    thinking_section = False
                                if not thinking_section:
                                    clean_lines.append(line)
                            answer = '\n'.join(clean_lines).strip()
                        answer = answer.replace("<think>", "").replace("</think>", "").strip()

                        if answer:
                            st.markdown("**查询回答:**")
                            st.info(answer)
                        else:
                            st.warning("答案为空或无法解析")

                    elif tab_type == 'documents':
                        # Document processing results
                        st.success(f"✅ 成功生成 {content} 个文档片段")
                        st.info("文档已成功处理并存储到向量数据库中，可以进行智能查询。")

                    elif tab_type == 'validation':
                        # Validation results
                        display_job_validation_summary(content)

                        if st.button("查看完整验证报告", key=f"modal_full_validation_{job_id}"):
                            st.session_state[f"modal_show_full_validation_{job_id}"] = True
                            st.rerun()

                        if st.session_state.get(f"modal_show_full_validation_{job_id}", False):
                            render_unified_validation_display(content)

                            if st.button("隐藏验证报告", key=f"modal_hide_validation_{job_id}"):
                                st.session_state[f"modal_show_full_validation_{job_id}"] = False
                                st.rerun()

    # Error information for failed jobs
    elif job_detail.get('status') == 'failed':
        st.markdown("---")
        st.subheader("❌ 错误信息")
        error = job_detail.get('error', '')
        if error:
            st.error(f"**错误详情:** {error}")
        else:
            st.error("任务失败，但未获取到具体错误信息")

    # Action buttons section
    st.markdown("---")
    st.subheader("🚀 操作选项")

    action_col1, action_col2, action_col3, action_col4 = st.columns(4)

    with action_col1:
        if st.button("🔄 刷新详情", key=f"modal_refresh_{job_id}", use_container_width=True):
            st.rerun()

    with action_col2:
        if st.button("❌ 关闭面板", key=f"modal_close_{job_id}", use_container_width=True):
            st.session_state.modal_job_id = None
            st.rerun()

    with action_col3:
        if job_detail.get('status') in ['completed', 'failed']:
            if st.button("🗑️ 删除任务", key=f"modal_delete_{job_id}", use_container_width=True):
                if st.session_state.get(f"confirm_delete_{job_id}", False):
                    try:
                        result = api_request(f"/ingest/jobs/{job_id}", method="DELETE")
                        if result:
                            st.success("任务已删除")
                            st.session_state.modal_job_id = None
                            time.sleep(1)
                            st.rerun()
                        else:
                            st.error("删除失败")
                    except:
                        st.error("删除操作失败")
                else:
                    st.session_state[f"confirm_delete_{job_id}"] = True
                    st.warning("再次点击确认删除")
                    st.rerun()

    with action_col4:
        if st.button("📋 查看所有任务", key=f"modal_view_all_{job_id}", use_container_width=True):
            st.session_state.modal_job_id = None
            st.rerun()

    # Add some spacing at the bottom
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("### 📋 任务详情")

    # Close button
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown(f"**任务ID:** {job_id[:12]}...")
    with col2:
        if st.button("❌", key="close_blade", help="关闭"):
            st.session_state.blade_job_id = None
            st.rerun()

    st.markdown("---")

    # Basic information
    job_type = job_detail.get('job_type', '')
    status = job_detail.get('status', '')
    created = job_detail.get('created_at', 0)
    updated = job_detail.get('updated_at', 0)

    st.write(f"**类型:** {format_job_type(job_type)}")
    st.write(f"**状态:** {status}")

    if created:
        st.write(f"**创建:** {time.strftime('%m-%d %H:%M:%S', time.localtime(created))}")
    if updated:
        st.write(f"**更新:** {time.strftime('%m-%d %H:%M:%S', time.localtime(updated))}")

    # Progress information
    progress_info = job_detail.get('progress_info', {})
    if progress_info:
        progress = progress_info.get('progress')
        message = progress_info.get('message', '')

        st.write("**当前进度:**")
        if progress is not None:
            st.progress(progress / 100.0)
            st.caption(f"{progress}% - {message}")
        else:
            st.caption(message or "处理中...")

    # Enhanced Metadata Display
    st.markdown("**📋 任务元数据:**")
    metadata = job_detail.get('metadata', {})
    result = job_detail.get('result', {})

    if metadata and isinstance(metadata, dict):
        if metadata.get('url'):
            st.write(f"**URL:** {metadata['url']}")
        if metadata.get('query'):
            st.write(f"**查询:** {metadata['query']}")
        if metadata.get('platform'):
            st.write(f"**平台:** {metadata['platform']}")

        # UPDATED: Show query mode information
        if metadata.get('query_mode'):
            st.write(f"**查询模式:** {metadata['query_mode']}")
        if metadata.get('mode_name'):
            st.write(f"**模式名称:** {metadata['mode_name']}")

    # Parse result properly
    if isinstance(result, str):
        try:
            import json
            result = json.loads(result)
        except:
            result = {}

    # Enhanced Results Display
    if job_detail.get('status') == 'completed':
        if result and isinstance(result, dict):

            # Show video metadata
            video_metadata = result.get('video_metadata', {})

            if video_metadata and isinstance(video_metadata, dict):
                st.markdown("**🎬 视频信息:**")

                if video_metadata.get('title'):
                    st.write(f"**标题:** {video_metadata['title']}")
                if video_metadata.get('author'):
                    st.write(f"**作者:** {video_metadata['author']}")
                if video_metadata.get('published_date'):
                    pub_date = video_metadata['published_date']
                    if isinstance(pub_date, str) and len(pub_date) == 8:
                        formatted_date = f"{pub_date[:4]}-{pub_date[4:6]}-{pub_date[6:8]}"
                        st.write(f"**发布日期:** {formatted_date}")
                    else:
                        st.write(f"**发布日期:** {pub_date}")

                if video_metadata.get('url'):
                    st.write(f"**链接:** [观看视频]({video_metadata['url']})")

                if video_metadata.get('length'):
                    duration_mins = video_metadata['length'] // 60
                    duration_secs = video_metadata['length'] % 60
                    st.write(f"**时长:** {duration_mins}分{duration_secs}秒")
                if video_metadata.get('views'):
                    views = video_metadata['views']
                    st.write(f"**观看次数:** {views:,}")
                if video_metadata.get('video_id'):
                    st.write(f"**视频ID:** {video_metadata['video_id']}")

                language = result.get('language') or video_metadata.get('language')
                if language:
                    lang_display = {"zh": "中文", "en": "英文"}.get(language, language)
                    st.write(f"**语言:** {lang_display}")

            # Show transcription with better formatting
            transcript = result.get('transcript', '')
            if transcript:
                st.markdown("**🎤 转录内容:**")

                # Show transcript stats
                word_count = len(transcript.split())
                char_count = len(transcript)
                language = result.get('language', '未知')
                duration = result.get('duration', 0)

                st.metric("字数", f"{word_count:,}")
                st.metric("字符数", f"{char_count:,}")
                lang_display = {"zh": "中文", "en": "英文"}.get(language, language)
                st.metric("语言", lang_display)
                if duration > 0:
                    st.metric("时长", f"{duration:.1f}秒")

                # Show transcript in expandable area
                with st.expander("查看完整转录内容"):
                    st.text_area(
                        "完整转录内容",
                        transcript,
                        height=300,
                        disabled=True,
                        key=f"blade_transcript_{job_id}"
                    )

            # Document processing results
            if 'document_count' in result:
                st.success(f"✅ 成功生成 {result['document_count']} 个文档片段")

            # Query results with validation
            if 'answer' in result:
                st.write("**❓ 查询答案:**")
                answer = result['answer']

                # Clean up LLM thinking artifacts
                if "</think>" in answer:
                    answer = answer.split("</think>")[-1].strip()
                if answer.startswith("<think>"):
                    lines = answer.split('\n')
                    clean_lines = []
                    thinking_section = True
                    for line in lines:
                        if thinking_section and (not line.strip().startswith('<') and line.strip()):
                            thinking_section = False
                        if not thinking_section:
                            clean_lines.append(line)
                    answer = '\n'.join(clean_lines).strip()
                answer = answer.replace("<think>", "").replace("</think>", "").strip()

                if answer:
                    st.info(answer)
                else:
                    st.warning("答案为空或无法解析")

                # UPDATED: Show validation results for completed LLM inference jobs
                if job_type == "llm_inference" and has_validation_data(result):
                    st.markdown("---")
                    st.markdown("### 🛡️ 验证结果")

                    # Quick validation summary
                    display_job_validation_summary(result)

                    # Option to view full validation details
                    if st.button(f"查看完整验证报告", key=f"blade_full_validation_{job_id}"):
                        st.session_state[f"blade_show_full_validation_{job_id}"] = True
                        st.rerun()

                    # Show full validation if requested
                    if st.session_state.get(f"blade_show_full_validation_{job_id}", False):
                        st.markdown("#### 完整验证报告")
                        render_unified_validation_display(result)

                        if st.button(f"隐藏验证报告", key=f"blade_hide_validation_{job_id}"):
                            st.session_state[f"blade_show_full_validation_{job_id}"] = False
                            st.rerun()

    # Error information (for failed jobs)
    elif job_detail.get('status') == 'failed':
        error = job_detail.get('error', '')
        if error:
            st.error(f"❌ **错误:** {error}")

    # Enhanced action buttons that remain functional
    st.markdown("---")
    st.markdown("**🚀 操作:**")

    if st.button("🔄 刷新", key=f"blade_refresh_{job_id}", use_container_width=True):
        st.rerun()

    if job_detail.get('status') in ['completed', 'failed']:
        if st.button("🗑️ 删除", key=f"blade_delete_{job_id}", use_container_width=True):
            try:
                result = api_request(f"/ingest/jobs/{job_id}", method="DELETE")
                if result:
                    st.success("任务已删除")
                    st.session_state.blade_job_id = None
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error("删除失败")
            except:
                st.error("删除操作失败")


# JavaScript to handle backdrop clicks and ensure proper modal behavior
st.markdown("""
    <script>
    // Handle backdrop clicks to close modal
    document.addEventListener('click', function(event) {
        if (event.target.classList.contains('blade-backdrop')) {
            // Signal to close the modal
            window.parent.postMessage({type: 'closeModal'}, '*');
        }
    });

    // Prevent interaction with underlying content when blade is open
    if (document.querySelector('.blade-backdrop')) {
        const mainContent = document.querySelector('[data-testid="main"]');
        if (mainContent) {
            mainContent.style.pointerEvents = 'none';
            // Re-enable pointer events for blade content
            const bladeModal = document.querySelector('.blade-modal');
            if (bladeModal) {
                bladeModal.style.pointerEvents = 'auto';
            }
        }
    }
    </script>
    """, unsafe_allow_html=True)

# === FILTER JOBS BY STATUS ===
processing_jobs = [j for j in jobs if j.get("status") in ["pending", "processing"]]
completed_jobs = [j for j in jobs if j.get("status") == "completed"]

# === TABBED INTERFACE WITH IMPROVED PAGINATION ===
# Manual tab implementation to preserve state during pagination
if "current_tab" not in st.session_state:
    st.session_state.current_tab = 0  # 0=processing, 1=completed, 2=all

# Create tab buttons
tab_col1, tab_col2, tab_col3 = st.columns(3)

with tab_col1:
    if st.button(f"⏳ 处理中 ({len(processing_jobs)})",
                 key="tab_processing",
                 use_container_width=True,
                 type="primary" if st.session_state.current_tab == 0 else "secondary"):
        st.session_state.current_tab = 0
        st.rerun()

with tab_col2:
    if st.button(f"✅ 已完成 ({len(completed_jobs)})",
                 key="tab_completed",
                 use_container_width=True,
                 type="primary" if st.session_state.current_tab == 1 else "secondary"):
        st.session_state.current_tab = 1
        st.rerun()

with tab_col3:
    if st.button(f"📋 全部任务 ({len(jobs)})",
                 key="tab_all",
                 use_container_width=True,
                 type="primary" if st.session_state.current_tab == 2 else "secondary"):
        st.session_state.current_tab = 2
        st.rerun()

st.markdown("---")

# Display content based on selected tab
if st.session_state.current_tab == 0:  # Processing jobs
    if processing_jobs:
        # Get jobs for current page
        page_jobs = paginate_jobs(processing_jobs, st.session_state.processing_page, jobs_per_page)

        # Display all jobs first
        for i, job in enumerate(page_jobs):
            # Calculate global index for unique keys
            global_index = (st.session_state.processing_page - 1) * jobs_per_page + i
            display_job_card(job, f"processing", global_index)

            # Only add divider if this is NOT the last job
            if i < len(page_jobs) - 1:
                st.divider()

        # Pagination at bottom
        if len(processing_jobs) > jobs_per_page:
            st.markdown("---")
            new_page = render_pagination(
                len(processing_jobs),
                st.session_state.processing_page,
                jobs_per_page,
                "processing"
            )
            # Only update and rerun if page actually changed
            if new_page != st.session_state.processing_page:
                st.session_state.processing_page = new_page
                st.rerun()

        # Auto-refresh option for processing jobs
        st.markdown("---")
        if st.checkbox("⚡ 自动刷新 (5秒)", key="auto_refresh_processing"):
            time.sleep(5)
            st.rerun()
    else:
        st.info("✨ 当前没有正在处理的任务")

elif st.session_state.current_tab == 1:  # Completed jobs
    if completed_jobs:
        # Get jobs for current page
        page_jobs = paginate_jobs(completed_jobs, st.session_state.completed_page, jobs_per_page)

        # Display all jobs first
        for i, job in enumerate(page_jobs):
            # Calculate global index for unique keys
            global_index = (st.session_state.completed_page - 1) * jobs_per_page + i
            display_job_card(job, f"completed", global_index)

            # Only add divider if this is NOT the last job
            if i < len(page_jobs) - 1:
                st.divider()

        # Pagination at bottom
        if len(completed_jobs) > jobs_per_page:
            st.markdown("---")
            new_page = render_pagination(
                len(completed_jobs),
                st.session_state.completed_page,
                jobs_per_page,
                "completed"
            )
            # Only update and rerun if page actually changed
            if new_page != st.session_state.completed_page:
                st.session_state.completed_page = new_page
                st.rerun()
    else:
        st.info("📭 暂无已完成的任务")

elif st.session_state.current_tab == 2:  # All jobs
    # Get jobs for current page
    page_jobs = paginate_jobs(jobs, st.session_state.all_jobs_page, jobs_per_page)

    # Display all jobs first
    for i, job in enumerate(page_jobs):
        # Calculate global index for unique keys
        global_index = (st.session_state.all_jobs_page - 1) * jobs_per_page + i
        display_job_card(job, f"all", global_index)

        # Only add divider if this is NOT the last job
        if i < len(page_jobs) - 1:
            st.divider()

    # Pagination at bottom
    if len(jobs) > jobs_per_page:
        st.markdown("---")
        new_page = render_pagination(
            len(jobs),
            st.session_state.all_jobs_page,
            jobs_per_page,
            "all_jobs"
        )
        # Only update and rerun if page actually changed
        if new_page != st.session_state.all_jobs_page:
            st.session_state.all_jobs_page = new_page
            st.rerun()

# Render the blade modal if a job is selected
render_blade_modal()

# === PAGE ACTIONS ===
st.markdown("---")

action_cols = st.columns(4)

with action_cols[0]:
    if st.button("🔄 刷新列表", use_container_width=True):
        st.rerun()

with action_cols[1]:
    if st.button("📤 新建任务", use_container_width=True):
        st.switch_page("pages/数据摄取.py")

with action_cols[2]:
    if st.button("🧠 智能查询", use_container_width=True):
        st.switch_page("pages/智能查询.py")

with action_cols[3]:
    if st.button("📊 系统状态", use_container_width=True):
        st.switch_page("pages/系统信息.py")

# Show active processing count
processing_count = len([j for j in jobs if j.get("status") in ["pending", "processing"]])
if processing_count > 0:
    st.info(f"ℹ️ 当前有 {processing_count} 个任务正在处理中")

st.caption("后台任务 - 跟踪您的处理任务进度，包括验证结果查看")