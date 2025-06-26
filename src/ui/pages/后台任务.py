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

# === INITIALIZE SINGLE DETAIL VIEW STATE ===
if "active_job_detail" not in st.session_state:
    st.session_state.active_job_detail = None


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
    """Display a job card with progress and actions, including validation results."""
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
            # For queries, show the mode name and query
            mode_name = metadata.get("mode_name", "")
            query = metadata.get("query") or result.get("query", "")

            if mode_name and query:
                return f"**{mode_name}** - {query[:60]}{'...' if len(query) > 60 else ''}"
            elif mode_name:
                return f"**{mode_name}** 查询"
            elif query:
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
            # UPDATED: Check if this job is currently active
            is_expanded = st.session_state.active_job_detail == job_id
            button_text = "🔼 收起" if is_expanded else "📄 详情"
            button_type = "secondary" if is_expanded else "primary"

            if st.button(button_text, key=f"detail_{context}_{index}_{job_id[:8]}",
                         type=button_type, use_container_width=True):
                # UPDATED: Toggle logic for single detail view
                if st.session_state.active_job_detail == job_id:
                    # Close current detail if it's already open
                    st.session_state.active_job_detail = None
                else:
                    # Open this job's detail (automatically closes any other open detail)
                    st.session_state.active_job_detail = job_id
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

        # UPDATED: Show details only if this job is the active one
        if st.session_state.active_job_detail == job_id:
            st.markdown("---")
            st.markdown("### 📋 任务详情")

            # Get full job details
            job_detail = get_job_details(job_id)

            if job_detail:
                # Basic information in columns
                detail_col1, detail_col2 = st.columns(2)

                with detail_col1:
                    st.write(f"**任务ID:** {job_id}")
                    st.write(f"**类型:** {format_job_type(job_detail.get('job_type', ''))}")
                    st.write(f"**状态:** {job_detail.get('status', '')}")

                with detail_col2:
                    created = job_detail.get('created_at', 0)
                    updated = job_detail.get('updated_at', 0)

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

                    if metadata.get('mode_name'):
                        st.write(f"**模式名称:** {metadata['mode_name']}")

                # Parse result properly
                if isinstance(result, str):
                    try:
                        import json
                        result = json.loads(result)
                    except:
                        result = {}

                display_enhanced_job_metadata_analysis(job_detail)

                # UPDATED: Show validation results for completed LLM inference jobs
                if job_detail.get('status') == 'completed' and job_type == "llm_inference":
                    if has_validation_data(result):
                        st.markdown("---")
                        st.markdown("### 🛡️ 验证结果")

                        # Quick validation summary
                        display_job_validation_summary(result)

                        # Option to view full validation details
                        if st.button(f"查看完整验证报告", key=f"full_validation_{job_id[:8]}"):
                            st.session_state[f"show_full_validation_{job_id}"] = True
                            st.rerun()

                        # Show full validation if requested
                        if st.session_state.get(f"show_full_validation_{job_id}", False):
                            st.markdown("#### 完整验证报告")
                            render_unified_validation_display(result)

                            if st.button(f"隐藏验证报告", key=f"hide_validation_{job_id[:8]}"):
                                st.session_state[f"show_full_validation_{job_id}"] = False
                                st.rerun()

                # Enhanced Results Display for Video Processing
                if job_detail.get('status') == 'completed':
                    if result and isinstance(result, dict):

                        # Show video metadata
                        video_metadata = result.get('video_metadata', {})

                        if video_metadata and isinstance(video_metadata, dict):
                            st.markdown("**🎬 视频信息:**")

                            video_col1, video_col2 = st.columns(2)
                            with video_col1:
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

                            with video_col2:
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

                            transcript_key = f"show_transcript_{job_id}"
                            if transcript_key not in st.session_state:
                                st.session_state[transcript_key] = False

                            # Show transcript stats
                            word_count = len(transcript.split())
                            char_count = len(transcript)
                            language = result.get('language', '未知')
                            duration = result.get('duration', 0)

                            trans_col1, trans_col2, trans_col3, trans_col4 = st.columns(4)
                            with trans_col1:
                                st.metric("字数", f"{word_count:,}")
                            with trans_col2:
                                st.metric("字符数", f"{char_count:,}")
                            with trans_col3:
                                lang_display = {"zh": "中文", "en": "英文"}.get(language, language)
                                st.metric("语言", lang_display)
                            with trans_col4:
                                if duration > 0:
                                    st.metric("时长", f"{duration:.1f}秒")

                            # Toggle transcript display
                            if st.button(f"{'隐藏' if st.session_state[transcript_key] else '显示'} 转录内容",
                                         key=f"toggle_transcript_{job_id[:8]}"):
                                st.session_state[transcript_key] = not st.session_state[transcript_key]
                                st.rerun()

                            if st.session_state[transcript_key]:
                                st.text_area(
                                    "完整转录内容",
                                    transcript,
                                    height=300,
                                    disabled=True,
                                    key=f"transcript_{job_id[:8]}"
                                )

                        # Document processing results
                        if 'document_count' in result:
                            st.success(f"✅ 成功生成 {result['document_count']} 个文档片段")

                        # Query results
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

                # Error information (for failed jobs)
                elif job_detail.get('status') == 'failed':
                    error = job_detail.get('error', '')
                    if error:
                        st.error(f"❌ **错误:** {error}")

                # Quick actions with unique keys
                action_col1, action_col2 = st.columns(2)
                with action_col1:
                    if st.button("🔄 刷新", key=f"refresh_{job_id[:8]}"):
                        st.rerun()
                with action_col2:
                    if job_detail.get('status') in ['completed', 'failed']:
                        if st.button("🗑️ 删除", key=f"delete_{job_id[:8]}"):
                            try:
                                result = api_request(f"/ingest/jobs/{job_id}", method="DELETE")
                                if result:
                                    st.success("任务已删除")
                                    # Clear the active detail if this job was deleted
                                    if st.session_state.active_job_detail == job_id:
                                        st.session_state.active_job_detail = None
                                    time.sleep(1)
                                    st.rerun()
                                else:
                                    st.error("删除失败")
                            except:
                                st.error("删除操作失败")
            else:
                st.error("无法获取任务详情")


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
        # Clear active detail when switching tabs
        st.session_state.active_job_detail = None
        st.rerun()

with tab_col2:
    if st.button(f"✅ 已完成 ({len(completed_jobs)})",
                 key="tab_completed",
                 use_container_width=True,
                 type="primary" if st.session_state.current_tab == 1 else "secondary"):
        st.session_state.current_tab = 1
        # Clear active detail when switching tabs
        st.session_state.active_job_detail = None
        st.rerun()

with tab_col3:
    if st.button(f"📋 全部任务 ({len(jobs)})",
                 key="tab_all",
                 use_container_width=True,
                 type="primary" if st.session_state.current_tab == 2 else "secondary"):
        st.session_state.current_tab = 2
        # Clear active detail when switching tabs
        st.session_state.active_job_detail = None
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

            # FIXED: Only add divider if this is NOT the last job
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
                # Clear active detail when changing pages
                st.session_state.active_job_detail = None
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

            # FIXED: Only add divider if this is NOT the last job
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
                # Clear active detail when changing pages
                st.session_state.active_job_detail = None
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

        # FIXED: Only add divider if this is NOT the last job
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
            # Clear active detail when changing pages
            st.session_state.active_job_detail = None
            st.rerun()

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