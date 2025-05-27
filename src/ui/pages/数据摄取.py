"""
Simplified upload page - src/ui/pages/数据摄取.py
Clean upload interface - jobs auto-queue, no capability checks needed
"""

import streamlit as st
import json
from src.ui.api_client import api_request
from src.ui.session_init import initialize_session_state

initialize_session_state()

st.title("📤 上传汽车资料")
st.markdown("支持视频链接和PDF文档上传，任务将自动处理")

def submit_upload(upload_type, data):
    """Submit upload job"""
    try:
        endpoints = {
            "video": "/ingest/video",
            "pdf": "/ingest/pdf",
            "text": "/ingest/text"
        }

        result = api_request(
            endpoint=endpoints[upload_type],
            method="POST",
            data=data.get("data"),
            files=data.get("files")
        )
        return result.get("job_id") if result else None
    except:
        return None

# Upload tabs - Always show all options (jobs will queue automatically)
tab1, tab2, tab3 = st.tabs(["🎬 视频链接", "📄 PDF文档", "✍️ 文字内容"])

with tab1:
    st.subheader("视频链接上传")
    st.markdown("支持YouTube、Bilibili等平台")

    video_url = st.text_input(
        "视频链接",
        placeholder="https://www.youtube.com/watch?v=...",
        help="粘贴视频链接，系统将自动提取语音并转换为文字"
    )

    # Simple metadata
    with st.expander("补充信息（可选）"):
        v_manufacturer = st.text_input("品牌", key="v_manufacturer")
        v_model = st.text_input("车型", key="v_model")
        v_year = st.text_input("年份", key="v_year")

    if st.button("上传视频", disabled=not video_url, key="upload_video"):
        metadata = {}
        if v_manufacturer:
            metadata["manufacturer"] = v_manufacturer
        if v_model:
            metadata["model"] = v_model
        if v_year:
            metadata["year"] = v_year

        with st.spinner("正在提交视频处理任务..."):
            job_id = submit_upload("video", {
                "data": {
                    "url": video_url,
                    "metadata": metadata if metadata else None
                }
            })

        if job_id:
            st.success(f"✅ 视频已提交处理，任务ID: {job_id[:8]}...")
            st.info("📋 任务已加入处理队列，您可以在\"后台任务\"页面跟踪进度")

            # Quick action buttons
            col1, col2 = st.columns(2)
            with col1:
                if st.button("查看任务状态", key="view_video_status"):
                    st.switch_page("pages/后台任务.py")
            with col2:
                if st.button("继续上传", key="continue_video"):
                    st.rerun()
        else:
            st.error("❌ 上传失败，请检查链接格式或稍后重试")

with tab2:
    st.subheader("PDF文档上传")

    pdf_file = st.file_uploader(
        "选择PDF文件",
        type=["pdf"],
        help="支持文字版和扫描版PDF，系统会自动识别"
    )

    # Simple metadata
    with st.expander("补充信息（可选）"):
        p_manufacturer = st.text_input("品牌", key="p_manufacturer")
        p_model = st.text_input("车型", key="p_model")
        p_year = st.text_input("年份", key="p_year")

        # Processing options
        st.markdown("**处理选项：**")
        use_ocr = st.checkbox("启用OCR识别 (扫描版PDF)", value=True, key="use_ocr")
        extract_tables = st.checkbox("提取表格数据", value=True, key="extract_tables")

    if st.button("上传PDF", disabled=not pdf_file, key="upload_pdf"):
        metadata = {}
        if p_manufacturer:
            metadata["manufacturer"] = p_manufacturer
        if p_model:
            metadata["model"] = p_model
        if p_year:
            metadata["year"] = p_year

        with st.spinner("正在上传和处理PDF文件..."):
            job_id = submit_upload("pdf", {
                "data": {
                    "metadata": json.dumps(metadata) if metadata else None,
                    "use_ocr": "true" if use_ocr else "false",
                    "extract_tables": "true" if extract_tables else "false"
                },
                "files": {"file": (pdf_file.name, pdf_file, "application/pdf")}
            })

        if job_id:
            st.success(f"✅ PDF已提交处理，任务ID: {job_id[:8]}...")
            st.info("📋 任务已加入处理队列，您可以在\"后台任务\"页面跟踪进度")

            # Show file info
            file_size_mb = pdf_file.size / 1024 / 1024
            st.write(f"📄 文件: {pdf_file.name} ({file_size_mb:.2f} MB)")

            # Quick action buttons
            col1, col2 = st.columns(2)
            with col1:
                if st.button("查看任务状态", key="view_pdf_status"):
                    st.switch_page("pages/后台任务.py")
            with col2:
                if st.button("继续上传", key="continue_pdf"):
                    st.rerun()
        else:
            st.error("❌ 上传失败，请重试")

with tab3:
    st.subheader("文字内容输入")

    text_content = st.text_area(
        "输入文字内容",
        height=200,
        placeholder="请输入汽车相关的文字信息，如规格参数、配置说明等...",
        help="支持多段落长文本，系统会自动分段处理"
    )

    # Character count
    if text_content:
        char_count = len(text_content)
        st.caption(f"字符数: {char_count:,}")

    # Simple metadata
    with st.expander("补充信息（可选）"):
        t_title = st.text_input("标题", key="t_title", placeholder="例如：2023款宝马X5技术规格")
        t_manufacturer = st.text_input("品牌", key="t_manufacturer")
        t_model = st.text_input("车型", key="t_model")
        t_year = st.text_input("年份", key="t_year")

    if st.button("提交文字", disabled=not text_content.strip(), key="upload_text"):
        metadata = {
            "source": "manual",
            "title": t_title or "手动输入内容"
        }
        if t_manufacturer:
            metadata["manufacturer"] = t_manufacturer
        if t_model:
            metadata["model"] = t_model
        if t_year:
            try:
                metadata["year"] = int(t_year)
            except:
                pass

        with st.spinner("正在处理文字内容..."):
            job_id = submit_upload("text", {
                "data": {
                    "content": text_content.strip(),
                    "metadata": metadata
                }
            })

        if job_id:
            st.success(f"✅ 文字已提交处理，任务ID: {job_id[:8]}...")
            st.info("📋 任务已加入处理队列，您可以在\"后台任务\"页面跟踪进度")

            # Quick action buttons
            col1, col2 = st.columns(2)
            with col1:
                if st.button("查看任务状态", key="view_text_status"):
                    st.switch_page("pages/后台任务.py")
            with col2:
                if st.button("继续输入", key="continue_text"):
                    st.rerun()
        else:
            st.error("❌ 提交失败，请重试")

# Upload tips
st.markdown("---")
with st.expander("💡 上传建议"):
    st.markdown("""
    **视频上传：**
    - 确保视频包含汽车相关内容
    - 支持中文和英文语音识别
    - 处理时间约为视频时长的1-2倍
    - 建议视频时长不超过30分钟
    
    **PDF上传：**
    - 支持文字版和扫描版PDF
    - 建议文件大小小于100MB
    - 清晰的文档会获得更好的识别效果
    - 表格数据会被自动提取
    
    **文字输入：**
    - 适合输入规格参数、配置信息
    - 支持多段落长文本
    - 建议提供详细的车型信息
    - 可以包含技术参数表格
    """)

# Navigation
st.markdown("---")
nav_cols = st.columns(3)

with nav_cols[0]:
    if st.button("📋 查看任务状态", use_container_width=True):
        st.switch_page("pages/后台任务.py")

with nav_cols[1]:
    if st.button("🔍 开始查询", use_container_width=True):
        st.switch_page("pages/查询.py")

with nav_cols[2]:
    if st.button("🏠 返回主页", use_container_width=True):
        st.switch_page("src/ui/主页.py")

st.caption("数据摄取 - 所有任务都会自动排队处理，请耐心等待")