"""
Clean upload page - src/ui/pages/数据摄取.py
"""

import streamlit as st
import json
from src.ui.api_client import api_request
from src.ui.session_init import initialize_session_state

initialize_session_state()

st.title("📤 上传汽车资料")
st.markdown("支持视频链接和PDF文档上传")

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

# Upload tabs
tab1, tab2, tab3 = st.tabs(["🎬 视频链接", "📄 PDF文档", "✍️ 文字内容"])

with tab1:
    st.subheader("视频链接上传")
    st.markdown("支持YouTube、Bilibili等平台")

    video_url = st.text_input(
        "视频链接",
        placeholder="https://www.youtube.com/watch?v=..."
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

        job_id = submit_upload("video", {
            "data": {
                "url": video_url,
                "metadata": metadata if metadata else None
            }
        })

        if job_id:
            st.success(f"✅ 视频已提交处理，任务ID: {job_id[:8]}...")
            st.info("您可以在\"查看状态\"页面跟踪处理进度")
        else:
            st.error("上传失败，请检查链接格式")

with tab2:
    st.subheader("PDF文档上传")

    pdf_file = st.file_uploader("选择PDF文件", type=["pdf"])

    # Simple metadata
    with st.expander("补充信息（可选）"):
        p_manufacturer = st.text_input("品牌", key="p_manufacturer")
        p_model = st.text_input("车型", key="p_model")
        p_year = st.text_input("年份", key="p_year")

    if st.button("上传PDF", disabled=not pdf_file, key="upload_pdf"):
        metadata = {}
        if p_manufacturer:
            metadata["manufacturer"] = p_manufacturer
        if p_model:
            metadata["model"] = p_model
        if p_year:
            metadata["year"] = p_year

        job_id = submit_upload("pdf", {
            "data": {
                "metadata": json.dumps(metadata) if metadata else None,
                "use_ocr": "true",
                "extract_tables": "true"
            },
            "files": {"file": (pdf_file.name, pdf_file, "application/pdf")}
        })

        if job_id:
            st.success(f"✅ PDF已提交处理，任务ID: {job_id[:8]}...")
            st.info("您可以在\"查看状态\"页面跟踪处理进度")
        else:
            st.error("上传失败，请重试")

with tab3:
    st.subheader("文字内容输入")

    text_content = st.text_area(
        "输入文字内容",
        height=200,
        placeholder="请输入汽车相关的文字信息..."
    )

    # Simple metadata
    with st.expander("补充信息（可选）"):
        t_title = st.text_input("标题", key="t_title")
        t_manufacturer = st.text_input("品牌", key="t_manufacturer")
        t_model = st.text_input("车型", key="t_model")
        t_year = st.text_input("年份", key="t_year")

    if st.button("提交文字", disabled=not text_content.strip(), key="upload_text"):
        metadata = {
            "source": "manual",
            "title": t_title or "手动输入"
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

        job_id = submit_upload("text", {
            "data": {
                "content": text_content.strip(),
                "metadata": metadata
            }
        })

        if job_id:
            st.success(f"✅ 文字已提交处理，任务ID: {job_id[:8]}...")
            st.info("您可以在\"查看状态\"页面跟踪处理进度")
        else:
            st.error("提交失败，请重试")

# Simple tips
st.markdown("---")
with st.expander("💡 上传建议"):
    st.markdown("""
    **视频上传：**
    - 确保视频包含汽车相关内容
    - 支持中文和英文语音
    
    **PDF上传：**
    - 支持扫描版PDF（自动OCR识别）
    - 建议上传清晰的文档
    
    **文字输入：**
    - 适合输入规格参数、配置信息等
    - 建议提供详细的车型信息
    """)

# Link to status page
st.markdown("---")
if st.button("查看上传状态", use_container_width=True):
    st.switch_page("pages/后台任务.py")