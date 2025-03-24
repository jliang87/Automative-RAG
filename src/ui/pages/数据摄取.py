"""
汽车数据导入页面（Streamlit UI）
"""

import json
import os
import streamlit as st
from src.ui.components import header, api_request, loading_spinner

# API 配置
API_URL = os.environ.get("API_URL", "http://localhost:8000")
API_KEY = os.environ.get("API_KEY", "default-api-key")

# 会话状态初始化
if "api_url" not in st.session_state:
    st.session_state.api_url = API_URL
if "api_key" not in st.session_state:
    st.session_state.api_key = API_KEY
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

header(
    "导入汽车数据",
    "从多个来源添加新的汽车规格数据。"
)

# 不同数据导入方式的选项卡
tab1, tab2, tab3, tab4, tab5 = st.tabs(["YouTube", "Bilibili", "优酷", "PDF", "手动输入"])

def render_youtube_tab():
    """渲染 YouTube 数据导入选项卡"""
    st.header("导入 YouTube 视频")

    youtube_url = st.text_input("YouTube 链接", placeholder="https://www.youtube.com/watch?v=...")

    force_whisper = st.checkbox(
        "强制使用 Whisper 转录",
        value=True,
        help="使用 Whisper 进行转录，而不是 YouTube 提供的字幕。这通常提供更高质量的转录结果。"
    )

    with st.expander("附加元数据"):
        yt_manufacturer = st.text_input("制造商", key="yt_manufacturer")
        yt_model = st.text_input("车型", key="yt_model")
        yt_year = st.text_input("年份", value="2023", key="yt_year")
        # Validate year format if needed
        if yt_year:
            try:
                year_value = int(yt_year)
                if year_value < 1900 or year_value > 2100:
                    st.warning("年份应该在1900-2100之间")
            except ValueError:
                st.warning("请输入有效的年份")

        yt_categories = ["", "轿车", "SUV", "卡车", "跑车", "MPV", "双门轿车", "敞篷车", "掀背车", "旅行车"]
        yt_category = st.selectbox("类别", yt_categories, key="yt_category")

        yt_engine_types = ["", "汽油", "柴油", "电动", "混合动力", "氢能"]
        yt_engine_type = st.selectbox("发动机类型", yt_engine_types, key="yt_engine_type")

        yt_transmission = ["", "自动挡", "手动挡", "CVT", "DCT"]
        yt_transmission_type = st.selectbox("变速箱", yt_transmission, key="yt_transmission")

    if st.button("导入 YouTube 视频", type="primary", key="youtube_btn"):
        if not youtube_url:
            st.warning("请输入 YouTube 视频链接")
            return

        # 构建元数据
        custom_metadata = {}
        if yt_manufacturer:
            custom_metadata["制造商"] = yt_manufacturer
        if yt_model:
            custom_metadata["车型"] = yt_model
        if yt_year:
            custom_metadata["年份"] = yt_year
        if yt_category:
            custom_metadata["类别"] = yt_category
        if yt_engine_type:
            custom_metadata["发动机类型"] = yt_engine_type
        if yt_transmission_type:
            custom_metadata["变速箱"] = yt_transmission_type

        # 发送 API 请求
        with loading_spinner("正在处理 YouTube 视频... 这可能需要几分钟。"):
            endpoint = f"/ingest/youtube{'?force_whisper=true' if force_whisper else ''}"

            result = api_request(
                endpoint=endpoint,
                method="POST",
                data={
                    "url": youtube_url,
                    "metadata": custom_metadata if custom_metadata else None,
                },
            )

        if result:
            st.success(f"成功导入 YouTube 视频：已创建 {result.get('document_count', 0)} 个数据块")

def render_bilibili_tab():
    """渲染 Bilibili 数据导入选项卡"""
    st.header("导入 Bilibili 视频")

    bilibili_url = st.text_input("Bilibili 视频链接", placeholder="https://www.bilibili.com/video/...")

    force_whisper = st.checkbox(
        "强制使用 Whisper 转录",
        value=True,
        help="Whisper 总是用于 Bilibili 视频的高质量转录",
        key="bilibili_force_whisper",
        disabled=True
    )

    if st.button("导入 Bilibili 视频", type="primary", key="bilibili_btn"):
        if not bilibili_url:
            st.warning("请输入 Bilibili 视频链接")
            return

        with loading_spinner("正在处理 Bilibili 视频... 这可能需要几分钟。"):
            endpoint = "/ingest/bilibili"

            result = api_request(
                endpoint=endpoint,
                method="POST",
                data={
                    "url": bilibili_url,
                },
            )

        if result:
            st.success(f"成功导入 Bilibili 视频：已创建 {result.get('document_count', 0)} 个数据块")

def render_youku_tab():
    """渲染优酷数据导入选项卡"""
    st.header("导入优酷视频")

    youku_url = st.text_input("优酷视频链接", placeholder="https://v.youku.com/v_show/id_...")

    if st.button("导入优酷视频", type="primary", key="youku_btn"):
        if not youku_url:
            st.warning("请输入优酷视频链接")
            return

        with loading_spinner("正在处理优酷视频... 这可能需要几分钟。"):
            endpoint = "/ingest/youku"

            result = api_request(
                endpoint=endpoint,
                method="POST",
                data={
                    "url": youku_url,
                },
            )

        if result:
            st.success(f"成功导入优酷视频：已创建 {result.get('document_count', 0)} 个数据块")

def render_pdf_tab():
    """渲染 PDF 数据导入选项卡"""
    st.header("导入 PDF 文件")

    pdf_file = st.file_uploader("上传 PDF 文件", type=["pdf"])

    if st.button("导入 PDF", type="primary", key="pdf_btn"):
        if not pdf_file:
            st.warning("请上传 PDF 文件")
            return

        with loading_spinner("正在处理 PDF 文件..."):
            files = {"file": (pdf_file.name, pdf_file, "application/pdf")}

            result = api_request(
                endpoint="/ingest/pdf",
                method="POST",
                files=files,
            )

        if result:
            st.success(f"成功导入 PDF：已创建 {result.get('document_count', 0)} 个数据块")

def render_manual_tab():
    """渲染手动输入选项卡"""
    st.header("手动输入数据")

    manual_content = st.text_area("输入内容", height=300)

    if st.button("导入手动输入数据", type="primary", key="manual_btn"):
        if not manual_content:
            st.warning("请输入内容")
            return

        with loading_spinner("正在处理手动输入内容..."):
            result = api_request(
                endpoint="/ingest/text",
                method="POST",
                data={
                    "content": manual_content,
                },
            )

        if result:
            st.success(f"成功导入手动输入内容：已创建 {result.get('document_count', 0)} 个数据块")

# 渲染各个选项卡
with tab1:
    render_youtube_tab()

with tab2:
    render_bilibili_tab()

with tab3:
    render_youku_tab()

with tab4:
    render_pdf_tab()

with tab5:
    render_manual_tab()
