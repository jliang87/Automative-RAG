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
tab1, tab2, tab3 = st.tabs(["视频", "PDF", "手动输入"])

def detect_platform(url):
    """检测视频平台"""
    if "youtube.com" in url or "youtu.be" in url:
        return "YouTube"
    elif "bilibili.com" in url:
        return "Bilibili"
    else:
        return "未知平台"

def render_video_tab():
    """渲染统一的视频数据导入选项卡"""
    st.header("导入视频")

    video_url = st.text_input("视频链接", placeholder="https://www.youtube.com/watch?v=... 或 https://www.bilibili.com/video/...")

    # 检测视频平台
    platform = "未知平台"
    if video_url:
        platform = detect_platform(video_url)
        st.info(f"检测到 {platform} 平台")

    st.info("所有视频将使用 Whisper AI 进行转录以提供高质量的字幕")

    with st.expander("附加元数据"):
        video_manufacturer = st.text_input("制造商", key="video_manufacturer")
        video_model = st.text_input("车型", key="video_model")
        video_year = st.text_input("年份", value="2023", key="video_year")
        # Validate year format if needed
        if video_year:
            try:
                year_value = int(video_year)
                if year_value < 1900 or year_value > 2100:
                    st.warning("年份应该在1900-2100之间")
            except ValueError:
                st.warning("请输入有效的年份")

        video_categories = ["", "轿车", "SUV", "卡车", "跑车", "MPV", "双门轿车", "敞篷车", "掀背车", "旅行车"]
        video_category = st.selectbox("类别", video_categories, key="video_category")

        video_engine_types = ["", "汽油", "柴油", "电动", "混合动力", "氢能"]
        video_engine_type = st.selectbox("发动机类型", video_engine_types, key="video_engine_type")

        video_transmission = ["", "自动挡", "手动挡", "CVT", "DCT"]
        video_transmission_type = st.selectbox("变速箱", video_transmission, key="video_transmission")

    if st.button("导入视频", type="primary", key="video_btn"):
        if not video_url:
            st.warning("请输入视频链接")
            return

        # 构建元数据
        custom_metadata = {}
        if video_manufacturer:
            custom_metadata["manufacturer"] = video_manufacturer
        if video_model:
            custom_metadata["model"] = video_model
        if video_year:
            try:
                custom_metadata["year"] = int(video_year)
            except ValueError:
                st.warning("请输入有效的年份")
                return
        if video_category:
            custom_metadata["category"] = video_category
        if video_engine_type:
            custom_metadata["engine_type"] = video_engine_type
        if video_transmission_type:
            custom_metadata["transmission"] = video_transmission_type

        # 发送 API 请求
        with loading_spinner(f"正在处理{platform}视频... 这可能需要几分钟。"):
            endpoint = "/ingest/video"

            result = api_request(
                endpoint=endpoint,
                method="POST",
                data={
                    "url": video_url,
                    "metadata": custom_metadata if custom_metadata else None,
                },
            )

        if result:
            st.success(f"成功导入{platform}视频：已创建 {result.get('document_count', 0)} 个数据块")

def render_pdf_tab():
    """渲染 PDF 数据导入选项卡"""
    st.header("导入 PDF 文件")

    with st.expander("附加元数据"):
        pdf_manufacturer = st.text_input("制造商", key="pdf_manufacturer")
        pdf_model = st.text_input("车型", key="pdf_model")
        pdf_year = st.text_input("年份", value="2023", key="pdf_year")
        # Validate year format if needed
        if pdf_year:
            try:
                year_value = int(pdf_year)
                if year_value < 1900 or year_value > 2100:
                    st.warning("年份应该在1900-2100之间")
            except ValueError:
                st.warning("请输入有效的年份")

        pdf_categories = ["", "轿车", "SUV", "卡车", "跑车", "MPV", "双门轿车", "敞篷车", "掀背车", "旅行车"]
        pdf_category = st.selectbox("类别", pdf_categories, key="pdf_category")

        pdf_engine_types = ["", "汽油", "柴油", "电动", "混合动力", "氢能"]
        pdf_engine_type = st.selectbox("发动机类型", pdf_engine_types, key="pdf_engine_type")

        pdf_transmission = ["", "自动挡", "手动挡", "CVT", "DCT"]
        pdf_transmission_type = st.selectbox("变速箱", pdf_transmission, key="pdf_transmission")

    pdf_file = st.file_uploader("上传 PDF 文件", type=["pdf"])

    if st.button("导入 PDF", type="primary", key="pdf_btn"):
        if not pdf_file:
            st.warning("请上传 PDF 文件")
            return

        # 构建元数据
        custom_metadata = {}
        if pdf_manufacturer:
            custom_metadata["manufacturer"] = pdf_manufacturer
        if pdf_model:
            custom_metadata["model"] = pdf_model
        if pdf_year:
            try:
                custom_metadata["year"] = int(pdf_year)
            except ValueError:
                st.warning("请输入有效的年份")
                return
        if pdf_category:
            custom_metadata["category"] = pdf_category
        if pdf_engine_type:
            custom_metadata["engine_type"] = pdf_engine_type
        if pdf_transmission_type:
            custom_metadata["transmission"] = pdf_transmission_type

        with loading_spinner("正在处理 PDF 文件..."):
            files = {"file": (pdf_file.name, pdf_file, "application/pdf")}
            data = {}
            if custom_metadata:
                data["metadata"] = json.dumps(custom_metadata)

            result = api_request(
                endpoint="/ingest/pdf",
                method="POST",
                data=data,
                files=files,
            )

        if result:
            st.success(f"成功导入 PDF：已创建 {result.get('document_count', 0)} 个数据块")

def render_manual_tab():
    """渲染手动输入选项卡"""
    st.header("手动输入数据")

    with st.expander("元数据"):
        manual_manufacturer = st.text_input("制造商", key="manual_manufacturer")
        manual_model = st.text_input("车型", key="manual_model")
        manual_year = st.text_input("年份", value="2023", key="manual_year")
        # Validate year format if needed
        if manual_year:
            try:
                year_value = int(manual_year)
                if year_value < 1900 or year_value > 2100:
                    st.warning("年份应该在1900-2100之间")
            except ValueError:
                st.warning("请输入有效的年份")

        manual_categories = ["", "轿车", "SUV", "卡车", "跑车", "MPV", "双门轿车", "敞篷车", "掀背车", "旅行车"]
        manual_category = st.selectbox("类别", manual_categories, key="manual_category")

        manual_engine_types = ["", "汽油", "柴油", "电动", "混合动力", "氢能"]
        manual_engine_type = st.selectbox("发动机类型", manual_engine_types, key="manual_engine_type")

        manual_transmission = ["", "自动挡", "手动挡", "CVT", "DCT"]
        manual_transmission_type = st.selectbox("变速箱", manual_transmission, key="manual_transmission")

    manual_content = st.text_area("输入内容", height=300)

    if st.button("导入手动输入数据", type="primary", key="manual_btn"):
        if not manual_content:
            st.warning("请输入内容")
            return

        # 构建元数据
        metadata = {
            "source": "manual",
            "title": f"{manual_manufacturer} {manual_model} 规格",
        }

        if manual_manufacturer:
            metadata["manufacturer"] = manual_manufacturer
        if manual_model:
            metadata["model"] = manual_model
        if manual_year:
            try:
                metadata["year"] = int(manual_year)
            except ValueError:
                st.warning("请输入有效的年份")
                return
        if manual_category:
            metadata["category"] = manual_category
        if manual_engine_type:
            metadata["engine_type"] = manual_engine_type
        if manual_transmission_type:
            metadata["transmission"] = manual_transmission_type

        with loading_spinner("正在处理手动输入内容..."):
            result = api_request(
                endpoint="/ingest/text",
                method="POST",
                data={
                    "content": manual_content,
                    "metadata": metadata
                },
            )

        if result:
            st.success(f"成功导入手动输入内容：已创建 {result.get('document_count', 0)} 个数据块")

# 渲染各个选项卡
with tab1:
    render_video_tab()

with tab2:
    render_pdf_tab()

with tab3:
    render_manual_tab()