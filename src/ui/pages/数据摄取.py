"""
汽车数据导入页面（Streamlit UI）- 支持后台处理
已更新为使用增强的错误处理
"""

import json
import os
import datetime
import streamlit as st
import time
from src.ui.components import header, loading_spinner
from src.ui.system_notifications import display_notifications_sidebar
from src.ui.enhanced_error_handling import robust_api_status_indicator, handle_worker_dependency
from src.ui.api_client import api_request

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
if "job_created" not in st.session_state:
    st.session_state.job_created = False
if "job_id" not in st.session_state:
    st.session_state.job_id = None
if "redirect_to_jobs" not in st.session_state:
    st.session_state.redirect_to_jobs = False

# 在侧边栏显示通知
display_notifications_sidebar(st.session_state.api_url, st.session_state.api_key)

# 在侧边栏检查 API 状态
with st.sidebar:
    api_available = robust_api_status_indicator(show_detail=True)

# 仅在 API 可用时继续主要内容
if api_available:
    header(
        "导入汽车数据",
        "从多个来源添加新的汽车规格数据。"
    )

    # 如果需要重定向到任务页面
    if st.session_state.redirect_to_jobs:
        st.success(f"已创建后台任务，任务ID: {st.session_state.job_id}")
        st.info("正在跳转到任务管理页面...")

        # 添加倒计时定时器，而不是无限期等待
        import time

        for i in range(3, 0, -1):
            st.text(f"将在 {i} 秒后跳转...")
            time.sleep(1)
            st.rerun()

        # 倒计时完成后，使用相对路径重定向
        st.switch_page("pages/后台任务.py")

        # 添加立即重定向按钮
        if st.button("立即跳转到任务页面"):
            st.switch_page("pages/后台任务.py")

        # 在此停止页面执行
        st.stop()

    st.info("""
    ### 所有数据处理均在后台进行
    
    所有数据处理（包括视频转录、PDF解析和文本处理）均在后台异步执行。
    提交后，您将被自动重定向到任务管理页面，以便跟踪处理进度。
    """)

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

        with st.expander("附加元数据"):
            video_manufacturer = st.text_input("制造商", key="video_manufacturer")
            video_model = st.text_input("车型", key="video_model")
            video_year = st.text_input("年份", key="video_year")
            # 验证年份格式（如果需要）
            if video_year:
                try:
                    year_value = int(video_year)
                    if year_value < 1900 or year_value > 2100:
                        st.warning("年份应该在1900-2100之间")
                    elif year_value > datetime.datetime.now().year + 1:
                        st.warning(f"年份超出合理范围 (当前年份: {datetime.datetime.now().year})")
                except ValueError:
                    st.warning("请输入有效的数字年份")

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

            # 检查视频处理的 worker 依赖
            if not handle_worker_dependency("video"):
                return

            # 构建元数据
            custom_metadata = {}
            if video_manufacturer:
                custom_metadata["manufacturer"] = video_manufacturer
            if video_model:
                custom_metadata["model"] = video_model
            if video_year:
                try:
                    year_value = int(video_year)
                    if year_value < 1900 or year_value > 2100:
                        st.warning("年份应该在1900-2100之间")
                        return
                    elif year_value > datetime.datetime.now().year + 1:
                        st.warning(f"年份超出合理范围 (当前年份: {datetime.datetime.now().year})")
                        return
                    custom_metadata["year"] = str(year_value)
                except ValueError:
                    st.warning("请输入有效的数字年份")
                    return
            if video_category:
                custom_metadata["category"] = video_category
            if video_engine_type:
                custom_metadata["engine_type"] = video_engine_type
            if video_transmission_type:
                custom_metadata["transmission"] = video_transmission_type

            # 发送 API 请求
            with loading_spinner("正在提交处理任务... 这可能需要一些时间。"):
                endpoint = "/ingest/video"

                result = api_request(
                    endpoint=endpoint,
                    method="POST",
                    data={
                        "url": video_url,
                        "metadata": custom_metadata if custom_metadata else None
                    },
                )

            if result:
                # 显示任务创建成功，并准备跳转到任务页面
                st.session_state.job_created = True
                st.session_state.job_id = result.get("job_id")
                st.session_state.redirect_to_jobs = True
                st.rerun()

    def render_pdf_tab():
        """渲染 PDF 数据导入选项卡"""
        st.header("导入 PDF 文件")

        with st.expander("附加元数据"):
            pdf_manufacturer = st.text_input("制造商", key="pdf_manufacturer")
            pdf_model = st.text_input("车型", key="pdf_model")
            pdf_year = st.text_input("年份", key="pdf_year")
            # 验证年份格式（如果需要）
            if pdf_year:
                try:
                    year_value = int(pdf_year)
                    if year_value < 1900 or year_value > 2100:
                        st.warning("年份应该在1900-2100之间")
                    elif year_value > datetime.datetime.now().year + 1:
                        st.warning(f"年份超出合理范围 (当前年份: {datetime.datetime.now().year})")
                except ValueError:
                    st.warning("请输入有效的数字年份")

            pdf_categories = ["", "轿车", "SUV", "卡车", "跑车", "MPV", "双门轿车", "敞篷车", "掀背车", "旅行车"]
            pdf_category = st.selectbox("类别", pdf_categories, key="pdf_category")

            pdf_engine_types = ["", "汽油", "柴油", "电动", "混合动力", "氢能"]
            pdf_engine_type = st.selectbox("发动机类型", pdf_engine_types, key="pdf_engine_type")

            pdf_transmission = ["", "自动挡", "手动挡", "CVT", "DCT"]
            pdf_transmission_type = st.selectbox("变速箱", pdf_transmission, key="pdf_transmission")

        # OCR选项
        use_ocr = st.checkbox("启用OCR（识别扫描PDF中的文字）", value=True)
        extract_tables = st.checkbox("提取表格", value=True)

        pdf_file = st.file_uploader("上传 PDF 文件", type=["pdf"])

        if st.button("导入 PDF", type="primary", key="pdf_btn"):
            if not pdf_file:
                st.warning("请上传 PDF 文件")
                return

            # 检查 PDF 处理的 worker 依赖
            if not handle_worker_dependency("pdf"):
                return

            # 构建元数据
            custom_metadata = {}
            if pdf_manufacturer:
                custom_metadata["manufacturer"] = pdf_manufacturer
            if pdf_model:
                custom_metadata["model"] = pdf_model
            if pdf_year:
                try:
                    year_value = int(pdf_year)
                    if year_value < 1900 or year_value > 2100:
                        st.warning("年份应该在1900-2100之间")
                        return
                    elif year_value > datetime.datetime.now().year + 1:
                        st.warning(f"年份超出合理范围 (当前年份: {datetime.datetime.now().year})")
                        return
                    custom_metadata["year"] = str(year_value)
                except ValueError:
                    st.warning("请输入有效的数字年份")
                    return
            if pdf_category:
                custom_metadata["category"] = pdf_category
            if pdf_engine_type:
                custom_metadata["engine_type"] = pdf_engine_type
            if pdf_transmission_type:
                custom_metadata["transmission"] = pdf_transmission_type

            with loading_spinner("正在提交PDF处理任务..."):
                files = {"file": (pdf_file.name, pdf_file, "application/pdf")}
                data = {
                    "metadata": json.dumps(custom_metadata) if custom_metadata else None,
                    "use_ocr": str(use_ocr).lower(),
                    "extract_tables": str(extract_tables).lower()
                }

                result = api_request(
                    endpoint="/ingest/pdf",
                    method="POST",
                    data=data,
                    files=files,
                )

            if result:
                # 显示任务创建成功，并准备跳转到任务页面
                st.session_state.job_created = True
                st.session_state.job_id = result.get("job_id")
                st.session_state.redirect_to_jobs = True
                st.rerun()

    def render_manual_tab():
        """渲染手动输入选项卡"""
        st.header("手动输入数据")

        with st.expander("元数据"):
            manual_manufacturer = st.text_input("制造商", key="manual_manufacturer")
            manual_model = st.text_input("车型", key="manual_model")
            manual_year = st.text_input("年份", key="manual_year")
            # 验证年份格式（如果需要）
            if manual_year:
                try:
                    year_value = int(manual_year)
                    if year_value < 1900 or year_value > 2100:
                        st.warning("年份应该在1900-2100之间")
                    elif year_value > datetime.datetime.now().year + 1:
                        st.warning(f"年份超出合理范围 (当前年份: {datetime.datetime.now().year})")
                except ValueError:
                    st.warning("请输入有效的数字年份")

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

            # 检查文本处理的 worker 依赖
            if not handle_worker_dependency("text"):
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
                    year_value = int(manual_year)
                    if year_value < 1900 or year_value > 2100:
                        st.warning("年份应该在1900-2100之间")
                        return
                    elif year_value > datetime.datetime.now().year + 1:
                        st.warning(f"年份超出合理范围 (当前年份: {datetime.datetime.now().year})")
                        return
                    metadata["year"] = year_value
                except ValueError:
                    st.warning("请输入有效的数字年份")
                    return
            if manual_category:
                metadata["category"] = manual_category
            if manual_engine_type:
                metadata["engine_type"] = manual_engine_type
            if manual_transmission_type:
                metadata["transmission"] = manual_transmission_type

            with loading_spinner("正在提交文本处理任务..."):
                result = api_request(
                    endpoint="/ingest/text",
                    method="POST",
                    data={
                        "content": manual_content,
                        "metadata": metadata
                    },
                )

            if result:
                # 显示任务创建成功，并准备跳转到任务页面
                st.session_state.job_created = True
                st.session_state.job_id = result.get("job_id")
                st.session_state.redirect_to_jobs = True
                st.rerun()

    # 渲染各个选项卡
    with tab1:
        render_video_tab()

    with tab2:
        render_pdf_tab()

    with tab3:
        render_manual_tab()
else:
    header(
        "导入汽车数据",
        "从多个来源添加新的汽车规格数据。"
    )

    st.error("无法连接到API服务或所需的Worker未运行。")
    st.info("请确保API服务和相关Worker正在运行，然后刷新页面。")

    if st.button("刷新"):
        st.rerun()