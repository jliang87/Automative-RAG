"""
Streamlit 界面 UI 组件

该模块包含 Streamlit 应用程序中的可复用 UI 组件。
"""

import streamlit as st
import httpx
from typing import Dict, List, Optional, Union, Callable, Any


def header(title: str, description: Optional[str] = None):
    """显示页面标题和描述"""
    st.title(title)
    if description:
        st.markdown(description)
    st.divider()


def api_status_indicator():
    """显示 API 连接状态"""
    try:
        with httpx.Client() as client:
            response = client.get(
                f"{st.session_state.api_url}/health",
                headers={"x-token": st.session_state.api_key},
                timeout=3.0
            )

        if response.status_code == 200:
            st.sidebar.success("✅ API 连接正常")
            return True
        else:
            st.sidebar.error(f"❌ API 错误: {response.status_code}")
            return False
    except Exception as e:
        st.sidebar.error(f"❌ 连接失败: {str(e)}")
        return False


def gpu_status_indicator():
    """显示 GPU 状态信息"""
    try:
        with httpx.Client() as client:
            response = client.get(
                f"{st.session_state.api_url}/ingest/status",
                headers={"x-token": st.session_state.api_key},
                timeout=3.0
            )

        if response.status_code == 200:
            data = response.json()

            if "gpu_info" in data and data["gpu_info"]:
                gpu_info = data["gpu_info"]

                # 创建 GPU 信息展开区域
                with st.sidebar.expander("GPU 状态"):
                    if "device_name" in gpu_info:
                        st.info(f"GPU: {gpu_info['device_name']}")

                    if "memory_allocated" in gpu_info:
                        st.metric("显存占用", gpu_info['memory_allocated'])

                    if "whisper_model" in gpu_info:
                        st.text(f"Whisper 版本: {gpu_info['whisper_model']}")

                    if "fp16_enabled" in gpu_info:
                        fp16 = gpu_info['fp16_enabled']
                        st.text(f"混合精度: {'✓' if fp16 else '✗'}")
            else:
                st.sidebar.warning("⚠️ 当前运行在 CPU 上")

    except Exception as e:
        st.sidebar.warning("⚠️ 无法获取 GPU 状态")


def llm_status_indicator():
    """显示 LLM（大语言模型）状态信息"""
    try:
        with httpx.Client() as client:
            response = client.get(
                f"{st.session_state.api_url}/query/llm-info",
                headers={"x-token": st.session_state.api_key},
                timeout=3.0
            )

        if response.status_code == 200:
            llm_info = response.json()

            with st.sidebar.expander("LLM 状态"):
                # 模型名称 - 若名称过长，则显示缩短版本
                model_name = llm_info.get("model_name", "未知")
                if len(model_name) > 30:
                    display_name = f"{model_name.split('/')[-1]}"
                else:
                    display_name = model_name
                st.info(f"模型: {display_name}")

                # 量化信息
                quant = llm_info.get("quantization", "无")
                st.text(f"量化方式: {quant}")

                # 显存占用信息
                if "vram_usage" in llm_info:
                    st.metric("显存占用", llm_info["vram_usage"])

                # 设备信息
                device = llm_info.get("device", "未知")
                st.text(f"运行设备: {device}")
    except Exception:
        pass  # 忽略错误


def metadata_filters():
    """渲染元数据筛选选项"""
    st.subheader("筛选条件")

    col1, col2 = st.columns(2)

    with col1:
        # 这些选项应该从 API 获取
        manufacturers = ["", "丰田", "本田", "福特", "宝马", "特斯拉"]
        manufacturer = st.selectbox("厂商", manufacturers)

        years = [""] + [str(year) for year in range(2010, 2026)]
        year = st.selectbox("年份", years)

        categories = ["", "轿车", "SUV", "卡车", "跑车", "MPV"]
        category = st.selectbox("车型", categories)

    with col2:
        models = [""]
        if manufacturer == "丰田":
            models += ["凯美瑞", "卡罗拉", "RAV4", "汉兰达"]
        elif manufacturer == "本田":
            models += ["思域", "雅阁", "CR-V", "Pilot"]
        elif manufacturer == "福特":
            models += ["野马", "F-150", "探险者", "Escape"]
        elif manufacturer == "宝马":
            models += ["3系", "5系", "X3", "X5"]
        elif manufacturer == "特斯拉":
            models += ["Model S", "Model 3", "Model X", "Model Y"]

        model = st.selectbox("车型", models)

        engine_types = ["", "汽油", "柴油", "电动", "混合动力"]
        engine_type = st.selectbox("发动机类型", engine_types)

        transmission_types = ["", "自动", "手动", "CVT", "DCT"]
        transmission = st.selectbox("变速箱", transmission_types)

    # 构建筛选条件
    metadata_filter = {}

    if manufacturer:
        metadata_filter["manufacturer"] = manufacturer
    if model:
        metadata_filter["model"] = model
    if year:
        metadata_filter["year"] = int(year)
    if category:
        metadata_filter["category"] = category
    if engine_type:
        metadata_filter["engine_type"] = engine_type
    if transmission:
        metadata_filter["transmission"] = transmission

    return metadata_filter if metadata_filter else None


def api_request(
    endpoint: str,
    method: str = "GET",
    data: Optional[Dict] = None,
    files: Optional[Dict] = None,
    params: Optional[Dict] = None,
    handle_error: Callable[[str], Any] = None,
) -> Optional[Dict]:
    """发送 API 请求并处理错误"""
    headers = {"x-token": st.session_state.api_key}
    url = f"{st.session_state.api_url}{endpoint}"

    try:
        with httpx.Client(timeout=150.0) as client:
            if method == "GET":
                response = client.get(url, headers=headers, params=params)
            elif method == "POST":
                if files:
                    response = client.post(url, headers=headers, data=data, files=files)
                else:
                    response = client.post(url, headers=headers, json=data)
            elif method == "DELETE":
                response = client.delete(url, headers=headers)
            else:
                st.error(f"不支持的请求方法: {method}")
                return None

            if response.status_code >= 400:
                error_msg = f"API 错误 ({response.status_code}): {response.text}"
                if handle_error:
                    return handle_error(error_msg)
                else:
                    st.error(error_msg)
                    return None

            return response.json()
    except Exception as e:
        error_msg = f"连接错误: {str(e)}"
        if handle_error:
            return handle_error(error_msg)
        else:
            st.error(error_msg)
            return None


def display_document(doc: Dict, index: int):
    """展示搜索结果文档"""
    metadata = doc.get("metadata", {})

    # 确定来源图标
    source_type = metadata.get("source", "未知")
    source_icon = "📚"
    if source_type == "youtube":
        source_icon = "🎬"
    elif source_type == "pdf":
        source_icon = "📄"
    elif source_type == "manual":
        source_icon = "📝"

    title = metadata.get("title", f"文档 {index+1}")

    with st.expander(f"{source_icon} {title}"):
        st.caption(f"来源: {source_type}")
        st.caption(f"相关度: {doc.get('relevance_score', 0):.4f}")

        auto_info = [metadata.get("manufacturer", ""), metadata.get("model", ""), str(metadata.get("year", ""))]
        st.markdown(f"**车型**: {' '.join([x for x in auto_info if x])}")

        st.markdown("---")
        st.markdown(doc.get("content", ""))

        if metadata.get("url"):
            st.markdown(f"[原始链接]({metadata['url']})")


def loading_spinner(text: str = "处理中..."):
    """创建加载动画"""
    return st.spinner(text)
