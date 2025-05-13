"""
Streamlit 界面 UI 组件

该模块包含 Streamlit 应用程序中的可复用 UI 组件。
"""

import streamlit as st
import httpx
from typing import Dict, List, Optional, Union, Callable, Any

# 导入统一的 API 客户端
from src.ui.api_client import api_request


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


def worker_status_indicator():
    """在侧边栏显示worker状态。"""
    try:
        # 从API获取worker信息
        response = api_request(
            endpoint="/query/llm-info",
            method="GET",
            timeout=2.0  # 短超时以避免UI阻塞
        )

        if not response:
            st.sidebar.warning("⚠️ 无法获取worker状态")
            return

        # 显示活动workers
        active_workers = response.get("active_workers", {})
        if active_workers:
            with st.sidebar.expander("Worker状态"):
                for worker_type, info in active_workers.items():
                    count = info.get("count", 0)
                    description = info.get("description", worker_type)
                    st.success(f"✅ {description} ({count} 个活动)")

                # 如果有队列信息，显示队列信息
                queue_stats = response.get("queue_stats", {})
                if queue_stats:
                    st.subheader("任务队列")
                    for queue, count in queue_stats.items():
                        if count > 0:
                            st.text(f"{queue}: {count} 个任务")
        else:
            st.sidebar.warning("⚠️ 未检测到活动的workers")

    except Exception as e:
        st.sidebar.warning(f"⚠️ 检查worker状态时出错: {str(e)}")


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