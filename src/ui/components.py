"""
Streamlit 界面 UI 组件 - 简化版本用于自触发作业链架构

该模块包含 Streamlit 应用程序中的可复用 UI 组件。
专为自触发作业链和专用Worker架构优化。
"""

import streamlit as st
import httpx
from typing import Dict, List, Optional, Union, Callable, Any
# Fix for Arrow serialization errors - Add this to components.py or create a new utils file
import pandas as pd

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


def job_chain_status_card(job_data: Dict[str, Any]):
    """显示作业链状态卡片"""
    job_id = job_data.get("job_id", "")
    status = job_data.get("status", "")
    job_type = job_data.get("job_type", "")

    # 状态颜色映射
    status_colors = {
        "pending": "🟡",
        "processing": "🔵",
        "completed": "🟢",
        "failed": "🔴"
    }

    status_icon = status_colors.get(status, "⚪")

    with st.container():
        col1, col2, col3 = st.columns([2, 1, 1])

        with col1:
            st.markdown(f"**{status_icon} {job_id[:8]}...**")
            st.caption(f"类型: {job_type}")

        with col2:
            st.markdown(f"**状态**")
            st.caption(status)

        with col3:
            if st.button("详情", key=f"details_{job_id}"):
                st.session_state.selected_job_id = job_id
                st.rerun()


def worker_health_indicator():
    """显示专用Worker健康状态指示器"""
    health_data = api_request(
        endpoint="/system/health/detailed",
        method="GET",
        silent=True
    )

    if not health_data:
        st.warning("⚠️ 无法获取Worker状态")
        return

    workers = health_data.get("workers", {})

    # 按Worker类型分组
    worker_types = {
        "gpu-whisper": {"name": "🎵 语音转录", "workers": []},
        "gpu-embedding": {"name": "🔢 向量嵌入", "workers": []},
        "gpu-inference": {"name": "🧠 LLM推理", "workers": []},
        "cpu": {"name": "💻 CPU处理", "workers": []}
    }

    # 分类Worker
    for worker_id, info in workers.items():
        worker_type = info.get("type", "unknown")
        if worker_type in worker_types:
            worker_types[worker_type]["workers"].append((worker_id, info))

    # 显示每种Worker类型的状态
    cols = st.columns(4)
    for i, (worker_type, type_info) in enumerate(worker_types.items()):
        with cols[i]:
            workers_of_type = type_info["workers"]
            healthy_count = sum(1 for _, info in workers_of_type if info.get("status") == "healthy")
            total_count = len(workers_of_type)

            if healthy_count == total_count and total_count > 0:
                st.success(f"✅ {type_info['name']}")
                st.caption(f"{healthy_count}/{total_count} 健康")
            elif healthy_count > 0:
                st.warning(f"⚠️ {type_info['name']}")
                st.caption(f"{healthy_count}/{total_count} 健康")
            else:
                st.error(f"❌ {type_info['name']}")
                st.caption("不可用")


def safe_dataframe_display(data: List[Dict[str, Any]], hide_index: bool = True, use_container_width: bool = True):
    """
    Safely display a DataFrame with proper type conversion to avoid Arrow serialization errors.

    Args:
        data: List of dictionaries to convert to DataFrame
        hide_index: Whether to hide the DataFrame index
        use_container_width: Whether to use container width
    """
    if not data:
        st.info("没有数据显示")
        return

    try:
        # Create DataFrame
        df = pd.DataFrame(data)

        # Convert problematic columns to strings to avoid Arrow serialization issues
        for col in df.columns:
            # Check if column contains mixed types or objects that could cause issues
            if df[col].dtype == 'object':
                # Convert all values to strings, handling None/NaN values
                df[col] = df[col].astype(str).replace(['None', 'nan', 'NaN'], '-')
            elif df[col].dtype == 'int64':
                # Ensure integers are properly formatted
                df[col] = df[col].astype('Int64')  # Pandas nullable integer
            elif df[col].dtype == 'float64':
                # Handle float precision issues
                df[col] = df[col].round(2)

        # Display the DataFrame
        st.dataframe(df, hide_index=hide_index, use_container_width=use_container_width)

    except Exception as e:
        st.error(f"显示数据时出错: {str(e)}")
        # Fallback: display as JSON
        st.json(data)


def clean_dataframe_for_display(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean a DataFrame to avoid Arrow serialization issues.

    Args:
        df: Input DataFrame

    Returns:
        Cleaned DataFrame safe for Streamlit display
    """
    df_clean = df.copy()

    for col in df_clean.columns:
        if df_clean[col].dtype == 'object':
            # Convert problematic object columns to strings
            df_clean[col] = df_clean[col].astype(str).replace(['None', 'nan', 'NaN'], '-')
        elif pd.api.types.is_numeric_dtype(df_clean[col]):
            # Handle numeric columns with potential null values
            if df_clean[col].isna().any():
                if pd.api.types.is_integer_dtype(df_clean[col]):
                    df_clean[col] = df_clean[col].astype('Int64')  # Nullable integer
                else:
                    df_clean[col] = df_clean[col].astype('float64')

    return df_clean


# Example usage in your existing code:
# Instead of:
# st.dataframe(pd.DataFrame(data), hide_index=True)
#
# Use:
# safe_dataframe_display(data)
#
# Or for existing DataFrames:
# df_clean = clean_dataframe_for_display(df)
# st.dataframe(df_clean, hide_index=True)

# For the specific issue in your code, update the queue display in enhanced_worker_status.py:
def safe_queue_display(queue_data: List[Dict[str, Any]]):
    """Safely display queue data avoiding Arrow serialization issues."""
    if not queue_data:
        st.info("没有队列数据")
        return

    # Clean the data before creating DataFrame
    cleaned_data = []
    for item in queue_data:
        cleaned_item = {}
        for key, value in item.items():
            if value is None:
                cleaned_item[key] = "-"
            elif isinstance(value, (int, float)):
                cleaned_item[key] = value
            else:
                cleaned_item[key] = str(value)
        cleaned_data.append(cleaned_item)

    try:
        df = pd.DataFrame(cleaned_data)
        st.dataframe(df, hide_index=True, use_container_width=True)
    except Exception as e:
        st.error(f"显示队列数据时出错: {str(e)}")
        # Fallback to simple display
        for i, item in enumerate(cleaned_data):
            with st.expander(f"队列 {i + 1}"):
                for key, value in item.items():
                    st.write(f"**{key}**: {value}")