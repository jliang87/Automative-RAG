"""
汽车规格查询页面（Streamlit UI）
"""

import streamlit as st
import time
import os
from src.ui.components import header, metadata_filters, api_request, display_document, loading_spinner

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

def process_query(query: str, metadata_filter, top_k: int = 5):
    """处理查询并显示结果"""
    # 准备请求数据
    request_data = {
        "query": query,
        "metadata_filter": metadata_filter,
        "top_k": top_k
    }

    # 记录时间
    start_time = time.time()

    # 使用加载动画发送 API 请求
    with loading_spinner("正在处理您的查询..."):
        result = api_request(
            endpoint="/query/",
            method="POST",
            data=request_data
        )

    if not result:
        return

    # 显示答案
    st.header("查询结果")

    # 计算总处理时间
    processing_time = time.time() - start_time
    api_time = result.get("execution_time", 0)

    # 显示答案和引用来源
    st.markdown(result["answer"])

    # 显示处理时间信息
    st.caption(f"总处理时间: {processing_time:.2f}s (API: {api_time:.2f}s)")

    # 显示数据来源
    st.header("数据来源")

    if "documents" in result and result["documents"]:
        for i, doc in enumerate(result["documents"]):
            display_document(doc, i)
    else:
        st.info("未找到相关的文档数据。")

    # 允许优化查询
    st.subheader("优化您的查询")

    # 根据返回的文档生成推荐查询
    suggestions = generate_suggestions(query, result.get("documents", []))

    if suggestions:
        st.write("您可以尝试以下查询:")
        cols = st.columns(len(suggestions))

        for i, suggestion in enumerate(suggestions):
            with cols[i]:
                if st.button(suggestion, key=f"suggestion_{i}"):
                    st.session_state.current_query = suggestion
                    st.experimental_rerun()


def generate_suggestions(query: str, documents: list) -> list:
    """根据原始查询和返回结果生成推荐查询"""
    suggestions = []

    # 从返回的文档中提取元数据
    manufacturers = set()
    models = set()
    years = set()

    for doc in documents:
        metadata = doc.get("metadata", {})
        if "manufacturer" in metadata and metadata["manufacturer"]:
            manufacturers.add(metadata["manufacturer"])
        if "model" in metadata and metadata["model"]:
            models.add(metadata["model"])
        if "year" in metadata and metadata["year"]:
            years.add(str(metadata["year"]))

    # 生成基于查询和元数据的推荐问题
    query_lower = query.lower()

    # 如果查询涉及车辆规格
    if "specs" in query_lower or "specifications" in query_lower:
        for model in models:
            for year in years:
                suggestions.append(f"{year} 款 {model} 的车身尺寸是多少？")
                break
            break

    # 如果查询涉及马力
    if "horsepower" in query_lower or "power" in query_lower:
        if models and years:
            model = next(iter(models))
            year = next(iter(years))
            suggestions.append(f"{year} 款 {model} 的扭矩是多少？")

    # 如果查询的是某一款车型，则建议进行对比
    if len(models) == 1 and manufacturers:
        model = next(iter(models))
        manufacturer = next(iter(manufacturers))

        # 查找可比较的车型
        comparable = {
            "Camry": "本田雅阁",
            "Corolla": "本田思域",
            "RAV4": "本田 CR-V",
            "F-150": "雪佛兰 Silverado",
            "Civic": "丰田卡罗拉",
            "Accord": "丰田凯美瑞",
            "CR-V": "丰田 RAV4",
            "3 Series": "奔驰 C 级",
            "Model 3": "宝马 i4"
        }

        if model in comparable:
            suggestions.append(f"对比 {manufacturer} {model} 和 {comparable[model]} 的参数")

    # 如果查询涉及特定功能
    features = ["安全性", "油耗", "MPG", "娱乐系统", "导航"]
    for feature in features:
        if feature in query_lower and models:
            model = next(iter(models))
            suggestions.append(f"{model} 具备哪些 {feature} 相关功能？")
            break

    # 限制最多 3 条推荐查询
    return suggestions[:3]


header(
    "查询汽车规格",
    "输入问题，获取汽车规格的精准答案。"
)

# 查询输入框
if "current_query" in st.session_state:
    query = st.text_area("输入您的问题", value=st.session_state.current_query, height=100)
    # 清除会话状态中的当前查询，避免持续显示
    st.session_state.current_query = ""
else:
    query = st.text_area("输入您的问题", height=100)

# 元数据筛选（可展开）
with st.expander("高级筛选"):
    filter_data = metadata_filters()

# Top-K 选择器
top_k = st.slider("最多检索文档数量", min_value=1, max_value=20, value=5)

# 提交按钮
if st.button("提交查询", type="primary"):
    if not query:
        st.warning("请输入查询内容")

    process_query(query, filter_data, top_k)
