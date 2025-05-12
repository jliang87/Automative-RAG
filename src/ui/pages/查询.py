"""
汽车规格查询页面（Streamlit UI）
支持异步查询，避免前端超时
"""

import streamlit as st
import time
import os
import json
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

# 异步查询相关状态
if "query_job_id" not in st.session_state:
    st.session_state.query_job_id = None
if "query_status" not in st.session_state:
    st.session_state.query_status = None
if "polling" not in st.session_state:
    st.session_state.polling = False
if "poll_count" not in st.session_state:
    st.session_state.poll_count = 0

def process_async_query(query: str, metadata_filter, top_k: int = 5):
    """处理异步查询并启动轮询"""
    # 准备请求数据
    request_data = {
        "query": query,
        "metadata_filter": metadata_filter,
        "top_k": top_k
    }

    # 使用加载动画发送 API 请求
    with st.spinner("正在提交您的查询..."):
        result = api_request(
            endpoint="/query",
            method="POST",
            data=request_data
        )

    if not result:
        st.error("提交查询失败，请稍后重试")
        return

    # 获取任务ID并开始轮询
    st.session_state.query_job_id = result.get("job_id")
    st.session_state.query_status = result.get("status", "pending")
    st.session_state.polling = True
    st.session_state.poll_count = 0

    # 刷新页面开始轮询
    st.rerun()

def poll_for_results():
    """轮询查询结果"""
    if not st.session_state.query_job_id or not st.session_state.polling:
        return

    job_id = st.session_state.query_job_id

    # 创建占位容器
    status_container = st.empty()
    answer_container = st.empty()
    docs_container = st.empty()

    # 更新状态
    st.session_state.poll_count += 1
    status_container.info(f"查询正在处理中... (已等待 {st.session_state.poll_count} 秒)")

    # 获取查询结果
    result = api_request(
        endpoint=f"/query/results/{job_id}",
        method="GET"
    )

    if not result:
        status_container.error("获取查询状态失败")
        st.session_state.polling = False
        return

    # 处理结果
    status = result.get("status", "")

    if status == "completed":
        # 查询完成
        status_container.success("✅ 查询完成！")
        answer_container.markdown(f"### 回答\n{result.get('answer', '')}")

        # 显示文档来源
        docs_container.subheader("数据来源")
        documents = result.get("documents", [])
        if documents:
            for i, doc in enumerate(documents):
                display_document(doc, i)
        else:
            docs_container.info("没有找到相关文档")

        # 停止轮询
        st.session_state.polling = False

    elif status == "failed":
        # 查询失败
        status_container.error(f"❌ 查询失败: {result.get('answer', '')}")
        st.session_state.polling = False

    elif st.session_state.poll_count >= 120:  # 2分钟超时
        # 轮询超时
        status_container.warning("⚠️ 查询处理时间过长，但仍在后台运行")
        status_container.info("您可以稍后在任务管理页面查看结果")
        st.session_state.polling = False

    else:
        # 继续轮询
        time.sleep(1)
        st.rerun()

def render_query_page():
    """渲染查询页面"""
    header(
        "查询汽车规格",
        "输入问题，获取汽车规格的精准答案。"
    )

    # 查询输入框
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
        else:
            process_async_query(query, filter_data, top_k)

    # 如果正在轮询，显示取消按钮
    if st.session_state.polling:
        if st.button("取消等待"):
            st.session_state.polling = False
            st.info("查询仍在后台处理中，您可以稍后在任务管理页面查看结果")
            st.session_state.query_job_id = None

    # 如果有任务ID但未在轮询，显示检查结果按钮
    elif st.session_state.query_job_id and not st.session_state.polling:
        if st.button("检查查询结果"):
            st.session_state.polling = True
            st.rerun()

    # 轮询结果
    if st.session_state.polling:
        poll_for_results()

# 渲染页面
render_query_page()