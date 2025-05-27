"""
Clean query page - src/ui/pages/查询.py
"""

import streamlit as st
import time
from src.ui.api_client import api_request
from src.ui.session_init import initialize_session_state

initialize_session_state()

st.title("🔍 汽车信息查询")

def submit_query(query_text, filters):
    """Submit query and return job ID"""
    try:
        result = api_request(
            endpoint="/query",
            method="POST",
            data={
                "query": query_text,
                "metadata_filter": filters,
                "top_k": 5
            }
        )
        return result.get("job_id") if result else None
    except:
        return None

def get_query_result(job_id):
    """Get query results"""
    try:
        return api_request(f"/query/results/{job_id}", method="GET")
    except:
        return None

# Query input
query = st.text_area(
    "请输入您的问题",
    placeholder="例如：2023年宝马X5的发动机参数是什么？",
    height=100
)

# Simple filters
with st.expander("筛选条件（可选）"):
    col1, col2 = st.columns(2)

    with col1:
        manufacturer = st.selectbox(
            "品牌",
            ["", "宝马", "奔驰", "奥迪", "丰田", "本田", "大众", "福特", "特斯拉"],
            key="manufacturer"
        )

        year = st.selectbox(
            "年份",
            [""] + [str(y) for y in range(2024, 2015, -1)],
            key="year"
        )

    with col2:
        category = st.selectbox(
            "车型",
            ["", "轿车", "SUV", "跑车", "MPV"],
            key="category"
        )

# Build filters
filters = {}
if manufacturer:
    filters["manufacturer"] = manufacturer
if year:
    filters["year"] = int(year)
if category:
    filters["category"] = category

# Submit button
if st.button("查询", type="primary", disabled=not query.strip()):
    if query.strip():
        with st.spinner("正在查询中..."):
            job_id = submit_query(query.strip(), filters if filters else None)

            if job_id:
                st.session_state.current_job_id = job_id
                st.session_state.query_text = query.strip()
                st.rerun()
            else:
                st.error("查询提交失败，请重试")

# Show results if we have a job
if hasattr(st.session_state, 'current_job_id') and st.session_state.current_job_id:
    job_id = st.session_state.current_job_id

    # Poll for results
    result = get_query_result(job_id)

    if result:
        status = result.get("status", "")

        if status == "completed":
            # Show answer
            st.subheader("📋 查询结果")
            st.write(result.get("answer", ""))

            # Show sources
            documents = result.get("documents", [])
            if documents:
                st.subheader("📚 信息来源")

                for i, doc in enumerate(documents[:3]):  # Show top 3 sources
                    with st.expander(f"来源 {i+1}: {doc.get('metadata', {}).get('title', '文档')}"):
                        st.write(doc.get("content", ""))

                        # Simple metadata
                        metadata = doc.get("metadata", {})
                        if metadata.get("manufacturer") or metadata.get("model"):
                            info = []
                            if metadata.get("manufacturer"):
                                info.append(metadata["manufacturer"])
                            if metadata.get("model"):
                                info.append(metadata["model"])
                            if metadata.get("year"):
                                info.append(str(metadata["year"]))

                            if info:
                                st.caption(" • ".join(info))

            # Clear job state
            if st.button("新建查询"):
                del st.session_state.current_job_id
                del st.session_state.query_text
                st.rerun()

        elif status == "failed":
            st.error("查询失败，请重试")
            if st.button("重试"):
                del st.session_state.current_job_id
                st.rerun()
        else:
            # Still processing
            st.info("正在处理您的查询，请稍候...")
            time.sleep(2)
            st.rerun()

# Recent queries (simple)
if st.checkbox("显示查询示例"):
    st.subheader("💡 查询示例")
    examples = [
        "2023年奔驰E级的安全配置有哪些？",
        "特斯拉Model 3的续航里程是多少？",
        "宝马X5和奥迪Q7哪个更省油？",
        "丰田卡罗拉的维修保养费用如何？"
    ]

    for example in examples:
        if st.button(example, key=f"example_{example[:10]}"):
            st.session_state.example_query = example
            st.rerun()

    # Use example query
    if hasattr(st.session_state, 'example_query'):
        st.text_area("选中的示例", st.session_state.example_query, key="example_display")
        if st.button("使用此查询"):
            # Set the query and clear example
            st.session_state.query_text = st.session_state.example_query
            del st.session_state.example_query
            st.rerun()