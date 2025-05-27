# src/ui/pages/查询.py - Improved Chinese messages

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
            endpoint="/query/",  # Use trailing slash version
            method="POST",
            data={
                "query": query_text,
                "metadata_filter": filters,
                "top_k": 5
            }
        )

        if result and "job_id" in result:
            return result["job_id"]
        else:
            st.error("查询提交失败：服务器未返回有效响应")
            return None

    except Exception as e:
        st.error(f"查询提交时发生错误: {str(e)}")
        return None

def get_query_result(job_id):
    """Get query results"""
    try:
        result = api_request(f"/query/results/{job_id}", method="GET")
        return result
    except Exception as e:
        st.error(f"获取查询结果时发生错误: {str(e)}")
        return None

# Query input
query = st.text_area(
    "请输入您的问题",
    placeholder="例如：2023年宝马X5的发动机参数是什么？",
    height=100,
    help="请描述您想了解的汽车信息，系统将为您搜索相关内容"
)

# Simple filters
with st.expander("筛选条件（可选）"):
    col1, col2 = st.columns(2)

    with col1:
        manufacturer = st.selectbox(
            "品牌",
            ["", "宝马", "奔驰", "奥迪", "丰田", "本田", "大众", "福特", "特斯拉"],
            key="manufacturer",
            help="选择特定的汽车品牌"
        )

        year = st.selectbox(
            "年份",
            [""] + [str(y) for y in range(2024, 2015, -1)],
            key="year",
            help="选择汽车年份"
        )

    with col2:
        category = st.selectbox(
            "车型",
            ["", "轿车", "SUV", "跑车", "MPV"],
            key="category",
            help="选择车辆类型"
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
if st.button("🔍 开始查询", type="primary", disabled=not query.strip(), use_container_width=True):
    if query.strip():
        with st.spinner("正在提交查询请求..."):
            job_id = submit_query(query.strip(), filters if filters else None)

            if job_id:
                st.session_state.current_job_id = job_id
                st.session_state.query_text = query.strip()
                st.success(f"✅ 查询已提交，任务ID: {job_id[:8]}...")
                st.info("⏳ 正在处理您的查询，请稍候...")
                time.sleep(1)  # Give user time to see the success message
                st.rerun()
            else:
                st.error("❌ 查询提交失败，请检查网络连接或稍后重试")

# Show results if we have a job
if hasattr(st.session_state, 'current_job_id') and st.session_state.current_job_id:
    job_id = st.session_state.current_job_id

    st.markdown("---")
    st.subheader("📋 查询进度")

    # Create a placeholder for dynamic updates
    status_placeholder = st.empty()
    result_placeholder = st.empty()

    # Poll for results
    with status_placeholder:
        with st.spinner("正在获取查询结果..."):
            result = get_query_result(job_id)

    if result:
        status = result.get("status", "")

        with status_placeholder:
            if status == "completed":
                st.success("✅ 查询完成！")
            elif status == "failed":
                st.error("❌ 查询失败")
            elif status in ["processing", "pending"]:
                st.info("⏳ 正在处理中...")
            else:
                st.warning(f"状态: {status}")

        with result_placeholder:
            if status == "completed":
                # Show answer
                st.subheader("💡 查询结果")
                answer = result.get("answer", "")
                if answer:
                    st.markdown(f"**回答:** {answer}")
                else:
                    st.warning("未找到相关信息")

                # Show sources
                documents = result.get("documents", [])
                if documents:
                    st.subheader("📚 信息来源")

                    for i, doc in enumerate(documents[:3]):  # Show top 3 sources
                        with st.expander(f"📄 来源 {i+1}: {doc.get('metadata', {}).get('title', '文档')}"):
                            content = doc.get("content", "")
                            if content:
                                st.write(content[:500] + ("..." if len(content) > 500 else ""))

                            # Simple metadata
                            metadata = doc.get("metadata", {})
                            if metadata.get("manufacturer") or metadata.get("model"):
                                info = []
                                if metadata.get("manufacturer"):
                                    info.append(f"品牌: {metadata['manufacturer']}")
                                if metadata.get("model"):
                                    info.append(f"车型: {metadata['model']}")
                                if metadata.get("year"):
                                    info.append(f"年份: {metadata['year']}")

                                if info:
                                    st.caption(" | ".join(info))

                            # Relevance score
                            score = doc.get("relevance_score", 0)
                            if score:
                                st.caption(f"相关度: {score:.3f}")

                # Action buttons
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("🔄 新建查询", use_container_width=True):
                        del st.session_state.current_job_id
                        if hasattr(st.session_state, 'query_text'):
                            del st.session_state.query_text
                        st.rerun()

                with col2:
                    if st.button("📋 查看任务详情", use_container_width=True):
                        st.switch_page("pages/后台任务.py")

            elif status == "failed":
                error_msg = result.get("answer", "未知错误")
                st.error(f"查询失败: {error_msg}")

                col1, col2 = st.columns(2)
                with col1:
                    if st.button("🔄 重新查询", use_container_width=True):
                        del st.session_state.current_job_id
                        st.rerun()
                with col2:
                    if st.button("📋 查看错误详情", use_container_width=True):
                        st.switch_page("pages/后台任务.py")

            else:
                # Still processing
                progress_msg = result.get("answer", "正在处理您的查询...")
                st.info(progress_msg)

                # Auto-refresh for processing jobs
                if st.button("🔄 刷新状态"):
                    st.rerun()

                # Auto-refresh every 3 seconds
                time.sleep(3)
                st.rerun()
    else:
        with status_placeholder:
            st.error("❌ 无法获取查询状态，请稍后重试")

        with result_placeholder:
            if st.button("🔄 重试", use_container_width=True):
                st.rerun()

# Query examples
st.markdown("---")
if st.checkbox("💡 显示查询示例"):
    st.subheader("查询示例")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**车辆规格查询:**")
        examples1 = [
            "2023年奔驰E级的安全配置有哪些？",
            "特斯拉Model 3的续航里程是多少？",
            "宝马X5的发动机参数"
        ]
        for example in examples1:
            if st.button(example, key=f"example1_{example[:10]}", use_container_width=True):
                st.session_state.example_query = example
                st.rerun()

    with col2:
        st.markdown("**对比查询:**")
        examples2 = [
            "宝马X5和奥迪Q7哪个更省油？",
            "丰田卡罗拉的维修保养费用如何？",
            "电动车和燃油车的优缺点对比"
        ]
        for example in examples2:
            if st.button(example, key=f"example2_{example[:10]}", use_container_width=True):
                st.session_state.example_query = example
                st.rerun()

    # Use example query
    if hasattr(st.session_state, 'example_query'):
        st.text_area("选中的示例查询", st.session_state.example_query, key="example_display", height=60)
        if st.button("✅ 使用此查询", use_container_width=True):
            # Set the query and clear example
            st.session_state.query_text = st.session_state.example_query
            del st.session_state.example_query
            st.rerun()

# Footer
st.markdown("---")
st.caption("💡 提示：使用具体的车型名称和年份可以获得更准确的查询结果")