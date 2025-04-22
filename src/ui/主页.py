import os
import streamlit as st
import httpx
from typing import Dict, List, Optional, Union
from src.ui.components import header, api_request

# 配置应用
st.set_page_config(
    page_title="汽车规格 RAG 系统",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

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

# API 设置（添加到侧边栏）
with st.sidebar:
    st.subheader("API 连接设置")
    api_url = st.text_input("API 地址", value=st.session_state.api_url)
    api_key = st.text_input("API 密钥", value=st.session_state.api_key, type="password")

    if st.button("保存设置"):
        st.session_state.api_url = api_url
        st.session_state.api_key = api_key
        st.success("设置已保存")

    # 添加任务管理快捷方式
    st.divider()
    st.subheader("快速导航")

    if st.button("✨ 查询汽车规格"):
        st.switch_page("src/ui/pages/查询.py")

    if st.button("📥 导入数据"):
        st.switch_page("src/ui/pages/数据摄取.py")

    if st.button("⚙️ 后台任务管理"):
        st.switch_page("src/ui/pages/后台任务.py")

header(
    "汽车规格 RAG 系统",
    "一个基于 GPU 加速的系统，用于使用延迟交互检索 (Late Interaction Retrieval) 和混合重排序进行汽车规格信息检索。"
)

# 系统概述
st.markdown("""
## 关于系统

该系统采用最先进的检索增强生成 (RAG) 技术，提供精确的汽车规格信息。主要特性包括：

- **多语言支持**：使用 BAAI/bge-m3 嵌入模型，同时支持中文和英文内容
- **两阶段混合重排序**：结合 ColBERT 和 BGE-Reranker-Large 实现高质量检索
  - ColBERT 提供令牌级别的精确交互分析（占比 80%）
  - BGE-Reranker 增强双语理解能力（占比 20%）
- **延迟交互检索**（ColBERT）实现更精准的语义匹配
- **本地 DeepSeek LLM** 提供无 API 依赖的答案生成
- **统一视频处理**：使用 Whisper AI 高质量转录 YouTube、Bilibili 等平台视频
- **全 GPU 加速**，支持转录、OCR、嵌入和推理计算
- **混合搜索** 结合向量相似度和元数据筛选
- **双语 OCR**：支持英文和简体中文文档的光学字符识别
- **后台处理系统**：资源密集型任务（如视频转录和PDF处理）在后台运行，提高系统响应能力
""")

# 添加新的部分：后台处理系统介绍
st.markdown("""
## 后台处理系统

系统现在包含一个强大的后台处理子系统，专门处理资源密集型任务：

### 主要特点

- **非阻塞处理**：视频转录和PDF处理等重任务在后台执行，UI保持响应
- **任务管理**：完整的任务管理界面，跟踪所有处理任务的状态
- **高并发支持**：同时处理多个任务，充分利用系统资源
- **自动资源管理**：优化CPU和内存使用，防止系统过载
- **任务恢复**：支持从意外中断中恢复任务

### 支持的后台任务类型

1. **视频处理**：YouTube和Bilibili视频的下载和转录
2. **PDF处理**：PDF解析、OCR和表格提取
3. **批量视频处理**：一次处理多个视频链接

### 如何使用

1. 提交任务后，系统会自动创建一个后台作业并跳转到任务管理页面
2. 在任务管理页面查看所有任务的状态和结果
""")

# 获取系统状态信息
try:
    status_info = api_request(
        endpoint="/ingest/status",
        method="GET"
    )

    if status_info:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("系统状态")
            st.write(f"状态: {status_info.get('status', '未知')}")
            document_count = status_info.get('document_count') or 0
            st.write(f"文档数量: {document_count}")

            # 显示集合信息（如果可用）
            collection = status_info.get("collection") or '未定义'
            st.write(f"数据集合: {collection}")

            # 显示后台任务状态（如果可用）
            if "job_stats" in status_info:
                job_stats = status_info.get("job_stats", {})
                st.subheader("后台任务状态")

                col1a, col1b, col1c, col1d = st.columns(4)
                with col1a:
                    st.metric("等待中", job_stats.get("pending_jobs", 0))
                with col1b:
                    st.metric("处理中", job_stats.get("processing_jobs", 0))
                with col1c:
                    st.metric("已完成", job_stats.get("completed_jobs", 0))
                with col1d:
                    st.metric("失败", job_stats.get("failed_jobs", 0))

                if job_stats.get("processing_jobs", 0) > 0 or job_stats.get("pending_jobs", 0) > 0:
                    st.info("有任务正在处理中! [前往任务管理页面查看详情](#)")

        with col2:
            st.subheader("GPU 信息")
            gpu_info = status_info.get("gpu_info", {})

            if gpu_info:
                st.write(f"设备: {gpu_info.get('device', '未知')}")
                if 'device_name' in gpu_info:
                    st.write(f"GPU: {gpu_info['device_name']}")
                if 'memory_allocated' in gpu_info and gpu_info['memory_allocated'] is not None:
                    st.write(f"已使用内存: {gpu_info['memory_allocated']}")
                else:
                    st.write("已使用内存: 未知")
                if 'whisper_model' in gpu_info:
                    st.write(f"Whisper 模型: {gpu_info['whisper_model']}")
            else:
                st.write("运行在 CPU 上")

    # 获取 LLM 信息
    llm_info = api_request(
        endpoint="/query/llm-info",
        method="GET"
    )

    if llm_info:
        st.subheader("LLM 信息")

        # 规范化模型名称
        model_name = llm_info.get("model_name", "未知")
        if "/" in model_name:
            model_name = model_name.split("/")[-1]

        col1, col2 = st.columns(2)

        with col1:
            st.write(f"模型: {model_name}")
            st.write(f"量化方式: {llm_info.get('quantization', '无')}")

        with col2:
            st.write(f"温度参数: {llm_info.get('temperature', 0.0)}")
            st.write(f"最大 Token 数: {llm_info.get('max_tokens', 0)}")

        if 'vram_usage' in llm_info:
            st.write(f"显存占用: {llm_info['vram_usage']}")

except Exception as e:
    st.error(f"获取系统信息出错: {str(e)}")

# 功能导航卡片
st.markdown("## 功能导航")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    ### 🔍 查询汽车规格

    通过自然语言查询汽车的详细规格信息。

    - 中英双语支持
    - 精确匹配相关信息
    - 智能元数据筛选
    """)
    if st.button("前往查询页面", key="query_btn"):
        st.switch_page("src/ui/pages/查询.py")

with col2:
    st.markdown("""
    ### 📥 导入数据

    从多种来源导入汽车规格数据。

    - 视频转录处理
    - PDF文档解析
    - 手动数据录入
    """)
    if st.button("前往导入页面", key="ingest_btn"):
        st.switch_page("src/ui/pages/数据摄取.py")

with col3:
    st.markdown("""
    ### ⚙️ 后台任务管理

    查看和管理后台运行的处理任务。

    - 实时任务状态
    - 详细处理结果
    - 任务历史记录
    """)
    if st.button("前往任务管理", key="jobs_btn"):
        st.switch_page("src/ui/pages/后台任务.py")

# 系统架构和模型信息（可展开）
with st.expander("系统架构"):
    st.markdown("""
    ### 主要组件

    1. **数据导入管道**
       - 视频转录 - 使用 Whisper AI 转录 YouTube、Bilibili 等平台视频
       - PDF 解析（支持中英双语OCR和表格提取）
       - 元数据提取与文档切分

    2. **向量数据库**
       - Qdrant 高效存储向量
       - 使用 BAAI/bge-m3 多语言嵌入模型
       - 混合搜索能力（向量 + 元数据）
       - 文档存储与管理

    3. **检索系统**
       - Qdrant 进行初步检索
       - 两阶段混合重排序:
         - ColBERT 进行令牌级别的精确匹配 (80%)
         - BGE-Reranker 增强双语理解能力 (20%)
       - 采用延迟交互模式进行 Token 级分析

    4. **生成层**
       - DeepSeek LLM 集成
       - 结合上下文信息生成答案
       - 响应带有来源引用

    5. **API 层**
       - FastAPI 后端，采用依赖注入
       - Swagger 文档
       - 认证与安全

    6. **用户界面**
       - Streamlit 可视化界面
       - 查询与筛选接口
       - 结果展示
       - 结果来源标注与解释

    7. **后台处理系统**
       - Dramatiq 任务队列
       - Redis 作为消息代理
       - 作业状态追踪与管理
    """)