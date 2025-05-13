import os
import streamlit as st
import httpx
from typing import Dict, List, Optional, Union

# 导入统一的 API 客户端
from src.ui.api_client import api_request
from src.ui.components import header

# 导入增强组件
from src.ui.system_notifications import display_notifications_sidebar
from src.ui.enhanced_error_handling import robust_api_status_indicator, handle_worker_dependency

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
if "system_notifications" not in st.session_state:
    st.session_state.system_notifications = []
if "last_notification_check" not in st.session_state:
    st.session_state.last_notification_check = 0

# 在侧边栏显示通知
display_notifications_sidebar(st.session_state.api_url, st.session_state.api_key)

# 使用增强的 API 状态指示器代替简单的状态检查
with st.sidebar:
    api_available = robust_api_status_indicator(show_detail=True)

# 仅在 API 可用时显示主要内容
if api_available:
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
- **系统通知**：重要事件通知系统，及时提醒管理员处理问题

### 支持的后台任务类型

1. **视频处理**：YouTube和Bilibili视频的下载和转录
2. **PDF处理**：PDF解析、OCR和表格提取
3. **批量视频处理**：一次处理多个视频链接

### 如何使用

1. 提交任务后，系统会自动创建一个后台作业并跳转到任务管理页面
2. 在任务管理页面查看所有任务的状态和结果
3. 系统通知页面可以查看所有系统警告和错误
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

            # 显示集合信息
            collection = status_info.get("collection") or '未定义'
            st.write(f"集合: {collection}")

            # 显示任务统计
            if "job_stats" in status_info:
                job_stats = status_info.get("job_stats", {})
                st.subheader("后台任务")

                col1a, col1b, col1c, col1d = st.columns(4)
                with col1a:
                    st.metric("等待中", job_stats.get("pending_jobs", 0))
                with col1b:
                    st.metric("处理中", job_stats.get("processing_jobs", 0))
                with col1c:
                    st.metric("已完成", job_stats.get("completed_jobs", 0))
                with col1d:
                    st.metric("失败", job_stats.get("failed_jobs", 0))

        with col2:
            st.subheader("服务状态")

            # 获取简单的服务状态信息
            service_status = api_request(
                endpoint="/system/health/detailed",
                method="GET",
                silent=True
            )

            if service_status:
                # 创建一个简洁的服务状态摘要
                services = {
                    "API 服务": "✅ 正常",
                    "LLM 服务": "⚠️ 未知",
                    "嵌入服务": "⚠️ 未知",
                    "转录服务": "⚠️ 未知",
                    "文本处理服务": "⚠️ 未知"
                }

                # 从健康检查更新服务状态
                workers = service_status.get("workers", {})
                for worker_id, info in workers.items():
                    worker_type = info.get("type", "unknown")
                    status = info.get("status", "unknown")

                    if "gpu-inference" in worker_type:
                        services["LLM 服务"] = "✅ 正常" if status == "healthy" else "❌ 异常"
                    elif "gpu-embedding" in worker_type:
                        services["嵌入服务"] = "✅ 正常" if status == "healthy" else "❌ 异常"
                    elif "gpu-whisper" in worker_type:
                        services["转录服务"] = "✅ 正常" if status == "healthy" else "❌ 异常"
                    elif "cpu" in worker_type:
                        services["文本处理服务"] = "✅ 正常" if status == "healthy" else "❌ 异常"

                # 显示服务状态表格
                for service, status in services.items():
                    st.text(f"{service}: {status}")

                # 添加查看详情链接
                st.markdown("[查看详细状态](/系统管理)")

except Exception as e:
    st.error(f"获取系统状态时出错: {str(e)}")
else:
    st.info("API服务可用，但未能获取系统状态。请检查系统配置。")