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
if "api_key" not in st.session_state:
    st.session_state.api_key = API_KEY
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "current_page" not in st.session_state:
    st.session_state.current_page = "query"


def make_api_request(
        endpoint: str,
        method: str = "GET",
        data: Optional[Dict] = None,
        files: Optional[Dict] = None,
        params: Optional[Dict] = None,
) -> httpx.Response:
    """向 API 发送请求"""
    headers = {"x-token": st.session_state.api_key}
    url = f"{API_URL}{endpoint}"

    try:
        with httpx.Client() as client:
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

            return response
    except Exception as e:
        st.error(f"API 请求出错: {str(e)}")
        return None

header(
    "汽车规格 RAG 系统",
    "一个基于 GPU 加速的系统，用于使用后期交互式检索 (Late Interaction Retrieval) 进行汽车规格信息检索。"
)

# 系统概述
st.markdown("""
## 关于系统

该系统采用最先进的检索增强生成 (RAG) 技术，提供精确的汽车规格信息。主要特性包括：

- **后期交互检索**（ColBERT）实现更精准的语义匹配
- **本地 DeepSeek LLM** 提供无 API 依赖的答案生成
- **多模态输入处理** 支持 YouTube、Bilibili、优酷视频和 PDF 文档
- **全 GPU 加速**，支持转录、OCR、嵌入和推理计算
- **混合搜索** 结合向量相似度和元数据筛选
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
            st.write(f"文档数量: {status_info.get('document_count', 0):,}")

            # 显示集合信息（如果可用）
            if "collection" in status_info:
                st.write(f"数据集合: {status_info['collection']}")

        with col2:
            st.subheader("GPU 信息")
            gpu_info = status_info.get("gpu_info", {})

            if gpu_info:
                st.write(f"设备: {gpu_info.get('device', '未知')}")
                if 'device_name' in gpu_info:
                    st.write(f"GPU: {gpu_info['device_name']}")
                if 'memory_allocated' in gpu_info:
                    st.write(f"已使用内存: {gpu_info['memory_allocated']}")
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

# 使用指南
st.markdown("""
## 入门指南

1. **数据导入**：进入“数据导入”选项卡，上传汽车相关数据：
   - YouTube 汽车评测与展示视频
   - Bilibili 汽车内容
   - 优酷汽车视频
   - PDF 规格书和说明书
   - 手动录入汽车规格数据

2. **查询数据**：进入“查询”选项卡，提问汽车相关问题：
   - 按 **品牌、车型、年份、类别等** 筛选
   - 精确回答并提供数据来源
   - 显示知识库中的相关片段

3. **示例查询**：
   - "2023 款丰田凯美瑞的马力是多少？"
   - "本田思域和丰田卡罗拉的油耗对比"
   - "福特 F-150 的车身尺寸是多少？"
   - "特斯拉 Model 3 的标配安全功能有哪些？"
""")

# 系统架构
with st.expander("系统架构"):
    st.markdown("""
    ### 主要组件

    1. **数据导入管道**
       - 视频转录（YouTube、Bilibili、优酷）使用 Whisper
       - PDF 解析（OCR 和表格提取）
       - 元数据提取与文档切分

    2. **向量数据库**
       - Qdrant 高效存储向量
       - 混合搜索能力（向量 + 元数据）
       - 文档存储与管理

    3. **检索系统**
       - Qdrant 进行初步检索
       - ColBERT 进行语义重排序
       - 采用后期交互模式进行 Token 级分析

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
    """)
