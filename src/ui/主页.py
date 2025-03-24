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
# if "current_page" not in st.session_state:
#     st.session_state.current_page = "query"

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
- **多模态输入处理** 支持 YouTube、Bilibili、优酷视频和 PDF 文档
- **全 GPU 加速**，支持转录、OCR、嵌入和推理计算
- **混合搜索** 结合向量相似度和元数据筛选
- **双语 OCR**：支持英文和简体中文文档的光学字符识别
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

# 重排序信息
st.subheader("重排序引擎信息")
col1, col2 = st.columns(2)

with col1:
    st.write("**主要重排序引擎**: ColBERT")
    st.write("- 提供令牌级别的精确匹配")
    st.write("- 基于延迟交互模式")
    st.write("- 重排序权重: 80%")

with col2:
    st.write("**辅助重排序引擎**: BAAI/bge-reranker-large")
    st.write("- 增强双语理解能力")
    st.write("- 句子级别的重排序")
    st.write("- 重排序权重: 20%")

# 多语言混合排序系统的优势和示例
st.markdown("""
## 多语言混合排序系统的优势

使用 **BGE-M3 + ColBERTv2 (80%) + BGE Reranker (20%)** 的混合方法，可以有效地处理各种类型的查询。以下是系统的优势所在：
""")

st.markdown("""
### 1️⃣ ColBERTv2 (80%) - 精确的令牌匹配

ColBERT 在**基于事实、实体密集或简短查询**方面表现出色，这些查询需要**逐字精确匹配**。
""")

col1, col2 = st.columns([1, 2])
with col1:
    st.markdown("**示例查询**")
    st.markdown("- 「2024年中国最畅销的SUV」")
    st.markdown("- 「吉利极氪001电池规格」")
    st.markdown("- 「特斯拉Model S Plaid最高速度」")
    st.markdown("- 「北京周边的电动车充电站」")
    st.markdown("- 「比亚迪海豚和名爵MG4 EV对比」")

with col2:
    st.markdown("**为什么ColBERT表现优异**")
    st.markdown("优先匹配含有**精确词汇**如\"SUV\"、\"中国\"和\"2024\"的文档")
    st.markdown("确保含有**精确技术细节**的文档获得更高排名")
    st.markdown("将\"最高速度\"与实际规格匹配，而非泛泛的速度讨论")
    st.markdown("优先考虑同时包含**\"充电站\"+\"北京\"**的文档")
    st.markdown("找到直接比较而非关于比亚迪或MG4的一般性文章")

st.markdown("""
🔹 **为什么需要混合排序?** 如果仅使用ColBERT，可能会错过**更广泛的上下文理解**（例如，"最快"而非"最高速度"）。BGE Reranker (20%)在这里能提供帮助。
""")

st.markdown("""
### 2️⃣ BGE Reranker (20%) - 上下文理解与更广泛的语义

BGE-Reranker更适合**自然语言、基于观点和需要推理的查询**，这些查询中**上下文比精确词匹配更重要**。
""")

col1, col2 = st.columns([1, 2])
with col1:
    st.markdown("**示例查询**")
    st.markdown("- 「电池退化如何影响电动车性能?」")
    st.markdown("- 「在中国拥有电动车比燃油车更便宜吗?」")
    st.markdown("- 「为什么中国电动车正主导市场?」")
    st.markdown("- 「2024年中国政府对电动车的补贴政策」")
    st.markdown("- 「磷酸铁锂电池和三元锂电池的安全隐患是什么?」")

with col2:
    st.markdown("**为什么BGE Reranker表现优异**")
    st.markdown("捕捉整体含义，而不仅仅是匹配\"电池\"和\"退化\"")
    st.markdown("理解\"更便宜\"涉及**总拥有成本**的讨论")
    st.markdown("优先考虑**解释市场趋势**的文档，而非仅列出电动车")
    st.markdown("捕捉与补贴相关的讨论，即使\"政府政策\"不是精确表述")
    st.markdown("查找**解释风险**的文章，而非仅列出电池类型")

st.markdown("""
🔹 **为什么需要混合排序?** 如果**仅使用BGE-Reranker**，排名可能过于宽泛。**ColBERT确保关键词如"电动车"、"电池"、"中国"仍然被优先考虑。**
""")

st.markdown("""
### 3️⃣ 多语言查询 - 跨语言匹配

借助**BGE-M3**，系统可以**同时检索中英文文档**，即使查询仅使用一种语言。
""")

col1, col2 = st.columns([1, 2])
with col1:
    st.markdown("**用户查询**")
    st.markdown("- 「中国最畅销的SUV 2024」")
    st.markdown("- 「比亚迪海豹电池寿命」")
    st.markdown("- 「哪家电动汽车公司在欧洲市场增长最快?」")
    st.markdown("- 「Tesla Model Y销量」")
    st.markdown("- 「中国的自动驾驶法规是什么?」")

with col2:
    st.markdown("**预期检索结果(中英文)**")
    st.markdown("检索标题为\"Top 10 SUVs in China 2024\"的英文文档及中文文章")
    st.markdown("匹配讨论\"BYD Seal battery longevity\"的英文资料")
    st.markdown("检索关于**蔚来、比亚迪、特斯拉市场增长**的中英文报告")
    st.markdown("同时检索英文(\"Tesla Model Y sales report\")和中文资料")
    st.markdown("匹配英文政策摘要和中文政府文件")

st.markdown("""
🔹 **混合排序的优势:**
* **ColBERT确保事实精确性**（例如，如果查询是关于比亚迪，不会检索特斯拉）
* **BGE Reranker改善上下文排序**（例如，如果用户询问"增长最快的电动车公司"，即使文档中没有"最快"一词，也会包含特斯拉/蔚来）
* **BGE-M3实现跨语言检索**，确保**英文查询可检索中文文档，反之亦然**
""")

st.markdown("""
### 📌 核心优势

* **ColBERTv2 (80%)**: **为事实性、技术性和特定查询提供精确度**
* **BGE Reranker (20%)**: **为基于观点、推理和较长查询提供更广泛的语义排序**
* **BGE-M3**: **多语言检索，确保跨语言文档匹配**

这种**混合排序方法平衡了准确性、令牌级精确度和上下文相关性**，非常适合**多语言搜索和检索**。
""")

# 使用指南
st.markdown("""
## 入门指南

1. **数据导入**：进入"数据导入"选项卡，上传汽车相关数据：
   - YouTube 汽车评测与展示视频
   - Bilibili 汽车内容
   - 优酷汽车视频
   - PDF 规格书和说明书（支持中英文OCR）
   - 手动录入汽车规格数据

2. **查询数据**：进入"查询"选项卡，提问汽车相关问题：
   - 支持中文和英文查询
   - 按 **品牌、车型、年份、类别等** 筛选
   - 精确回答并提供数据来源
   - 显示知识库中的相关片段

3. **中英文查询示例**：
   - "2023 款丰田凯美瑞的马力是多少？"
   - "What is the horsepower of the 2024 Tesla Model Y?"
   - "比亚迪海豹和特斯拉Model 3哪个加速更快？"
   - "Which EV has better range: BYD Seal or NIO ET5?"
   - "北京地区有多少特斯拉超级充电站？"
   - "Compare safety features of Audi e-tron GT and Porsche Taycan"
""")

# 系统架构
with st.expander("系统架构"):
    st.markdown("""
    ### 主要组件

    1. **数据导入管道**
       - 视频转录（YouTube、Bilibili、优酷）使用 Whisper
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
    """)

# 模型信息
with st.expander("模型信息"):
    st.markdown("""
    ### 主要模型

    1. **嵌入模型**: BAAI/bge-m3
       - 多语言嵌入模型，同时支持中文和英文
       - 768维向量表示
       - 支持跨语言语义检索

    2. **重排序模型**:
       - **ColBERT**: 令牌级别的精确匹配
       - **BAAI/bge-reranker-large**: 增强双语重排序能力
       - 使用加权混合方法: ColBERT (80%) + BGE-Reranker (20%)

    3. **大语言模型**: DeepSeek-R1-Distill-Qwen-7B
       - 本地部署的高效推理
       - 支持中英双语回答
       - 4位量化减少内存占用

    4. **语音转文字**: Whisper-medium
       - 支持多语言视频转录
       - GPU加速处理
    """)

# 本地部署信息
with st.expander("本地部署信息"):
    st.markdown("""
    ### 本地部署

    系统设计为完全本地部署，所有组件都可以离线运行:

    1. **所有模型本地存储**:
       - 嵌入模型、重排序模型、LLM和Whisper模型都存储在本地
       - 使用download_models.sh脚本进行下载

    2. **向量数据库**:
       - Qdrant在本地Docker容器中运行
       - 所有索引和数据都存储在本地卷中

    3. **GPU加速**:
       - 利用本地GPU进行加速
       - 提供CPU备用选项

    4. **数据隐私**:
       - 所有数据都在本地处理，不发送到云端
       - 适合处理敏感或专有数据
    """)