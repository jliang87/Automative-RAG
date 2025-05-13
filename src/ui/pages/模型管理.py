"""
Model management UI for changing/updating models without editing configuration files
"""

import streamlit as st
import pandas as pd
import time
import os
from typing import Dict, List, Any, Optional
import json
import httpx


def render_model_management():
    """Render a UI for managing models."""
    st.title("模型管理")
    st.markdown("管理和配置系统使用的AI模型")

    # Add a note about model paths
    st.info("""
    注意：修改模型配置需要确保模型文件已下载到正确位置。
    如果您更改模型但文件不存在，系统将无法正常工作。
    """)

    # Get current model configuration
    model_config = get_model_config()

    if not model_config:
        st.error("无法获取模型配置")
        if st.button("重试"):
            st.rerun()
        return

    # Create tabs for different model types
    tab_llm, tab_embedding, tab_reranker, tab_whisper = st.tabs([
        "语言模型", "嵌入模型", "重排序模型", "语音识别模型"
    ])

    with tab_llm:
        render_llm_config(model_config)

    with tab_embedding:
        render_embedding_config(model_config)

    with tab_reranker:
        render_reranker_config(model_config)

    with tab_whisper:
        render_whisper_config(model_config)


def render_llm_config(model_config: Dict[str, Any]):
    """Render LLM configuration section."""
    st.header("语言模型 (LLM) 配置")

    # Extract LLM config
    current_model = model_config.get("default_llm_model")
    model_path = model_config.get("llm_model_full_path")
    use_4bit = model_config.get("llm_use_4bit", True)
    use_8bit = model_config.get("llm_use_8bit", False)
    temperature = model_config.get("llm_temperature", 0.1)
    max_tokens = model_config.get("llm_max_tokens", 512)

    # Show current model information
    st.subheader("当前模型")

    st.markdown(f"**模型名称:** {current_model}")
    st.markdown(f"**模型路径:** {model_path}")

    # Model status
    model_status = get_model_status("llm")

    if model_status.get("loaded", False):
        st.success("✅ 模型已加载")
        if "loading_time" in model_status:
            st.markdown(f"**加载用时:** {model_status['loading_time']:.1f}秒")
    else:
        st.warning("⚠️ 模型未加载")

    # Model configuration options
    st.subheader("模型配置")

    # Model selection
    available_models = [
        "DeepSeek-Coder-V2",
        "DeepSeek-R1-Distill-Qwen-7B",
        "Qwen-14B-Chat",
        "Qwen-7B-Chat",
        "Phi-3-mini-4k",
        current_model  # Include current model
    ]

    # Remove duplicates and sort
    available_models = sorted(list(set(available_models)))

    new_model = st.selectbox("选择模型", available_models, index=available_models.index(current_model))

    # Custom model path
    custom_path = st.text_input("自定义模型路径 (可选)", "",
                                help="如果您有自定义模型路径，请在此输入。否则将使用默认路径。")

    # Inference settings
    st.subheader("推理设置")

    col1, col2 = st.columns(2)

    with col1:
        new_temperature = st.slider("温度", 0.0, 1.0, temperature, 0.05,
                                    help="较低的值使输出更确定，较高的值增加随机性。")

    with col2:
        new_max_tokens = st.slider("最大生成长度", 64, 2048, max_tokens, 64,
                                   help="模型最多生成的token数量。")

    # Quantization options
    st.subheader("量化选项")

    new_use_4bit = st.checkbox("使用4位量化", use_4bit,
                               help="使用4位量化可以减少显存使用，但可能影响模型质量。")

    new_use_8bit = st.checkbox("使用8位量化", use_8bit,
                               help="使用8位量化可以适度减少显存使用，影响较小。")

    # Warning if both are selected
    if new_use_4bit and new_use_8bit:
        st.warning("不能同时使用4位和8位量化，将优先使用4位量化。")

    # Apply changes button
    if st.button("应用更改", key="apply_llm_changes"):
        # Create updated config
        updates = {
            "default_llm_model": new_model,
            "llm_temperature": new_temperature,
            "llm_max_tokens": new_max_tokens,
            "llm_use_4bit": new_use_4bit,
            "llm_use_8bit": new_use_8bit
        }

        # Add custom path if provided
        if custom_path:
            updates["llm_model_path"] = custom_path

        # Apply updates
        result = update_model_config(updates)

        if result:
            st.success("更新成功！")

            # Ask if user wants to reload model
            if st.button("重新加载模型", key="reload_llm"):
                reload_result = reload_model("llm")

                if reload_result:
                    st.success("已发送重新加载指令！")
                else:
                    st.error("重新加载指令发送失败")
        else:
            st.error("更新失败")

    # Show advanced information
    with st.expander("高级信息", expanded=False):
        st.markdown("### 模型信息")

        # Get detailed model info if loaded
        model_info = get_model_info()

        if model_info and "llm" in model_info:
            llm_info = model_info["llm"]

            # Create a formatted display of model info
            info_data = []

            for key, value in llm_info.items():
                info_data.append({"属性": key, "值": value})

            if info_data:
                st.dataframe(pd.DataFrame(info_data), hide_index=True)
            else:
                st.info("无详细信息")
        else:
            st.info("无法获取模型详细信息")


def render_embedding_config(model_config: Dict[str, Any]):
    """Render embedding model configuration section."""
    st.header("嵌入模型配置")

    # Extract embedding config
    current_model = model_config.get("default_embedding_model")
    model_path = model_config.get("embedding_model_full_path")
    batch_size = model_config.get("batch_size", 16)

    # Show current model information
    st.subheader("当前模型")

    st.markdown(f"**模型名称:** {current_model}")
    st.markdown(f"**模型路径:** {model_path}")

    # Model status
    model_status = get_model_status("embedding")

    if model_status.get("loaded", False):
        st.success("✅ 模型已加载")
        if "loading_time" in model_status:
            st.markdown(f"**加载用时:** {model_status['loading_time']:.1f}秒")
    else:
        st.warning("⚠️ 模型未加载")

    # Model configuration options
    st.subheader("模型配置")

    # Model selection
    available_models = [
        "bge-m3",
        "bge-large-zh-v1.5",
        "bge-base-zh-v1.5",
        "bge-small-zh-v1.5",
        "bge-large-en-v1.5",
        "text-embedding-3-small",
        current_model  # Include current model
    ]

    # Remove duplicates and sort
    available_models = sorted(list(set(available_models)))

    new_model = st.selectbox("选择模型", available_models, index=available_models.index(current_model))

    # Custom model path
    custom_path = st.text_input("自定义模型路径 (可选)", "",
                                help="如果您有自定义模型路径，请在此输入。否则将使用默认路径。")

    # Batch size setting
    new_batch_size = st.slider("批处理大小", 1, 64, batch_size, 1,
                               help="嵌入计算的批处理大小。较大的值可能加快处理但需要更多显存。")

    # Apply changes button
    if st.button("应用更改", key="apply_embedding_changes"):
        # Create updated config
        updates = {
            "default_embedding_model": new_model,
            "batch_size": new_batch_size
        }

        # Add custom path if provided
        if custom_path:
            updates["embedding_model_path"] = custom_path

        # Apply updates
        result = update_model_config(updates)

        if result:
            st.success("更新成功！")

            # Ask if user wants to reload model
            if st.button("重新加载模型", key="reload_embedding"):
                reload_result = reload_model("embedding")

                if reload_result:
                    st.success("已发送重新加载指令！")
                else:
                    st.error("重新加载指令发送失败")
        else:
            st.error("更新失败")

    # Show advanced information
    with st.expander("高级信息", expanded=False):
        st.markdown("### 模型信息")

        # Get detailed model info if loaded
        model_info = get_model_info()

        if model_info and "embedding" in model_info:
            embedding_info = model_info["embedding"]

            # Create a formatted display of model info
            info_data = []

            for key, value in embedding_info.items():
                info_data.append({"属性": key, "值": value})

            if info_data:
                st.dataframe(pd.DataFrame(info_data), hide_index=True)
            else:
                st.info("无详细信息")
        else:
            st.info("无法获取模型详细信息")


def render_reranker_config(model_config: Dict[str, Any]):
    """Render reranker model configuration section."""
    st.header("重排序模型配置")

    # Extract config values
    colbert_model = model_config.get("default_colbert_model")
    bge_model = model_config.get("default_bge_reranker_model")
    colbert_batch_size = model_config.get("colbert_batch_size", 16)
    use_bge = model_config.get("use_bge_reranker", True)
    colbert_weight = model_config.get("colbert_weight", 0.8)
    bge_weight = model_config.get("bge_weight", 0.2)

    # Show current configuration
    st.subheader("当前配置")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"**ColBERT模型:** {colbert_model}")
        st.markdown(f"**BGE重排序模型:** {bge_model}")

    with col2:
        st.markdown(f"**ColBERT权重:** {colbert_weight}")
        st.markdown(f"**BGE权重:** {bge_weight}")

    # Model status
    model_status = get_model_status("colbert")

    if model_status.get("loaded", False):
        st.success("✅ 重排序模型已加载")
        if "loading_time" in model_status:
            st.markdown(f"**加载用时:** {model_status['loading_time']:.1f}秒")
    else:
        st.warning("⚠️ 重排序模型未加载")

    # Configuration options
    st.subheader("重排序配置")

    # ColBERT model selection
    colbert_options = ["colbertv2.0", colbert_model]
    colbert_options = sorted(list(set(colbert_options)))
    new_colbert = st.selectbox("ColBERT模型", colbert_options,
                               index=colbert_options.index(colbert_model))

    # BGE model selection
    bge_options = [
        "bge-reranker-large",
        "bge-reranker-base",
        "BAAI/bge-reranker-large",
        "BAAI/bge-reranker-base",
        bge_model
    ]
    bge_options = sorted(list(set(bge_options)))
    new_bge = st.selectbox("BGE重排序模型", bge_options,
                           index=bge_options.index(bge_model))

    # Hybrid reranking settings
    st.subheader("混合重排序设置")

    new_use_bge = st.checkbox("启用BGE重排序", use_bge,
                              help="启用BGE重排序可以提高对中文的支持。")

    col1, col2 = st.columns(2)

    with col1:
        new_colbert_weight = st.slider("ColBERT权重", 0.0, 1.0, colbert_weight, 0.05,
                                       help="ColBERT得分的权重。")

    with col2:
        new_bge_weight = st.slider("BGE权重", 0.0, 1.0, bge_weight, 0.05,
                                   help="BGE重排序得分的权重。")

    # Total weight should be 1.0
    total_weight = new_colbert_weight + new_bge_weight
    if total_weight != 1.0:
        st.warning(f"权重总和应为1.0，当前为{total_weight:.2f}。将自动调整。")

    # Batch size setting
    new_batch_size = st.slider("批处理大小", 1, 64, colbert_batch_size, 1,
                               help="重排序的批处理大小。较大的值可能加快处理但需要更多显存。")

    # Apply changes button
    if st.button("应用更改", key="apply_reranker_changes"):
        # Normalize weights if needed
        if total_weight != 1.0:
            new_colbert_weight = new_colbert_weight / total_weight
            new_bge_weight = new_bge_weight / total_weight

        # Create updated config
        updates = {
            "default_colbert_model": new_colbert,
            "default_bge_reranker_model": new_bge,
            "colbert_batch_size": new_batch_size,
            "use_bge_reranker": new_use_bge,
            "colbert_weight": new_colbert_weight,
            "bge_weight": new_bge_weight
        }

        # Apply updates
        result = update_model_config(updates)

        if result:
            st.success("更新成功！")

            # Ask if user wants to reload model
            if st.button("重新加载模型", key="reload_reranker"):
                reload_result = reload_model("colbert")

                if reload_result:
                    st.success("已发送重新加载指令！")
                else:
                    st.error("重新加载指令发送失败")
        else:
            st.error("更新失败")


def render_whisper_config(model_config: Dict[str, Any]):
    """Render Whisper model configuration section."""
    st.header("语音识别模型配置")

    # Extract Whisper config
    current_model = model_config.get("default_whisper_model")
    model_path = model_config.get("whisper_model_full_path", "")
    use_youtube_captions = model_config.get("use_youtube_captions", False)
    use_whisper_as_fallback = model_config.get("use_whisper_as_fallback", False)
    force_whisper = model_config.get("force_whisper", True)

    # Show current model information
    st.subheader("当前模型")

    st.markdown(f"**模型尺寸:** {current_model}")
    if model_path:
        st.markdown(f"**模型路径:** {model_path}")

    # Model status
    model_status = get_model_status("whisper")

    if model_status.get("loaded", False):
        st.success("✅ 模型已加载")
        if "loading_time" in model_status:
            st.markdown(f"**加载用时:** {model_status['loading_time']:.1f}秒")
    else:
        st.warning("⚠️ 模型未加载")

    # Model configuration options
    st.subheader("模型配置")

    # Model size selection
    model_sizes = ["tiny", "base", "small", "medium", "large", current_model]
    model_sizes = sorted(list(set(model_sizes)))
    new_model = st.selectbox("选择模型尺寸", model_sizes,
                             index=model_sizes.index(current_model))

    # Custom model path
    custom_path = st.text_input("自定义模型路径 (可选)", model_path,
                                help="如果您有自定义模型路径，请在此输入。否则将使用默认模型。")

    # Transcription settings
    st.subheader("转录设置")

    new_use_youtube = st.checkbox("使用YouTube字幕（如可用）", use_youtube_captions,
                                  help="如果可用，使用YouTube提供的字幕而不是转录。")

    new_use_fallback = st.checkbox("将Whisper作为备选", use_whisper_as_fallback,
                                   help="仅在无法获取平台字幕时使用Whisper。")

    new_force_whisper = st.checkbox("始终使用Whisper", force_whisper,
                                    help="始终使用Whisper进行转录，忽略平台字幕。")

    # Warning for conflicting settings
    if new_use_youtube and new_force_whisper:
        st.warning("设置冲突：无法同时使用YouTube字幕并强制使用Whisper。")

    # Apply changes button
    if st.button("应用更改", key="apply_whisper_changes"):
        # Create updated config
        updates = {
            "default_whisper_model": new_model,
            "use_youtube_captions": new_use_youtube,
            "use_whisper_as_fallback": new_use_fallback,
            "force_whisper": new_force_whisper
        }

        # Add custom path if provided
        if custom_path:
            updates["whisper_model_path"] = custom_path

        # Apply updates
        result = update_model_config(updates)

        if result:
            st.success("更新成功！")

            # Ask if user wants to reload model
            if st.button("重新加载模型", key="reload_whisper"):
                reload_result = reload_model("whisper")

                if reload_result:
                    st.success("已发送重新加载指令！")
                else:
                    st.error("重新加载指令发送失败")
        else:
            st.error("更新失败")

    # Show advanced information
    with st.expander("高级信息", expanded=False):
        st.markdown("### 模型信息")

        # Memory requirements for different model sizes
        memory_reqs = {
            "tiny": "~1 GB VRAM",
            "base": "~1 GB VRAM",
            "small": "~2 GB VRAM",
            "medium": "~5 GB VRAM",
            "large": "~10 GB VRAM"
        }

        st.markdown("#### 显存需求")
        memory_data = []

        for size, req in memory_reqs.items():
            memory_data.append({"模型尺寸": size, "显存需求": req})

        st.dataframe(pd.DataFrame(memory_data), hide_index=True)

        st.markdown("#### 特点比较")
        st.markdown("""
        - **tiny/base**: 较快但准确度较低
        - **small**: 速度和准确度的良好平衡
        - **medium**: 大多数情况下的推荐选择
        - **large**: 最高准确度但需要更多资源
        """)


def get_model_config() -> Optional[Dict[str, Any]]:
    """Get current model configuration from the API."""
    try:
        response = httpx.get(
            f"{st.session_state.api_url}/system/config",
            headers={"x-token": st.session_state.api_key},
            timeout=5.0
        )

        if response.status_code == 200:
            return response.json()
        return None
    except Exception as e:
        st.error(f"获取模型配置失败: {str(e)}")
        return None


def update_model_config(updates: Dict[str, Any]) -> bool:
    """Update model configuration."""
    try:
        response = httpx.post(
            f"{st.session_state.api_url}/system/update-config",
            headers={"x-token": st.session_state.api_key},
            json=updates,
            timeout=5.0
        )

        return response.status_code == 200
    except Exception as e:
        st.error(f"更新模型配置失败: {str(e)}")
        return False


def reload_model(model_type: str) -> bool:
    """Send command to reload a specific model."""
    try:
        response = httpx.post(
            f"{st.session_state.api_url}/system/reload-model",
            headers={"x-token": st.session_state.api_key},
            json={"model_type": model_type},
            timeout=5.0
        )

        return response.status_code == 200
    except Exception as e:
        st.error(f"发送重新加载命令失败: {str(e)}")
        return False


def get_model_status(model_type: str) -> Dict[str, Any]:
    """Get status of a specific model."""
    try:
        response = httpx.get(
            f"{st.session_state.api_url}/system/health/detailed",
            headers={"x-token": st.session_state.api_key},
            timeout=3.0
        )

        if response.status_code == 200:
            data = response.json()
            model_status = data.get("model_status", {}).get(model_type, {})
            return model_status
        return {}
    except Exception:
        return {}


def get_model_info() -> Optional[Dict[str, Any]]:
    """Get detailed information about loaded models."""
    try:
        response = httpx.get(
            f"{st.session_state.api_url}/system/model-info",
            headers={"x-token": st.session_state.api_key},
            timeout=3.0
        )

        if response.status_code == 200:
            return response.json()
        return None
    except Exception:
        return None


# Add this page to src/ui/pages/模型管理.py
render_model_management()