"""
Model loading status indicator component
"""

import streamlit as st
import time
from typing import Dict, Any, List, Optional
from src.ui.api_client import api_request, check_worker_availability

def model_loading_status(detailed: bool = False):
    """
    Display the loading status of models with progress indicators.

    Args:
        detailed: Whether to show detailed loading information
    """
    # Get model status information
    model_info = api_request(
        endpoint="/system/health/detailed",
        method="GET",
        silent=True
    )

    if not model_info or "model_status" not in model_info:
        if detailed:
            st.warning("无法获取模型加载状态")
        return

    model_status = model_info.get("model_status", {})

    # Map model types to friendly names
    model_names = {
        "embedding": "嵌入模型 (BGE-M3)",
        "llm": "语言模型 (DeepSeek)",
        "colbert": "重排序模型 (ColBERT)",
        "whisper": "语音识别模型 (Whisper)"
    }

    # If any models are currently loading, show status
    loading_models = [model for model, status in model_status.items()
                      if not status.get("loaded", False)]

    if loading_models:
        st.warning("⚠️ 模型正在加载中，某些功能可能暂时不可用")

        if detailed:
            st.write("正在加载以下模型:")

            for model_type in loading_models:
                model_name = model_names.get(model_type, model_type)
                st.text(f"- {model_name}")

            # Show a progress spinner
            with st.spinner("请耐心等待..."):
                # Check status every few seconds until all models are loaded
                for _ in range(5):  # Check up to 5 times with small delays
                    time.sleep(1)

                    # Re-fetch model status
                    updated_info = api_request(
                        endpoint="/system/health/detailed",
                        method="GET",
                        silent=True
                    )

                    if updated_info and "model_status" in updated_info:
                        updated_status = updated_info.get("model_status", {})
                        still_loading = [model for model, status in updated_status.items()
                                         if not status.get("loaded", False)]

                        if not still_loading:
                            st.success("✅ 所有模型加载完成！")
                            return
    elif detailed:
        # Show loaded models
        st.success("✅ 所有需要的模型已加载完成")

        if detailed:
            with st.expander("已加载模型详情", expanded=False):
                for model_type, status in model_status.items():
                    if status.get("loaded", False):
                        model_name = model_names.get(model_type, model_type)
                        loading_time = status.get("loading_time", 0)
                        st.text(f"✓ {model_name} (加载用时: {loading_time:.1f}秒)")
