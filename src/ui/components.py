"""
UI Components for the Streamlit interface.

This module contains reusable UI components used across different pages
of the Streamlit application.
"""

import streamlit as st
import httpx
from typing import Dict, List, Optional, Union, Callable, Any


def header(title: str, description: Optional[str] = None):
    """Display a header with optional description."""
    st.title(title)
    if description:
        st.markdown(description)
    st.divider()


def api_status_indicator():
    """Display API connection status indicator."""
    try:
        with httpx.Client() as client:
            response = client.get(
                f"{st.session_state.api_url}/health",
                headers={"x-token": st.session_state.api_key},
                timeout=3.0
            )
            
        if response.status_code == 200:
            st.sidebar.success("✅ API Connected")
            return True
        else:
            st.sidebar.error(f"❌ API Error: {response.status_code}")
            return False
    except Exception as e:
        st.sidebar.error(f"❌ Connection Error: {str(e)}")
        return False


def gpu_status_indicator():
    """Display GPU status information if available."""
    try:
        with httpx.Client() as client:
            response = client.get(
                f"{st.session_state.api_url}/ingest/status",
                headers={"x-token": st.session_state.api_key},
                timeout=3.0
            )
            
        if response.status_code == 200:
            data = response.json()
            
            if "gpu_info" in data and data["gpu_info"]:
                gpu_info = data["gpu_info"]
                
                # Create expandable section for GPU info
                with st.sidebar.expander("GPU Status"):
                    if "device_name" in gpu_info:
                        st.info(f"GPU: {gpu_info['device_name']}")
                    
                    if "memory_allocated" in gpu_info:
                        st.metric("VRAM Used", gpu_info['memory_allocated'])
                    
                    if "whisper_model" in gpu_info:
                        st.text(f"Whisper: {gpu_info['whisper_model']}")
                        
                    if "fp16_enabled" in gpu_info:
                        fp16 = gpu_info['fp16_enabled']
                        st.text(f"Mixed Precision: {'✓' if fp16 else '✗'}")
            else:
                st.sidebar.warning("⚠️ Running on CPU")
                
    except Exception as e:
        st.sidebar.warning("⚠️ GPU status unavailable")


def llm_status_indicator():
    """Display LLM information if available."""
    try:
        with httpx.Client() as client:
            response = client.get(
                f"{st.session_state.api_url}/query/llm-info",
                headers={"x-token": st.session_state.api_key},
                timeout=3.0
            )
            
        if response.status_code == 200:
            llm_info = response.json()
            
            with st.sidebar.expander("LLM Status"):
                # Model name - show shortened version if too long
                model_name = llm_info.get("model_name", "Unknown")
                if len(model_name) > 30:
                    display_name = f"{model_name.split('/')[-1]}"
                else:
                    display_name = model_name
                st.info(f"Model: {display_name}")
                
                # Quantization info
                quant = llm_info.get("quantization", "none")
                st.text(f"Quantization: {quant}")
                
                # VRAM usage if available
                if "vram_usage" in llm_info:
                    st.metric("VRAM Used", llm_info["vram_usage"])
                    
                # Device info
                device = llm_info.get("device", "Unknown")
                st.text(f"Device: {device}")
    except Exception:
        pass  # Silently ignore errors


def metadata_filters():
    """Render metadata filter inputs."""
    st.subheader("Filters")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # These would ideally be populated from the API
        manufacturers = ["", "Toyota", "Honda", "Ford", "BMW", "Tesla"]
        manufacturer = st.selectbox("Manufacturer", manufacturers)
        
        years = [""] + [str(year) for year in range(2010, 2026)]
        year = st.selectbox("Year", years)
        
        categories = ["", "sedan", "suv", "truck", "sports", "minivan"]
        category = st.selectbox("Category", categories)
    
    with col2:
        models = [""]
        if manufacturer == "Toyota":
            models += ["Camry", "Corolla", "RAV4", "Highlander"]
        elif manufacturer == "Honda":
            models += ["Civic", "Accord", "CR-V", "Pilot"]
        elif manufacturer == "Ford":
            models += ["Mustang", "F-150", "Explorer", "Escape"]
        elif manufacturer == "BMW":
            models += ["3 Series", "5 Series", "X3", "X5"]
        elif manufacturer == "Tesla":
            models += ["Model S", "Model 3", "Model X", "Model Y"]
            
        model = st.selectbox("Model", models)
        
        engine_types = ["", "gasoline", "diesel", "electric", "hybrid"]
        engine_type = st.selectbox("Engine Type", engine_types)
        
        transmission_types = ["", "automatic", "manual", "cvt", "dct"]
        transmission = st.selectbox("Transmission", transmission_types)
        
    # Build metadata filter
    metadata_filter = {}
    
    if manufacturer:
        metadata_filter["manufacturer"] = manufacturer
    if model:
        metadata_filter["model"] = model
    if year:
        metadata_filter["year"] = int(year)
    if category:
        metadata_filter["category"] = category
    if engine_type:
        metadata_filter["engine_type"] = engine_type
    if transmission:
        metadata_filter["transmission"] = transmission
        
    return metadata_filter if metadata_filter else None


def api_request(
    endpoint: str,
    method: str = "GET",
    data: Optional[Dict] = None,
    files: Optional[Dict] = None,
    params: Optional[Dict] = None,
    handle_error: Callable[[str], Any] = None,
) -> Optional[Dict]:
    """Make a request to the API with error handling."""
    headers = {"x-token": st.session_state.api_key}
    url = f"{st.session_state.api_url}{endpoint}"
    
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
                st.error(f"Unsupported method: {method}")
                return None
                
            if response.status_code >= 400:
                error_msg = f"API Error ({response.status_code}): {response.text}"
                if handle_error:
                    return handle_error(error_msg)
                else:
                    st.error(error_msg)
                    return None
                    
            return response.json()
    except Exception as e:
        error_msg = f"Connection error: {str(e)}"
        if handle_error:
            return handle_error(error_msg)
        else:
            st.error(error_msg)
            return None


def display_document(doc: Dict, index: int):
    """Display a document from search results."""
    metadata = doc.get("metadata", {})
    
    # Determine source icon
    source_type = metadata.get("source", "unknown")
    if source_type == "youtube":
        source_icon = "🎬"
    elif source_type == "pdf":
        source_icon = "📄"
    elif source_type == "manual":
        source_icon = "📝"
    else:
        source_icon = "📚"
        
    # Format title
    title = metadata.get("title", f"Document {index+1}")
    
    # Create expandable result card
    with st.expander(f"{source_icon} {title}"):
        # Source information
        st.caption(f"Source: {source_type}")
        st.caption(f"Relevance Score: {doc.get('relevance_score', 0):.4f}")
        
        # Vehicle information if available
        auto_info = []
        if metadata.get("manufacturer"):
            auto_info.append(metadata["manufacturer"])
        if metadata.get("model"):
            auto_info.append(metadata["model"])
        if metadata.get("year"):
            auto_info.append(str(metadata["year"]))
            
        if auto_info:
            st.markdown(f"**Vehicle**: {' '.join(auto_info)}")
            
        # Display attributes like engine type if available
        attributes = []
        if metadata.get("category"):
            attributes.append(f"Category: {metadata['category']}")
        if metadata.get("engine_type"):
            attributes.append(f"Engine: {metadata['engine_type']}")
        if metadata.get("transmission"):
            attributes.append(f"Transmission: {metadata['transmission']}")
            
        if attributes:
            st.markdown(", ".join(attributes))
            
        # Document content
        st.markdown("---")
        st.markdown(doc.get("content", ""))
        
        # URL if available
        if metadata.get("url"):
            st.markdown(f"[Source Link]({metadata['url']})")
            
            
def loading_spinner(text: str = "Processing..."):
    """Creates a context manager for displaying a spinner."""
    return st.spinner(text)
