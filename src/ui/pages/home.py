"""
Home page for the Streamlit UI.
"""

import streamlit as st
from src.ui.components import header, api_request


def render_home_page():
    """Render the home page with system information."""
    header(
        "Automotive Specifications RAG System",
        "A GPU-accelerated system for retrieving automotive specifications using late interaction retrieval."
    )
    
    # System overview
    st.markdown("""
    ## About the System
    
    This system uses state-of-the-art retrieval augmented generation (RAG) to provide accurate answers 
    about automotive specifications. It features:
    
    - **Late Interaction Retrieval** with ColBERT for precise semantic matching
    - **Local DeepSeek LLM** for generating responses without API dependencies
    - **Multi-Modal Input Processing** including YouTube videos, Bilibili videos, and PDFs
    - **Full GPU Acceleration** for all