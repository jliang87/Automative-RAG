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
    - **Multi-Modal Input Processing** including YouTube, Bilibili, and Youku videos, and PDFs
    - **Full GPU Acceleration** for all components including transcription, OCR, embedding, and inference
    - **Hybrid Search** combining vector similarity with metadata filtering
    """)

    # Get system status information
    try:
        status_info = api_request(
            endpoint="/ingest/status",
            method="GET"
        )

        if status_info:
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("System Status")
                st.write(f"Status: {status_info.get('status', 'Unknown')}")
                st.write(f"Document Count: {status_info.get('document_count', 0):,}")

                # Display collection information if available
                if "collection" in status_info:
                    st.write(f"Collection: {status_info['collection']}")

            with col2:
                st.subheader("GPU Information")
                gpu_info = status_info.get("gpu_info", {})

                if gpu_info:
                    st.write(f"Device: {gpu_info.get('device', 'Unknown')}")
                    if 'device_name' in gpu_info:
                        st.write(f"GPU: {gpu_info['device_name']}")
                    if 'memory_allocated' in gpu_info:
                        st.write(f"Memory Used: {gpu_info['memory_allocated']}")
                    if 'whisper_model' in gpu_info:
                        st.write(f"Whisper Model: {gpu_info['whisper_model']}")
                else:
                    st.write("Running on CPU")

        # Get LLM information
        llm_info = api_request(
            endpoint="/query/llm-info",
            method="GET"
        )

        if llm_info:
            st.subheader("LLM Information")

            # Format model name nicely
            model_name = llm_info.get("model_name", "Unknown")
            if "/" in model_name:
                model_name = model_name.split("/")[-1]

            col1, col2 = st.columns(2)

            with col1:
                st.write(f"Model: {model_name}")
                st.write(f"Quantization: {llm_info.get('quantization', 'None')}")

            with col2:
                st.write(f"Temperature: {llm_info.get('temperature', 0.0)}")
                st.write(f"Max Tokens: {llm_info.get('max_tokens', 0)}")

            if 'vram_usage' in llm_info:
                st.write(f"VRAM Usage: {llm_info['vram_usage']}")

    except Exception as e:
        st.error(f"Error fetching system information: {str(e)}")

    # Usage instructions
    st.markdown("""
    ## Getting Started
    
    1. **Ingest Data**: Go to the Ingest tab to add automotive data from:
       - YouTube vehicle reviews and walkthroughs
       - Bilibili automotive content
       - Youku automotive videos
       - PDF manuals and spec sheets
       - Manual text entry for specifications
       
    2. **Query Data**: Use the Query tab to ask questions about vehicles:
       - Filter by manufacturer, model, year, category, etc.
       - Get precise answers with source attribution
       - View relevant snippets from the knowledge base
       
    3. **Examples**:
       - "What is the horsepower of the 2023 Toyota Camry?"
       - "Compare the fuel efficiency of the Honda Civic and Toyota Corolla"
       - "What are the dimensions of the Ford F-150?"
       - "What safety features come standard on the Tesla Model 3?"
    """)

    # System architecture
    with st.expander("System Architecture"):
        st.markdown("""
        ### Components
        
        1. **Data Ingestion Pipeline**
           - Video transcription (YouTube, Bilibili, Youku) with Whisper
           - PDF parsing with OCR and table extraction
           - Metadata extraction and document chunking
        
        2. **Vector Database**
           - Qdrant for efficient vector storage
           - Hybrid search capabilities (vector + metadata)
           - Document storage and management
        
        3. **Retrieval System**
           - Initial retrieval via Qdrant
           - ColBERT reranking for precise semantic matching
           - Late interaction patterns for token-level analysis
        
        4. **Generation Layer**
           - DeepSeek LLM integration
           - Context assembly with metadata
           - Response generation with source attribution
        
        5. **API Layer**
           - FastAPI backend with dependency injection
           - Swagger documentation
           - Authentication and security
        
        6. **User Interface**
           - Streamlit dashboard
           - Query and filtering interface
           - Results visualization
           - Source attribution and explanation
        """)