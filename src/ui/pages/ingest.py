"""
Ingest page for the Streamlit UI.
"""

import json
import streamlit as st
from src.ui.components import header, api_request, loading_spinner


def render_ingest_page():
    """Render the ingest page."""
    header(
        "Ingest Automotive Data",
        "Add new automotive specifications data from various sources."
    )
    
    # Tabs for different ingestion methods
    tab1, tab2, tab3, tab4 = st.tabs(["YouTube", "Bilibili", "PDF", "Manual Entry"])
    
    # YouTube ingestion
    with tab1:
        render_youtube_tab()
    
    # Bilibili ingestion
    with tab2:
        render_bilibili_tab()
    
    # PDF ingestion
    with tab3:
        render_pdf_tab()
    
    # Manual entry
    with tab4:
        render_manual_tab()


def render_youtube_tab():
    """Render the YouTube ingestion tab."""
    st.header("Ingest YouTube Video")
    
    youtube_url = st.text_input("YouTube URL", placeholder="https://www.youtube.com/watch?v=...")
    
    force_whisper = st.checkbox(
        "Force Whisper transcription", 
        help="Enable to use Whisper even if YouTube captions are available. This may provide higher quality transcription but takes longer."
    )
    
    with st.expander("Additional Metadata"):
        yt_manufacturer = st.text_input("Manufacturer", key="yt_manufacturer")
        yt_model = st.text_input("Model", key="yt_model")
        yt_year = st.number_input("Year", min_value=1900, max_value=2100, value=2023, key="yt_year")
        
        yt_categories = ["", "sedan", "suv", "truck", "sports", "minivan", "coupe", "convertible", "hatchback", "wagon"]
        yt_category = st.selectbox("Category", yt_categories, key="yt_category")
        
        yt_engine_types = ["", "gasoline", "diesel", "electric", "hybrid", "hydrogen"]
        yt_engine_type = st.selectbox("Engine Type", yt_engine_types, key="yt_engine_type")
        
        yt_transmission = ["", "automatic", "manual", "cvt", "dct"]
        yt_transmission_type = st.selectbox("Transmission", yt_transmission, key="yt_transmission")
    
    if st.button("Ingest YouTube Video", type="primary", key="youtube_btn"):
        if not youtube_url:
            st.warning("Please enter a YouTube URL")
            return
            
        # Prepare metadata
        custom_metadata = {}
        if yt_manufacturer:
            custom_metadata["manufacturer"] = yt_manufacturer
        if yt_model:
            custom_metadata["model"] = yt_model
        if yt_year:
            custom_metadata["year"] = yt_year
        if yt_category and yt_category != "":
            custom_metadata["category"] = yt_category
        if yt_engine_type and yt_engine_type != "":
            custom_metadata["engine_type"] = yt_engine_type
        if yt_transmission_type and yt_transmission_type != "":
            custom_metadata["transmission"] = yt_transmission_type
            
        # Make API request
        with loading_spinner("Processing YouTube video... This may take several minutes."):
            # Add query parameter for force_whisper if selected
            endpoint = f"/ingest/youtube{'?force_whisper=true' if force_whisper else ''}"
            
            result = api_request(
                endpoint=endpoint,
                method="POST",
                data={
                    "url": youtube_url,
                    "metadata": custom_metadata if custom_metadata else None,
                },
            )
            
        if result:
            st.success(f"Successfully ingested YouTube video: {result.get('document_count', 0)} chunks created")


def render_bilibili_tab():
    """Render the Bilibili ingestion tab."""
    st.header("Ingest Bilibili Video")
    
    bilibili_url = st.text_input("Bilibili URL", placeholder="https://www.bilibili.com/video/...")
    
    with st.expander("Additional Metadata"):
        bl_manufacturer = st.text_input("Manufacturer", key="bl_manufacturer")
        bl_model = st.text_input("Model", key="bl_model")
        bl_year = st.number_input("Year", min_value=1900, max_value=2100, value=2023, key="bl_year")
        
        bl_categories = ["", "sedan", "suv", "truck", "sports", "minivan", "coupe", "convertible", "hatchback", "wagon"]
        bl_category = st.selectbox("Category", bl_categories, key="bl_category")
        
        bl_engine_types = ["", "gasoline", "diesel", "electric", "hybrid", "hydrogen"]
        bl_engine_type = st.selectbox("Engine Type", bl_engine_types, key="bl_engine_type")
        
        bl_transmission = ["", "automatic", "manual", "cvt", "dct"]
        bl_transmission_type = st.selectbox("Transmission", bl_transmission, key="bl_transmission")
    
    if st.button("Ingest Bilibili Video", type="primary", key="bilibili_btn"):
        if not bilibili_url:
            st.warning("Please enter a Bilibili URL")
            return
            
        # Prepare metadata
        custom_metadata = {}
        if bl_manufacturer:
            custom_metadata["manufacturer"] = bl_manufacturer
        if bl_model:
            custom_metadata["model"] = bl_model
        if bl_year:
            custom_metadata["year"] = bl_year
        if bl_category and bl_category != "":
            custom_metadata["category"] = bl_category
        if bl_engine_type and bl_engine_type != "":
            custom_metadata["engine_type"] = bl_engine_type
        if bl_transmission_type and bl_transmission_type != "":
            custom_metadata["transmission"] = bl_transmission_type
            
        # Make API request
        with loading_spinner("Processing Bilibili video... This may take several minutes."):
            result = api_request(
                endpoint="/ingest/bilibili",
                method="POST",
                data={
                    "url": bilibili_url,
                    "metadata": custom_metadata if custom_metadata else None,
                },
            )
            
        if result:
            st.success(f"Successfully ingested Bilibili video: {result.get('document_count', 0)} chunks created")


def render_pdf_tab():
    """Render the PDF ingestion tab."""
    st.header("Ingest PDF")
    
    pdf_file = st.file_uploader("Upload PDF", type=["pdf"])
    
    use_ocr = st.checkbox(
        "Use OCR", value=True,
        help="Enable OCR for scanned documents. This uses GPU-accelerated recognition."
    )
    
    extract_tables = st.checkbox(
        "Extract tables", value=True,
        help="Enable table extraction from the PDF."
    )
    
    with st.expander("Additional Metadata"):
        pdf_title = st.text_input("Title", key="pdf_title")
        pdf_manufacturer = st.text_input("Manufacturer", key="pdf_manufacturer")
        pdf_model = st.text_input("Model", key="pdf_model")
        pdf_year = st.number_input("Year", min_value=1900, max_value=2100, value=2023, key="pdf_year")
        
        pdf_categories = ["", "sedan", "suv", "truck", "sports", "minivan", "coupe", "convertible", "hatchback", "wagon"]
        pdf_category = st.selectbox("Category", pdf_categories, key="pdf_category")
        
        pdf_engine_types = ["", "gasoline", "diesel", "electric", "hybrid", "hydrogen"]
        pdf_engine_type = st.selectbox("Engine Type", pdf_engine_types, key="pdf_engine_type")
        
        pdf_transmission = ["", "automatic", "manual", "cvt", "dct"]
        pdf_transmission_type = st.selectbox("Transmission", pdf_transmission, key="pdf_transmission")
    
    if st.button("Ingest PDF", type="primary", key="pdf_btn"):
        if not pdf_file:
            st.warning("Please upload a PDF file")
            return
            
        # Prepare metadata
        custom_metadata = {}
        if pdf_title:
            custom_metadata["title"] = pdf_title
        if pdf_manufacturer:
            custom_metadata["manufacturer"] = pdf_manufacturer
        if pdf_model:
            custom_metadata["model"] = pdf_model
        if pdf_year:
            custom_metadata["year"] = pdf_year
        if pdf_category and pdf_category != "":
            custom_metadata["category"] = pdf_category
        if pdf_engine_type and pdf_engine_type != "":
            custom_metadata["engine_type"] = pdf_engine_type
        if pdf_transmission_type and pdf_transmission_type != "":
            custom_metadata["transmission"] = pdf_transmission_type
            
        # Make API request
        with loading_spinner("Processing PDF... This may take some time."):
            # Prepare form data
            files = {"file": (pdf_file.name, pdf_file, "application/pdf")}
            data = {
                "metadata": json.dumps(custom_metadata) if custom_metadata else None,
                "use_ocr": json.dumps(use_ocr),
                "extract_tables": json.dumps(extract_tables)
            }
            
            result = api_request(
                endpoint="/ingest/pdf",
                method="POST",
                data=data,
                files=files,
            )
            
        if result:
            st.success(f"Successfully ingested PDF: {result.get('document_count', 0)} chunks created")


def render_manual_tab():
    """Render the manual entry tab."""
    st.header("Manual Entry")
    
    manual_content = st.text_area("Enter Content", height=300)
    
    with st.expander("Metadata"):
        man_title = st.text_input("Title", key="man_title")
        man_manufacturer = st.text_input("Manufacturer", key="man_manufacturer")
        man_model = st.text_input("Model", key="man_model")
        man_year = st.number_input("Year", min_value=1900, max_value=2100, value=2023, key="man_year")
        
        man_categories = ["", "sedan", "suv", "truck", "sports", "minivan", "coupe", "convertible", "hatchback", "wagon"]
        man_category = st.selectbox("Category", man_categories, key="man_category")
        
        man_engine_types = ["", "gasoline", "diesel", "electric", "hybrid", "hydrogen"]
        man_engine_type = st.selectbox("Engine Type", man_engine_types, key="man_engine_type")
        
        man_transmission = ["", "automatic", "manual", "cvt", "dct"]
        man_transmission_type = st.selectbox("Transmission", man_transmission, key="man_transmission")
    
    if st.button("Ingest Manual Content", type="primary", key="manual_btn"):
        if not manual_content:
            st.warning("Please enter content")
            return
            
        # Build metadata
        metadata = {
            "source": "manual",
            "source_id": "manual-entry",
            "title": man_title if man_title else "Manual Entry",
        }
        
        if man_manufacturer:
            metadata["manufacturer"] = man_manufacturer
        if man_model:
            metadata["model"] = man_model
        if man_year:
            metadata["year"] = man_year
        if man_category and man_category != "":
            metadata["category"] = man_category
        if man_engine_type and man_engine_type != "":
            metadata["engine_type"] = man_engine_type
        if man_transmission_type and man_transmission_type != "":
            metadata["transmission"] = man_transmission_type
            
        # Make API request
        with loading_spinner("Processing manual content..."):
            result = api_request(
                endpoint="/ingest/text",
                method="POST",
                data={
                    "content": manual_content,
                    "metadata": metadata,
                },
            )
            
        if result:
            st.success(f"Successfully ingested manual content: {result.get('document_count', 0)} chunks created")
