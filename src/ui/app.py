import os
import streamlit as st
import httpx
import json
from typing import Dict, List, Optional, Union

# Configure the app
st.set_page_config(
    page_title="Automotive Specs RAG",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# API configuration
API_URL = os.environ.get("API_URL", "http://localhost:8000")
API_KEY = os.environ.get("API_KEY", "default-api-key")

# Session state initialization
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
    """Make a request to the API."""
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
                st.error(f"Unsupported method: {method}")
                return None