# src/ui/session_init.py
import streamlit as st
import os


def initialize_session_state():
    """Initialize common session state variables."""
    API_URL = os.environ.get("API_URL", "http://localhost:8000")
    API_KEY = os.environ.get("API_KEY", "default-api-key")

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