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

    # Add missing attributes needed for data ingestion page
    if "redirect_to_jobs" not in st.session_state:
        st.session_state.redirect_to_jobs = False
    if "job_id" not in st.session_state:
        st.session_state.job_id = None
    if "job_created" not in st.session_state:
        st.session_state.job_created = False

    # Add attributes needed for query page
    if "query_job_id" not in st.session_state:
        st.session_state.query_job_id = None
    if "query_status" not in st.session_state:
        st.session_state.query_status = None
    if "polling" not in st.session_state:
        st.session_state.polling = False
    if "poll_count" not in st.session_state:
        st.session_state.poll_count = 0

    # Add attribute for task management
    if "selected_job_id" not in st.session_state:
        st.session_state.selected_job_id = None