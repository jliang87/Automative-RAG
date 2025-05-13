import streamlit as st
import httpx
import time
import pandas as pd
from typing import Dict, List, Optional, Any


def enhanced_worker_status():
    """
    Display enhanced worker status with health information in the sidebar.
    """
    try:
        # Get detailed health information from API
        response = api_request(
            endpoint="/system/health/detailed",
            method="GET",
            timeout=3.0  # Short timeout to avoid UI blocking
        )

        if not response:
            st.sidebar.warning("⚠️ Unable to get system health information")
            return

        # Display overall system status
        status = response.get("status", "unknown")
        if status == "healthy":
            st.sidebar.success("✅ System is healthy")
        else:
            st.sidebar.warning("⚠️ System status: " + status)

        # Display active workers
        workers = response.get("workers", {})
        if workers:
            with st.sidebar.expander("Worker Status", expanded=True):
                worker_data = []

                # Group workers by type
                worker_types = {}
                for worker_id, info in workers.items():
                    worker_type = info.get("type", "unknown")
                    if worker_type not in worker_types:
                        worker_types[worker_type] = []
                    worker_types[worker_type].append((worker_id, info))

                # Display workers by type with status indicators
                for worker_type, workers_of_type in worker_types.items():
                    healthy_count = sum(1 for _, info in workers_of_type if info.get("status") == "healthy")
                    total_count = len(workers_of_type)

                    # Create a nice display name
                    display_name = {
                        "gpu-inference": "LLM & Reranking",
                        "gpu-embedding": "Vector Embeddings",
                        "gpu-whisper": "Speech Transcription",
                        "cpu": "Text Processing",
                        "system": "System Management"
                    }.get(worker_type, worker_type)

                    # Show status with color
                    if healthy_count == total_count:
                        st.success(f"✅ {display_name}: {healthy_count}/{total_count} workers healthy")
                    elif healthy_count > 0:
                        st.warning(f"⚠️ {display_name}: {healthy_count}/{total_count} workers healthy")
                    else:
                        st.error(f"❌ {display_name}: 0/{total_count} workers healthy")

                    # Add restart button for offline workers
                    if healthy_count < total_count and worker_type != "system":
                        if st.button(f"Restart {display_name} workers", key=f"restart_{worker_type}"):
                            restart_response = api_request(
                                endpoint=f"/system/restart-worker/{worker_type}",
                                method="POST"
                            )
                            if restart_response:
                                st.success(f"Restart signal sent to {display_name} workers")

                # Display queue info if available
                queue_stats = response.get("queues", {})
                if queue_stats:
                    st.subheader("Task Queues")
                    queue_data = []
                    for queue, count in queue_stats.items():
                        display_name = {
                            "inference_tasks": "LLM Generation",
                            "embedding_tasks": "Embeddings",
                            "transcription_tasks": "Transcription",
                            "cpu_tasks": "Text Processing",
                            "system_tasks": "System Tasks"
                        }.get(queue, queue)

                        queue_data.append({
                            "Queue": display_name,
                            "Pending Tasks": count
                        })

                    # Create a DataFrame and display
                    if queue_data:
                        queue_df = pd.DataFrame(queue_data)
                        st.dataframe(queue_df, hide_index=True)
        else:
            st.sidebar.warning("⚠️ No active workers detected")
            st.sidebar.info("Start required worker services using docker-compose")

        # Display GPU status if available
        gpu_health = response.get("gpu_health", {})
        if gpu_health:
            with st.sidebar.expander("GPU Status", expanded=False):
                for gpu_id, gpu_info in gpu_health.items():
                    # Display name and health
                    if gpu_info.get("is_healthy", False):
                        st.success(f"✅ {gpu_info.get('device_name', gpu_id)}")
                    else:
                        st.error(
                            f"❌ {gpu_info.get('device_name', gpu_id)}: {gpu_info.get('health_message', 'Not healthy')}")

                    # Memory usage
                    free_pct = gpu_info.get("free_percentage", 0)
                    memory_color = "normal"
                    if free_pct < 10:
                        memory_color = "off"
                    elif free_pct < 30:
                        memory_color = "warning"

                    # Create a progress bar showing GPU memory usage
                    st.progress(100 - free_pct, text=f"Memory: {100 - free_pct:.1f}% used")

                    # Show memory details
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Used", f"{gpu_info.get('allocated_memory_gb', 0):.1f} GB")
                    with col2:
                        st.metric("Free", f"{gpu_info.get('free_memory_gb', 0):.1f} GB")

        # Display model loading status
        model_status = response.get("model_status", {})
        if model_status:
            with st.sidebar.expander("Model Status", expanded=False):
                for model_type, status in model_status.items():
                    display_name = {
                        "embedding": "Embedding Model",
                        "llm": "Language Model",
                        "colbert": "Reranking Model",
                        "whisper": "Speech Recognition"
                    }.get(model_type, model_type)

                    if status.get("loaded", False):
                        st.success(f"✅ {display_name} loaded")
                        if status.get("loading_time", 0) > 0:
                            st.caption(f"Loaded in {status.get('loading_time', 0):.1f}s")
                    else:
                        st.warning(f"⚠️ {display_name} not loaded")

        # Simple refresh button
        if st.sidebar.button("Refresh Status", key="refresh_worker_status"):
            st.rerun()

    except Exception as e:
        st.sidebar.warning(f"⚠️ Error checking worker status: {str(e)}")

        # Add button to try again
        if st.sidebar.button("Try Again", key="try_worker_status_again"):
            st.rerun()


def api_request(
        endpoint: str,
        method: str = "GET",
        data: Optional[Dict] = None,
        files: Optional[Dict] = None,
        params: Optional[Dict] = None,
        timeout: float = 10.0,
        handle_error: callable = None,
) -> Optional[Dict]:
    """Send API request and handle errors."""
    headers = {"x-token": st.session_state.api_key}
    url = f"{st.session_state.api_url}{endpoint}"

    try:
        with httpx.Client(timeout=timeout) as client:
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
                if handle_error:
                    return handle_error(f"Unsupported method: {method}")
                return None

            if response.status_code >= 400:
                error_msg = f"API Error ({response.status_code}): {response.text}"
                if handle_error:
                    return handle_error(error_msg)
                else:
                    return None

            return response.json()
    except httpx.TimeoutException:
        error_msg = f"Request timed out after {timeout}s"
        if handle_error:
            return handle_error(error_msg)
        return None
    except Exception as e:
        error_msg = f"Connection error: {str(e)}"
        if handle_error:
            return handle_error(error_msg)
        return None