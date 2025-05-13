"""
Enhanced task visualization component with progress bars and estimated completion times.
"""

import streamlit as st
import pandas as pd
import time
from typing import Dict, Any, List, Optional


def display_task_progress(job_data: Dict[str, Any]):
    """Display a task with progress bar and estimated completion time."""
    # Extract key information
    job_id = job_data.get("job_id", "")
    status = job_data.get("status", "")
    job_type = job_data.get("job_type", "")

    # Get progress information
    progress_info = job_data.get("progress_info", {})
    progress = progress_info.get("progress", 0)
    progress_message = progress_info.get("message", "")

    # Get timing information
    created_at = job_data.get("created_at", 0)
    updated_at = job_data.get("updated_at", 0)
    elapsed_time = time.time() - created_at
    estimated_remaining = job_data.get("estimated_remaining_seconds")

    # Display basic job information
    st.subheader(f"Task: {job_id}")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"**Type:** {job_type}")
        st.markdown(f"**Status:** {status}")

        # Format elapsed time
        if elapsed_time < 60:
            elapsed_str = f"{elapsed_time:.1f} seconds"
        elif elapsed_time < 3600:
            elapsed_str = f"{elapsed_time / 60:.1f} minutes"
        else:
            elapsed_str = f"{elapsed_time / 3600:.1f} hours"

        st.markdown(f"**Elapsed time:** {elapsed_str}")

    with col2:
        # Display created and updated times
        created_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(created_at))
        updated_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(updated_at))

        st.markdown(f"**Created:** {created_str}")
        st.markdown(f"**Last updated:** {updated_str}")

        # Show estimated completion time if available
        if estimated_remaining and status == "processing":
            if estimated_remaining < 60:
                remaining_str = f"{estimated_remaining:.1f} seconds"
            elif estimated_remaining < 3600:
                remaining_str = f"{estimated_remaining / 60:.1f} minutes"
            else:
                remaining_str = f"{estimated_remaining / 3600:.1f} hours"

            st.markdown(f"**Estimated completion in:** {remaining_str}")

    # Display progress bar
    if status == "processing":
        st.progress(int(progress), text=f"{int(progress)}% - {progress_message}")
    elif status == "completed":
        st.progress(100, text="100% - Task completed")
    elif status in ["failed", "timeout"]:
        st.error("Task failed")
    else:
        st.progress(0, text="0% - Task pending")

    # Show current stage if available
    current_stage = job_data.get("current_stage")
    if current_stage:
        st.info(f"Current stage: {current_stage}")

    # For completed tasks, show result summary
    if status == "completed":
        st.subheader("Result Summary")
        result = job_data.get("result", {})

        if isinstance(result, dict):
            # Display different results based on job type
            if job_type == "llm_inference":
                if "answer" in result:
                    st.markdown("### Answer")
                    st.markdown(result.get("answer", ""))

                if "documents" in result:
                    st.markdown("### Referenced Documents")
                    st.markdown(f"Found {len(result.get('documents', []))} relevant documents")

                if "execution_time" in result:
                    st.markdown(f"Query processed in {result.get('execution_time', 0):.2f} seconds")

            elif job_type in ["video_processing", "batch_video_processing"]:
                if "transcript" in result:
                    st.markdown("### Transcript Summary")
                    transcript = result.get("transcript", "")
                    if len(transcript) > 500:
                        st.markdown(f"{transcript[:500]}...")
                        if st.button("Show full transcript"):
                            st.text_area("Full transcript", transcript, height=300)
                    else:
                        st.markdown(transcript)

                if "document_count" in result:
                    st.markdown(f"Generated {result.get('document_count', 0)} document chunks")

            elif job_type == "pdf_processing":
                if "chunk_count" in result:
                    st.markdown(f"Extracted {result.get('chunk_count', 0)} text chunks from PDF")

                if "processing_time" in result:
                    st.markdown(f"Processing completed in {result.get('processing_time', 0):.2f} seconds")

        elif isinstance(result, str):
            st.markdown(result)

    # For failed tasks, show error
    elif status in ["failed", "timeout"]:
        st.subheader("Error Details")
        error = job_data.get("error", "Unknown error")
        st.error(error)

        # Add retry button
        if st.button("Retry Task", key=f"retry_{job_id}"):
            return {"action": "retry", "job_id": job_id}

    # For processing tasks, show cancel button
    elif status == "processing":
        if st.button("Cancel Task", key=f"cancel_{job_id}"):
            return {"action": "cancel", "job_id": job_id}

    # Add divider
    st.divider()

    return {"action": None}


def display_stage_timeline(job_data: Dict[str, Any]):
    """Display a timeline visualization of job stages."""
    stage_history = job_data.get("stage_history", [])

    if not stage_history:
        st.info("No stage information available")
        return

    # Create timeline data
    timeline_data = []
    created_at = job_data.get("created_at", 0)

    for i, stage in enumerate(stage_history):
        stage_name = stage.get("stage", "unknown")
        start_time = stage.get("started_at", 0)

        # Calculate end time (either next stage start or current time)
        if i < len(stage_history) - 1:
            end_time = stage_history[i + 1].get("started_at", time.time())
        else:
            end_time = time.time()

        # Calculate relative position and width for timeline
        start_pos = (start_time - created_at) / (time.time() - created_at) * 100
        width = (end_time - start_time) / (time.time() - created_at) * 100

        # Format times
        start_str = time.strftime("%H:%M:%S", time.localtime(start_time))
        duration = end_time - start_time

        if duration < 60:
            duration_str = f"{duration:.1f}s"
        elif duration < 3600:
            duration_str = f"{duration / 60:.1f}m"
        else:
            duration_str = f"{duration / 3600:.1f}h"

        timeline_data.append({
            "stage": stage_name,
            "start_time": start_time,
            "end_time": end_time,
            "start_str": start_str,
            "duration": duration,
            "duration_str": duration_str,
            "start_pos": start_pos,
            "width": width
        })

    # Display timeline
    st.subheader("Processing Timeline")

    # Create container for custom timeline
    timeline_container = st.container()

    # Display the timeline visually
    total_duration = time.time() - created_at
    if total_duration > 0:
        # Calculate total duration in a readable format
        if total_duration < 60:
            total_duration_str = f"{total_duration:.1f} seconds"
        elif total_duration < 3600:
            total_duration_str = f"{total_duration / 60:.1f} minutes"
        else:
            total_duration_str = f"{total_duration / 3600:.1f} hours"

        st.caption(f"Total processing time: {total_duration_str}")

        # Display each stage as a progress bar with custom colors
        colors = ["#1E88E5", "#FFC107", "#43A047", "#E53935", "#8E24AA"]

        for i, stage in enumerate(timeline_data):
            col1, col2 = st.columns([stage["start_pos"] / 100, 1 - stage["start_pos"] / 100]) if stage[
                                                                                                     "start_pos"] > 0 else (
            None, st.columns([1])[0])

            if col1 is not None:
                col1.write("")  # Empty space for positioning

            # Calculate color index
            color_idx = i % len(colors)
            color = colors[color_idx]

            # Display the stage bar with appropriate width
            if stage["width"] > 0:
                col2.progress(min(100, stage["width"]), text=f"{stage['stage']} ({stage['duration_str']})")

    # Display as table as well
    timeline_table = pd.DataFrame([
        {
            "Stage": t["stage"],
            "Start Time": t["start_str"],
            "Duration": t["duration_str"],
        } for t in timeline_data
    ])

    st.dataframe(timeline_table, hide_index=True, use_container_width=True)


def render_priority_queue_visualization(priority_queue_data: Dict[str, Any]):
    """Render a visualization of the priority queue system."""
    if not priority_queue_data:
        st.warning("No priority queue data available")
        return

    st.subheader("Priority Queue Status")

    # Display active task
    active_task = priority_queue_data.get("active_task")
    if active_task:
        st.markdown("### Active GPU Task")

        active_task_info = {
            "Queue": active_task.get("queue_name", "unknown"),
            "Priority": active_task.get("priority", "-"),
            "Task ID": active_task.get("task_id", "unknown"),
            "Job ID": active_task.get("job_id", "unknown"),
        }

        # Calculate how long the task has been active
        registered_at = active_task.get("registered_at")
        if registered_at:
            active_duration = time.time() - registered_at
            if active_duration < 60:
                duration_str = f"{active_duration:.1f} seconds"
            elif active_duration < 3600:
                duration_str = f"{active_duration / 60:.1f} minutes"
            else:
                duration_str = f"{active_duration / 3600:.1f} hours"

            active_task_info["Active For"] = duration_str

        # Display as DataFrame
        active_df = pd.DataFrame([active_task_info])
        st.dataframe(active_df, hide_index=True, use_container_width=True)
    else:
        st.info("No active GPU task currently running")

    # Display waiting tasks
    st.markdown("### Waiting Tasks")

    tasks_by_queue = priority_queue_data.get("tasks_by_queue", {})
    if tasks_by_queue:
        # Prepare data for visualization
        queue_data = []

        for queue, count in tasks_by_queue.items():
            if count > 0:
                # Get the priority level for this queue
                priority_level = priority_queue_data.get("priority_levels", {}).get(queue, "-")

                queue_data.append({
                    "Queue": queue,
                    "Priority": priority_level,
                    "Waiting Tasks": count
                })

        if queue_data:
            # Sort by priority (lower number = higher priority)
            queue_data.sort(key=lambda x: x["Priority"])

            # Display as DataFrame
            queue_df = pd.DataFrame(queue_data)
            st.dataframe(queue_df, hide_index=True, use_container_width=True)

            # Visualize as horizontal bars
            for queue_info in queue_data:
                st.text(f"{queue_info['Queue']} (Priority {queue_info['Priority']})")
                st.progress(min(100, queue_info["Waiting Tasks"] * 10), text=f"{queue_info['Waiting Tasks']} tasks")
        else:
            st.info("No tasks waiting in queue")
    else:
        st.info("No tasks waiting in queue")