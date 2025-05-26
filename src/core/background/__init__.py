# src/core/background/__init__.py (Simplified for Job Chain)

"""
Simplified background tasks system using job chains.

This package provides a streamlined background processing system that uses
event-driven job chains instead of complex priority queues and coordination.
"""

from .common import JobStatus
from .job_tracker import job_tracker
from .job_chain import (
    job_chain,
    JobType,
    download_video_task,
    transcribe_video_task,
    process_pdf_task,
    process_text_task,
    generate_embeddings_task,
    retrieve_documents_task,
    llm_inference_task
)

__all__ = [
    # Common
    "JobStatus",

    # Job tracking and chains
    "job_tracker",
    "job_chain",
    "JobType",

    # Task actors
    "download_video_task",
    "transcribe_video_task",
    "process_pdf_task",
    "process_text_task",
    "generate_embeddings_task",
    "retrieve_documents_task",
    "llm_inference_task"
]
