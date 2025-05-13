"""
Background tasks system for the RAG application.

This package provides a comprehensive background processing system for
handling GPU-intensive tasks like embedding generation, inference, and
transcription in a priority-based manner.
"""

from .common import JobStatus
from .job_tracker import job_tracker
from .priority_queue import priority_queue
from .actors.ingestion import (
    generate_embeddings_gpu,
    transcribe_video_gpu,
    process_text,
    process_transcript,
    process_pdf_cpu,
    process_video_gpu,
    batch_process_videos,
    delete_document_gpu
)
from .actors.inference import (
    perform_llm_inference,
    rerank_documents,
    retrieve_documents,
    process_query_request
)
from .actors.system import (
    cleanup_old_jobs,
    cleanup_stalled_tasks,
    check_priority_queue_health,
    monitor_gpu_memory,
    reload_models_periodically,
    balance_task_queues,
    collect_system_statistics,
    optimize_databases,
    analyze_error_patterns,
    system_watchdog
)
from .monitoring import (
    get_system_status,
    get_priority_queue_status,
    generate_system_report
)

__all__ = [
    # Common
    "JobStatus",

    # Job tracking
    "job_tracker",

    # Priority queue
    "priority_queue",

    # Ingestion actors
    "generate_embeddings_gpu",
    "transcribe_video_gpu",
    "process_text",
    "process_transcript",
    "process_pdf_cpu",
    "process_video_gpu",
    "batch_process_videos",
    "delete_document_gpu",

    # Inference actors
    "perform_llm_inference",
    "rerank_documents",
    "retrieve_documents",
    "process_query_request",

    # System actors
    "cleanup_old_jobs",
    "cleanup_stalled_tasks",
    "check_priority_queue_health",
    "monitor_gpu_memory",
    "reload_models_periodically",
    "balance_task_queues",
    "collect_system_statistics",
    "optimize_databases",
    "analyze_error_patterns",
    "system_watchdog",

    # Monitoring
    "get_system_status",
    "get_priority_queue_status",
    "generate_system_report"
]