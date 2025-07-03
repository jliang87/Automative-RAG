"""
Ingestion Tasks
Dramatiq task definitions for content ingestion workflows.
"""

# Video processing tasks
from .video_tasks import (
    download_video_task,
    transcribe_video_task,
    start_video_processing
)

# PDF processing tasks
from .pdf_tasks import (
    process_pdf_task,
    start_pdf_processing
)

# Text processing tasks
from .text_tasks import (
    process_text_task,
    start_text_processing
)

# Embedding generation tasks
from .embedding_tasks import (
    generate_embeddings_task,
    start_embedding_generation
)

__all__ = [
    # Video tasks
    "download_video_task",
    "transcribe_video_task",
    "start_video_processing",

    # PDF tasks
    "process_pdf_task",
    "start_pdf_processing",

    # Text tasks
    "process_text_task",
    "start_text_processing",

    # Embedding tasks
    "generate_embeddings_task",
    "start_embedding_generation"
]