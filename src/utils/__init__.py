"""
Utility functions for the Automotive Specs RAG system.
"""

from .helpers import (
    generate_unique_id,
    clean_text,
    extract_year_from_text,
    extract_metadata_from_text,
    format_time,
    parse_youtube_url,
    parse_bilibili_url,
    chunk_text,
    get_file_extension,
    is_valid_file_type,
    safe_json_loads,
    retry_with_backoff
)

from .logging import setup_logger, GPULogger
from .model_paths import (
    get_model_path,
    get_embedding_model_path,
    get_colbert_model_path,
    get_llm_model_path,
    get_whisper_model_path,
    get_model_paths
)

__all__ = [
    # From helpers
    "generate_unique_id",
    "clean_text",
    "extract_year_from_text",
    "extract_metadata_from_text",
    "format_time",
    "parse_youtube_url",
    "parse_bilibili_url",
    "chunk_text",
    "get_file_extension",
    "is_valid_file_type",
    "safe_json_loads",
    "retry_with_backoff",

    # From logging
    "setup_logger",
    "GPULogger",

    # From model_paths
    "get_model_path",
    "get_embedding_model_path",
    "get_colbert_model_path",
    "get_llm_model_path",
    "get_whisper_model_path",
    "get_model_paths"
]