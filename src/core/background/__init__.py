"""
Background infrastructure - Redis, models, worker management only.
NO task imports - use specific task modules directly.
"""

# Core infrastructure
from .common import JobStatus, get_redis_client
from .models import (
    get_vector_store,
    get_llm_model,
    get_colbert_reranker,
    get_whisper_model,
    preload_models,
    reload_models
)
from .worker_status import (
    get_worker_heartbeats,
    get_active_worker_counts,
    get_worker_summary,
    get_worker_status_for_ui,
    debug_redis_keys,
    clean_problematic_redis_keys
)

__all__ = [
    # Status constants
    "JobStatus",

    # Infrastructure
    "get_redis_client",

    # Model management
    "get_vector_store",
    "get_llm_model",
    "get_colbert_reranker",
    "get_whisper_model",
    "preload_models",
    "reload_models",

    # Worker management
    "get_worker_heartbeats",
    "get_active_worker_counts",
    "get_worker_summary",
    "get_worker_status_for_ui",
    "debug_redis_keys",
    "clean_problematic_redis_keys"
]