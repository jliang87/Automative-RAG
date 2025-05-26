"""
Simplified background tasks system using job chains with dedicated workers.

This package provides a streamlined background processing system that uses
event-driven job chains with specialized GPU workers.
"""

import os
import logging
import dramatiq
from dramatiq.brokers.redis import RedisBroker
from dramatiq.middleware import TimeLimit, Retries

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure Redis broker
redis_host = os.environ.get("REDIS_HOST", "localhost")
redis_port = int(os.environ.get("REDIS_PORT", "6379"))
redis_url = f"redis://{redis_host}:{redis_port}"

broker = RedisBroker(url=redis_url)
dramatiq.set_broker(broker)

# Add middleware
broker.add_middleware(TimeLimit(time_limit=300_000))  # 5 minutes
broker.add_middleware(Retries(max_retries=3))

def main():
    """Main entry point for the worker."""
    worker_type = os.environ.get("WORKER_TYPE", "")
    logger.info(f"Starting worker with type: {worker_type}")

    # Import the job_chain module to register the tasks
    from . import job_chain

    # Preload models based on worker type
    if worker_type.startswith("gpu-"):
        logger.info("GPU worker detected, preloading models...")
        from .models import preload_models
        preload_models()
        logger.info("Model preloading complete")

    # Import dramatiq CLI and run
    from dramatiq.__main__ import main as dramatiq_main
    dramatiq_main()

if __name__ == "__main__":
    main()

# Import the job_chain module to ensure tasks are registered
from . import job_chain
from .common import JobStatus

# Import task functions for external use
from .job_chain import (
    download_video_task,
    transcribe_video_task,
    process_pdf_task,
    process_text_task,
    generate_embeddings_task,
    retrieve_documents_task,
    llm_inference_task,
    JobChain,
    JobType
)

# Export key components
__all__ = [
    # Common
    "JobStatus",
    # Job tracking and chains
    "JobChain",
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