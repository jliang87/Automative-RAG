# src/core/background/common.py (Simplified)

"""
Simplified common components for the background processing system.

This module sets up the Redis broker and basic utilities for the job chain system.
"""

import os
import time
import logging
from typing import Dict, List, Optional, Union, Any
import torch
import redis
import dramatiq
from dramatiq.brokers.redis import RedisBroker
from dramatiq.middleware.callbacks import Callbacks
from dramatiq.middleware.age_limit import AgeLimit
from dramatiq.middleware.retries import Retries
from dramatiq.results import Results
from dramatiq.results.backends import RedisBackend as ResultsRedisBackend
from datetime import datetime

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Redis configuration
redis_host = os.environ.get("REDIS_HOST", "localhost")
redis_port = int(os.environ.get("REDIS_PORT", "6379"))
redis_password = os.environ.get("REDIS_PASSWORD", None)
worker_type = os.environ.get("WORKER_TYPE", "unknown")

# Initialize Redis broker
broker_kwargs = {
    "host": redis_host,
    "port": redis_port,
    "max_connections": 20,
    "client_name": f"dramatiq-{worker_type}-{os.getpid()}"
}
if redis_password:
    broker_kwargs["password"] = redis_password

# Create broker with simplified middleware
redis_broker = RedisBroker(**broker_kwargs)

# Create Results backend
results_backend = ResultsRedisBackend(
    client=redis_broker.client,
    prefix="dramatiq:results",
    ttl=3600000  # 1 hour
)


# Define job status constants
class JobStatus:
    """Job status constants"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"


# Add simplified middleware
redis_broker.add_middleware(Callbacks())
redis_broker.add_middleware(AgeLimit())
redis_broker.add_middleware(Retries(max_retries=2))  # Reduced retries
redis_broker.add_middleware(Results(backend=results_backend))

# Set the broker as default
dramatiq.set_broker(redis_broker)

# Simplified worker setup
from dramatiq.middleware import Middleware


class SimpleWorkerSetup(Middleware):
    """Simplified worker setup middleware."""

    def before_worker_boot(self, broker, worker):
        """Run setup code before the worker boots."""
        worker_id = f"{worker_type}-{os.getpid()}"
        logger.info(f"Initializing worker {worker_id} of type {worker_type}")

        # Import here to avoid circular imports
        from .models import (
            preload_embedding_model,
            preload_llm_model,
            preload_colbert_reranker,
            preload_whisper_model
        )

        # Preload appropriate models based on worker type
        if worker_type == "gpu-inference":
            logger.info("Preloading LLM and reranking models for gpu-inference worker")
            preload_llm_model()
            preload_colbert_reranker()

        elif worker_type == "gpu-embedding":
            logger.info("Preloading embedding model for gpu-embedding worker")
            preload_embedding_model()

        elif worker_type == "gpu-whisper":
            logger.info("Preloading Whisper model for gpu-whisper worker")
            preload_whisper_model()

        # Log GPU memory after preloading
        if torch.cuda.is_available() and worker_type.startswith("gpu-"):
            for i in range(torch.cuda.device_count()):
                device_name = torch.cuda.get_device_name(i)
                allocated = torch.cuda.memory_allocated(i) / (1024 ** 3)
                reserved = torch.cuda.memory_reserved(i) / (1024 ** 3)
                logger.info(
                    f"GPU {i} ({device_name}) after init: Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")

        # Start simplified heartbeat
        start_worker_heartbeat(worker_id)


# Add the worker setup middleware
redis_broker.add_middleware(SimpleWorkerSetup())


def get_redis_client():
    """Get the Redis client from the broker."""
    return redis_broker.client


def start_worker_heartbeat(worker_id: str):
    """Start a simple worker heartbeat."""
    try:
        redis_client = get_redis_client()
        heartbeat_key = f"dramatiq:__heartbeats__:{worker_id}"

        def update_heartbeat():
            while True:
                try:
                    redis_client.set(heartbeat_key, str(time.time()), ex=60)
                    time.sleep(15)  # Update every 15 seconds
                except Exception as e:
                    logger.error(f"Failed to update heartbeat: {str(e)}")
                    time.sleep(5)

        # Start heartbeat thread
        import threading
        heartbeat_thread = threading.Thread(target=update_heartbeat, daemon=True)
        heartbeat_thread.start()
        logger.info(f"Started heartbeat for {worker_id}")
    except Exception as e:
        logger.error(f"Failed to set up worker heartbeat: {str(e)}")