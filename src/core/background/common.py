"""
Common components for the background processing system.

This module sets up the Redis broker, middleware, and common utilities
used by all background tasks.
"""

import os
import time
import json
import logging
from typing import Dict, List, Optional, Union, Any
import torch
import redis
import dramatiq
from dramatiq.brokers.redis import RedisBroker
from dramatiq.middleware.time_limit import TimeLimitExceeded
from dramatiq.middleware.callbacks import Callbacks
from dramatiq.middleware.age_limit import AgeLimit
from dramatiq.middleware.retries import Retries
from dramatiq.rate_limits import ConcurrentRateLimiter
from dramatiq.rate_limits.backends import RedisBackend
from dramatiq.results import Results
from dramatiq.results.backends import RedisBackend as ResultsRedisBackend
from datetime import datetime
import logging

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
    "max_connections": 20,  # Connection pool size
    "client_name": f"dramatiq-{worker_type}-{os.getpid()}"  # Unique client name
}
if redis_password:
    broker_kwargs["password"] = redis_password

# Create broker with middleware
redis_broker = RedisBroker(**broker_kwargs)

# Rate limiter backend
rate_limiter_backend = RedisBackend(
    client=redis_broker.client
)

# Create Results backend using the same Redis client
results_backend = ResultsRedisBackend(
    client=redis_broker.client,
    prefix="dramatiq:results",
    # Set a reasonable ttl for results (1 hour)
    ttl=3600000
)

# Define job status constants
class JobStatus:
    """Job status constants"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"

# Add middleware
redis_broker.add_middleware(Callbacks())
redis_broker.add_middleware(AgeLimit())
redis_broker.add_middleware(Retries(max_retries=3))
# Add Results middleware
redis_broker.add_middleware(Results(backend=results_backend))

# Set the broker as default
dramatiq.set_broker(redis_broker)

# Worker setup middleware
class WorkerSetupMiddleware:
    """Middleware to run setup code when worker processes boot."""

    # Required by Dramatiq - empty set of actor options
    actor_options = set()

    def before_worker_boot(self, broker, worker):
        """Run setup code before the worker boots."""
        logger.info(f"Initializing worker {worker.worker_id} of type {worker_type}")

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

        # Log GPU memory statistics after preloading
        if torch.cuda.is_available() and worker_type.startswith("gpu-"):
            for i in range(torch.cuda.device_count()):
                device_name = torch.cuda.get_device_name(i)
                allocated = torch.cuda.memory_allocated(i) / (1024 ** 3)
                reserved = torch.cuda.memory_reserved(i) / (1024 ** 3)
                logger.info(
                    f"GPU {i} ({device_name}) after worker init: Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")

# Add the worker setup middleware
worker_setup = WorkerSetupMiddleware()
redis_broker.add_middleware(worker_setup)

def check_gpu_health():
    """
    Perform proactive GPU health check with adaptive memory threshold.
    Returns tuple of (is_healthy, status_message)
    """
    try:
        if not torch.cuda.is_available():
            return False, "CUDA not available"

        # Check if we can create and manipulate a simple tensor
        try:
            # Try to allocate a significant but not huge tensor (100MB)
            # This tests if we can still allocate new memory, even if most is used
            test_size = 25 * 1024 * 1024  # ~100MB in float32
            test_tensor = torch.ones(test_size, dtype=torch.float32, device="cuda")
            test_result = test_tensor.sum().item()
            del test_tensor
            torch.cuda.empty_cache()

            # If we get here, we can still allocate memory
        except RuntimeError as e:
            if "out of memory" in str(e):
                return False, "GPU out of memory - cannot allocate test tensor"
            else:
                return False, f"CUDA runtime error during basic tensor operations: {str(e)}"

        # Check memory status for logging/monitoring purposes
        try:
            # Get memory information
            total_memory = torch.cuda.get_device_properties(0).total_memory
            allocated_memory = torch.cuda.memory_allocated(0)
            reserved_memory = torch.cuda.memory_reserved(0)

            # Calculate actual free memory (at CUDA driver level)
            free_memory = total_memory - reserved_memory

            # Calculate what percentage of total memory is free
            free_percent = (free_memory / total_memory) * 100

            # Log the memory status but don't fail just based on percentage
            memory_status = (
                f"Memory status: {allocated_memory / (1024 ** 3):.2f}GB allocated, "
                f"{reserved_memory / (1024 ** 3):.2f}GB reserved, "
                f"{free_memory / (1024 ** 3):.2f}GB free ({free_percent:.1f}%)"
            )

            # If we can't allocate a small tensor for LLM processing, that's a problem
            # This is handled by the actual tensor allocation test above, not by percentage
        except Exception as e:
            return False, f"Error checking GPU memory: {str(e)}"

        return True, f"GPU health check passed. {memory_status}"
    except Exception as e:
        return False, f"Unexpected error in GPU health check: {str(e)}"

def get_redis_client():
    """Get the Redis client from the broker."""
    return redis_broker.client