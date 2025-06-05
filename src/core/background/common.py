# src/core/background/common.py - Fixed heartbeat section

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
import threading

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
    "client_name": f"dramatiq-{worker_type}-{os.getpid()}",
    # CRITICAL FIXES for UTF-8:
    "encoding": "utf-8",
    "decode_responses": True,  # This ensures strings are returned as str, not bytes
    "charset": "utf-8"
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

# Global variables for heartbeat management
_heartbeat_thread = None
_heartbeat_stop_event = None


def get_redis_client():
    """Get the Redis client from the broker."""
    return redis_broker.client


def start_worker_heartbeat(worker_id: str):
    """Start an improved worker heartbeat system."""
    global _heartbeat_thread, _heartbeat_stop_event

    try:
        redis_client = get_redis_client()

        # Use the standard dramatiq heartbeat key pattern
        heartbeat_key = f"dramatiq:__heartbeats__:{worker_id}"

        logger.info(f"Starting heartbeat for {worker_id} with key: {heartbeat_key}")

        # Create stop event
        _heartbeat_stop_event = threading.Event()

        def update_heartbeat():
            """Heartbeat update function"""
            heartbeat_interval = 15  # Update every 15 seconds

            while not _heartbeat_stop_event.is_set():
                try:
                    current_time = time.time()

                    # Set heartbeat with expiration
                    redis_client.set(heartbeat_key, str(current_time), ex=60)

                    logger.debug(f"Updated heartbeat for {worker_id}: {current_time}")

                    # Wait for next update or stop event
                    if _heartbeat_stop_event.wait(timeout=heartbeat_interval):
                        break  # Stop event was set

                except Exception as e:
                    logger.error(f"Failed to update heartbeat for {worker_id}: {e}")
                    # Wait a bit before retrying
                    if _heartbeat_stop_event.wait(timeout=5):
                        break

            logger.info(f"Heartbeat stopped for {worker_id}")

        # Start heartbeat thread
        _heartbeat_thread = threading.Thread(target=update_heartbeat, daemon=True)
        _heartbeat_thread.start()

        logger.info(f"Heartbeat thread started for {worker_id}")

        # Test the heartbeat immediately
        test_heartbeat = redis_client.get(heartbeat_key)
        logger.info(f"Initial heartbeat test for {worker_id}: {test_heartbeat}")

    except Exception as e:
        logger.error(f"Failed to set up worker heartbeat for {worker_id}: {e}")


def stop_worker_heartbeat():
    """Stop the worker heartbeat."""
    global _heartbeat_thread, _heartbeat_stop_event

    if _heartbeat_stop_event:
        _heartbeat_stop_event.set()

    if _heartbeat_thread and _heartbeat_thread.is_alive():
        _heartbeat_thread.join(timeout=5)


# Simplified worker setup
from dramatiq.middleware import Middleware


class SimpleWorkerSetup(Middleware):
    """Simplified worker setup middleware for dedicated workers."""

    def before_worker_boot(self, broker, worker):
        """Run setup code before the worker boots."""
        worker_id = f"{worker_type}-{os.getpid()}"
        logger.info(f"Initializing worker {worker_id} of type {worker_type}")

        # Import here to avoid circular imports
        from .models import preload_models

        # Preload appropriate models based on worker type
        if worker_type.startswith("gpu-"):
            logger.info(f"GPU worker detected: {worker_type}, preloading models...")
            preload_models()

            # Log GPU memory after preloading
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    device_name = torch.cuda.get_device_name(i)
                    allocated = torch.cuda.memory_allocated(i) / (1024 ** 3)
                    reserved = torch.cuda.memory_reserved(i) / (1024 ** 3)
                    logger.info(
                        f"GPU {i} ({device_name}) after init: Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")

        # Start improved heartbeat
        start_worker_heartbeat(worker_id)

    def after_worker_shutdown(self, broker, worker):
        """Clean up after worker shutdown."""
        logger.info("Worker shutting down, stopping heartbeat...")
        stop_worker_heartbeat()


# Add the worker setup middleware
redis_broker.add_middleware(SimpleWorkerSetup())