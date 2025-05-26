# src/api/dependencies.py (SIMPLIFIED - Remove Priority Queue References)

from typing import Dict, Optional
import logging
import os
import redis
from fastapi import Depends, HTTPException, Header, status
from qdrant_client import QdrantClient

from src.config.settings import settings
from src.core.vectorstore import QdrantStore
from src.core.background.job_tracker import JobTracker

# Configure logging
logger = logging.getLogger(__name__)

# Redis client configuration
redis_host = os.environ.get("REDIS_HOST", "localhost")
redis_port = int(os.environ.get("REDIS_PORT", "6379"))
redis_password = os.environ.get("REDIS_PASSWORD", None)

# Global instances
redis_client = None
job_tracker = None
qdrant_client = None
vector_store = None

# Determine if we're in API-only mode
IS_API_MODE = os.environ.get("WORKER_TYPE", "") == "api"


def init_vector_store():
    """Initialize Qdrant client and vector store."""
    global qdrant_client, vector_store
    if qdrant_client is None:
        logger.info("🚀 Initializing Qdrant client...")
        qdrant_client = QdrantClient(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
        )
        logger.info("✅ Qdrant Client Initialized!")

    if vector_store is None:
        logger.info("🚀 Initializing Vector Store...")

        # API service only needs metadata-only mode
        if IS_API_MODE:
            from src.core.vectorstore import QdrantStore
            vector_store = QdrantStore(
                client=qdrant_client,
                collection_name=settings.qdrant_collection,
                embedding_function=None,  # No embedding function needed for API
            )
            logger.info("✅ Vector Store Initialized (metadata-only mode)!")
        else:
            # Full initialization with embedding model for workers
            from src.core.vectorstore import QdrantStore
            vector_store = QdrantStore(
                client=qdrant_client,
                collection_name=settings.qdrant_collection,
                embedding_function=settings.embedding_function,
            )
            logger.info("✅ Vector Store Initialized with embedding function!")


def init_redis_client():
    """Initialize Redis client."""
    global redis_client
    if redis_client is None:
        redis_kwargs = {
            "host": redis_host,
            "port": redis_port,
            "decode_responses": True,
            "socket_connect_timeout": 5,
            "socket_timeout": 5,
            "retry_on_timeout": True,
        }
        if redis_password:
            redis_kwargs["password"] = redis_password

        redis_client = redis.Redis(**redis_kwargs)
        logger.info("✅ Redis Client Initialized!")
    return redis_client


def init_job_tracker():
    """Initialize job tracker."""
    global job_tracker
    if job_tracker is None:
        logger.info("🚀 Initializing Job Tracker...")
        # Use the existing Redis client
        client = init_redis_client()
        job_tracker = JobTracker(redis_client=client)
        logger.info("✅ Job Tracker Initialized!")
    return job_tracker


# Redis client dependency
def get_redis_client() -> redis.Redis:
    """Get the cached Redis client instance."""
    if redis_client is None:
        raise HTTPException(status_code=500, detail="Redis client not initialized yet.")
    return redis_client


# JobTracker dependency
def get_job_tracker() -> JobTracker:
    """Get the cached JobTracker instance."""
    if job_tracker is None:
        raise HTTPException(status_code=500, detail="Job tracker not initialized yet.")
    return job_tracker


def load_all_components():
    """Initialize only necessary components at application startup."""
    if IS_API_MODE:
        logger.info("🔄 Initializing API service with minimal components...")

        # Only initialize necessary components for API service
        init_vector_store()
        init_redis_client()
        init_job_tracker()

        logger.info("✅ API service initialized with minimal components")
    else:
        # For worker services, load required components based on worker type
        worker_type = os.environ.get("WORKER_TYPE", "unknown")
        logger.info(f"🔄 Initializing worker service: {worker_type}")

        # All worker types need these
        init_vector_store()
        init_redis_client()
        init_job_tracker()

        logger.info(f"✅ Base components initialized for worker: {worker_type}")


# Authentication dependency
async def get_token_header(x_token: str = Header(...)):
    if x_token != settings.api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token",
        )
    return x_token


# Qdrant client dependency - reuses the global instance
def get_qdrant_client() -> QdrantClient:
    if qdrant_client is None:
        raise HTTPException(status_code=500, detail="Qdrant client not initialized yet.")
    return qdrant_client


# Vector store dependency - reuses the global instance
def get_vector_store() -> QdrantStore:
    if vector_store is None:
        raise HTTPException(status_code=500, detail="Vector store not initialized yet.")
    return vector_store


# Function to handle missing model dependencies
def api_mode_only_handler(model_name: str):
    """Handler for endpoints that require models in API-only mode."""
    if IS_API_MODE:
        raise HTTPException(
            status_code=501,
            detail=f"{model_name} is not available in API-only mode. Use job chain system for processing."
        )
