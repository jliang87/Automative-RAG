# src/api/dependencies.py - Fixed initialization

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
        from src.core.vectorstore import QdrantStore

        # Always use metadata-only mode for API
        vector_store = QdrantStore(
            client=qdrant_client,
            collection_name=settings.qdrant_collection,
            embedding_function=None,  # No embedding function needed for API
        )
        logger.info("✅ Vector Store Initialized (metadata-only mode)!")


def init_redis_client():
    """Initialize Redis client."""
    global redis_client
    if redis_client is None:
        logger.info("🚀 Initializing Redis client...")
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

        try:
            redis_client = redis.Redis(**redis_kwargs)
            # Test the connection
            redis_client.ping()
            logger.info("✅ Redis Client Initialized!")
        except Exception as e:
            logger.error(f"❌ Failed to initialize Redis client: {e}")
            # Don't raise exception, let the app start but warn about Redis issues
            redis_client = None
    return redis_client


def init_job_tracker():
    """Initialize job tracker."""
    global job_tracker
    if job_tracker is None:
        logger.info("🚀 Initializing Job Tracker...")
        # Use the existing Redis client
        client = init_redis_client()
        if client is not None:
            job_tracker = JobTracker(redis_client=client)
            logger.info("✅ Job Tracker Initialized!")
        else:
            logger.warning("⚠️ Job Tracker not initialized - Redis unavailable")
    return job_tracker


# Redis client dependency
def get_redis_client() -> redis.Redis:
    """Get the cached Redis client instance."""
    if redis_client is None:
        # Try to initialize on-demand
        client = init_redis_client()
        if client is None:
            raise HTTPException(status_code=500, detail="Redis client not available")
    return redis_client


# JobTracker dependency
def get_job_tracker() -> JobTracker:
    """Get the cached JobTracker instance."""
    if job_tracker is None:
        # Try to initialize on-demand
        tracker = init_job_tracker()
        if tracker is None:
            raise HTTPException(status_code=500, detail="Job tracker not available")
    return job_tracker


def load_all_components():
    """Initialize only necessary components at application startup."""
    logger.info("🔄 Initializing API service components...")

    try:
        # Initialize Redis client FIRST
        init_redis_client()

        # Initialize job tracker (depends on Redis)
        init_job_tracker()

        # Initialize vector store last (can fail without breaking Redis)
        init_vector_store()

        logger.info("✅ API service initialization complete")

    except Exception as e:
        logger.error(f"❌ Error during API initialization: {str(e)}")
        # Don't re-raise - let the API start even if some components fail
        logger.warning("⚠️ API starting with limited functionality")


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