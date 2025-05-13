from typing import Dict, Optional
import logging
import os
import redis
from fastapi import HTTPException, Header, status
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
        from src.core.vectorstore import QdrantStore
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
    logger.info("🔄 Initializing API service with minimal components...")

    # Only initialize necessary components for API service
    init_vector_store()
    init_redis_client()
    init_job_tracker()

    logger.info("✅ API service initialized with minimal components")


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


# Placeholder function for model-related endpoints
def model_not_available(model_name: str) -> None:
    """Helper function to handle requests for models that are not available in API mode."""
    raise HTTPException(
        status_code=501,
        detail=f"{model_name} is not available in API-only mode. Use worker services for processing."
    )


# For endpoints that need access to models, provide these helper functions
def get_document_processor():
    """Placeholder for document processor - API server doesn't load this."""
    # When endpoints requiring the document processor are called,
    # they'll receive this error instead of trying to load the model
    return model_not_available("Document processor")


def get_colbert_reranker():
    """Placeholder for ColBERT reranker - API server doesn't load this."""
    return model_not_available("ColBERT reranker")


def get_llm():
    """Placeholder for LLM - API server doesn't load this."""
    return model_not_available("LLM")


def get_video_transcriber():
    """Placeholder for video transcriber - API server doesn't load this."""
    return model_not_available("Video transcriber")


def get_pdf_loader():
    """Placeholder for PDF loader - API server doesn't load this."""
    return model_not_available("PDF loader")