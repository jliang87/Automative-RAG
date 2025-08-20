import logging
import os
import redis
from fastapi import HTTPException, Header, status
from qdrant_client import QdrantClient

from src.config.settings import settings
from src.core.query.retrieval.vectorstore import QdrantStore
from src.core.orchestration.job_tracker import JobTracker

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

# Service instances
query_service = None
validation_service = None
system_service = None
orchestration_service = None
document_processing_service = None
response_processing_service = None

# Controller instances
query_controller = None
validation_controller = None
system_controller = None


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

        try:
            embedding_function = settings.embedding_function
            vector_store = QdrantStore(
                client=qdrant_client,
                collection_name=settings.qdrant_collection,
                embedding_function=embedding_function,
            )
            logger.info("✅ Vector Store Initialized with embedding function!")
        except Exception as e:
            logger.error(f"❌ Failed to initialize vector store: {e}")
            logger.error("Vector store requires embedding function. No metadata-only mode supported.")
            raise HTTPException(
                status_code=500,
                detail="Vector store initialization failed. Embedding function required."
            )


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
            redis_client.ping()
            logger.info("✅ Redis Client Initialized!")
        except Exception as e:
            logger.error(f"❌ Failed to initialize Redis client: {e}")
            redis_client = None
    return redis_client


def init_job_tracker():
    """Initialize job tracker."""
    global job_tracker
    if job_tracker is None:
        logger.info("🚀 Initializing Job Tracker...")
        client = init_redis_client()
        if client is not None:
            job_tracker = JobTracker(redis_client=client)
            logger.info("✅ Job Tracker Initialized!")
        else:
            logger.warning("⚠️ Job Tracker not initialized - Redis unavailable")
    return job_tracker


def init_services():
    """Initialize all service layer components."""
    global
    (query_service, validation_service, system_service,
     orchestration_service, document_processing_service, response_processing_service)

    # Import services dynamically to avoid circular imports
    try:

        if document_processing_service is None:
            logger.info("🚀 Initializing Document Processing Service...")
            from src.services.document_service import DocumentProcessingService
            document_processing_service = DocumentProcessingService()
            logger.info("✅ Document Processing Service Initialized!")

        if system_service is None:
            logger.info("🚀 Initializing System Service...")
            from src.services.system_service import SystemService
            system_service = SystemService(
                vector_store=get_vector_store(),
                job_tracker=get_job_tracker(),
                orchestration_service=orchestration_service
            )
            logger.info("✅ System Service Initialized!")

        if query_service is None:
            logger.info("🚀 Initializing Query Service...")
            # Query service is already imported from existing file
            query_service = get_query_service_instance()
            logger.info("✅ Query Service Initialized!")

    except ImportError as e:
        logger.error(f"Missing service module: {e}")
        raise HTTPException(status_code=500, detail=f"Service initialization failed: {e}")


def get_query_service_instance():
    """Get QueryService instance with all dependencies."""
    from src.services.query_service import QueryService
    return QueryService(
        validation_service=validation_service,
        orchestration_service=orchestration_service,
        document_service=document_processing_service,
        response_service=response_processing_service
    )


def init_controllers():
    """Initialize all controller layer components."""
    global query_controller, validation_controller, system_controller

    # Ensure services are initialized first
    init_services()

    if query_controller is None:
        logger.info("🚀 Initializing Query Controller...")
        query_controller = QueryController(
            query_service=query_service,
            validation_service=validation_service,
            system_service=system_service
        )
        logger.info("✅ Query Controller Initialized!")

    if validation_controller is None:
        logger.info("🚀 Initializing Validation Controller...")
        validation_controller = ValidationController(
            validation_service=validation_service
        )
        logger.info("✅ Validation Controller Initialized!")

    if system_controller is None:
        logger.info("🚀 Initializing System Controller...")
        system_controller = SystemController(
            system_service=system_service
        )
        logger.info("✅ System Controller Initialized!")


def load_all_components():
    """Initialize all components at application startup."""
    logger.info("🔄 Initializing API service components...")

    try:
        # Initialize Redis client FIRST
        init_redis_client()

        # Initialize job tracker (depends on Redis)
        init_job_tracker()

        # Initialize vector store (now always requires embedding function)
        init_vector_store()

        # Initialize services layer
        init_services()

        # Initialize controllers layer
        init_controllers()

        logger.info("✅ API service initialization complete")

    except Exception as e:
        logger.error(f"❌ Error during API initialization: {str(e)}")
        raise


# Authentication dependency
async def get_token_header(x_token: str = Header(...)):
    if x_token != settings.api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token",
        )
    return x_token


# Infrastructure dependencies
def get_redis_client() -> redis.Redis:
    """Get the cached Redis client instance."""
    if redis_client is None:
        client = init_redis_client()
        if client is None:
            raise HTTPException(status_code=500, detail="Redis client not available")
    return redis_client


def get_job_tracker() -> JobTracker:
    """Get the cached JobTracker instance."""
    if job_tracker is None:
        tracker = init_job_tracker()
        if tracker is None:
            raise HTTPException(status_code=500, detail="Job tracker not available")
    return job_tracker


def get_qdrant_client() -> QdrantClient:
    """Get the cached Qdrant client instance."""
    if qdrant_client is None:
        raise HTTPException(status_code=500, detail="Qdrant client not initialized yet.")
    return qdrant_client


def get_vector_store() -> QdrantStore:
    """Get the cached vector store instance."""
    if vector_store is None:
        raise HTTPException(status_code=500, detail="Vector store not initialized yet.")
    return vector_store


# Service dependencies
def get_query_service() -> 'QueryService':
    """Get the cached Query Service instance."""
    if query_service is None:
        init_services()
    return query_service


def get_validation_service() -> 'ValidationService':
    """Get the cached Validation Service instance."""
    if validation_service is None:
        init_services()
    return validation_service


def get_system_service() -> 'SystemService':
    """Get the cached System Service instance."""
    if system_service is None:
        init_services()
    return system_service


def get_orchestration_service() -> 'OrchestrationService':
    """Get the cached Orchestration Service instance."""
    if orchestration_service is None:
        init_services()
    return orchestration_service


def get_document_processing_service() -> 'DocumentProcessingService':
    """Get the cached Document Processing Service instance."""
    if document_processing_service is None:
        init_services()
    return document_processing_service


def get_response_processing_service() -> 'ResponseProcessingService':
    """Get the cached Response Processing Service instance."""
    if response_processing_service is None:
        init_services()
    return response_processing_service


# Controller dependencies - This is what was missing!
def get_controllers():
    """Get all controller instances - this was the missing function!"""
    if query_controller is None or validation_controller is None or system_controller is None:
        init_controllers()

    # Return a simple object that provides access to all controllers
    class Controllers:
        def __init__(self):
            self.query_controller = query_controller
            self.validation_controller = validation_controller
            self.system_controller = system_controller

    return Controllers()


# Individual controller dependencies
def get_query_controller() -> QueryController:
    """Get the cached Query Controller instance."""
    if query_controller is None:
        init_controllers()
    return query_controller


def get_validation_controller() -> ValidationController:
    """Get the cached Validation Controller instance."""
    if validation_controller is None:
        init_controllers()
    return validation_controller


def get_system_controller() -> SystemController:
    """Get the cached System Controller instance."""
    if system_controller is None:
        init_controllers()
    return system_controller