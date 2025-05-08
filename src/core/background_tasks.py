import os
import json
import logging
import time
from typing import Dict, List, Optional, Union, Any
import torch
from sentence_transformers import SentenceTransformer
from src.config.settings import settings
from langchain_huggingface import HuggingFaceEmbeddings

import dramatiq
from dramatiq.brokers.redis import RedisBroker
from dramatiq.middleware.time_limit import TimeLimitExceeded
from dramatiq.middleware.callbacks import Callbacks
from dramatiq.middleware.age_limit import AgeLimit
from dramatiq.middleware.retries import Retries
from dramatiq.rate_limits import ConcurrentRateLimiter
from dramatiq.rate_limits.backends import RedisBackend
# Add these imports for Results middleware
from dramatiq.results import Results
from dramatiq.results.backends.redis import RedisBackend as ResultsRedisBackend


# Create a worker middleware to run the preload function
class WorkerSetupMiddleware:
    """Middleware to run setup code when worker processes boot."""

    def before_worker_boot(self, broker, worker):
        """Run setup code before the worker boots."""
        logger.info(f"Initializing worker {worker.worker_id} of type {worker_type}")

        if worker_type == "gpu":
            # Only preload models on GPU workers
            logger.info("Preloading models for GPU worker")
            preload_embedding_model()

            # Log GPU memory statistics after preloading
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    device_name = torch.cuda.get_device_name(i)
                    allocated = torch.cuda.memory_allocated(i) / (1024 ** 3)
                    reserved = torch.cuda.memory_reserved(i) / (1024 ** 3)
                    logger.info(
                        f"GPU {i} ({device_name}) after worker init: Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")


# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Configure Redis broker
redis_host = os.environ.get("REDIS_HOST", "localhost")
redis_port = int(os.environ.get("REDIS_PORT", "6379"))
redis_password = os.environ.get("REDIS_PASSWORD", None)

# Get worker type
worker_type = os.environ.get("WORKER_TYPE", "unknown")
logger.info(f"Initializing worker of type: {worker_type}")

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

# Add middleware
redis_broker.add_middleware(Callbacks())
redis_broker.add_middleware(AgeLimit())
redis_broker.add_middleware(Retries(max_retries=3))
# Add Results middleware
redis_broker.add_middleware(Results(backend=results_backend))
# Add the middleware to the broker
worker_setup = WorkerSetupMiddleware()
redis_broker.add_middleware(worker_setup)

dramatiq.set_broker(redis_broker)

# Create GPU rate limiter - limits to 1 concurrent GPU task
gpu_limiter = ConcurrentRateLimiter(
    backend=rate_limiter_backend,
    key="gpu_task_limiter",
    limit=1,
    ttl=600000  # 10 minute
)

# Define job status constants
class JobStatus:
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"


# Job tracking
class JobTracker:
    """Track and manage background jobs."""

    def __init__(self, redis_client=None):
        """Initialize the job tracker with Redis."""
        if redis_client is None:
            import redis
            self.redis = redis.Redis(
                host=redis_host,
                port=redis_port,
                password=redis_password,
                decode_responses=True
            )
        else:
            self.redis = redis_client

        # Use a Redis hash to store job information
        self.job_key = "rag_system:jobs"

    def create_job(self, job_id: str, job_type: str, metadata: Dict[str, Any]) -> None:
        """Create a new job record."""
        job_data = {
            "job_id": job_id,
            "job_type": job_type,
            "status": JobStatus.PENDING,
            "created_at": time.time(),
            "updated_at": time.time(),
            "result": None,
            "error": None,
            "metadata": json.dumps(metadata)
        }
        # Store as a hash field with job_id as field name
        self.redis.hset(self.job_key, job_id, json.dumps(job_data))
        logger.info(f"Created job {job_id} of type {job_type}")

    def update_job_status(self, job_id: str, status: str, result: Any = None, error: str = None) -> None:
        """Update the status of a job."""
        job_data_json = self.redis.hget(self.job_key, job_id)
        if not job_data_json:
            logger.warning(f"Job {job_id} not found when updating status to {status}")
            return

        job_data = json.loads(job_data_json)
        job_data["status"] = status
        job_data["updated_at"] = time.time()

        if result is not None:
            if isinstance(result, (list, dict)):
                job_data["result"] = json.dumps(result)
            else:
                job_data["result"] = str(result)

        if error is not None:
            job_data["error"] = str(error)

        self.redis.hset(self.job_key, job_id, json.dumps(job_data))
        logger.info(f"Updated job {job_id} status to {status}")

    # Add a method to check if a job is already completed
    def is_job_completed(self, job_id: str) -> bool:
        """Check if a job has already been completed."""
        job_data_json = self.redis.hget(self.job_key, job_id)
        if not job_data_json:
            return False

        job_data = json.loads(job_data_json)
        return job_data.get("status") == JobStatus.COMPLETED

    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get job information by ID."""
        job_data_json = self.redis.hget(self.job_key, job_id)
        if not job_data_json:
            return None

        job_data = json.loads(job_data_json)
        # Parse JSON metadata back to dict if it exists
        if "metadata" in job_data and job_data["metadata"]:
            job_data["metadata"] = json.loads(job_data["metadata"])
        # Parse result if it's JSON
        if "result" in job_data and job_data["result"] and isinstance(job_data["result"], str) and job_data["result"].startswith('{'):
            try:
                job_data["result"] = json.loads(job_data["result"])
            except:
                pass  # Keep as string if it's not valid JSON

        return job_data

    def get_all_jobs(self, limit: int = 100, job_type: str = None) -> List[Dict[str, Any]]:
        """Get all jobs, optionally filtered by type."""
        all_jobs = self.redis.hgetall(self.job_key)
        jobs = []

        for _, job_data_json in all_jobs.items():
            job_data = json.loads(job_data_json)

            # Filter by job type if specified
            if job_type and job_data.get("job_type") != job_type:
                continue

            # Parse JSON metadata back to dict if it exists
            if "metadata" in job_data and job_data["metadata"]:
                job_data["metadata"] = json.loads(job_data["metadata"])

            # Parse result if it's JSON
            if "result" in job_data and job_data["result"] and isinstance(job_data["result"], str) and job_data["result"].startswith('{'):
                try:
                    job_data["result"] = json.loads(job_data["result"])
                except:
                    pass  # Keep as string if it's not valid JSON

            jobs.append(job_data)

        # Sort by creation time (newest first) and limit results
        jobs.sort(key=lambda x: x.get("created_at", 0), reverse=True)
        return jobs[:limit]

    def delete_job(self, job_id: str) -> bool:
        """Delete a job by ID."""
        deleted = self.redis.hdel(self.job_key, job_id)
        return deleted > 0

    def clean_old_jobs(self, days: int = 7) -> int:
        """Clean up jobs older than specified days."""
        all_jobs = self.redis.hgetall(self.job_key)
        cutoff_time = time.time() - (days * 24 * 60 * 60)
        deleted_count = 0

        for job_id, job_data_json in all_jobs.items():
            job_data = json.loads(job_data_json)
            if job_data.get("created_at", 0) < cutoff_time:
                self.redis.hdel(self.job_key, job_id)
                deleted_count += 1

        return deleted_count


# Initialize the job tracker
job_tracker = JobTracker()


def check_gpu_health():
    """
    Perform proactive GPU health check with adaptive memory threshold.
    Returns tuple of (is_healthy, status_message)
    """
    try:
        import torch

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


# Global variable to hold the preloaded embedding model
_PRELOADED_EMBEDDING_MODEL = None


def preload_embedding_model():
    """
    Preload the embedding model at worker startup to avoid loading it for each job.
    This function should be called during worker initialization.
    """
    global _PRELOADED_EMBEDDING_MODEL

    # Skip if already loaded or if we're on a CPU worker
    if _PRELOADED_EMBEDDING_MODEL is not None or worker_type == "cpu":
        return

    logger.info(f"Preloading embedding model {settings.default_embedding_model} on {settings.device}")

    try:
        # Clear CUDA cache before loading
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            for i in range(torch.cuda.device_count()):
                torch.cuda.synchronize(i)

            # Log GPU memory status before loading
            for i in range(torch.cuda.device_count()):
                device_name = torch.cuda.get_device_name(i)
                allocated = torch.cuda.memory_allocated(i) / (1024 ** 3)
                reserved = torch.cuda.memory_reserved(i) / (1024 ** 3)
                logger.info(
                    f"GPU {i} ({device_name}) before model loading: Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")

        # Load the model
        _PRELOADED_EMBEDDING_MODEL = HuggingFaceEmbeddings(
            model_name=settings.embedding_model_full_path,
            model_kwargs={"device": settings.device},
            encode_kwargs={"batch_size": settings.batch_size, "normalize_embeddings": True},
            cache_folder=os.path.join(settings.models_dir, settings.embedding_model_path)
        )

        # Test the model with a simple embedding to ensure it works
        test_embedding = _PRELOADED_EMBEDDING_MODEL.embed_query("Test sentence for embedding model")
        embedding_dim = len(test_embedding)

        # Log GPU memory status after loading
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                allocated = torch.cuda.memory_allocated(i) / (1024 ** 3)
                reserved = torch.cuda.memory_reserved(i) / (1024 ** 3)
                logger.info(f"GPU {i} after model loading: Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")

        logger.info(f"Successfully preloaded embedding model with dimension {embedding_dim}")
    except Exception as e:
        logger.error(f"Failed to preload embedding model: {str(e)}")
        # We don't raise the exception here - the worker should still start
        # Individual jobs will attempt to load the model as needed


@dramatiq.actor(
    queue_name="inference_tasks",
    max_retries=3,  # Increase retries
    time_limit=300000,
    min_backoff=10000,  # 10 seconds minimum backoff
    max_backoff=300000  # 5 minutes maximum backoff
)
def perform_llm_inference(job_id: str, query: str, documents: List[Dict], metadata_filter: Optional[Dict] = None):
    """Perform LLM inference using GPU with high priority and resource management."""
    try:
        # Use the GPU rate limiter to ensure only one GPU task runs at a time
        with gpu_limiter.acquire():
            # Try to acquire GPU resources
            logger.info(f"Acquiring GPU resources for job {job_id}")

            # Update job status to processing
            job_tracker.update_job_status(job_id, JobStatus.PROCESSING)
            logger.info(f"GPU worker performing inference for query: {query}")

            # Import here to avoid circular imports
            from src.core.llm import LocalLLM
            from src.config.settings import settings
            from langchain_core.documents import Document
            import torch

            # Try to clean up CUDA memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                initial_memory = torch.cuda.memory_allocated() / (1024 ** 3)
                logger.info(f"Initial GPU memory allocated: {initial_memory:.2f} GB")

            # Convert document dictionaries back to Document objects
            doc_objects = []
            for doc_dict in documents:
                doc = Document(
                    page_content=doc_dict["content"],
                    metadata=doc_dict.get("metadata", {})
                )
                score = doc_dict.get("relevance_score", 0)
                doc_objects.append((doc, score))

            # Initialize LLM with memory efficient settings
            llm = LocalLLM(
                model_name=settings.default_llm_model,
                device="cuda",
                temperature=settings.llm_temperature,
                max_tokens=settings.llm_max_tokens,
                use_4bit=settings.llm_use_4bit,
                use_8bit=settings.llm_use_8bit,
                torch_dtype=torch.float16 if settings.use_fp16 and settings.device.startswith("cuda") else None
            )

            # Log memory usage after loading model
            if torch.cuda.is_available():
                model_memory = torch.cuda.memory_allocated() / (1024 ** 3)
                logger.info(f"GPU memory after loading model: {model_memory:.2f} GB")
                logger.info(f"Model memory delta: {model_memory - initial_memory:.2f} GB")

            start_time = time.time()

            # Perform inference
            answer = llm.answer_query(
                query=query,
                documents=doc_objects,
                metadata_filter=metadata_filter
            )

            inference_time = time.time() - start_time
            logger.info(f"Inference completed in {inference_time:.2f}s")

            # Extract source information
            sources = []
            for doc, score in doc_objects:
                source = {
                    "id": doc.metadata.get("id", ""),
                    "title": doc.metadata.get("title", "Unknown"),
                    "source_type": doc.metadata.get("source", "unknown"),
                    "url": doc.metadata.get("url"),
                    "relevance_score": score,
                }
                sources.append(source)

            # Prepare formatted documents for response
            formatted_documents = []
            for doc, score in doc_objects:
                formatted_doc = {
                    "id": doc.metadata.get("id", ""),
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "relevance_score": score,
                }
                formatted_documents.append(formatted_doc)

            # Clean up GPU memory after inference
            if torch.cuda.is_available():
                # Explicitly delete model to free memory
                del llm.model
                del llm.pipe
                del llm
                torch.cuda.empty_cache()
                final_memory = torch.cuda.memory_allocated() / (1024 ** 3)
                logger.info(f"Final GPU memory after cleanup: {final_memory:.2f} GB")

            # Update job with success result
            job_tracker.update_job_status(
                job_id,
                JobStatus.COMPLETED,
                result={
                    "query": query,
                    "answer": answer,
                    "documents": formatted_documents,
                    "metadata_filters_used": metadata_filter,
                    "execution_time": inference_time,
                    "sources": sources,
                }
            )

            return {
                "answer": answer,
                "documents": formatted_documents,
                "execution_time": inference_time
            }

    except TimeLimitExceeded:
        logger.error(f"Time limit exceeded for job {job_id}")
        job_tracker.update_job_status(
            job_id,
            JobStatus.TIMEOUT,
            error="Inference timeout exceeded"
        )
        raise
    except Exception as e:
        import traceback
        error_detail = f"Error performing LLM inference: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_detail)

        # Update job with error
        job_tracker.update_job_status(
            job_id,
            JobStatus.FAILED,
            error=error_detail
        )

        # Clean up GPU memory on error
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info(f"Cleaned up GPU memory after error")

        # Re-raise for dramatiq retry mechanism
        raise


# Now modify the get_vector_store function to use the preloaded model when available
def get_vector_store(force_cpu=False):
    """
    Get a vector store instance for background tasks with support for preloaded models.

    Args:
        force_cpu: Whether to force CPU usage for embeddings
    """
    global _PRELOADED_EMBEDDING_MODEL

    from src.core.vectorstore import QdrantStore
    from src.config.settings import settings
    import torch

    # Initialize qdrant client
    from qdrant_client import QdrantClient
    qdrant_client = QdrantClient(
        host=settings.qdrant_host,
        port=settings.qdrant_port,
    )

    # If we have a preloaded model and we're not forcing CPU, use the preloaded model
    if _PRELOADED_EMBEDDING_MODEL is not None and not force_cpu and worker_type != "cpu":
        logger.info("Using preloaded embedding model")
        return QdrantStore(
            client=qdrant_client,
            collection_name=settings.qdrant_collection,
            embedding_function=_PRELOADED_EMBEDDING_MODEL,
        )

    # If forcing CPU or on a CPU worker, use CPU model
    if force_cpu or worker_type == "cpu":
        logger.info("Using CPU embedding model")
        from langchain_huggingface import HuggingFaceEmbeddings
        embedding_function = HuggingFaceEmbeddings(
            model_name=settings.embedding_model_full_path,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"batch_size": settings.batch_size, "normalize_embeddings": True},
            cache_folder=os.path.join(settings.models_dir, settings.embedding_model_path)
        )
        return QdrantStore(
            client=qdrant_client,
            collection_name=settings.qdrant_collection,
            embedding_function=embedding_function,
        )

    # If we reach here, we need to create a new GPU model instance
    # This should only happen if preloading failed or was skipped
    logger.info("Creating new GPU embedding model instance")
    # For GPU workers, implement robust GPU handling
    if torch.cuda.is_available():
        # Maximum number of GPU attempts - but fail faster on model loading
        max_gpu_attempts = 2  # Reduced from 3 to fail faster
        gpu_attempts = 0

        while gpu_attempts < max_gpu_attempts:
            try:
                # Aggressive GPU memory cleanup
                torch.cuda.empty_cache()
                for i in range(torch.cuda.device_count()):
                    torch.cuda.synchronize(i)

                # Add a cooling-off period to allow GPU to stabilize, but shorter
                time.sleep(1)  # Reduced from 2s to make failure faster

                # Check GPU memory status and log it for debugging
                allocated = torch.cuda.memory_allocated() / (1024 ** 3)
                reserved = torch.cuda.memory_reserved() / (1024 ** 3)
                logger.info(
                    f"GPU memory before embedding model: Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")

                # Try to get embedding function on GPU - with timeout to fail faster
                from langchain_huggingface import HuggingFaceEmbeddings
                try:
                    # Try to import the model with a timeout to fail faster
                    import signal
                    import functools

                    # Define a timeout handler
                    class TimeoutError(Exception):
                        pass

                    def timeout_handler(signum, frame):
                        raise TimeoutError("Model loading timed out")

                    # Set timeout - 30 seconds should be enough for model loading
                    # If it takes longer, something is likely wrong
                    signal.signal(signal.SIGALRM, timeout_handler)
                    signal.alarm(30)

                    # Try to load the model
                    embedding_function = HuggingFaceEmbeddings(
                        model_name=settings.embedding_model_full_path,
                        model_kwargs={"device": settings.device},
                        encode_kwargs={"batch_size": settings.batch_size, "normalize_embeddings": True},
                        cache_folder=os.path.join(settings.models_dir, settings.embedding_model_path)
                    )

                    # Cancel the alarm if successful
                    signal.alarm(0)
                except TimeoutError:
                    logger.error("Embedding model loading timed out - failing fast")
                    raise RuntimeError("Embedding model loading timed out")

                logger.info("Successfully initialized GPU embedding model")

                # Initialize vector store with GPU embeddings
                return QdrantStore(
                    client=qdrant_client,
                    collection_name=settings.qdrant_collection,
                    embedding_function=embedding_function,
                )
            except Exception as e:
                gpu_attempts += 1
                logger.warning(f"GPU embedding attempt {gpu_attempts}/{max_gpu_attempts} failed: {str(e)}")

                # Fail fast on certain error types that suggest hardware issues
                if "CUDA out of memory" in str(e) or "CUDA error" in str(e) or "invalid argument" in str(e):
                    logger.error(f"Critical CUDA error detected, failing fast: {str(e)}")
                    # Don't retry on these fundamental errors
                    break

                # Try more aggressive CUDA reset approach for next attempt
                if gpu_attempts < max_gpu_attempts:
                    try:
                        # Force CUDA reset by creating and destroying a dummy tensor
                        dummy = torch.ones(1).cuda()
                        del dummy
                        torch.cuda.empty_cache()

                        # Wait between attempts, but not too long to fail fast
                        wait_time = 3 * gpu_attempts  # Reduced wait time
                        logger.info(f"Waiting {wait_time} seconds before next GPU attempt")
                        time.sleep(wait_time)
                    except Exception as reset_error:
                        logger.warning(f"Error during GPU reset: {str(reset_error)}")

        # If we've exhausted all GPU attempts, this is a critical error
        # Don't fall back to CPU for embedding - that would be too slow
        error_msg = f"Critical error: Failed to initialize GPU embedding model after {gpu_attempts} attempts"
        logger.error(error_msg)
        raise RuntimeError(error_msg)
    else:
        # No GPU available but we're on a GPU worker - this is unexpected
        error_msg = "Critical error: GPU worker but no CUDA available"
        logger.error(error_msg)
        raise RuntimeError(error_msg)


@dramatiq.actor(
    queue_name="gpu_tasks",
    max_retries=5,
    time_limit=3600000,
    min_backoff=60000,
    max_backoff=300000,
    store_results=True  # Enable result storage with Results middleware
)
def process_video_gpu(job_id: str, url: str, custom_metadata: Optional[Dict[str, Any]] = None):
    """Process a video using GPU for transcription and embedding with improved error handling."""
    try:
        # First, check if this job has already been completed
        if job_tracker.is_job_completed(job_id):
            logger.info(f"Job {job_id} has already been completed. Skipping processing.")
            return {"message": "Job already completed", "job_id": job_id, "status": "already_completed"}

        # Perform proactive GPU health check BEFORE trying to acquire the limiter
        is_healthy, health_message = check_gpu_health()
        if not is_healthy:
            # Don't even try to acquire the GPU limiter if the GPU is in a bad state
            error_msg = f"GPU health check failed: {health_message}"
            logger.error(error_msg)

            # Update job status to reflect GPU health issues
            job_tracker.update_job_status(
                job_id,
                JobStatus.FAILED,
                error=error_msg
            )

            # Raise an exception to trigger retry with backoff
            raise RuntimeError(error_msg)

        # GPU health check passed, now try to acquire the rate limiter
        logger.info(f"GPU health check passed. Attempting to acquire GPU limiter for job {job_id}")

        try:
            # Update job status to reflect waiting for GPU
            job_tracker.update_job_status(
                job_id,
                JobStatus.PROCESSING,
                result={"message": "Waiting for GPU resources to become available"}
            )

            # Attempt to acquire the limiter
            with gpu_limiter.acquire():
                # Once we have the limiter, check GPU health again - it might have changed
                is_still_healthy, health_message = check_gpu_health()
                if not is_still_healthy:
                    # GPU was healthy before, but is unhealthy now that we have the lock
                    error_msg = f"GPU health deteriorated while waiting for lock: {health_message}"
                    logger.error(error_msg)

                    # Update job status
                    job_tracker.update_job_status(
                        job_id,
                        JobStatus.FAILED,
                        error=error_msg
                    )

                    # Raise an exception to release the lock and trigger retry
                    raise RuntimeError(error_msg)

                # Update job status to processing
                job_tracker.update_job_status(
                    job_id,
                    JobStatus.PROCESSING,
                    result={"message": f"Processing video with GPU: {url}"}
                )
                logger.info(f"GPU resources acquired for job {job_id}")

                # Import here to avoid circular imports
                from src.core.document_processor import DocumentProcessor
                from src.core.video_transcriber import VideoTranscriber
                import torch

                # Advanced GPU preparation
                if torch.cuda.is_available():
                    # Log GPU state before processing
                    for i in range(torch.cuda.device_count()):
                        device_name = torch.cuda.get_device_name(i)
                        allocated = torch.cuda.memory_allocated(i) / (1024 ** 3)
                        reserved = torch.cuda.memory_reserved(i) / (1024 ** 3)
                        logger.info(
                            f"GPU {i} ({device_name}) before processing: Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")

                    # Reset CUDA context
                    torch.cuda.empty_cache()
                    for i in range(torch.cuda.device_count()):
                        torch.cuda.synchronize(i)
                    time.sleep(1)  # Short pause

                # Get vector store with improved GPU handling
                vector_store = get_vector_store(force_cpu=False)

                # Create transcriber with GPU
                transcriber = VideoTranscriber(
                    whisper_model_size=os.environ.get("WHISPER_MODEL_SIZE", "medium"),
                    device="cuda",
                    num_workers=1  # Use single worker thread to avoid oversubscription
                )

                # Initialize document processor
                processor = DocumentProcessor(
                    vector_store=vector_store,
                    video_transcriber=transcriber
                )

                # Process the video - this uses GPU for transcription and embedding
                document_ids = processor.process_video(url=url, custom_metadata=custom_metadata)

                # Clean up GPU memory
                if torch.cuda.is_available():
                    # Log GPU state after processing
                    for i in range(torch.cuda.device_count()):
                        allocated = torch.cuda.memory_allocated(i) / (1024 ** 3)
                        reserved = torch.cuda.memory_reserved(i) / (1024 ** 3)
                        logger.info(
                            f"GPU {i} after processing: Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")

                    # Thorough cleanup
                    torch.cuda.empty_cache()
                    for i in range(torch.cuda.device_count()):
                        torch.cuda.synchronize(i)

                # Update job with success result
                job_tracker.update_job_status(
                    job_id,
                    JobStatus.COMPLETED,
                    result={
                        "message": f"Successfully processed video using GPU: {url}",
                        "document_count": len(document_ids),
                        "document_ids": document_ids,
                    }
                )

                logger.info(f"Released GPU resources for job {job_id}")
                return {
                    "message": "Successfully processed video",
                    "document_count": len(document_ids),
                    "document_ids": document_ids,
                    "job_id": job_id,
                    "status": "completed"
                }

        except dramatiq.errors.RateLimitExceeded:
            # This will trigger a retry with backoff thanks to the actor decorator settings
            logger.warning(f"Rate limit exceeded for job {job_id}, will retry with backoff")
            # Update job status to indicate it's pending retry
            job_tracker.update_job_status(
                job_id,
                JobStatus.PENDING,
                result={"message": "Waiting in queue for GPU resources, will retry automatically"}
            )
            raise  # Raise to trigger dramatiq's built-in retry

    except TimeLimitExceeded:
        logger.error(f"Time limit exceeded for job {job_id}")
        job_tracker.update_job_status(
            job_id,
            JobStatus.TIMEOUT,
            error="Processing timeout exceeded"
        )
        raise
    except Exception as e:
        import traceback
        error_detail = f"Error processing video with GPU: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_detail)

        # Update job with error
        job_tracker.update_job_status(
            job_id,
            JobStatus.FAILED,
            error=error_detail
        )

        # Clean up GPU memory on error
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            for i in range(torch.cuda.device_count()):
                torch.cuda.synchronize(i)
            logger.info(f"Cleaned up GPU memory after error")

        # Re-raise for dramatiq retry mechanism
        raise