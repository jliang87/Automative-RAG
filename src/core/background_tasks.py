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

    # Modify the JobTracker class in background_tasks.py to track stage transitions

    def update_job_status(self, job_id: str, status: str, result: Any = None, error: str = None,
                          stage: str = None) -> None:
        """Update the status of a job, including stage transitions."""
        job_data_json = self.redis.hget(self.job_key, job_id)
        if not job_data_json:
            logger.warning(f"Job {job_id} not found when updating status to {status}")
            return

        job_data = json.loads(job_data_json)
        job_data["status"] = status
        job_data["updated_at"] = time.time()

        # Track stage transitions with timestamps
        if stage:
            # Initialize stage_history if it doesn't exist
            if "stage_history" not in job_data:
                job_data["stage_history"] = []

            # Add stage transition with timestamp
            current_time = time.time()
            job_data["stage_history"].append({
                "stage": stage,
                "started_at": current_time,
                "status": status
            })

            # Update current stage
            job_data["current_stage"] = stage
            job_data["stage_started_at"] = current_time

        # Add result and error
        if result is not None:
            if isinstance(result, (list, dict)):
                job_data["result"] = json.dumps(result)
            else:
                job_data["result"] = str(result)

        if error is not None:
            job_data["error"] = str(error)

        self.redis.hset(self.job_key, job_id, json.dumps(job_data))
        logger.info(f"Updated job {job_id} status to {status}" + (f", stage to {stage}" if stage else ""))

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
        if "result" in job_data and job_data["result"] and isinstance(job_data["result"], str) and job_data[
            "result"].startswith('{'):
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
            if "result" in job_data and job_data["result"] and isinstance(job_data["result"], str) and job_data[
                "result"].startswith('{'):
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


# Initialize the job tracker
job_tracker = JobTracker()


class PriorityQueueManager:
    """Manages task priorities across different queues."""

    def __init__(self, redis_client, priority_levels=None):
        self.redis = redis_client
        self.priority_queue_key = "priority_task_queue"
        self.active_task_key = "active_gpu_task"
        # Set priority levels - lower number = higher priority
        self.priority_levels = priority_levels or {
            "inference_tasks": 1,  # Highest priority
            "reranking_tasks": 2,
            "gpu_tasks": 3,
            "transcription_tasks": 4
        }

    def register_task(self, queue_name, task_id, metadata=None):
        """Register a task in the priority system."""
        priority = self.priority_levels.get(queue_name, 999)
        task_data = {
            "task_id": task_id,
            "queue_name": queue_name,
            "priority": priority,
            "registered_at": time.time(),
            "metadata": metadata or {}
        }

        # Add to sorted set with priority as score
        self.redis.zadd(self.priority_queue_key, {json.dumps(task_data): priority})
        logger.info(f"Registered task {task_id} from queue {queue_name} with priority {priority}")
        return task_id

    def get_active_task(self):
        """Get the currently active GPU task, if any."""
        active_task = self.redis.get(self.active_task_key)
        if active_task:
            return json.loads(active_task)
        return None

    def mark_task_active(self, task_data):
        """Mark a task as currently active on GPU."""
        self.redis.set(self.active_task_key, json.dumps(task_data), ex=3600)  # 1 hour expiry as safety
        logger.info(f"Marked task {task_data.get('task_id')} as active on GPU")

    def mark_task_completed(self, task_id):
        """Mark a task as completed and remove from priority queue."""
        # Remove from active task if it's this task
        active_task = self.get_active_task()
        if active_task and active_task.get("task_id") == task_id:
            self.redis.delete(self.active_task_key)
            logger.info(f"Removed task {task_id} from active GPU task")

        # Remove from priority queue by searching for task with this ID
        all_tasks = self.redis.zrange(self.priority_queue_key, 0, -1, withscores=True)
        for task_json, _ in all_tasks:
            task = json.loads(task_json)
            if task.get("task_id") == task_id:
                self.redis.zrem(self.priority_queue_key, task_json)
                logger.info(f"Removed task {task_id} from priority queue")
                break

    def get_next_task(self):
        """Get the highest priority task that should run next."""
        # Get the task with the lowest score (highest priority)
        tasks = self.redis.zrange(self.priority_queue_key, 0, 0, withscores=True)
        if not tasks:
            return None

        task_json, priority = tasks[0]
        return json.loads(task_json)

    def can_run_task(self, queue_name, task_id):
        """Determine if a task can run according to priorities, with anti-starvation measures."""
        # Get the task details
        task_details = None
        all_tasks = self.redis.zrange(self.priority_queue_key, 0, -1, withscores=True)
        for task_json, _ in all_tasks:
            task = json.loads(task_json)
            if task.get("task_id") == task_id:
                task_details = task
                break

        if not task_details:
            # Task not found in queue
            return False

        # For inference tasks, always allow them to run immediately
        # This ensures they have absolute priority
        if queue_name == "inference_tasks":
            active_task = self.get_active_task()
            # If no active task, or active task is not inference, allow to run
            if not active_task or active_task.get("queue_name") != "inference_tasks":
                return True

        # Check if task has been waiting too long (5 minutes)
        current_time = time.time()
        registered_time = task_details.get("registered_at", current_time)

        if current_time - registered_time > 300:  # 5 minutes
            # Task has waited too long, allow it to run regardless of priority
            logger.info(f"Task {task_id} has waited over 5 minutes, allowing to run")
            return True

        # If no active task, can run if it's the highest priority task
        active_task = self.get_active_task()
        if not active_task:
            # Check if this is the highest priority task
            next_task = self.get_next_task()

            # If no tasks in queue or this is the highest priority task, run it
            if not next_task or next_task.get("task_id") == task_id:
                return True

            # If this task's queue has higher priority than next task, run it
            task_priority = self.priority_levels.get(queue_name, 999)
            next_priority = next_task.get("priority", 999)
            return task_priority <= next_priority

        # If there's an active task of higher priority, wait
        active_priority = active_task.get("priority", 999)
        task_priority = self.priority_levels.get(queue_name, 999)

        # Only allow if this task has strictly higher priority than the running task
        return task_priority < active_priority


# Initialize the priority queue manager
priority_queue = PriorityQueueManager(redis_client=redis_broker.client)

# Add middleware
redis_broker.add_middleware(Callbacks())
redis_broker.add_middleware(AgeLimit())
redis_broker.add_middleware(Retries(max_retries=3))
# Add Results middleware
redis_broker.add_middleware(Results(backend=results_backend))

dramatiq.set_broker(redis_broker)

# Global variables to hold the preloaded models
_PRELOADED_EMBEDDING_MODEL = None
_PRELOADED_LLM_MODEL = None
_PRELOADED_COLBERT_RERANKER = None
_PRELOADED_WHISPER_MODEL = None


# Create worker middleware to run the preload function
class WorkerSetupMiddleware:
    """Middleware to run setup code when worker processes boot."""

    def before_worker_boot(self, broker, worker):
        """Run setup code before the worker boots."""
        logger.info(f"Initializing worker {worker.worker_id} of type {worker_type}")

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


# Add the middleware to the broker
worker_setup = WorkerSetupMiddleware()
redis_broker.add_middleware(worker_setup)


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


def preload_embedding_model():
    """
    Preload the embedding model at worker startup to avoid loading it for each job.
    """
    global _PRELOADED_EMBEDDING_MODEL

    # Skip if already loaded or if we're not on the right worker type
    if _PRELOADED_EMBEDDING_MODEL is not None or worker_type != "gpu-embedding":
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
                    f"GPU {i} ({device_name}) before embedding model loading: Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")

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
                logger.info(
                    f"GPU {i} after embedding model loading: Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")

        logger.info(f"Successfully preloaded embedding model with dimension {embedding_dim}")
    except Exception as e:
        logger.error(f"Failed to preload embedding model: {str(e)}")


def preload_llm_model():
    """
    Preload the LLM model at worker startup.
    """
    global _PRELOADED_LLM_MODEL

    # Skip if already loaded or if we're not on the right worker type
    if _PRELOADED_LLM_MODEL is not None or worker_type != "gpu-inference":
        return

    logger.info(f"Preloading LLM model {settings.default_llm_model} on {settings.device}")

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
                    f"GPU {i} ({device_name}) before LLM model loading: Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")

        # Import here to avoid circular imports
        from src.core.llm import LocalLLM

        # Load the model
        _PRELOADED_LLM_MODEL = LocalLLM(
            model_name=settings.default_llm_model,
            device=settings.device,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
            use_4bit=settings.llm_use_4bit,
            use_8bit=settings.llm_use_8bit,
            torch_dtype=torch.float16 if settings.use_fp16 and settings.device.startswith("cuda") else None
        )

        # Test the LLM with a simple prompt to ensure it works
        test_response = _PRELOADED_LLM_MODEL.answer_query(
            query="What is 2+2?",
            documents=[]
        )

        # Log GPU memory status after loading
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                allocated = torch.cuda.memory_allocated(i) / (1024 ** 3)
                reserved = torch.cuda.memory_reserved(i) / (1024 ** 3)
                logger.info(
                    f"GPU {i} after LLM model loading: Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")

        logger.info(f"Successfully preloaded LLM model")
    except Exception as e:
        logger.error(f"Failed to preload LLM model: {str(e)}")


def preload_colbert_reranker():
    """
    Preload the ColBERT and BGE reranking models.
    """
    global _PRELOADED_COLBERT_RERANKER

    # Skip if already loaded or if we're not on the right worker type
    if _PRELOADED_COLBERT_RERANKER is not None or worker_type != "gpu-inference":
        return

    logger.info(f"Preloading ColBERT and BGE reranking models on {settings.device}")

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
                    f"GPU {i} ({device_name}) before reranker loading: Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")

        # Import here to avoid circular imports
        from src.core.colbert_reranker import ColBERTReranker

        # Load the reranking models
        _PRELOADED_COLBERT_RERANKER = ColBERTReranker(
            model_name=settings.default_colbert_model,
            device=settings.device,
            batch_size=settings.colbert_batch_size,
            use_fp16=settings.use_fp16,
            use_bge_reranker=settings.use_bge_reranker,
            colbert_weight=settings.colbert_weight,
            bge_weight=settings.bge_weight,
            bge_model_name=settings.default_bge_reranker_model
        )

        # Test the reranker with a simple query and document to ensure it works
        from langchain_core.documents import Document
        test_doc = Document(page_content="This is a test document.", metadata={})
        test_results = _PRELOADED_COLBERT_RERANKER.rerank("test query", [test_doc], 1)

        # Log GPU memory status after loading
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                allocated = torch.cuda.memory_allocated(i) / (1024 ** 3)
                reserved = torch.cuda.memory_reserved(i) / (1024 ** 3)
                logger.info(
                    f"GPU {i} after reranker loading: Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")

        logger.info(f"Successfully preloaded ColBERT and BGE reranking models")
    except Exception as e:
        logger.error(f"Failed to preload reranking models: {str(e)}")


def preload_whisper_model():
    """
    Preload the Whisper model for transcription.
    """
    global _PRELOADED_WHISPER_MODEL

    # Skip if already loaded or if we're not on the right worker type
    if _PRELOADED_WHISPER_MODEL is not None or worker_type != "gpu-whisper":
        return

    logger.info(f"Preloading Whisper model {settings.whisper_model_size} on {settings.device}")

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
                    f"GPU {i} ({device_name}) before Whisper model loading: Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")

        # Import the WhisperModel
        from faster_whisper import WhisperModel

        # Get the model path
        model_path = settings.whisper_model_full_path if hasattr(settings,
                                                                 'whisper_model_full_path') else settings.whisper_model_size

        # Load the Whisper model
        _PRELOADED_WHISPER_MODEL = WhisperModel(
            model_path,
            device=settings.device,
            compute_type="float16" if settings.use_fp16 else "float32",
            cpu_threads=4,  # Use multiple CPU threads for pre/post-processing
            num_workers=2  # Number of workers for parallel processing
        )

        # Log GPU memory status after loading
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                allocated = torch.cuda.memory_allocated(i) / (1024 ** 3)
                reserved = torch.cuda.memory_reserved(i) / (1024 ** 3)
                logger.info(
                    f"GPU {i} after Whisper model loading: Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")

        logger.info(f"Successfully preloaded Whisper model")
    except Exception as e:
        logger.error(f"Failed to preload Whisper model: {str(e)}")


def get_vector_store():
    """
    Get a vector store instance for background tasks with support for preloaded models.
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

    # If we have a preloaded model and we're on the embedding worker, use it
    if _PRELOADED_EMBEDDING_MODEL is not None and worker_type == "gpu-embedding":
        logger.info("Using preloaded embedding model for vector store")
        return QdrantStore(
            client=qdrant_client,
            collection_name=settings.qdrant_collection,
            embedding_function=_PRELOADED_EMBEDDING_MODEL,
        )

    # Create a new embedding model instance
    logger.info("Creating new embedding model instance for vector store")
    from langchain_huggingface import HuggingFaceEmbeddings

    # Determine the device (default to CPU for non-embedding workers)
    device = settings.device if worker_type == "gpu-embedding" else "cpu"

    embedding_function = HuggingFaceEmbeddings(
        model_name=settings.embedding_model_full_path,
        model_kwargs={"device": device},
        encode_kwargs={"batch_size": settings.batch_size, "normalize_embeddings": True},
        cache_folder=os.path.join(settings.models_dir, settings.embedding_model_path)
    )

    return QdrantStore(
        client=qdrant_client,
        collection_name=settings.qdrant_collection,
        embedding_function=embedding_function,
    )


# Define a periodic task to clean up stalled tasks in the priority system
@dramatiq.actor(queue_name="system_tasks")
def cleanup_stalled_tasks():
    """Clean up stalled tasks in the priority system."""
    redis_client = redis_broker.client

    # Check active task
    active_task_json = redis_client.get("active_gpu_task")
    if not active_task_json:
        logger.info("No active GPU task to clean up")
        return

    active_task = json.loads(active_task_json)
    task_id = active_task.get("task_id")
    job_id = active_task.get("job_id", "unknown")

    # Get when this task was marked active
    task_age = time.time() - active_task.get("registered_at", time.time())

    # If task has been active for more than 30 minutes, it's likely stalled
    if task_age > 1800:  # 30 minutes
        logger.warning(f"Found stalled task {task_id} for job {job_id} (active for {task_age:.2f} seconds)")

        # Check if the job still exists and its status
        job_data = job_tracker.get_job(job_id)
        if not job_data or job_data.get("status") in ["completed", "failed", "timeout"]:
            # Job is no longer running or doesn't exist, but task is still marked active
            redis_client.delete("active_gpu_task")
            logger.warning(f"Cleaned up stalled task: {task_id} for job {job_id}")

            # Try to update job status if it exists but isn't marked as failed
            if job_data and job_data.get("status") == "processing":
                job_tracker.update_job_status(
                    job_id,
                    JobStatus.FAILED,
                    error="Task appears to be stalled and was terminated"
                )
        else:
            # Job is still in processing state - check GPU health
            is_healthy, status_message = check_gpu_health()
            if not is_healthy:
                # GPU is unhealthy, terminate the task
                redis_client.delete("active_gpu_task")
                logger.error(f"Terminating stalled task due to GPU health issues: {status_message}")

                # Update job status
                job_tracker.update_job_status(
                    job_id,
                    JobStatus.FAILED,
                    error=f"Task terminated due to GPU health issues: {status_message}"
                )
            else:
                # GPU is healthy but task is still running - log but don't terminate yet
                logger.warning(f"Found long-running task {task_id} for job {job_id}, but GPU appears healthy")


# Specialized actor for LLM inference with priority handling
@dramatiq.actor(
    queue_name="inference_tasks",
    max_retries=3,
    time_limit=300000,  # 5 minutes
    min_backoff=10000,  # 10 seconds
    max_backoff=300000  # 5 minutes
)
def perform_llm_inference(job_id: str, query: str, documents: List[Dict], metadata_filter: Optional[Dict] = None):
    """Perform LLM inference using reranking and local LLM with priority handling."""
    global _PRELOADED_LLM_MODEL, _PRELOADED_COLBERT_RERANKER

    try:
        # Register task with the priority system
        task_id = f"inference_{job_id}"
        priority_queue.register_task("inference_tasks", task_id, {"job_id": job_id, "query": query})

        # Update job status to show we're queued
        job_tracker.update_job_status(
            job_id,
            JobStatus.PROCESSING,
            result={"message": "In priority queue for inference resources"}
        )

        # Wait for priority system to allow this task to run
        # For inference tasks, can_run_task will always return True if there's no active inference task
        wait_start = time.time()
        while not priority_queue.can_run_task("inference_tasks", task_id):
            # Log every 10 seconds of waiting
            if int(time.time() - wait_start) % 10 == 0:
                logger.info(f"Inference task {task_id} waiting in priority queue")
            time.sleep(0.5)

        # Mark this task as now active on GPU
        priority_queue.mark_task_active({
            "task_id": task_id,
            "queue_name": "inference_tasks",
            "priority": 1,
            "job_id": job_id,
            "registered_at": time.time()
        })

        logger.info(f"Starting inference for job {job_id} with priority handling")

        try:
            # Update job status to processing
            job_tracker.update_job_status(
                job_id,
                JobStatus.PROCESSING,
                result={"message": "Processing query with LLM"}
            )

            # Convert document dictionaries back to Document objects
            from langchain_core.documents import Document
            doc_objects = []
            for doc_dict in documents:
                doc = Document(
                    page_content=doc_dict["content"],
                    metadata=doc_dict.get("metadata", {})
                )
                score = doc_dict.get("relevance_score", 0)
                doc_objects.append((doc, score))

            # Perform reranking if available
            reranker = _PRELOADED_COLBERT_RERANKER
            if reranker is not None:
                logger.info(f"Reranking {len(doc_objects)} documents")
                reranked_docs = reranker.rerank(query, [doc for doc, _ in doc_objects], 5)
            else:
                logger.warning("Reranker not available, using original document order")
                reranked_docs = doc_objects[:5]

            # Use preloaded LLM model if available
            llm = _PRELOADED_LLM_MODEL
            if llm is None:
                logger.warning("Preloaded LLM not available, loading on demand")
                # Import here to avoid circular imports
                from src.core.llm import LocalLLM
                import torch

                # Initialize LLM with memory efficient settings
                llm = LocalLLM(
                    model_name=settings.default_llm_model,
                    device=settings.device,
                    temperature=settings.llm_temperature,
                    max_tokens=settings.llm_max_tokens,
                    use_4bit=settings.llm_use_4bit,
                    use_8bit=settings.llm_use_8bit,
                    torch_dtype=torch.float16 if settings.use_fp16 and settings.device.startswith("cuda") else None
                )

            # Perform inference
            start_time = time.time()
            answer = llm.answer_query(
                query=query,
                documents=reranked_docs,
                metadata_filter=metadata_filter
            )
            inference_time = time.time() - start_time
            logger.info(f"Inference completed in {inference_time:.2f}s")

            # Prepare formatted documents for response
            formatted_documents = []
            for doc, score in reranked_docs:
                formatted_doc = {
                    "id": doc.metadata.get("id", ""),
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "relevance_score": score,
                }
                formatted_documents.append(formatted_doc)

            # Update job with success result
            job_tracker.update_job_status(
                job_id,
                JobStatus.COMPLETED,
                result={
                    "query": query,
                    "answer": answer,
                    "documents": formatted_documents,
                    "metadata_filters_used": metadata_filter,
                    "execution_time": inference_time
                }
            )

            return {
                "answer": answer,
                "documents": formatted_documents,
                "execution_time": inference_time
            }
        finally:
            # Always mark task as completed, even if it failed
            priority_queue.mark_task_completed(task_id)

            # Clear cache if needed
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

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

        # Make sure to mark task as completed even in error case
        priority_queue.mark_task_completed(f"inference_{job_id}")

        # Re-raise for dramatiq retry mechanism
        raise


# Specialized actor for GPU-based embedding with priority handling
@dramatiq.actor(
    queue_name="gpu_tasks",
    max_retries=3,
    time_limit=600000,  # 10 minutes
    min_backoff=10000,
    max_backoff=300000
)
def generate_embeddings_gpu(job_id: str, chunks: List[Dict], metadata: Optional[Dict] = None):
    """Generate embeddings with priority handling."""
    try:
        # Register task with the priority system
        task_id = f"embedding_{job_id}"
        priority_queue.register_task("gpu_tasks", task_id, {"job_id": job_id})

        # Update job status to show we're queued
        job_tracker.update_job_status(
            job_id,
            JobStatus.PROCESSING,
            result={"message": "In priority queue for GPU resources"}
        )

        # Wait for priority system to allow this task to run
        wait_start = time.time()
        while not priority_queue.can_run_task("gpu_tasks", task_id):
            # Log every 30 seconds of waiting
            if int(time.time() - wait_start) % 30 == 0:
                logger.info(f"Embedding task {task_id} waiting in priority queue")
            time.sleep(1)

        # Mark this task as now active on GPU
        priority_queue.mark_task_active({
            "task_id": task_id,
            "queue_name": "gpu_tasks",
            "priority": 3,
            "job_id": job_id,
            "registered_at": time.time()
        })

        logger.info(f"Starting embedding generation for job {job_id} with priority handling")

        try:
            # Update job status to processing
            job_tracker.update_job_status(
                job_id,
                JobStatus.PROCESSING,
                result={"message": "Generating embeddings"}
            )

            # Convert chunk dictionaries to Document objects
            from langchain_core.documents import Document
            documents = []
            for chunk in chunks:
                doc = Document(
                    page_content=chunk["content"],
                    metadata=chunk.get("metadata", {})
                )
                documents.append(doc)

            # Get vector store with preloaded embeddings
            vector_store = get_vector_store()

            # Generate embeddings and add to vector store
            start_time = time.time()
            doc_ids = vector_store.add_documents(documents)
            embedding_time = time.time() - start_time

            logger.info(f"Generated embeddings for {len(documents)} documents in {embedding_time:.2f}s")

            # Update job status upon completion
            job_tracker.update_job_status(
                job_id,
                JobStatus.COMPLETED,
                result={
                    "message": f"Generated embeddings for {len(documents)} documents",
                    "document_count": len(documents),
                    "document_ids": doc_ids,
                    "execution_time": embedding_time
                }
            )

            return {
                "document_count": len(documents),
                "document_ids": doc_ids,
                "execution_time": embedding_time
            }
        finally:
            # Always mark task as completed, even if it failed
            priority_queue.mark_task_completed(task_id)

            # Clear cache if needed
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    except Exception as e:
        import traceback
        error_detail = f"Error generating embeddings: {str(e)}\n{traceback.format_exc()}"
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

        # Make sure to mark task as completed even in error case
        priority_queue.mark_task_completed(f"embedding_{job_id}")

        # Re-raise for dramatiq retry mechanism
        raise


# Specialized actor for Whisper transcription with priority handling
@dramatiq.actor(
    queue_name="transcription_tasks",
    max_retries=3,
    time_limit=3600000,  # 1 hour for long videos
    min_backoff=10000,
    max_backoff=300000
)
def transcribe_video_gpu(job_id: str, media_path: str):
    """Process a video using GPU-accelerated Whisper transcription with priority handling."""
    global _PRELOADED_WHISPER_MODEL

    try:
        # Register task with the priority system
        task_id = f"transcription_{job_id}"
        priority_queue.register_task("transcription_tasks", task_id, {"job_id": job_id, "media_path": media_path})

        # Update job status to show we're queued
        job_tracker.update_job_status(
            job_id,
            JobStatus.PROCESSING,
            result={"message": "In priority queue for GPU transcription resources"}
        )

        # Wait for priority system to allow this task to run
        wait_start = time.time()
        while not priority_queue.can_run_task("transcription_tasks", task_id):
            # Log every 60 seconds of waiting
            if int(time.time() - wait_start) % 60 == 0:
                logger.info(f"Transcription task {task_id} waiting in priority queue")
            time.sleep(1)

        # Mark this task as now active on GPU
        priority_queue.mark_task_active({
            "task_id": task_id,
            "queue_name": "transcription_tasks",
            "priority": 4,
            "job_id": job_id,
            "registered_at": time.time()
        })

        logger.info(f"Starting transcription for job {job_id} with priority handling")

        try:
            # Update job status to processing
            job_tracker.update_job_status(
                job_id,
                JobStatus.PROCESSING,
                result={"message": "Transcribing video"}
            )

            # Use preloaded Whisper model if available
            whisper_model = _PRELOADED_WHISPER_MODEL
            if whisper_model is None:
                logger.warning("Preloaded Whisper model not available, loading on demand")
                from faster_whisper import WhisperModel

                # Get the model path
                model_path = settings.whisper_model_full_path if hasattr(settings,
                                                                         'whisper_model_full_path') else settings.whisper_model_size

                # Load the Whisper model
                whisper_model = WhisperModel(
                    model_path,
                    device=settings.device,
                    compute_type="float16" if settings.use_fp16 else "float32"
                )

            # Perform transcription
            start_time = time.time()
            segments, info = whisper_model.transcribe(
                media_path,
                beam_size=5,
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=500)
            )

            # Collect all segments
            all_text = [segment.text for segment in segments]
            transcript = " ".join(all_text)

            # Apply Chinese conversion if needed
            if info.language == "zh" and hasattr(settings, 'chinese_converter') and settings.chinese_converter:
                import opencc
                converter = opencc.OpenCC('t2s')
                transcript = converter.convert(transcript)

            transcription_time = time.time() - start_time

            logger.info(f"Transcription completed in {transcription_time:.2f}s, detected language: {info.language}")

            # Update job with success result
            job_tracker.update_job_status(
                job_id,
                JobStatus.COMPLETED,
                result={
                    "transcript": transcript,
                    "language": info.language,
                    "duration": info.duration,
                    "processing_time": transcription_time
                }
            )

            # Chain to process_transcript actor for embedding
            process_transcript.send(job_id, transcript, info.language)

            return transcript

        finally:
            # Always mark task as completed, even if it failed
            priority_queue.mark_task_completed(task_id)

            # Clear cache if needed
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    except Exception as e:
        import traceback
        error_detail = f"Error performing transcription: {str(e)}\n{traceback.format_exc()}"
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

        # Make sure to mark task as completed even in error case
        priority_queue.mark_task_completed(f"transcription_{job_id}")

        # Re-raise for dramatiq retry mechanism
        raise


# Actor for processing text chunks (CPU task)
@dramatiq.actor(queue_name="cpu_tasks")
def process_text(job_id: str, text: str, metadata: Dict[str, Any] = None):
    """Process text and prepare for embedding."""
    try:
        logger.info(f"Processing text for job {job_id}")

        # Update job status
        job_tracker.update_job_status(
            job_id,
            JobStatus.PROCESSING,
            result={"message": "Processing text"}
        )

        # Split text into chunks
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )

        chunks = text_splitter.split_text(text)
        logger.info(f"Split text into {len(chunks)} chunks")

        # Convert chunks to documents with metadata
        from langchain_core.documents import Document
        documents = []
        for i, chunk_text in enumerate(chunks):
            # Create a document with metadata
            doc = Document(
                page_content=chunk_text,
                metadata={
                    "chunk_id": i,
                    "source": "manual",
                    "source_id": job_id,
                    **metadata
                }
            )
            documents.append(doc)

        # Convert documents to format for embedding
        document_dicts = []
        for doc in documents:
            document_dicts.append({
                "content": doc.page_content,
                "metadata": doc.metadata
            })

        # Chain to embedding task
        embedding_job_id = f"{job_id}_embed"
        job_tracker.create_job(
            job_id=embedding_job_id,
            job_type="embedding",
            metadata={
                "parent_job_id": job_id,
                "chunk_count": len(document_dicts)
            }
        )

        # Send to embedding worker
        generate_embeddings_gpu.send(embedding_job_id, document_dicts, metadata)

        # Update original job status
        job_tracker.update_job_status(
            job_id,
            JobStatus.PROCESSING,
            result={
                "message": "Text processed, embedding in progress",
                "chunk_count": len(chunks),
                "embedding_job_id": embedding_job_id
            }
        )

        return {
            "chunk_count": len(chunks),
            "embedding_job_id": embedding_job_id
        }
    except Exception as e:
        import traceback
        error_detail = f"Error processing text: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_detail)

        # Update job with error
        job_tracker.update_job_status(
            job_id,
            JobStatus.FAILED,
            error=error_detail
        )

        # Re-raise for dramatiq retry mechanism
        raise


# Actor for processing transcript and adding to vector store
@dramatiq.actor(queue_name="cpu_tasks")
def process_transcript(job_id: str, transcript: str, language: str):
    """Process transcript and prepare for embedding."""
    try:
        logger.info(f"Processing transcript for job {job_id}")

        # Update job status
        job_tracker.update_job_status(
            job_id,
            JobStatus.PROCESSING,
            result={"message": "Processing transcript"}
        )

        # Get job metadata
        job_data = job_tracker.get_job(job_id)
        if not job_data:
            raise ValueError(f"Job {job_id} not found")

        metadata = job_data.get("metadata", {})

        # Split transcript into chunks
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )

        chunks = text_splitter.split_text(transcript)
        logger.info(f"Split transcript into {len(chunks)} chunks")

        # Convert chunks to documents with metadata
        from langchain_core.documents import Document
        documents = []
        for i, chunk_text in enumerate(chunks):
            # Create document metadata
            doc_metadata = {
                "chunk_id": i,
                "language": language,
                "source": "video",
                "source_id": job_id,
            }

            # Add any metadata from the job
            if "url" in metadata:
                doc_metadata["url"] = metadata["url"]
            if "platform" in metadata:
                doc_metadata["platform"] = metadata["platform"]

            # Create document
            doc = Document(
                page_content=chunk_text,
                metadata=doc_metadata
            )
            documents.append(doc)

        # Convert documents to format for embedding
        document_dicts = []
        for doc in documents:
            document_dicts.append({
                "content": doc.page_content,
                "metadata": doc.metadata
            })

        # Chain to embedding task
        embedding_job_id = f"{job_id}_embed"
        job_tracker.create_job(
            job_id=embedding_job_id,
            job_type="embedding",
            metadata={
                "parent_job_id": job_id,
                "chunk_count": len(document_dicts)
            }
        )

        # Send to embedding worker
        generate_embeddings_gpu.send(embedding_job_id, document_dicts, metadata)

        # Update original job status
        job_tracker.update_job_status(
            job_id,
            JobStatus.PROCESSING,
            result={
                "message": "Transcript processed, embedding in progress",
                "chunk_count": len(chunks),
                "embedding_job_id": embedding_job_id
            }
        )

        return {
            "chunk_count": len(chunks),
            "embedding_job_id": embedding_job_id
        }
    except Exception as e:
        import traceback
        error_detail = f"Error processing transcript: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_detail)

        # Update job with error
        job_tracker.update_job_status(
            job_id,
            JobStatus.FAILED,
            error=error_detail
        )

        # Re-raise for dramatiq retry mechanism
        raise


# Actor for processing PDFs with OCR on CPU
@dramatiq.actor(queue_name="cpu_tasks")
def process_pdf_cpu(job_id: str, file_path: str, custom_metadata: Optional[Dict[str, Any]] = None):
    """Process a PDF file using CPU for OCR and text extraction."""
    try:
        logger.info(f"Processing PDF {file_path} for job {job_id}")

        # Update job status
        job_tracker.update_job_status(
            job_id,
            JobStatus.PROCESSING,
            result={"message": "Processing PDF"}
        )

        # Import PDF loader
        from src.core.pdf_loader import PDFLoader

        # Create PDF loader (on CPU only)
        pdf_loader = PDFLoader(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            device="cpu",
            use_ocr=settings.use_pdf_ocr,
            ocr_languages=settings.ocr_languages
        )

        # Process PDF
        start_time = time.time()
        documents = pdf_loader.process_pdf(
            file_path=file_path,
            custom_metadata=custom_metadata,
        )
        processing_time = time.time() - start_time

        logger.info(f"Processed PDF into {len(documents)} documents in {processing_time:.2f}s")

        # Convert documents to format for embedding
        document_dicts = []
        for doc in documents:
            document_dicts.append({
                "content": doc.page_content,
                "metadata": doc.metadata
            })

        # Chain to embedding task
        embedding_job_id = f"{job_id}_embed"
        job_tracker.create_job(
            job_id=embedding_job_id,
            job_type="embedding",
            metadata={
                "parent_job_id": job_id,
                "file_path": file_path,
                "chunk_count": len(document_dicts)
            }
        )

        # Send to embedding worker
        generate_embeddings_gpu.send(embedding_job_id, document_dicts, custom_metadata)

        # Update original job status
        job_tracker.update_job_status(
            job_id,
            JobStatus.PROCESSING,
            result={
                "message": "PDF processed, embedding in progress",
                "chunk_count": len(documents),
                "processing_time": processing_time,
                "embedding_job_id": embedding_job_id
            }
        )

        return {
            "chunk_count": len(documents),
            "processing_time": processing_time,
            "embedding_job_id": embedding_job_id
        }
    except Exception as e:
        import traceback
        error_detail = f"Error processing PDF: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_detail)

        # Update job with error
        job_tracker.update_job_status(
            job_id,
            JobStatus.FAILED,
            error=error_detail
        )

        # Re-raise for dramatiq retry mechanism
        raise


# Utility actor to get priority queue status
@dramatiq.actor(queue_name="system_tasks")
def get_priority_queue_status():
    """Get the status of the priority queue system."""
    redis_client = redis_broker.client

    # Get all tasks in priority queue
    all_tasks = redis_client.zrange(
        "priority_task_queue", 0, -1,
        withscores=True
    )

    # Get active task
    active_task_json = redis_client.get("active_gpu_task")
    active_task = json.loads(active_task_json) if active_task_json else None

    # Compile stats
    stats = {
        "total_tasks": len(all_tasks),
        "tasks_by_priority": {},
        "tasks_by_queue": {},
        "active_task": active_task,
        "timestamp": time.time()
    }

    # Group tasks by priority and queue
    for task_json, priority in all_tasks:
        task = json.loads(task_json)
        queue_name = task.get("queue_name", "unknown")

        # Count by priority
        if priority not in stats["tasks_by_priority"]:
            stats["tasks_by_priority"][priority] = 0
        stats["tasks_by_priority"][priority] += 1

        # Count by queue
        if queue_name not in stats["tasks_by_queue"]:
            stats["tasks_by_queue"][queue_name] = 0
        stats["tasks_by_queue"][queue_name] += 1

    return stats


@dramatiq.actor(queue_name="system_tasks")
def cleanup_old_jobs(retention_days=7):
    """Clean up old completed jobs that are beyond the retention period."""
    job_key = "rag_system:jobs"
    all_jobs = redis_client.hgetall(job_key)

    # Get current time minus retention period
    cutoff_time = time.time() - (retention_days * 24 * 60 * 60)
    deleted_count = 0

    # Check each job
    for job_id, job_data_json in all_jobs.items():
        try:
            job_data = json.loads(job_data_json)
            job_status = job_data.get("status", "")
            created_at = job_data.get("created_at", 0)

            # Delete if it's old and completed/failed
            if created_at < cutoff_time and job_status in ["completed", "failed", "timeout"]:
                redis_client.hdel(job_key, job_id)
                deleted_count += 1

                # Log every 50 deletions to avoid excessive logging
                if deleted_count % 50 == 0:
                    logger.info(f"Deleted {deleted_count} old jobs so far")
        except Exception as e:
            logger.warning(f"Error processing job {job_id}: {str(e)}")

    logger.info(f"Old job cleanup completed: deleted {deleted_count} jobs")
    return deleted_count


@dramatiq.actor(queue_name="system_tasks")
def check_priority_queue_health():
    """Perform a health check on the priority queue system."""
    redis_client = redis_broker.client

    # Check for inconsistencies in the priority queue
    active_task_json = redis_client.get("active_gpu_task")
    active_task = json.loads(active_task_json) if active_task_json else None

    # Get the priority queue
    queue_tasks = redis_client.zrange("priority_task_queue", 0, -1, withscores=True)

    # Check if there's an active task but no GPU worker is running
    if active_task:
        # Check how long the task has been active
        task_age = time.time() - active_task.get("registered_at", time.time())

        # If task has been active too long, check if the worker is still running
        if task_age > 900:  # 15 minutes
            task_id = active_task.get("task_id")
            job_id = active_task.get("job_id")

            # Check if GPU is healthy
            is_healthy, _ = check_gpu_health()

            if not is_healthy:
                # GPU is unhealthy but there's an active task - reset it
                logger.warning(f"GPU unhealthy but task {task_id} is marked active. Resetting priority system.")
                redis_client.delete("active_gpu_task")

                # Mark the job as failed
                if job_id:
                    job_tracker.update_job_status(
                        job_id,
                        JobStatus.FAILED,
                        error="Task reset due to GPU health issues"
                    )

    # Check if there are tasks in queue but no active task
    if not active_task and queue_tasks:
        logger.info("Priority queue has tasks but no active task - system may need attention")


@dramatiq.actor(queue_name="system_tasks")
def monitor_gpu_memory():
    """Monitor GPU memory usage and optimize when needed."""
    if not torch.cuda.is_available():
        return

    # Get memory usage for each GPU
    for i in range(torch.cuda.device_count()):
        # Get memory stats
        total_memory = torch.cuda.get_device_properties(i).total_memory
        allocated_memory = torch.cuda.memory_allocated(i)
        reserved_memory = torch.cuda.memory_reserved(i)

        # Calculate free memory
        free_memory = total_memory - reserved_memory
        free_percentage = (free_memory / total_memory) * 100

        # Log memory status
        logger.info(
            f"GPU {i} memory: {allocated_memory / 1e9:.2f} GB allocated, {free_memory / 1e9:.2f} GB free ({free_percentage:.1f}%)")

        # If memory usage is too high, trigger cache clearing
        if free_percentage < 20:  # Less than 20% free memory
            logger.warning(f"GPU {i} memory is running low ({free_percentage:.1f}% free). Clearing cache.")
            torch.cuda.empty_cache()


@dramatiq.actor(queue_name="system_tasks")
def collect_system_statistics():
    """Collect and store system performance statistics."""
    stats = {
        "timestamp": time.time(),
        "jobs": {
            "total": redis_client.hlen("rag_system:jobs"),
            "pending": 0,
            "processing": 0,
            "completed": 0,
            "failed": 0
        },
        "queues": {},
        "gpu": {}
    }

    # Count jobs by status
    all_jobs = redis_client.hgetall("rag_system:jobs")
    for _, job_data_json in all_jobs.items():
        try:
            job_data = json.loads(job_data_json)
            status = job_data.get("status", "unknown")
            if status in stats["jobs"]:
                stats["jobs"][status] += 1
        except:
            pass

    # Get queue lengths
    for queue in ["inference_tasks", "gpu_tasks", "transcription_tasks", "cpu_tasks"]:
        queue_key = f"dramatiq:{queue}:msgs"
        stats["queues"][queue] = redis_client.llen(queue_key)

    # Get GPU stats if available
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            stats["gpu"][f"gpu_{i}"] = {
                "allocated": torch.cuda.memory_allocated(i) / 1e9,  # GB
                "reserved": torch.cuda.memory_reserved(i) / 1e9,  # GB
                "total": torch.cuda.get_device_properties(i).total_memory / 1e9  # GB
            }

    # Store statistics (last 24 hours worth, one entry per minute)
    stats_key = "rag_system:stats"
    redis_client.lpush(stats_key, json.dumps(stats))
    redis_client.ltrim(stats_key, 0, 60 * 24 - 1)  # Keep 24 hours of 1-minute stats

    return stats


@dramatiq.actor(queue_name="system_tasks")
def reload_models_periodically():
    """Periodically reload models to avoid memory issues from long-running processes."""
    # Get the specific worker type
    if worker_type.startswith("gpu-"):
        logger.info(f"Initiating periodic model reload for {worker_type} worker")

        # Specific reload logic based on worker type
        if worker_type == "gpu-inference":
            # Clear any in-memory caches first
            if hasattr(torch.cuda, 'empty_cache'):
                torch.cuda.empty_cache()

            # Reload LLM model
            global _PRELOADED_LLM_MODEL
            if _PRELOADED_LLM_MODEL is not None:
                # Delete old model reference first
                del _PRELOADED_LLM_MODEL
                _PRELOADED_LLM_MODEL = None

                # Force garbage collection
                import gc
                gc.collect()

                # Reload model
                preload_llm_model()

                logger.info("LLM model successfully reloaded")

            # Reload reranking models
            global _PRELOADED_COLBERT_RERANKER
            if _PRELOADED_COLBERT_RERANKER is not None:
                # Delete old model reference
                del _PRELOADED_COLBERT_RERANKER
                _PRELOADED_COLBERT_RERANKER = None

                # Force garbage collection
                gc.collect()

                # Reload model
                preload_colbert_reranker()

                logger.info("Reranking models successfully reloaded")


@dramatiq.actor(queue_name="system_tasks")
def balance_task_queues():
    """Balance workload across queues to prevent starvation."""
    # Get current queue lengths
    queue_lengths = {}
    for queue in ["inference_tasks", "gpu_tasks", "transcription_tasks", "cpu_tasks"]:
        queue_key = f"dramatiq:{queue}:msgs"
        queue_lengths[queue] = redis_client.llen(queue_key)

    # Check for queue imbalances
    high_priority_queues = ["inference_tasks", "reranking_tasks"]
    low_priority_queues = ["gpu_tasks", "transcription_tasks"]

    # If lower priority queues have too many tasks waiting,
    # temporarily boost their priority to prevent starvation
    if all(queue_lengths.get(q, 0) == 0 for q in high_priority_queues) and \
            any(queue_lengths.get(q, 0) > 10 for q in low_priority_queues):

        # Update priority levels temporarily
        logger.info("Temporarily boosting priority for lower-priority queues to prevent starvation")

        # Example: If transcription tasks are piling up, adjust their priority
        if queue_lengths.get("transcription_tasks", 0) > 10:
            # Record the temporary priority boost
            redis_client.set(
                "priority_boost:transcription_tasks",
                json.dumps({"original": 4, "boosted": 2, "expires_at": time.time() + 600}),
                ex=600  # 10 minute expiry
            )

            logger.info("Temporarily boosted transcription_tasks priority for 10 minutes")


@dramatiq.actor(queue_name="system_tasks")
def optimize_databases():
    """Perform maintenance on the Redis and Qdrant databases."""
    # Get Redis info
    redis_info = redis_client.info()

    # Check if Redis memory usage is high
    used_memory = redis_info.get("used_memory", 0)
    used_memory_peak = redis_info.get("used_memory_peak", 0)

    # If memory usage is over 80% of peak, trigger optimization
    if used_memory > 0.8 * used_memory_peak:
        logger.info("Redis memory usage is high. Performing memory optimization.")
        # Run Redis memory optimization if needed
        redis_client.config_set("maxmemory-policy", "allkeys-lru")

    # Optimize Qdrant collection if needed
    try:
        from qdrant_client import QdrantClient
        qdrant_client = QdrantClient(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
        )

        # Get collection info
        collection_info = qdrant_client.get_collection(settings.qdrant_collection)

        # Check if optimization needed
        if collection_info.vectors_count > 10000:
            logger.info(f"Running optimization on Qdrant collection {settings.qdrant_collection}")
            qdrant_client.update_collection(
                collection_name=settings.qdrant_collection,
                optimizer_config={
                    "indexing_threshold": 0  # Force re-indexing
                }
            )
    except Exception as e:
        logger.error(f"Error optimizing Qdrant: {str(e)}")


@dramatiq.actor(queue_name="system_tasks")
def analyze_error_patterns():
    """Analyze error patterns in failed jobs to detect systemic issues."""
    # Get all failed jobs
    all_jobs = redis_client.hgetall("rag_system:jobs")
    failed_jobs = []

    for job_id, job_data_json in all_jobs.items():
        try:
            job_data = json.loads(job_data_json)
            if job_data.get("status") in ["failed", "timeout"]:
                failed_jobs.append(job_data)
        except:
            continue

    # Skip if no failed jobs
    if not failed_jobs:
        return

    # Count error types
    error_counts = {}
    for job in failed_jobs:
        error = job.get("error", "")

        # Extract error type (first line or first 50 chars)
        error_type = error.split('\n')[0][:50] if error else "Unknown error"

        if error_type not in error_counts:
            error_counts[error_type] = 0
        error_counts[error_type] += 1

    # Find common patterns
    common_errors = sorted(error_counts.items(), key=lambda x: x[1], reverse=True)

    # Log patterns
    logger.info(f"Error pattern analysis: Found {len(failed_jobs)} failed jobs")
    for error_type, count in common_errors[:5]:  # Top 5 errors
        logger.info(f"Common error: '{error_type}' occurred {count} times")

    # Check for spikes in specific error types
    # (Could trigger alerts or automated recovery actions)


@dramatiq.actor(queue_name="system_tasks", periodic=True)
def system_watchdog():
    """
    Main watchdog service that periodically checks system health and coordinates other maintenance tasks.
    This task runs every minute to perform quick checks and dispatch other maintenance tasks as needed.
    """
    # Check GPU health
    if torch.cuda.is_available():
        is_healthy, message = check_gpu_health()
        if not is_healthy:
            logger.warning(f"GPU health check failed: {message}")

            # May need to take recovery actions here

    # Check if any maintenance tasks need to be run
    current_hour = datetime.now().hour

    # Run daily cleanup during off-hours (3 AM)
    if current_hour == 3:
        # Schedule daily maintenance tasks
        cleanup_old_jobs.send(retention_days=7)
        optimize_databases.send()

    # Run hourly tasks
    if datetime.now().minute < 5:  # In the first 5 minutes of each hour
        collect_system_statistics.send()
        analyze_error_patterns.send()

    # Always check for stalled tasks
    cleanup_stalled_tasks.send()

    # Always check priority queue health
    check_priority_queue_health.send()

    # Collect basic system stats
    stats = {
        "timestamp": time.time(),
        "gpu_available": torch.cuda.is_available(),
        "queues": {
            "inference_tasks": redis_client.llen("dramatiq:inference_tasks:msgs"),
            "gpu_tasks": redis_client.llen("dramatiq:gpu_tasks:msgs"),
            "transcription_tasks": redis_client.llen("dramatiq:transcription_tasks:msgs"),
            "cpu_tasks": redis_client.llen("dramatiq:cpu_tasks:msgs"),
        }
    }

    return stats