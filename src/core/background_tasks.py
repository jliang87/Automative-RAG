"""
Background task processing system using Dramatiq and Redis.

This module provides the functionality to run resource-intensive tasks like
video transcription and LLM inference in the background, with dedicated
worker types for GPU and CPU processing.
"""

import os
import json
import logging
import time
from typing import Dict, List, Optional, Union, Any

import dramatiq
from dramatiq.brokers.redis import RedisBroker
from dramatiq.middleware.time_limit import TimeLimitExceeded
from dramatiq.middleware.callbacks import Callbacks
from dramatiq.middleware.age_limit import AgeLimit
from dramatiq.middleware.retries import Retries
from dramatiq.rate_limits import ConcurrentRateLimiter
from dramatiq.rate_limits.backends import RedisBackend

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
dramatiq.set_broker(redis_broker)

# Rate limiter backend
rate_limiter_backend = RedisBackend(
    client=redis_broker.client
)

# Add middleware
redis_broker.add_middleware(Callbacks())
redis_broker.add_middleware(AgeLimit())
redis_broker.add_middleware(Retries(max_retries=3))

# Create GPU rate limiter - limits to 1 concurrent GPU task
gpu_limiter = ConcurrentRateLimiter(
    backend=rate_limiter_backend,
    key="gpu_task_limiter",
    limit=1,
    ttl=60000  # 1 minute
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


# Centralized function to get vector store for background tasks
def get_vector_store(force_cpu=False):
    """
    Get a vector store instance for background tasks.

    Args:
        force_cpu: Whether to force CPU usage for embeddings
    """
    from src.core.vectorstore import QdrantStore
    from src.config.settings import settings

    # Initialize qdrant client
    from qdrant_client import QdrantClient
    qdrant_client = QdrantClient(
        host=settings.qdrant_host,
        port=settings.qdrant_port,
    )

    if force_cpu or worker_type == "cpu":
        # Create a custom embedding function that uses CPU
        from langchain_huggingface import HuggingFaceEmbeddings
        try:
            # Try to load the embedding model explicitly on CPU
            embedding_function = HuggingFaceEmbeddings(
                model_name=settings.embedding_model_full_path,
                model_kwargs={"device": "cpu"},  # Force CPU usage
                encode_kwargs={"batch_size": settings.batch_size, "normalize_embeddings": True},
                cache_folder=os.path.join(settings.models_dir, settings.embedding_model_path)
            )
            logger.info("Using CPU embedding model for background task worker")
        except Exception as e:
            logger.error(f"Error loading embedding model for background task: {str(e)}")
            raise

        # Initialize vector store with CPU embeddings
        return QdrantStore(
            client=qdrant_client,
            collection_name=settings.qdrant_collection,
            embedding_function=embedding_function,
        )
    else:
        # Use regular embedding function from settings (could use GPU)
        return QdrantStore(
            client=qdrant_client,
            collection_name=settings.qdrant_collection,
            embedding_function=settings.embedding_function,
        )


# Improved GPU resource management using rate limiters

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


# Video processing actor - GPU version
@dramatiq.actor(queue_name="gpu_tasks", max_retries=3, time_limit=3600000)
def process_video_gpu(job_id: str, url: str, custom_metadata: Optional[Dict[str, Any]] = None):
    """Process a video using GPU for transcription and embedding."""
    try:
        # Use the GPU rate limiter to ensure only one GPU task runs at a time
        with gpu_limiter.acquire():
            # Update job status to processing
            job_tracker.update_job_status(job_id, JobStatus.PROCESSING)
            logger.info(f"GPU worker processing video: {url}")

            # Import here to avoid circular imports
            from src.core.document_processor import DocumentProcessor
            from src.core.video_transcriber import VideoTranscriber
            import torch

            # Get vector store with GPU embeddings
            vector_store = get_vector_store(force_cpu=False)

            # Create transcriber with GPU
            transcriber = VideoTranscriber(
                whisper_model_size=os.environ.get("WHISPER_MODEL_SIZE", "medium"),
                device="cuda",
                num_workers=1  # Just use 1 worker thread to avoid oversubscription
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
                torch.cuda.empty_cache()

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

            return document_ids

    except TimeLimitExceeded:
        job_tracker.update_job_status(
            job_id,
            JobStatus.TIMEOUT,
            error="Processing timeout exceeded"
        )
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

        # Re-raise for dramatiq retry mechanism
        raise


# PDF processing actor - CPU version
@dramatiq.actor(queue_name="cpu_tasks", max_retries=2, time_limit=1800000)
def process_pdf_cpu(job_id: str, file_path: str, custom_metadata: Optional[Dict[str, Any]] = None):
    """Process a PDF file in the background using CPU."""
    try:
        # Update job status to processing
        job_tracker.update_job_status(job_id, JobStatus.PROCESSING)
        logger.info(f"CPU worker processing PDF: {file_path}")

        # Import here to avoid circular imports
        from src.core.document_processor import DocumentProcessor
        from src.core.pdf_loader import PDFLoader

        # Get vector store with CPU embeddings
        vector_store = get_vector_store(force_cpu=True)

        # Create PDF loader with CPU
        pdf_loader = PDFLoader(
            chunk_size=int(os.environ.get("CHUNK_SIZE", "1000")),
            chunk_overlap=int(os.environ.get("CHUNK_OVERLAP", "200")),
            device="cpu",
            use_ocr=os.environ.get("USE_PDF_OCR", "true").lower() == "true",
            ocr_languages=os.environ.get("OCR_LANGUAGES", "en+ch_doc")
        )

        # Initialize document processor
        processor = DocumentProcessor(
            vector_store=vector_store,
            pdf_loader=pdf_loader
        )

        # Process the PDF
        document_ids = processor.process_pdf(file_path=file_path, custom_metadata=custom_metadata)

        # Update job with success result
        job_tracker.update_job_status(
            job_id,
            JobStatus.COMPLETED,
            result={
                "message": f"Successfully processed PDF: {os.path.basename(file_path)}",
                "document_count": len(document_ids),
                "document_ids": document_ids,
            }
        )

        return document_ids

    except TimeLimitExceeded:
        job_tracker.update_job_status(
            job_id,
            JobStatus.TIMEOUT,
            error="Processing timeout exceeded"
        )
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


# Text processing actor - CPU version
@dramatiq.actor(queue_name="cpu_tasks", max_retries=2, time_limit=300000)
def process_text(job_id: str, content: str, metadata: Dict[str, Any]):
    """Process manual text input in the background using CPU."""
    try:
        # Update job status to processing
        job_tracker.update_job_status(job_id, JobStatus.PROCESSING)
        logger.info(f"CPU worker processing text input")

        # Import here to avoid circular imports
        from src.core.document_processor import DocumentProcessor
        from src.models.schema import ManualIngestRequest, DocumentMetadata

        # Get vector store with CPU embeddings
        vector_store = get_vector_store(force_cpu=True)

        # Initialize document processor
        processor = DocumentProcessor(
            vector_store=vector_store
        )

        # Create a proper request object
        request = ManualIngestRequest(
            content=content,
            metadata=DocumentMetadata(**metadata)
        )

        # Process the text
        document_ids = processor.process_text(request)

        # Update job with success result
        job_tracker.update_job_status(
            job_id,
            JobStatus.COMPLETED,
            result={
                "message": "Successfully processed manual text input",
                "document_count": len(document_ids),
                "document_ids": document_ids,
            }
        )

        return document_ids

    except TimeLimitExceeded:
        job_tracker.update_job_status(
            job_id,
            JobStatus.TIMEOUT,
            error="Processing timeout exceeded"
        )
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


# Router function for video processing
@dramatiq.actor(max_retries=1, time_limit=60000)
def process_video(job_id: str, url: str, custom_metadata: Optional[Dict[str, Any]] = None):
    """Route video processing to the GPU worker."""
    logger.info(f"Routing video processing to GPU worker: {url}")

    # Update job status to processing before sending to GPU queue
    job_tracker.update_job_status(
        job_id,
        JobStatus.PROCESSING,
        result=None,
        error=None
    )

    # Then send to GPU tasks queue
    process_video_gpu.send(job_id, url, custom_metadata)


# Router function for PDF processing
@dramatiq.actor(max_retries=1, time_limit=60000)
def process_pdf(job_id: str, file_path: str, custom_metadata: Optional[Dict[str, Any]] = None):
    """Route PDF processing to the CPU worker."""
    logger.info(f"Routing PDF processing to CPU worker: {file_path}")

    # Send to CPU tasks queue
    process_pdf_cpu.send(job_id, file_path, custom_metadata)

    # Update job status to pending in the new queue
    job_tracker.update_job_status(
        job_id,
        JobStatus.PENDING,
        result=None,
        error=None
    )


# Batch videos processing router
@dramatiq.actor(max_retries=1, time_limit=60000)
def batch_process_videos(job_id: str, urls: List[str], custom_metadata: Optional[List[Dict[str, Any]]] = None):
    """Process multiple videos, routing each to GPU worker."""
    try:
        # Update job status to processing
        job_tracker.update_job_status(job_id, JobStatus.PROCESSING)

        # Track jobs for each URL
        sub_job_ids = []

        # Create a job for each URL
        for i, url in enumerate(urls):
            # Get metadata for this URL if provided
            url_metadata = None
            if custom_metadata and i < len(custom_metadata):
                url_metadata = custom_metadata[i]

            # Create a sub job ID
            sub_job_id = f"{job_id}-{i}"
            sub_job_ids.append(sub_job_id)

            # Create job record
            job_tracker.create_job(
                job_id=sub_job_id,
                job_type="video_processing",
                metadata={
                    "url": url,
                    "parent_job_id": job_id,
                    "custom_metadata": url_metadata
                }
            )

            # Send to GPU worker
            process_video_gpu.send(sub_job_id, url, url_metadata)

        # Update job with sub job IDs
        job_tracker.update_job_status(
            job_id,
            JobStatus.PROCESSING,
            result={
                "message": f"Started processing {len(urls)} videos",
                "sub_job_ids": sub_job_ids
            }
        )

        # Return sub job IDs
        return sub_job_ids

    except Exception as e:
        import traceback
        error_detail = f"Error batch processing videos: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_detail)

        job_tracker.update_job_status(
            job_id,
            JobStatus.FAILED,
            error=error_detail
        )

        raise