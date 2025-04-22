"""
Background task processing system using Dramatiq and Redis.

This module provides the functionality to run resource-intensive tasks like
video transcription and PDF OCR in the background, freeing up the web servers
to handle more requests.
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

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Configure Redis broker
redis_host = os.environ.get("REDIS_HOST", "localhost")
redis_port = int(os.environ.get("REDIS_PORT", "6379"))
redis_password = os.environ.get("REDIS_PASSWORD", None)

# Initialize Redis broker
broker_kwargs = {
    "host": redis_host,
    "port": redis_port,
    "max_connections": 20,  # Connection pool size
}
if redis_password:
    broker_kwargs["password"] = redis_password

# Create broker with middleware
redis_broker = RedisBroker(**broker_kwargs)
redis_broker.add_middleware(Callbacks())
dramatiq.set_broker(redis_broker)


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
        if "result" in job_data and job_data["result"] and job_data["result"].startswith('{'):
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
            if "result" in job_data and job_data["result"] and job_data["result"].startswith('{'):
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


# Video processing actor
@dramatiq.actor(max_retries=3, time_limit=3600000)  # 1 hour timeout
def process_video(job_id: str, url: str, custom_metadata: Optional[Dict[str, Any]] = None):
    """Process a video in the background."""
    try:
        # Update job status to processing
        job_tracker.update_job_status(job_id, JobStatus.PROCESSING)

        # Import here to avoid circular imports
        from src.core.document_processor import DocumentProcessor
        from src.config.settings import settings

        # Initialize components
        processor = DocumentProcessor()

        # Process the video
        document_ids = processor.process_video(url=url, custom_metadata=custom_metadata)

        # Update job with success result
        job_tracker.update_job_status(
            job_id,
            JobStatus.COMPLETED,
            result={
                "message": f"Successfully processed video: {url}",
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
        error_detail = f"Error processing video: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_detail)

        # Update job with error
        job_tracker.update_job_status(
            job_id,
            JobStatus.FAILED,
            error=error_detail
        )

        # Re-raise for dramatiq retry mechanism
        raise


# PDF processing actor
@dramatiq.actor(max_retries=2, time_limit=1800000)  # 30 minutes timeout
def process_pdf(job_id: str, file_path: str, custom_metadata: Optional[Dict[str, Any]] = None):
    """Process a PDF file in the background."""
    try:
        # Update job status to processing
        job_tracker.update_job_status(job_id, JobStatus.PROCESSING)

        # Import here to avoid circular imports
        from src.core.document_processor import DocumentProcessor

        # Initialize components
        processor = DocumentProcessor()

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


# Text processing actor
@dramatiq.actor(max_retries=2, time_limit=300000)  # 5 minutes timeout
def process_text(job_id: str, content: str, metadata: Dict[str, Any]):
    """Process manual text input in the background."""
    try:
        # Update job status to processing
        job_tracker.update_job_status(job_id, JobStatus.PROCESSING)

        # Import here to avoid circular imports
        from src.core.document_processor import DocumentProcessor
        from src.models.schema import ManualIngestRequest, DocumentMetadata

        # Initialize components
        processor = DocumentProcessor()

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