import json
import time
import logging
from typing import Dict, List, Optional, Any, Union
import redis

from .common import JobStatus

logger = logging.getLogger(__name__)


class JobTracker:
    def __init__(self, redis_client=None):
        if redis_client is None:
            from .common import get_redis_client
            self.redis = get_redis_client()  # This client supports UTF-8
        else:
            self.redis = redis_client

        self.job_key = "rag_system:jobs"
        self.progress_key = "rag_system:job_progress"

    def create_job(self, job_id: str, job_type: str, metadata: Dict[str, Any]) -> None:
        """Create a new job record with proper UTF-8 encoding."""

        job_data = {
            "job_id": job_id,
            "job_type": job_type,
            "status": "pending",
            "created_at": time.time(),
            "updated_at": time.time(),
            "result": None,
            "error": None,
            "metadata": metadata,  # Store as dict, not JSON string
            "progress": 0.0
        }

        job_json = json.dumps(job_data)
        self.redis.hset(self.job_key, job_id, job_json)
        logging.info(f"Created job {job_id} with UTF-8 encoding")

    def update_job_status(self, job_id: str, status: str, result: Any = None,
                          error: str = None, stage: str = None, replace_result: bool = False) -> None:
        """Update job status with proper UTF-8 encoding."""

        job_data_json = self.redis.hget(self.job_key, job_id)
        if not job_data_json:
            logging.warning(f"Job {job_id} not found")
            return

        job_data = json.loads(job_data_json)
        job_data["status"] = status
        job_data["updated_at"] = time.time()

        if stage:
            job_data["current_stage"] = stage
            job_data["stage_updated_at"] = time.time()

        if result is not None:
            if replace_result or not job_data.get("result"):
                # Store result as dict/object, not JSON string
                job_data["result"] = result
            else:
                # Merge with existing result
                existing_result = job_data.get("result", {})
                if isinstance(existing_result, str):
                    try:
                        existing_result = json.loads(existing_result)
                    except:
                        existing_result = {}

                if isinstance(existing_result, dict) and isinstance(result, dict):
                    existing_result.update(result)
                    job_data["result"] = existing_result
                else:
                    job_data["result"] = result

        if error is not None:
            job_data["error"] = str(error)

        job_json = json.dumps(job_data)
        self.redis.hset(self.job_key, job_id, job_json)
        logger.info(f"Updated job {job_id} status to {status}" + (f", stage: {stage}" if stage else ""))

    def update_job_progress(self, job_id: str, progress: Union[int, float, None], message: str = "") -> None:
        """Update the progress percentage and message for a job with Unicode handling."""
        if progress is not None:
            # Ensure progress is between 0 and 100
            progress = max(0, min(100, progress))

        # Create progress entry
        progress_data = {
            "progress": progress,
            "message": message,
            "timestamp": time.time()
        }

        # Store in Redis with proper Unicode encoding
        progress_key = f"{self.progress_key}:{job_id}"
        self.redis.set(progress_key, json.dumps(progress_data), ex=86400)  # Expire after 24 hours

        # Also update progress in the main job data
        job_data_json = self.redis.hget(self.job_key, job_id)
        if job_data_json:
            try:
                job_data = json.loads(job_data_json)
                if progress is not None:
                    job_data["progress"] = progress
                job_data["progress_message"] = message
                job_data["progress_updated_at"] = time.time()
                self.redis.hset(self.job_key, job_id, json.dumps(job_data))
            except:
                pass

        if progress is not None:
            logger.debug(f"Updated job {job_id} progress to {progress}%: {message}")

    def get_job_progress(self, job_id: str) -> Dict[str, Any]:
        """Get the current progress of a job with Unicode decoding."""
        progress_key = f"{self.progress_key}:{job_id}"
        progress_data_json = self.redis.get(progress_key)

        if not progress_data_json:
            # Check job status for default progress
            job_data = self.get_job(job_id, include_progress=False)
            if not job_data:
                return {"progress": 0, "message": "Job not found", "timestamp": time.time()}

            status = job_data.get("status", "")
            if status == JobStatus.COMPLETED:
                return {"progress": 100, "message": "Job completed", "timestamp": time.time()}
            elif status in [JobStatus.FAILED, JobStatus.TIMEOUT]:
                return {"progress": 0, "message": "Job failed", "timestamp": time.time()}
            else:
                return {"progress": 0, "message": f"Status: {status}", "timestamp": time.time()}

        try:
            progress_data = json.loads(progress_data_json)
            return progress_data
        except:
            return {"progress": 0, "message": "Invalid progress data", "timestamp": time.time()}

    def get_job(self, job_id: str, include_progress: bool = True) -> Optional[Dict[str, Any]]:
        """Get job information by ID with comprehensive Unicode decoding."""
        job_data_json = self.redis.hget(self.job_key, job_id)
        if not job_data_json:
            return None

        job_data = json.loads(job_data_json)

        # Parse metadata if it's JSON string
        if "metadata" in job_data and isinstance(job_data["metadata"], str):
            try:
                job_data["metadata"] = json.loads(job_data["metadata"])
            except:
                pass

        # Parse result if it's JSON string
        if "result" in job_data and isinstance(job_data["result"], str):
            try:
                job_data["result"] = json.loads(job_data["result"])
            except:
                pass

        if include_progress:
            progress_data = self.get_job_progress(job_id)
            job_data["progress_info"] = progress_data

        return job_data

    def get_all_jobs(self, limit: int = 100, job_type: str = None) -> List[Dict[str, Any]]:
        """Get all jobs with Unicode decoding, optionally filtered by type."""
        all_jobs = self.redis.hgetall(self.job_key)
        jobs = []

        for _, job_data_json in all_jobs.items():
            job_data = json.loads(job_data_json)

            # Filter by job type if specified
            if job_type and job_data.get("job_type") != job_type:
                continue

            jobs.append(job_data)

        # Sort by creation time (newest first) and limit results
        jobs.sort(key=lambda x: x.get("created_at", 0), reverse=True)
        return jobs[:limit]

    def delete_job(self, job_id: str) -> bool:
        """Delete a job by ID."""
        # Delete main job data
        deleted = self.redis.hdel(self.job_key, job_id)

        # Delete progress data
        progress_key = f"{self.progress_key}:{job_id}"
        self.redis.delete(progress_key)

        return deleted > 0

    def count_jobs_by_status(self) -> Dict[str, int]:
        """Count jobs by status."""
        all_jobs = self.redis.hgetall(self.job_key)

        status_counts = {
            "pending": 0,
            "processing": 0,
            "completed": 0,
            "failed": 0,
            "timeout": 0,
            "total": len(all_jobs)
        }

        for _, job_data_json in all_jobs.items():
            try:
                job_data = json.loads(job_data_json)
                status = job_data.get("status", "unknown")

                if status in status_counts:
                    status_counts[status] += 1
            except:
                pass

        return status_counts

    def cleanup_old_jobs(self, retention_days: int = 7) -> int:
        """Clean up old completed jobs that are beyond the retention period."""
        all_jobs = self.redis.hgetall(self.job_key)

        # Get current time minus retention period
        cutoff_time = time.time() - (retention_days * 24 * 60 * 60)
        deleted_count = 0

        # Process each job
        for job_id, job_data_json in all_jobs.items():
            try:
                job_data = json.loads(job_data_json)
                job_status = job_data.get("status", "")
                created_at = job_data.get("created_at", 0)

                # Delete if it's old and completed/failed
                if created_at < cutoff_time and job_status in ["completed", "failed", "timeout"]:
                    self.redis.hdel(self.job_key, job_id)
                    # Also delete progress data
                    progress_key = f"{self.progress_key}:{job_id}"
                    self.redis.delete(progress_key)
                    deleted_count += 1

                    if deleted_count % 50 == 0:
                        logger.info(f"Deleted {deleted_count} old jobs so far")
            except:
                pass

        logger.info(f"Old job cleanup completed: deleted {deleted_count} jobs")
        return deleted_count


# Initialize the global job tracker instance
job_tracker = JobTracker()