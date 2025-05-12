"""
Job tracking system for background tasks.

This module provides a system for tracking the status and metadata of
background jobs, including updating job status, storing results, and
retrieving job information.
"""

import json
import time
import logging
from typing import Dict, List, Optional, Any
import redis

from .common import JobStatus

logger = logging.getLogger(__name__)

class JobTracker:
    """Track and manage background jobs."""

    def __init__(self, redis_client=None):
        """Initialize the job tracker with Redis."""
        self.redis = redis_client
        if self.redis is None:
            from .common import get_redis_client
            self.redis = get_redis_client()

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
                    deleted_count += 1

                    # Log every 50 deletions to avoid excessive logging
                    if deleted_count % 50 == 0:
                        logger.info(f"Deleted {deleted_count} old jobs so far")
            except:
                pass

        logger.info(f"Old job cleanup completed: deleted {deleted_count} jobs")
        return deleted_count

    def clean_stalled_jobs(self, stall_threshold_hours: int = 3) -> int:
        """Clean up stalled jobs that have been processing for too long."""
        all_jobs = self.redis.hgetall(self.job_key)
        if not all_jobs:
            return 0

        # Get current time minus stall threshold
        cutoff_time = time.time() - (stall_threshold_hours * 60 * 60)
        reset_count = 0

        # Process each job
        for job_id, job_data_json in all_jobs.items():
            try:
                job_data = json.loads(job_data_json)
                job_status = job_data.get("status", "")
                updated_at = job_data.get("updated_at", 0)

                # Reset if it's stalled in "processing" state
                if job_status == "processing" and updated_at < cutoff_time:
                    # Set status back to pending for retry
                    job_data["status"] = "pending"
                    job_data["updated_at"] = time.time()
                    job_data["error"] = "Job appeared to be stalled and was reset for retry"

                    # Save updated job data
                    self.redis.hset(self.job_key, job_id, json.dumps(job_data))
                    reset_count += 1

                    logger.info(f"Reset stalled job: {job_id}")
            except:
                continue

        if reset_count > 0:
            logger.info(f"Reset {reset_count} stalled jobs")

        return reset_count

# Initialize the global job tracker instance
job_tracker = JobTracker()