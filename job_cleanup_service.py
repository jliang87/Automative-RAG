#!/usr/bin/env python3
"""
Job cleanup service for the background task system.

This script provides a service that periodically cleans up old completed jobs
to prevent Redis memory from filling up over time. It should be run as a
separate process or container.
"""

import os
import time
import redis
import json
import logging
import argparse
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/job_cleanup.log"),
    ]
)
logger = logging.getLogger("job_cleanup")

# Redis configuration from environment
REDIS_HOST = os.environ.get("REDIS_HOST", "localhost")
REDIS_PORT = int(os.environ.get("REDIS_PORT", "6379"))
REDIS_PASSWORD = os.environ.get("REDIS_PASSWORD", None)
JOB_RETENTION_DAYS = int(os.environ.get("JOB_RETENTION_DAYS", "7"))
CLEANUP_INTERVAL = int(os.environ.get("CLEANUP_INTERVAL", "3600"))  # 1 hour by default


def connect_to_redis():
    """Create a Redis connection."""
    try:
        redis_client = redis.Redis(
            host=REDIS_HOST,
            port=REDIS_PORT,
            password=REDIS_PASSWORD,
            decode_responses=True,
            socket_connect_timeout=5,
            socket_timeout=5,
            retry_on_timeout=True,
        )
        # Test the connection
        redis_client.ping()
        return redis_client
    except Exception as e:
        logger.error(f"Error connecting to Redis: {str(e)}")
        return None


def clean_old_jobs(redis_client, retention_days=7):
    """Clean up old completed jobs."""
    try:
        job_key = "rag_system:jobs"

        # Get current time minus retention period
        cutoff_time = time.time() - (retention_days * 24 * 60 * 60)

        # Get all jobs
        all_jobs = redis_client.hgetall(job_key)
        if not all_jobs:
            logger.info("No jobs found to clean up")
            return 0

        # Count jobs before cleanup
        total_jobs = len(all_jobs)
        logger.info(f"Found {total_jobs} total jobs")

        # Track deleted jobs
        deleted_count = 0

        # Process each job
        for job_id, job_data_json in all_jobs.items():
            try:
                job_data = json.loads(job_data_json)
                job_status = job_data.get("status", "")
                created_at = job_data.get("created_at", 0)

                # Delete if it's old and completed or failed
                if created_at < cutoff_time and job_status in ["completed", "failed", "timeout"]:
                    redis_client.hdel(job_key, job_id)
                    deleted_count += 1

                    # Log every 50 deletions to avoid excessive logging
                    if deleted_count % 50 == 0:
                        logger.info(f"Deleted {deleted_count} old jobs so far")
            except json.JSONDecodeError:
                logger.warning(f"Invalid JSON data for job {job_id}")
                continue

        # Log final result
        logger.info(f"Cleanup complete: Deleted {deleted_count} old jobs out of {total_jobs} total jobs")
        logger.info(f"Remaining jobs: {total_jobs - deleted_count}")

        return deleted_count
    except Exception as e:
        logger.error(f"Error cleaning up old jobs: {str(e)}")
        return 0


def clean_stalled_jobs(redis_client, stall_threshold_hours=3):
    """Clean up stalled jobs that have been processing for too long."""
    try:
        job_key = "rag_system:jobs"

        # Get current time minus stall threshold
        cutoff_time = time.time() - (stall_threshold_hours * 60 * 60)

        # Get all jobs
        all_jobs = redis_client.hgetall(job_key)
        if not all_jobs:
            return 0

        # Track reset jobs
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
                    redis_client.hset(job_key, job_id, json.dumps(job_data))
                    reset_count += 1

                    logger.info(f"Reset stalled job: {job_id}")
            except json.JSONDecodeError:
                continue

        if reset_count > 0:
            logger.info(f"Reset {reset_count} stalled jobs")

        return reset_count
    except Exception as e:
        logger.error(f"Error cleaning up stalled jobs: {str(e)}")
        return 0


def get_redis_stats(redis_client):
    """Get Redis memory stats."""
    try:
        info = redis_client.info("memory")
        used_memory = info.get("used_memory_human", "unknown")
        used_memory_peak = info.get("used_memory_peak_human", "unknown")
        total_system_memory = info.get("total_system_memory_human", "unknown")

        logger.info(f"Redis memory: Used={used_memory}, Peak={used_memory_peak}, System Total={total_system_memory}")

        return info
    except Exception as e:
        logger.error(f"Error getting Redis stats: {str(e)}")
        return {}


def run_cleanup_service(retention_days=7, interval=3600, stall_threshold_hours=3):
    """Run the cleanup service continuously."""
    logger.info(f"Starting job cleanup service")
    logger.info(f"Redis connection: {REDIS_HOST}:{REDIS_PORT}")
    logger.info(f"Job retention period: {retention_days} days")
    logger.info(f"Cleanup interval: {interval} seconds")
    logger.info(f"Stall threshold: {stall_threshold_hours} hours")

    while True:
        # Connect to Redis (create new connection each time to handle reconnection)
        redis_client = connect_to_redis()

        if redis_client:
            # Perform cleanup
            try:
                # Get Redis stats before cleanup
                get_redis_stats(redis_client)

                # Clean up old completed/failed jobs
                deleted_count = clean_old_jobs(redis_client, retention_days)
                logger.info(f"Cleaned up {deleted_count} old jobs")

                # Reset stalled jobs
                reset_count = clean_stalled_jobs(redis_client, stall_threshold_hours)
                if reset_count > 0:
                    logger.info(f"Reset {reset_count} stalled jobs")

                # Get Redis stats after cleanup
                get_redis_stats(redis_client)

                # Close Redis connection
                redis_client.close()
            except Exception as e:
                logger.error(f"Error during cleanup cycle: {str(e)}")

        # Wait for next cleanup cycle
        next_run = datetime.now() + timedelta(seconds=interval)
        logger.info(f"Next cleanup scheduled at: {next_run.strftime('%Y-%m-%d %H:%M:%S')}")

        # Sleep until next cycle
        time.sleep(interval)


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Job cleanup service")
    parser.add_argument("--retention-days", type=int, default=JOB_RETENTION_DAYS,
                        help="Number of days to retain completed jobs")
    parser.add_argument("--interval", type=int, default=CLEANUP_INTERVAL,
                        help="Cleanup interval in seconds")
    parser.add_argument("--stall-threshold", type=int, default=3,
                        help="Threshold in hours to consider a job stalled")
    args = parser.parse_args()

    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)

    # Run the service
    run_cleanup_service(
        retention_days=args.retention_days,
        interval=args.interval,
        stall_threshold_hours=args.stall_threshold
    )