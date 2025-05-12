"""
Dramatiq actors for system maintenance tasks.

This module defines actors for system maintenance tasks like cleanup,
monitoring, and optimization. These tasks are typically executed by
system workers on a periodic schedule.
"""

import os
import time
import json
import logging
from typing import Dict, Any
import datetime
import torch
from qdrant_client import QdrantClient

import dramatiq

from ..common import get_redis_client, check_gpu_health
from ..job_tracker import job_tracker
from ..priority_queue import priority_queue
from ..models import reload_models

logger = logging.getLogger(__name__)


# Cleanup old jobs after retention period
@dramatiq.actor(
    queue_name="system_tasks",
    store_results=True
)
def cleanup_old_jobs(retention_days: int = 7):
    """Clean up old completed jobs that are beyond the retention period."""
    deleted_count = job_tracker.cleanup_old_jobs(retention_days)
    return {"deleted_count": deleted_count}


# Cleanup stalled tasks in the priority system
@dramatiq.actor(
    queue_name="system_tasks",
    store_results=True
)
def cleanup_stalled_tasks():
    """Clean up stalled tasks in the priority system."""
    redis_client = get_redis_client()

    # Check active task
    active_task_json = redis_client.get("active_gpu_task")
    if not active_task_json:
        logger.info("No active GPU task to clean up")
        return {"result": "no_active_task"}

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
                    "failed",
                    error="Task appears to be stalled and was terminated"
                )

            return {
                "result": "stalled_task_cleaned",
                "task_id": task_id,
                "job_id": job_id,
                "task_age": task_age
            }
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
                    "failed",
                    error=f"Task terminated due to GPU health issues: {status_message}"
                )

                return {
                    "result": "gpu_unhealthy_task_terminated",
                    "task_id": task_id,
                    "job_id": job_id,
                    "task_age": task_age,
                    "gpu_status": status_message
                }
            else:
                # GPU is healthy but task is still running - log but don't terminate yet
                logger.warning(f"Found long-running task {task_id} for job {job_id}, but GPU appears healthy")

                return {
                    "result": "long_running_task",
                    "task_id": task_id,
                    "job_id": job_id,
                    "task_age": task_age
                }

    return {
        "result": "active_task_normal",
        "task_id": task_id,
        "job_id": job_id,
        "task_age": task_age
    }


# Check priority queue health
@dramatiq.actor(
    queue_name="system_tasks",
    store_results=True
)
def check_priority_queue_health():
    """Perform a health check on the priority queue system."""
    health_result = priority_queue.check_health()

    # Take actions based on health check
    if health_result["status"] != "healthy":
        logger.warning(f"Priority queue health check failed: {health_result['issues']}")

        # Apply recommended fixes
        for recommendation in health_result["recommendations"]:
            if recommendation == "Reset active GPU task":
                # Reset active task
                redis_client = get_redis_client()
                active_task_json = redis_client.get("active_gpu_task")
                if active_task_json:
                    active_task = json.loads(active_task_json)
                    job_id = active_task.get("job_id", "unknown")
                    redis_client.delete("active_gpu_task")
                    logger.info(f"Reset active GPU task for job {job_id}")

                    # Also update job status
                    job_data = job_tracker.get_job(job_id)
                    if job_data and job_data.get("status") == "processing":
                        job_tracker.update_job_status(
                            job_id,
                            "failed",
                            error="Task reset due to priority queue health check"
                        )

    return health_result


# Monitor GPU memory
@dramatiq.actor(
    queue_name="system_tasks",
    store_results=True
)
def monitor_gpu_memory():
    """Monitor GPU memory usage and optimize when needed."""
    if not torch.cuda.is_available():
        return {"result": "no_gpu_available"}

    memory_stats = {}

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

        # Store stats
        memory_stats[f"gpu_{i}"] = {
            "total_memory_gb": total_memory / 1e9,
            "allocated_memory_gb": allocated_memory / 1e9,
            "reserved_memory_gb": reserved_memory / 1e9,
            "free_memory_gb": free_memory / 1e9,
            "free_percentage": free_percentage
        }

        # If memory usage is too high, trigger cache clearing
        if free_percentage < 20:  # Less than 20% free memory
            logger.warning(f"GPU {i} memory is running low ({free_percentage:.1f}% free). Clearing cache.")
            torch.cuda.empty_cache()

            # Check if clearing cache helped
            new_reserved = torch.cuda.memory_reserved(i)
            new_free = total_memory - new_reserved
            new_percentage = (new_free / total_memory) * 100

            memory_stats[f"gpu_{i}"]["action"] = "cache_cleared"
            memory_stats[f"gpu_{i}"]["new_free_percentage"] = new_percentage

    return {"memory_stats": memory_stats}


# Reload models periodically
@dramatiq.actor(
    queue_name="system_tasks",
    store_results=True
)
def reload_models_periodically():
    """Periodically reload models to avoid memory issues from long-running processes."""
    worker_type = os.environ.get("WORKER_TYPE", "")

    # Only reload on GPU workers
    if not worker_type.startswith("gpu-"):
        return {"result": "not_gpu_worker"}

    logger.info(f"Initiating periodic model reload for {worker_type} worker")

    # First, check if there's an active task
    redis_client = get_redis_client()
    active_task_json = redis_client.get("active_gpu_task")

    if active_task_json:
        logger.info("Active task found, skipping model reload")
        return {"result": "active_task_skip_reload"}

    # Also check if there are pending tasks in the queue
    queue_length = redis_client.zcard("priority_task_queue")
    if queue_length > 0:
        logger.info(f"Found {queue_length} tasks in queue, skipping model reload")
        return {"result": "pending_tasks_skip_reload"}

    # Safe to reload models
    reload_models()

    return {
        "result": "models_reloaded",
        "worker_type": worker_type
    }


# Balance task queues
@dramatiq.actor(
    queue_name="system_tasks",
    store_results=True
)
def balance_task_queues():
    """Balance workload across queues to prevent starvation."""
    balance_result = priority_queue.balance_queues()
    return balance_result


# Collect system statistics
@dramatiq.actor(
    queue_name="system_tasks",
    store_results=True
)
def collect_system_statistics():
    """Collect and store system performance statistics."""
    redis_client = get_redis_client()

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
    job_stats = job_tracker.count_jobs_by_status()
    stats["jobs"].update(job_stats)

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


# Optimize databases
@dramatiq.actor(
    queue_name="system_tasks",
    store_results=True
)
def optimize_databases():
    """Perform maintenance on the Redis and Qdrant databases."""
    results = {
        "redis": {},
        "qdrant": {}
    }

    # Get Redis info
    redis_client = get_redis_client()
    redis_info = redis_client.info()

    # Check if Redis memory usage is high
    used_memory = redis_info.get("used_memory", 0)
    used_memory_peak = redis_info.get("used_memory_peak", 0)

    # If memory usage is over 80% of peak, trigger optimization
    if used_memory > 0.8 * used_memory_peak:
        logger.info("Redis memory usage is high. Performing memory optimization.")
        # Run Redis memory optimization if needed
        redis_client.config_set("maxmemory-policy", "allkeys-lru")
        results["redis"]["action"] = "memory_optimization"
        results["redis"]["memory_usage"] = f"{used_memory / 1024 / 1024:.2f} MB"
        results["redis"]["peak_memory"] = f"{used_memory_peak / 1024 / 1024:.2f} MB"

    # Optimize Qdrant collection if needed
    try:
        from src.config.settings import settings

        # Initialize qdrant client
        qdrant_client = QdrantClient(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
        )

        # Get collection info
        collection_info = qdrant_client.get_collection(settings.qdrant_collection)
        results["qdrant"]["collection"] = settings.qdrant_collection
        results["qdrant"]["vectors_count"] = collection_info.vectors_count

        # Check if optimization needed
        if collection_info.vectors_count > 10000:
            logger.info(f"Running optimization on Qdrant collection {settings.qdrant_collection}")
            qdrant_client.update_collection(
                collection_name=settings.qdrant_collection,
                optimizer_config={
                    "indexing_threshold": 0  # Force re-indexing
                }
            )
            results["qdrant"]["action"] = "forced_reindexing"
        else:
            results["qdrant"]["action"] = "none_needed"
    except Exception as e:
        logger.error(f"Error optimizing Qdrant: {str(e)}")
        results["qdrant"]["error"] = str(e)

    return results


# Analyze error patterns
@dramatiq.actor(
    queue_name="system_tasks",
    store_results=True
)
def analyze_error_patterns():
    """Analyze error patterns in failed jobs to detect systemic issues."""
    # Get all failed jobs
    all_jobs = job_tracker.get_all_jobs(limit=1000)
    failed_jobs = [job for job in all_jobs if job.get("status") in ["failed", "timeout"]]

    # Skip if no failed jobs
    if not failed_jobs:
        return {"result": "no_failed_jobs"}

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

    return {
        "total_failed_jobs": len(failed_jobs),
        "common_errors": common_errors[:5]
    }


# Main watchdog service
@dramatiq.actor(
    queue_name="system_tasks",
    periodic=True
)
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

    # Get current time
    current_time = datetime.datetime.now()
    current_hour = current_time.hour

    # Run daily cleanup during off-hours (3 AM)
    if current_hour == 3 and current_time.minute < 5:
        # Schedule daily maintenance tasks
        cleanup_old_jobs.send(retention_days=7)
        optimize_databases.send()

    # Run hourly tasks
    if current_time.minute < 5:  # In the first 5 minutes of each hour
        collect_system_statistics.send()
        analyze_error_patterns.send()
        monitor_gpu_memory.send()

    # Run every 4 hours
    if current_hour % 4 == 0 and current_time.minute < 5:
        reload_models_periodically.send()

    # Run every hour
    if current_time.minute < 5:
        balance_task_queues.send()

    # Always check for stalled tasks
    cleanup_stalled_tasks.send()

    # Always check priority queue health
    check_priority_queue_health.send()

    # Collect basic system stats
    redis_client = get_redis_client()
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