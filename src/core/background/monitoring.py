"""
Monitoring utilities for the background tasks system.

This module provides utilities for monitoring the health and performance
of the background tasks system, including the priority queue, GPU usage,
and job status.
"""

import os
import time
import json
import logging
from typing import Dict, List, Optional, Any
import torch

from .common import get_redis_client
from .job_tracker import job_tracker
from .priority_queue import priority_queue

logger = logging.getLogger(__name__)


def get_system_status() -> Dict[str, Any]:
    """
    Get comprehensive system status, including job, queue, and GPU statistics.

    Returns:
        Dictionary with system status information
    """
    redis_client = get_redis_client()

    status = {
        "timestamp": time.time(),
        "jobs": job_tracker.count_jobs_by_status(),
        "queues": get_queue_status(),
        "priority_queue": priority_queue.get_queue_status(),
        "gpu": get_gpu_status(),
        "workers": get_worker_status()
    }

    return status


def get_queue_status() -> Dict[str, int]:
    """
    Get the status of all Dramatiq queues.

    Returns:
        Dictionary mapping queue names to message counts
    """
    redis_client = get_redis_client()
    queue_status = {}

    # Get all queue lengths
    for queue in ["inference_tasks", "embedding_tasks", "transcription_tasks", "cpu_tasks", "system_tasks", "default"]:
        queue_key = f"dramatiq:{queue}:msgs"
        queue_status[queue] = redis_client.llen(queue_key)

    return queue_status


def get_gpu_status() -> Dict[str, Any]:
    """
    Get GPU status information.

    Returns:
        Dictionary with GPU status information
    """
    if not torch.cuda.is_available():
        return {"available": False}

    gpu_status = {
        "available": True,
        "device_count": torch.cuda.device_count(),
        "devices": {}
    }

    # Get information for each GPU
    for i in range(torch.cuda.device_count()):
        device_info = {
            "name": torch.cuda.get_device_name(i),
            "total_memory_gb": torch.cuda.get_device_properties(i).total_memory / 1e9,
            "allocated_memory_gb": torch.cuda.memory_allocated(i) / 1e9,
            "reserved_memory_gb": torch.cuda.memory_reserved(i) / 1e9,
            "free_memory_gb": (torch.cuda.get_device_properties(i).total_memory - torch.cuda.memory_reserved(i)) / 1e9
        }

        # Calculate utilization percentage
        device_info["utilization_percent"] = (device_info["allocated_memory_gb"] / device_info["total_memory_gb"]) * 100

        gpu_status["devices"][f"gpu_{i}"] = device_info

    return gpu_status


def get_worker_status() -> Dict[str, Any]:
    """
    Get information about running worker processes.

    Returns:
        Dictionary with worker status information
    """
    redis_client = get_redis_client()

    # Get worker information from Redis
    workers = []
    worker_keys = redis_client.keys("dramatiq:__heartbeats__:*")

    for key in worker_keys:
        worker_id = key.split(":")[-1]
        last_heartbeat = redis_client.get(key)

        if last_heartbeat:
            heartbeat_age = time.time() - float(last_heartbeat)
            workers.append({
                "worker_id": worker_id,
                "last_heartbeat_seconds_ago": heartbeat_age
            })

    # Group workers by type
    worker_types = {}
    for worker in workers:
        worker_id = worker["worker_id"]

        # Extract worker type from ID (assumes format like 'worker-gpu-inference-1')
        parts = worker_id.split("-")
        if len(parts) >= 3:
            worker_type = "-".join(parts[1:-1])  # Get middle parts
        else:
            worker_type = "unknown"

        if worker_type not in worker_types:
            worker_types[worker_type] = []

        worker_types[worker_type].append(worker)

    return {
        "total_workers": len(workers),
        "worker_types": worker_types
    }


def generate_system_report() -> Dict[str, Any]:
    """
    Generate a comprehensive system report for diagnostics.

    Returns:
        Dictionary with system report information
    """
    # Get basic system status
    report = get_system_status()

    # Add additional diagnostic information
    report["diagnostics"] = {
        "job_distribution": analyze_job_distribution(),
        "error_summary": analyze_error_patterns(),
        "bottlenecks": identify_bottlenecks(),
        "recommendations": generate_recommendations()
    }

    return report


def analyze_job_distribution() -> Dict[str, Any]:
    """
    Analyze the distribution of jobs by type and status.

    Returns:
        Dictionary with job distribution analysis
    """
    # Get all jobs
    all_jobs = job_tracker.get_all_jobs(limit=1000)

    # Count jobs by type
    job_types = {}
    for job in all_jobs:
        job_type = job.get("job_type", "unknown")

        if job_type not in job_types:
            job_types[job_type] = {
                "total": 0,
                "status": {
                    "pending": 0,
                    "processing": 0,
                    "completed": 0,
                    "failed": 0,
                    "timeout": 0
                }
            }

        job_types[job_type]["total"] += 1

        # Count by status
        status = job.get("status", "unknown")
        if status in job_types[job_type]["status"]:
            job_types[job_type]["status"][status] += 1

    return {
        "total_jobs": len(all_jobs),
        "job_types": job_types
    }


def analyze_error_patterns() -> Dict[str, Any]:
    """
    Analyze error patterns in failed jobs.

    Returns:
        Dictionary with error pattern analysis
    """
    # Get all failed jobs
    all_jobs = job_tracker.get_all_jobs(limit=1000)
    failed_jobs = [job for job in all_jobs if job.get("status") in ["failed", "timeout"]]

    # Skip if no failed jobs
    if not failed_jobs:
        return {"total_failed_jobs": 0}

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

    return {
        "total_failed_jobs": len(failed_jobs),
        "common_errors": common_errors[:5]  # Top 5 errors
    }


def identify_bottlenecks() -> List[Dict[str, Any]]:
    """
    Identify potential bottlenecks in the system.

    Returns:
        List of dictionaries describing identified bottlenecks
    """
    bottlenecks = []

    # Check GPU utilization
    gpu_status = get_gpu_status()
    if gpu_status.get("available"):
        for device_id, device_info in gpu_status.get("devices", {}).items():
            if device_info.get("utilization_percent", 0) > 90:
                bottlenecks.append({
                    "type": "high_gpu_utilization",
                    "device": device_id,
                    "utilization": device_info["utilization_percent"],
                    "severity": "high"
                })

    # Check queue lengths
    queue_status = get_queue_status()
    for queue, length in queue_status.items():
        if length > 100:
            bottlenecks.append({
                "type": "long_queue",
                "queue": queue,
                "length": length,
                "severity": "medium"
            })

    # Check stalled tasks
    priority_status = priority_queue.get_queue_status()
    active_task = priority_status.get("active_task")
    if active_task:
        task_age = time.time() - active_task.get("registered_at", time.time())
        if task_age > 1800:  # 30 minutes
            bottlenecks.append({
                "type": "stalled_task",
                "task_id": active_task.get("task_id"),
                "job_id": active_task.get("job_id"),
                "age_seconds": task_age,
                "severity": "high"
            })

    # Check if we have many tasks waiting but no active task
    if not active_task and priority_status.get("total_tasks", 0) > 10:
        bottlenecks.append({
            "type": "idle_workers",
            "waiting_tasks": priority_status.get("total_tasks", 0),
            "severity": "high"
        })

    return bottlenecks


def generate_recommendations() -> List[str]:
    """
    Generate recommendations based on system status.

    Returns:
        List of recommendation strings
    """
    recommendations = []

    # Analyze bottlenecks
    bottlenecks = identify_bottlenecks()

    for bottleneck in bottlenecks:
        if bottleneck["type"] == "high_gpu_utilization":
            recommendations.append(f"Consider adding more GPU workers to handle the load on {bottleneck['device']}")

        elif bottleneck["type"] == "long_queue":
            recommendations.append(f"Add more workers to process the backlog in the '{bottleneck['queue']}' queue")

        elif bottleneck["type"] == "stalled_task":
            recommendations.append(f"Reset stalled task {bottleneck['task_id']} for job {bottleneck['job_id']}")

        elif bottleneck["type"] == "idle_workers":
            recommendations.append(
                "Check if GPU workers are running properly, as tasks are waiting but none are active")

    # Check job error patterns
    error_patterns = analyze_error_patterns()
    if error_patterns.get("total_failed_jobs", 0) > 10:
        recommendations.append(
            f"Investigate common error pattern: {error_patterns.get('common_errors', [('None', 0)])[0][0]}")

    # Add general recommendations
    job_counts = job_tracker.count_jobs_by_status()
    if job_counts.get("total", 0) > 5000:
        recommendations.append("Consider increasing job cleanup frequency to reduce database size")

    # If no specific recommendations, add general ones
    if not recommendations:
        recommendations.append("System appears to be operating normally")
        recommendations.append("Consider periodic model reloading to prevent GPU memory fragmentation")

    return recommendations


def get_priority_queue_status():
    """Get status of the priority queue system."""
    return priority_queue.get_queue_status()