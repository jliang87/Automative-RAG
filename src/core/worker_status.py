"""
Centralized worker status management module.

This module provides functions for checking worker status, health and availability
across the application.
"""

import time
from typing import Dict, List, Any, Optional, Tuple
import redis
import logging

logger = logging.getLogger(__name__)

def normalize_redis_key(key):
    """
    Normalize a Redis key to string format regardless of whether it's bytes or string.

    Args:
        key: A Redis key that could be either bytes or string

    Returns:
        String representation of the key
    """
    if isinstance(key, bytes):
        return key.decode("utf-8")
    return key

def get_worker_heartbeats(redis_client: redis.Redis) -> Dict[str, Dict[str, Any]]:
    """
    Get all worker heartbeats from Redis.

    Args:
        redis_client: Redis client instance

    Returns:
        Dictionary mapping worker IDs to worker information
    """
    worker_info = {}
    worker_heartbeats = redis_client.keys("dramatiq:__heartbeats__:*")

    for key in worker_heartbeats:
        normalized_key = normalize_redis_key(key)
        worker_id = normalized_key.split(":")[-1]
        last_heartbeat = redis_client.get(key)

        # Try to parse heartbeat
        try:
            heartbeat_value = float(last_heartbeat) if last_heartbeat else 0
            heartbeat_age = time.time() - heartbeat_value

            # Determine worker type
            worker_type = "unknown"
            for wtype in ["gpu-inference", "gpu-embedding", "gpu-whisper", "cpu", "system"]:
                if wtype in worker_id:
                    worker_type = wtype
                    break

            # Determine status based on heartbeat age
            status = "healthy" if heartbeat_age < 60 else "stalled"

            worker_info[worker_id] = {
                "type": worker_type,
                "last_heartbeat_seconds_ago": heartbeat_age,
                "status": status
            }
        except (ValueError, TypeError):
            # Handle case where heartbeat value is invalid
            worker_info[worker_id] = {
                "type": "unknown",
                "last_heartbeat_seconds_ago": 0,
                "status": "unknown"
            }

    return worker_info

def get_active_worker_counts(redis_client: redis.Redis) -> Dict[str, int]:
    """
    Get counts of active workers grouped by type.

    Args:
        redis_client: Redis client instance

    Returns:
        Dictionary mapping worker types to counts
    """
    worker_info = get_worker_heartbeats(redis_client)

    # Initialize counter for each worker type
    worker_count = {
        "gpu-inference": 0,
        "gpu-embedding": 0,
        "gpu-whisper": 0,
        "cpu": 0,
        "system": 0,
    }

    # Count workers by type
    for worker_id, info in worker_info.items():
        worker_type = info.get("type", "unknown")
        if worker_type in worker_count and info.get("status") == "healthy":
            worker_count[worker_type] += 1

    return worker_count


def get_worker_status_for_ui(redis_client: redis.Redis) -> Dict[str, Any]:
    """
    Get worker status information formatted for UI display with Chinese localization.

    Args:
        redis_client: Redis client instance

    Returns:
        Dictionary with worker status information for UI in Chinese
    """
    worker_counts = get_active_worker_counts(redis_client)

    # Format for UI display with Chinese descriptions
    active_workers = {}
    worker_types = {
        "gpu-inference": "LLM 和重排序",
        "gpu-embedding": "向量嵌入",
        "gpu-whisper": "语音转录",
        "cpu": "文本处理",
        "system": "系统管理"
    }

    for worker_type, description in worker_types.items():
        count = worker_counts.get(worker_type, 0)
        if count > 0:
            active_workers[worker_type] = {
                "count": count,
                "description": description,
                "status": "活跃"  # "active" in Chinese
            }

    # Get queue information
    queue_stats = {}
    queue_names = {
        "inference_tasks": "推理任务",
        "embedding_tasks": "嵌入任务",
        "transcription_tasks": "转录任务",
        "cpu_tasks": "CPU任务",
        "system_tasks": "系统任务"
    }

    for queue_key, queue_name in queue_names.items():
        redis_queue_key = f"dramatiq:{queue_key}:msgs"
        queue_stats[queue_name] = redis_client.llen(redis_queue_key)

    return {
        "active_workers": active_workers,
        "queue_stats": queue_stats
    }

def get_worker_status(redis_client: redis.Redis) -> Dict[str, Any]:
    """
    Get comprehensive information about running worker processes.
    Incorporates the functionality from monitoring.py

    Args:
        redis_client: Redis client instance

    Returns:
        Dictionary with worker status information
    """
    try:
        # Get worker information from Redis
        workers = []
        worker_keys = redis_client.keys("dramatiq:__heartbeats__:*")

        for key in worker_keys:
            normalized_key = normalize_redis_key(key)
            worker_id = normalized_key.split(":")[-1]
            last_heartbeat = redis_client.get(key)

            if last_heartbeat:
                try:
                    heartbeat_age = time.time() - float(last_heartbeat)
                    workers.append({
                        "worker_id": worker_id,
                        "last_heartbeat_seconds_ago": heartbeat_age
                    })
                except (ValueError, TypeError):
                    # Handle invalid heartbeat format
                    workers.append({
                        "worker_id": worker_id,
                        "last_heartbeat_seconds_ago": None
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
    except Exception as e:
        logger.error(f"Error getting worker status: {str(e)}")
        return {
            "total_workers": 0,
            "worker_types": {},
            "error": str(e)
        }

def check_worker_availability(redis_client: redis.Redis, required_type: str) -> bool:
    """
    Check if a specific type of worker is available.

    Args:
        redis_client: Redis client instance
        required_type: Type of worker to check for

    Returns:
        True if at least one worker of the required type is active
    """
    worker_counts = get_active_worker_counts(redis_client)
    return worker_counts.get(required_type, 0) > 0