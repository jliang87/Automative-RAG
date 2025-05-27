"""
Simplified worker status management module.

Core functions for checking worker status and availability.
UI-specific formatting is now handled by the API layer.
"""

import time
from typing import Dict, Any, Optional
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

    try:
        worker_heartbeats = redis_client.keys("dramatiq:__heartbeats__:*")

        for key in worker_heartbeats:
            normalized_key = normalize_redis_key(key)
            worker_id = normalized_key.split(":")[-1]
            last_heartbeat = redis_client.get(key)

            # Try to parse heartbeat
            try:
                heartbeat_value = float(last_heartbeat) if last_heartbeat else 0
                heartbeat_age = time.time() - heartbeat_value

                # Determine worker type from worker ID
                worker_type = "unknown"
                for wtype in ["gpu-inference", "gpu-embedding", "gpu-whisper", "cpu"]:
                    if wtype in worker_id:
                        worker_type = wtype
                        break

                # Determine status based on heartbeat age (60 second threshold)
                status = "healthy" if heartbeat_age < 60 else "stale"

                worker_info[worker_id] = {
                    "type": worker_type,
                    "last_heartbeat_seconds_ago": heartbeat_age,
                    "status": status,
                    "last_heartbeat_time": heartbeat_value
                }
            except (ValueError, TypeError):
                # Handle case where heartbeat value is invalid
                worker_info[worker_id] = {
                    "type": "unknown",
                    "last_heartbeat_seconds_ago": 0,
                    "status": "error",
                    "last_heartbeat_time": 0
                }
    except Exception as e:
        logger.error(f"Error getting worker heartbeats: {str(e)}")

    return worker_info


def get_active_worker_counts(redis_client: redis.Redis) -> Dict[str, int]:
    """
    Get counts of active workers grouped by type.

    Args:
        redis_client: Redis client instance

    Returns:
        Dictionary mapping worker types to counts of healthy workers
    """
    worker_info = get_worker_heartbeats(redis_client)

    # Initialize counter for each worker type
    worker_count = {
        "gpu-inference": 0,
        "gpu-embedding": 0,
        "gpu-whisper": 0,
        "cpu": 0,
        "total": 0
    }

    # Count workers by type (only healthy ones)
    for worker_id, info in worker_info.items():
        if info.get("status") == "healthy":
            worker_type = info.get("type", "unknown")
            if worker_type in worker_count:
                worker_count[worker_type] += 1
            worker_count["total"] += 1

    return worker_count


def check_worker_availability(redis_client: redis.Redis, worker_type: str) -> bool:
    """
    Check if a specific type of worker is available.

    Args:
        redis_client: Redis client instance
        worker_type: Type of worker to check for

    Returns:
        True if at least one worker of the required type is healthy
    """
    try:
        worker_counts = get_active_worker_counts(redis_client)
        return worker_counts.get(worker_type, 0) > 0
    except Exception as e:
        logger.error(f"Error checking worker availability for {worker_type}: {str(e)}")
        return False


def get_worker_summary(redis_client: redis.Redis) -> Dict[str, Any]:
    """
    Get a summary of worker status for system monitoring.

    Args:
        redis_client: Redis client instance

    Returns:
        Dictionary with worker summary information
    """
    try:
        worker_counts = get_active_worker_counts(redis_client)
        worker_heartbeats = get_worker_heartbeats(redis_client)

        # Calculate summary stats
        total_workers = len(worker_heartbeats)
        healthy_workers = worker_counts["total"]

        # Check if all essential worker types are available
        essential_types = ["gpu-inference", "gpu-embedding", "gpu-whisper", "cpu"]
        all_types_available = all(worker_counts.get(wtype, 0) > 0 for wtype in essential_types)

        return {
            "total_workers": total_workers,
            "healthy_workers": healthy_workers,
            "worker_type_counts": {k: v for k, v in worker_counts.items() if k != "total"},
            "all_types_available": all_types_available,
            "health_ratio": healthy_workers / total_workers if total_workers > 0 else 0
        }
    except Exception as e:
        logger.error(f"Error getting worker summary: {str(e)}")
        return {
            "total_workers": 0,
            "healthy_workers": 0,
            "worker_type_counts": {},
            "all_types_available": False,
            "health_ratio": 0
        }