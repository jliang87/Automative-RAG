# src/core/worker_status.py - Fixed version

"""
Fixed worker status management module with improved heartbeat detection.
"""

import time
from typing import Dict, Any, Optional
import redis
import logging

logger = logging.getLogger(__name__)


def normalize_redis_key(key):
    """
    Normalize a Redis key to string format regardless of whether it's bytes or string.
    """
    if isinstance(key, bytes):
        return key.decode("utf-8")
    return key


def get_worker_heartbeats(redis_client: redis.Redis) -> Dict[str, Dict[str, Any]]:
    """
    Get all worker heartbeats from Redis with improved detection.
    """
    worker_info = {}

    try:
        # Try multiple heartbeat key patterns
        heartbeat_patterns = [
            "dramatiq:__heartbeats__:*",
            "dramatiq:heartbeats:*",
            "worker:heartbeat:*",
            "*heartbeat*"
        ]

        all_heartbeat_keys = set()

        # Collect all possible heartbeat keys
        for pattern in heartbeat_patterns:
            try:
                keys = redis_client.keys(pattern)
                all_heartbeat_keys.update(keys)
                logger.debug(f"Found {len(keys)} keys for pattern {pattern}")
            except Exception as e:
                logger.debug(f"Error searching pattern {pattern}: {e}")

        logger.info(f"Total heartbeat keys found: {len(all_heartbeat_keys)}")

        # If no heartbeat keys found, check if workers are using a different pattern
        if not all_heartbeat_keys:
            # Check all keys in Redis that might be heartbeats
            try:
                all_keys = redis_client.keys("*")
                logger.info(f"Total Redis keys: {len(all_keys)}")

                # Look for any keys that might be heartbeats
                potential_heartbeats = []
                for key in all_keys:
                    key_str = normalize_redis_key(key)
                    if any(word in key_str.lower() for word in ['heartbeat', 'worker', 'dramatiq']):
                        potential_heartbeats.append(key_str)

                logger.info(f"Potential heartbeat keys: {potential_heartbeats}")
                all_heartbeat_keys.update(potential_heartbeats)
            except Exception as e:
                logger.error(f"Error getting all Redis keys: {e}")

        # Process heartbeat keys
        current_time = time.time()

        for key in all_heartbeat_keys:
            try:
                normalized_key = normalize_redis_key(key)

                # Extract worker ID from different key patterns
                worker_id = None
                if ":__heartbeats__:" in normalized_key:
                    worker_id = normalized_key.split(":__heartbeats__:")[-1]
                elif ":heartbeats:" in normalized_key:
                    worker_id = normalized_key.split(":heartbeats:")[-1]
                elif ":heartbeat:" in normalized_key:
                    worker_id = normalized_key.split(":heartbeat:")[-1]
                else:
                    # Extract from end of key
                    worker_id = normalized_key.split(":")[-1]

                if not worker_id:
                    continue

                # Get heartbeat value
                last_heartbeat = redis_client.get(key)

                if last_heartbeat is None:
                    continue

                # Parse heartbeat timestamp
                try:
                    if isinstance(last_heartbeat, bytes):
                        heartbeat_value = float(last_heartbeat.decode())
                    else:
                        heartbeat_value = float(last_heartbeat)
                except (ValueError, AttributeError):
                    logger.warning(f"Invalid heartbeat value for {worker_id}: {last_heartbeat}")
                    continue

                heartbeat_age = current_time - heartbeat_value

                # Determine worker type from worker ID
                worker_type = "unknown"
                for wtype in ["gpu-inference", "gpu-embedding", "gpu-whisper", "cpu"]:
                    if wtype in worker_id:
                        worker_type = wtype
                        break

                # Determine status (60 second threshold)
                status = "healthy" if heartbeat_age < 60 else "stale"

                worker_info[worker_id] = {
                    "type": worker_type,
                    "last_heartbeat_seconds_ago": heartbeat_age,
                    "status": status,
                    "last_heartbeat_time": heartbeat_value,
                    "key_pattern": normalized_key  # For debugging
                }

                logger.debug(f"Worker {worker_id}: type={worker_type}, age={heartbeat_age:.1f}s, status={status}")

            except Exception as e:
                logger.error(f"Error processing heartbeat key {key}: {e}")

    except Exception as e:
        logger.error(f"Error getting worker heartbeats: {e}")

    logger.info(f"Found {len(worker_info)} workers with heartbeats")
    return worker_info


def debug_redis_keys(redis_client: redis.Redis) -> Dict[str, Any]:
    """
    Debug function to inspect Redis keys and help diagnose heartbeat issues.
    """
    debug_info = {
        "all_keys_count": 0,
        "dramatiq_keys": [],
        "heartbeat_keys": [],
        "worker_keys": [],
        "sample_keys": []
    }

    try:
        # Get all keys
        all_keys = redis_client.keys("*")
        debug_info["all_keys_count"] = len(all_keys)

        # Categorize keys
        for key in all_keys:
            key_str = normalize_redis_key(key)

            if "dramatiq" in key_str.lower():
                debug_info["dramatiq_keys"].append(key_str)

            if "heartbeat" in key_str.lower():
                debug_info["heartbeat_keys"].append(key_str)

            if "worker" in key_str.lower():
                debug_info["worker_keys"].append(key_str)

        # Get sample of all keys
        debug_info["sample_keys"] = [normalize_redis_key(k) for k in all_keys[:20]]

        logger.info(f"Redis debug info: {debug_info}")

    except Exception as e:
        logger.error(f"Error debugging Redis keys: {e}")
        debug_info["error"] = str(e)

    return debug_info


def get_active_worker_counts(redis_client: redis.Redis) -> Dict[str, int]:
    """
    Get counts of active workers grouped by type.
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

    logger.info(f"Active worker counts: {worker_count}")
    return worker_count


def check_worker_availability(redis_client: redis.Redis, worker_type: str) -> bool:
    """
    Check if a specific type of worker is available.
    """
    try:
        worker_counts = get_active_worker_counts(redis_client)
        return worker_counts.get(worker_type, 0) > 0
    except Exception as e:
        logger.error(f"Error checking worker availability for {worker_type}: {e}")
        return False


def get_worker_summary(redis_client: redis.Redis) -> Dict[str, Any]:
    """
    Get a summary of worker status for system monitoring.
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

        summary = {
            "total_workers": total_workers,
            "healthy_workers": healthy_workers,
            "worker_type_counts": {k: v for k, v in worker_counts.items() if k != "total"},
            "all_types_available": all_types_available,
            "health_ratio": healthy_workers / total_workers if total_workers > 0 else 0
        }

        logger.info(f"Worker summary: {summary}")
        return summary

    except Exception as e:
        logger.error(f"Error getting worker summary: {e}")
        return {
            "total_workers": 0,
            "healthy_workers": 0,
            "worker_type_counts": {},
            "all_types_available": False,
            "health_ratio": 0
        }


# Additional function for the API to help debug
def get_worker_status_for_ui(redis_client: redis.Redis) -> Dict[str, Any]:
    """
    Get comprehensive worker status for UI display with debugging info.
    """
    try:
        # Get basic worker info
        worker_summary = get_worker_summary(redis_client)
        worker_heartbeats = get_worker_heartbeats(redis_client)

        # Add debug info if no workers found
        debug_info = None
        if worker_summary["total_workers"] == 0:
            debug_info = debug_redis_keys(redis_client)

        return {
            "summary": worker_summary,
            "workers": worker_heartbeats,
            "debug_info": debug_info
        }

    except Exception as e:
        logger.error(f"Error getting worker status for UI: {e}")
        return {
            "summary": {
                "total_workers": 0,
                "healthy_workers": 0,
                "worker_type_counts": {},
                "all_types_available": False,
                "health_ratio": 0
            },
            "workers": {},
            "debug_info": {"error": str(e)}
        }