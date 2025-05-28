# src/core/worker_status.py - Complete fixed version

"""
Simplified worker status management - keeps only what's needed.
"""

import time
from typing import Dict, Any, Optional
import redis
import logging

logger = logging.getLogger(__name__)


def normalize_redis_key(key):
    """Normalize a Redis key to string format."""
    if isinstance(key, bytes):
        return key.decode("utf-8")
    return key


def safe_redis_get(client: redis.Redis, key) -> Optional[str]:
    """
    Safely get a Redis value, handling the WRONGTYPE error we actually encountered.
    This fix is still needed because of the problematic parent key.
    """
    try:
        value = client.get(key)
        if value is None:
            return None

        if isinstance(value, bytes):
            return value.decode("utf-8")
        return str(value)

    except redis.ResponseError as e:
        if "WRONGTYPE" in str(e):
            # This was the actual problem we solved
            logger.debug(f"Key {key} has wrong type, skipping")
            return None
        else:
            logger.error(f"Redis error for key {key}: {e}")
            return None
    except Exception as e:
        logger.error(f"Unexpected error getting key {key}: {e}")
        return None


def get_worker_heartbeats(redis_client: redis.Redis) -> Dict[str, Dict[str, Any]]:
    """
    Get worker heartbeats - simplified version that keeps the actual fix we needed.
    """
    worker_info = {}

    try:
        # Simple approach: just get the keys we know workers create
        heartbeat_keys = redis_client.keys("dramatiq:__heartbeats__:*")

        logger.info(f"Found {len(heartbeat_keys)} heartbeat keys")

        current_time = time.time()

        for key in heartbeat_keys:
            try:
                normalized_key = normalize_redis_key(key)

                # Skip the problematic parent key (this was the real issue)
                if normalized_key == "dramatiq:__heartbeats__":
                    logger.debug(f"Skipping parent key: {normalized_key}")
                    continue

                # Extract worker ID (simple approach since pattern is consistent)
                worker_id = normalized_key.split(":")[-1]

                if not worker_id or len(worker_id) < 3:
                    continue

                # Get heartbeat value safely (this prevents the WRONGTYPE error)
                last_heartbeat = safe_redis_get(redis_client, key)

                if last_heartbeat is None:
                    continue

                # Parse heartbeat timestamp
                try:
                    heartbeat_value = float(last_heartbeat)
                except (ValueError, TypeError):
                    logger.warning(f"Invalid heartbeat value for {worker_id}: {last_heartbeat}")
                    continue

                heartbeat_age = current_time - heartbeat_value

                # Skip very old heartbeats
                if heartbeat_age > 300:  # 5 minutes
                    continue

                # Determine worker type from worker ID
                worker_type = "unknown"
                for wtype in ["gpu-inference", "gpu-embedding", "gpu-whisper", "cpu"]:
                    if wtype in worker_id:
                        worker_type = wtype
                        break

                # Determine status
                status = "healthy" if heartbeat_age < 60 else "stale"

                worker_info[worker_id] = {
                    "type": worker_type,
                    "last_heartbeat_seconds_ago": heartbeat_age,
                    "status": status,
                    "last_heartbeat_time": heartbeat_value
                }

            except Exception as e:
                logger.error(f"Error processing heartbeat key {key}: {e}")
                continue

    except Exception as e:
        logger.error(f"Error getting worker heartbeats: {e}")

    logger.info(f"Found {len(worker_info)} active workers")
    return worker_info


def get_active_worker_counts(redis_client: redis.Redis) -> Dict[str, int]:
    """Get counts of active workers grouped by type."""
    worker_info = get_worker_heartbeats(redis_client)

    worker_count = {
        "gpu-inference": 0,
        "gpu-embedding": 0,
        "gpu-whisper": 0,
        "cpu": 0,
        "total": 0
    }

    for worker_id, info in worker_info.items():
        if info.get("status") == "healthy":
            worker_type = info.get("type", "unknown")
            if worker_type in worker_count:
                worker_count[worker_type] += 1
            worker_count["total"] += 1

    return worker_count


def get_worker_summary(redis_client: redis.Redis) -> Dict[str, Any]:
    """Get a summary of worker status for system monitoring."""
    try:
        worker_counts = get_active_worker_counts(redis_client)
        worker_heartbeats = get_worker_heartbeats(redis_client)

        total_workers = len(worker_heartbeats)
        healthy_workers = worker_counts["total"]

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
        logger.error(f"Error getting worker summary: {e}")
        return {
            "total_workers": 0,
            "healthy_workers": 0,
            "worker_type_counts": {},
            "all_types_available": False,
            "health_ratio": 0
        }


def get_worker_status_for_ui(redis_client: redis.Redis) -> Dict[str, Any]:
    """Get comprehensive worker status for UI display."""
    try:
        worker_summary = get_worker_summary(redis_client)
        worker_heartbeats = get_worker_heartbeats(redis_client)

        return {
            "summary": worker_summary,
            "workers": worker_heartbeats
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
            "workers": {}
        }


def debug_redis_keys(redis_client: redis.Redis) -> Dict[str, Any]:
    """
    Debug function to analyze Redis keys for worker heartbeat issues.
    """
    try:
        # Get all heartbeat-related keys
        heartbeat_keys = redis_client.keys("dramatiq:__heartbeats__*")

        debug_info = {
            "total_heartbeat_keys": len(heartbeat_keys),
            "heartbeat_keys": [normalize_redis_key(key) for key in heartbeat_keys],
            "key_types": {},
            "problematic_keys": []
        }

        # Check each key type
        for key in heartbeat_keys:
            normalized_key = normalize_redis_key(key)
            try:
                key_type = redis_client.type(key)
                if isinstance(key_type, bytes):
                    key_type = key_type.decode('utf-8')

                debug_info["key_types"][normalized_key] = key_type

                # Flag problematic keys
                if key_type not in ['string', 'none']:
                    debug_info["problematic_keys"].append(normalized_key)

            except Exception as e:
                debug_info["key_types"][normalized_key] = f"error: {e}"

        return debug_info

    except Exception as e:
        logger.error(f"Error in debug_redis_keys: {e}")
        return {
            "error": str(e),
            "total_heartbeat_keys": 0,
            "heartbeat_keys": [],
            "key_types": {},
            "problematic_keys": []
        }


def clean_problematic_redis_keys(redis_client: redis.Redis) -> Dict[str, Any]:
    """
    Clean up the specific problematic key we identified.
    This is still useful for the specific issue we found.
    """
    cleaned_keys = []
    errors = []

    # Clean up the specific problematic key
    problematic_key = "dramatiq:__heartbeats__"

    try:
        key_type = redis_client.type(problematic_key)
        if isinstance(key_type, bytes):
            key_type = key_type.decode('utf-8')

        if key_type not in ['string', 'none']:
            redis_client.delete(problematic_key)
            cleaned_keys.append(problematic_key)
            logger.info(f"Deleted problematic key: {problematic_key}")
    except Exception as e:
        errors.append(f"Error cleaning {problematic_key}: {e}")

    return {
        "cleaned_keys": cleaned_keys,
        "errors": errors,
        "total_cleaned": len(cleaned_keys)
    }