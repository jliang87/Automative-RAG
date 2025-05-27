# src/core/worker_status.py - Fixed to handle Redis key type issues

"""
Fixed worker status management with proper Redis key type handling.
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
    """Safely get a Redis value, handling type errors."""
    try:
        value = client.get(key)
        if value is None:
            return None

        if isinstance(value, bytes):
            return value.decode("utf-8")
        return str(value)

    except redis.ResponseError as e:
        if "WRONGTYPE" in str(e):
            # Key exists but is wrong type (hash, list, etc.)
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
    Get all worker heartbeats from Redis with improved error handling.
    """
    worker_info = {}

    try:
        # Get all possible heartbeat keys using SCAN for better performance
        heartbeat_keys = set()

        # Use SCAN to find keys matching heartbeat patterns
        patterns = [
            "dramatiq:__heartbeats__:*",
            "dramatiq:heartbeats:*",
            "worker:heartbeat:*"
        ]

        for pattern in patterns:
            try:
                for key in redis_client.scan_iter(match=pattern, count=100):
                    heartbeat_keys.add(key)
                logger.debug(f"Found {len(heartbeat_keys)} keys for pattern {pattern}")
            except Exception as e:
                logger.debug(f"Error scanning pattern {pattern}: {e}")

        logger.info(f"Total heartbeat keys found: {len(heartbeat_keys)}")

        # Process heartbeat keys
        current_time = time.time()

        for key in heartbeat_keys:
            try:
                normalized_key = normalize_redis_key(key)

                # Skip keys that don't contain worker IDs (like parent keys)
                if normalized_key.endswith(":__heartbeats__") or normalized_key.endswith(":heartbeats"):
                    logger.debug(f"Skipping parent key: {normalized_key}")
                    continue

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
                    parts = normalized_key.split(":")
                    if len(parts) > 1:
                        worker_id = parts[-1]

                if not worker_id or len(worker_id) < 3:  # Skip very short worker IDs
                    logger.debug(f"Skipping key with invalid worker ID: {normalized_key}")
                    continue

                # Get heartbeat value safely
                last_heartbeat = safe_redis_get(redis_client, key)

                if last_heartbeat is None:
                    logger.debug(f"No value for heartbeat key: {normalized_key}")
                    continue

                # Parse heartbeat timestamp
                try:
                    heartbeat_value = float(last_heartbeat)
                except (ValueError, TypeError):
                    logger.warning(f"Invalid heartbeat value for {worker_id}: {last_heartbeat}")
                    continue

                heartbeat_age = current_time - heartbeat_value

                # Skip very old heartbeats (older than 10 minutes)
                if heartbeat_age > 600:
                    logger.debug(f"Skipping old heartbeat for {worker_id}: {heartbeat_age:.1f}s")
                    continue

                # Determine worker type from worker ID
                worker_type = "unknown"
                for wtype in ["gpu-inference", "gpu-embedding", "gpu-whisper", "cpu"]:
                    if wtype in worker_id:
                        worker_type = wtype
                        break

                # Determine status (60 second threshold for healthy)
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
                continue

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
        "sample_keys": [],
        "key_types": {},
        "problematic_keys": []
    }

    try:
        # Use SCAN instead of KEYS for better performance
        all_keys = []
        for key in redis_client.scan_iter(count=1000):
            all_keys.append(key)

        debug_info["all_keys_count"] = len(all_keys)

        # Categorize keys and check their types
        for key in all_keys:
            key_str = normalize_redis_key(key)

            # Check key type
            try:
                key_type = redis_client.type(key).decode('utf-8') if isinstance(redis_client.type(key),
                                                                                bytes) else redis_client.type(key)
                debug_info["key_types"][key_str] = key_type

                # Flag problematic keys
                if "heartbeat" in key_str.lower() and key_type != "string":
                    debug_info["problematic_keys"].append({
                        "key": key_str,
                        "type": key_type,
                        "expected": "string"
                    })

            except Exception as e:
                debug_info["key_types"][key_str] = f"error: {e}"

            if "dramatiq" in key_str.lower():
                debug_info["dramatiq_keys"].append(key_str)

            if "heartbeat" in key_str.lower():
                debug_info["heartbeat_keys"].append(key_str)

            if "worker" in key_str.lower():
                debug_info["worker_keys"].append(key_str)

        # Get sample of all keys
        debug_info["sample_keys"] = [normalize_redis_key(k) for k in all_keys[:20]]

        logger.info(f"Redis debug info: found {len(debug_info['problematic_keys'])} problematic keys")

    except Exception as e:
        logger.error(f"Error debugging Redis keys: {e}")
        debug_info["error"] = str(e)

    return debug_info


def clean_problematic_redis_keys(redis_client: redis.Redis) -> Dict[str, Any]:
    """
    Clean up problematic Redis keys that might be interfering with heartbeats.
    This should be called carefully and only in development/debugging.
    """
    debug_info = debug_redis_keys(redis_client)
    problematic_keys = debug_info.get("problematic_keys", [])

    cleaned_keys = []
    errors = []

    for problem in problematic_keys:
        key = problem["key"]
        try:
            # Only delete keys that are definitely problematic
            if key.endswith(":__heartbeats__") or key.endswith(":heartbeats"):
                redis_client.delete(key)
                cleaned_keys.append(key)
                logger.info(f"Deleted problematic key: {key}")
        except Exception as e:
            errors.append(f"Error deleting {key}: {e}")
            logger.error(f"Error deleting {key}: {e}")

    return {
        "cleaned_keys": cleaned_keys,
        "errors": errors,
        "total_cleaned": len(cleaned_keys)
    }


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