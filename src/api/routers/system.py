# Add this to src/api/routers/system.py

import os
import time
import torch
from fastapi import APIRouter, Depends, HTTPException
from typing import Dict, Any, List
import psutil

from src.core.background.job_tracker import job_tracker
from src.api.dependencies import get_redis_client, get_token_header
from src.core.background.common import check_gpu_health

router = APIRouter()


@router.get("/health/detailed")
async def detailed_health_check():
    """
    Get detailed health information about the system, workers, and resources.
    """
    redis_client = get_redis_client()

    # Basic system info
    system_info = {
        "timestamp": time.time(),
        "hostname": os.uname().nodename if hasattr(os, "uname") else "unknown",
        "cpu_usage": psutil.cpu_percent(),
        "memory_usage": psutil.virtual_memory().percent,
        "uptime": time.time() - psutil.boot_time()
    }

    # Worker health information
    worker_info = {}
    worker_heartbeats = redis_client.keys("dramatiq:__heartbeats__:*")

    for key in worker_heartbeats:
        # Check if key is bytes or string (depending on Redis client configuration)
        worker_id = key.split(":")[-1] if isinstance(key, str) else key.decode("utf-8").split(":")[-1]
        last_heartbeat = redis_client.get(key)

        if last_heartbeat:
            heartbeat_age = time.time() - float(last_heartbeat)
            worker_type = "unknown"

            # Extract worker type from ID
            for wtype in ["gpu-inference", "gpu-embedding", "gpu-whisper", "cpu", "system"]:
                if wtype in worker_id:
                    worker_type = wtype
                    break

            worker_info[worker_id] = {
                "type": worker_type,
                "last_heartbeat_seconds_ago": heartbeat_age,
                "status": "healthy" if heartbeat_age < 60 else "stalled"
            }

    # Queue status
    queue_stats = {}
    for queue in ["inference_tasks", "embedding_tasks", "transcription_tasks", "cpu_tasks", "system_tasks"]:
        queue_key = f"dramatiq:{queue}:msgs"
        queue_stats[queue] = redis_client.llen(queue_key)

    # Priority queue status
    from src.core.background.priority_queue import priority_queue
    priority_status = priority_queue.get_queue_status()

    # GPU health
    gpu_health = {}
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            is_healthy, message = check_gpu_health()

            # Get memory information
            total_memory = torch.cuda.get_device_properties(i).total_memory
            allocated_memory = torch.cuda.memory_allocated(i)
            reserved_memory = torch.cuda.memory_reserved(i)
            free_memory = total_memory - reserved_memory

            gpu_health[f"gpu_{i}"] = {
                "device_name": torch.cuda.get_device_name(i),
                "is_healthy": is_healthy,
                "health_message": message,
                "total_memory_gb": total_memory / 1e9,
                "allocated_memory_gb": allocated_memory / 1e9,
                "reserved_memory_gb": reserved_memory / 1e9,
                "free_memory_gb": free_memory / 1e9,
                "free_percentage": (free_memory / total_memory) * 100
            }

    # Model loading status
    model_status = {}
    for model_type in ["embedding", "llm", "colbert", "whisper"]:
        status_key = f"model_loaded:{model_type}"
        status = redis_client.get(status_key)
        model_status[model_type] = {
            "loaded": status == b"1" if status else False,
            "loading_time": float(redis_client.get(f"model_loading_time:{model_type}") or 0)
        }

    # Assemble the complete health data
    health_data = {
        "system": system_info,
        "workers": worker_info,
        "queues": queue_stats,
        "priority_queue": priority_status,
        "gpu_health": gpu_health,
        "model_status": model_status,
        "status": "healthy" if all(w["status"] == "healthy" for w in worker_info.values()) else "degraded"
    }

    return health_data


@router.post("/watchdog", dependencies=[Depends(get_token_header)])
async def trigger_system_watchdog():
    """Trigger the system watchdog task."""
    from src.core.background.actors.system import system_watchdog
    system_watchdog.send()
    return {"status": "watchdog_triggered"}


@router.post("/restart-worker/{worker_type}", dependencies=[Depends(get_token_header)])
async def restart_worker(worker_type: str):
    """Signal a worker to restart itself."""
    redis_client = get_redis_client()

    # Validate worker type
    if worker_type not in ["gpu-inference", "gpu-embedding", "gpu-whisper", "cpu", "system"]:
        raise HTTPException(400, "Invalid worker type")

    # Set a restart flag in Redis
    restart_key = f"worker:restart:{worker_type}"
    redis_client.set(restart_key, "1", ex=300)  # Expires in 5 minutes

    return {"status": f"restart signal sent to {worker_type} workers"}