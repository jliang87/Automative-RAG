# Updated src/api/routers/system.py with additional endpoints

import os
import time
import torch
import psutil
import logging
import json
from fastapi import APIRouter, Depends, HTTPException, File, UploadFile
from typing import Dict, Any, List, Optional
import redis

from src.core.background.job_tracker import job_tracker
from src.api.dependencies import get_redis_client, get_token_header
from src.core.background.common import check_gpu_health
from src.core.worker_status import get_worker_heartbeats, get_worker_status, get_worker_status_for_ui
from src.config.settings import settings

router = APIRouter()
logger = logging.getLogger(__name__)

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

    # Worker health information using the centralized function
    worker_info = get_worker_heartbeats(redis_client)

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


@router.get("/workers")
async def get_workers_status(redis_client: redis.Redis = Depends(get_redis_client)):
    """
    Get detailed information about worker processes.
    """
    return get_worker_status(redis_client)


@router.get("/worker-ui-status")
async def get_worker_ui_status(redis_client: redis.Redis = Depends(get_redis_client)):
    """
    Get worker status formatted for UI display.
    """
    return get_worker_status_for_ui(redis_client)


@router.get("/queue-stats")
async def get_queue_stats(redis_client: redis.Redis = Depends(get_redis_client)):
    """
    Get statistics about Dramatiq message queues.
    """
    queue_stats = {}
    queues = {
        "inference_tasks": "LLM Inference Tasks",
        "embedding_tasks": "Embedding Tasks",
        "transcription_tasks": "Transcription Tasks",
        "cpu_tasks": "CPU Tasks",
        "system_tasks": "System Tasks",
        "default": "Default Queue"
    }

    for queue_name, display_name in queues.items():
        # Get queued messages count
        queue_key = f"dramatiq:{queue_name}:msgs"
        queued = redis_client.llen(queue_key)

        # Get processed and failed counts if available
        processed_key = f"dramatiq:{queue_name}:processed"
        processed = int(redis_client.get(processed_key) or 0)

        failed_key = f"dramatiq:{queue_name}:failed"
        failed = int(redis_client.get(failed_key) or 0)

        retried_key = f"dramatiq:{queue_name}:retried"
        retried = int(redis_client.get(retried_key) or 0)

        queue_stats[queue_name] = {
            "display_name": display_name,
            "messages": queued,
            "processed": processed,
            "failed": failed,
            "retried": retried
        }

    return {"queues": queue_stats}


@router.get("/config")
async def get_system_config():
    """
    Get current system configuration settings.
    """
    config = {
        # Basic settings
        "host": getattr(settings, "host", "0.0.0.0"),
        "port": getattr(settings, "port", 8000),
        "api_auth_enabled": getattr(settings, "api_auth_enabled", True),

        # Model settings
        "default_embedding_model": getattr(settings, "default_embedding_model", "bge-m3"),
        "default_colbert_model": getattr(settings, "default_colbert_model", "colbertv2.0"),
        "default_llm_model": getattr(settings, "default_llm_model", "DeepSeek-Coder-V2"),
        "default_whisper_model": getattr(settings, "default_whisper_model", "medium"),

        # GPU settings
        "device": getattr(settings, "device", "cuda:0"),
        "use_fp16": getattr(settings, "use_fp16", True),
        "batch_size": getattr(settings, "batch_size", 16),
        "llm_use_4bit": getattr(settings, "llm_use_4bit", True),
        "llm_use_8bit": getattr(settings, "llm_use_8bit", False),

        # Retrieval settings
        "retriever_top_k": getattr(settings, "retriever_top_k", 30),
        "reranker_top_k": getattr(settings, "reranker_top_k", 10),
        "colbert_batch_size": getattr(settings, "colbert_batch_size", 16),
        "colbert_weight": getattr(settings, "colbert_weight", 0.8),
        "bge_weight": getattr(settings, "bge_weight", 0.2),

        # Chunking settings
        "chunk_size": getattr(settings, "chunk_size", 1000),
        "chunk_overlap": getattr(settings, "chunk_overlap", 200),

        # Directory settings
        "data_dir": getattr(settings, "data_dir", "data"),
        "models_dir": getattr(settings, "models_dir", "models"),
        "upload_dir": getattr(settings, "upload_dir", "data/uploads"),
    }

    return config


@router.post("/update-config", dependencies=[Depends(get_token_header)])
async def update_system_config(config_updates: Dict[str, Any]):
    """
    Update system configuration settings.
    For demonstration purposes only - in a real system, this would actually modify settings.
    """
    # In a production system, this would update a configuration file or database
    # For this demo, we'll just acknowledge the request
    return {
        "status": "success",
        "message": "Config updates acknowledged",
        "updated_fields": list(config_updates.keys())
    }


@router.post("/reload-model", dependencies=[Depends(get_token_header)])
async def reload_model(request: Dict[str, str]):
    """
    Send a command to reload a specific model.
    """
    model_type = request.get("model_type")
    if not model_type or model_type not in ["embedding", "llm", "colbert", "whisper"]:
        raise HTTPException(400, "Invalid model type")

    redis_client = get_redis_client()

    # Set a reload flag in Redis
    reload_key = f"model:reload:{model_type}"
    redis_client.set(reload_key, "1", ex=300)  # Expires in 5 minutes

    return {
        "status": "success",
        "message": f"Reload signal sent for {model_type} model"
    }


@router.get("/disk-usage")
async def get_disk_usage():
    """
    Get system disk usage information.
    """
    # Get partition information
    partitions = {}
    for partition in psutil.disk_partitions():
        try:
            usage = psutil.disk_usage(partition.mountpoint)
            partitions[partition.mountpoint] = {
                "total": usage.total,
                "used": usage.used,
                "free": usage.free,
                "percent": usage.percent,
                "fstype": partition.fstype
            }
        except PermissionError:
            # Skip partitions we can't access
            continue

    # Get sizes of important data directories
    data_dirs = {}
    important_dirs = [
        ("uploads", os.path.join("data", "uploads")),
        ("models", "models"),
        ("logs", "logs"),
        ("temp", os.path.join("data", "temp"))
    ]

    for name, path in important_dirs:
        if os.path.exists(path):
            dir_size = get_directory_size(path)
            data_dirs[name] = {
                "path": path,
                "size": dir_size
            }

    return {
        "partitions": partitions,
        "data_dirs": data_dirs
    }


@router.get("/logs/{log_type}")
async def get_system_logs(log_type: str):
    """
    Get system logs by type.
    """
    # Map log type to file path
    log_files = {
        "system": "logs/system.log",
        "worker": "logs/worker.log",
        "api": "logs/api.log",
        "error": "logs/error.log"
    }

    if log_type not in log_files:
        raise HTTPException(400, "Invalid log type")

    log_path = log_files[log_type]

    # Check if log file exists
    if not os.path.exists(log_path):
        return {
            "status": "error",
            "message": f"Log file not found: {log_path}",
            "content": f"Log file {log_path} does not exist or is not accessible."
        }

    # Read the tail of the log file (last 100 lines)
    try:
        with open(log_path, "r") as f:
            # Read last 100 lines
            lines = f.readlines()[-100:]
            content = "".join(lines)

        return {
            "status": "success",
            "log_type": log_type,
            "content": content
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error reading log file: {str(e)}",
            "content": f"Could not read log file due to error: {str(e)}"
        }


@router.post("/clear-gpu-cache")
async def clear_gpu_cache(request: Dict[str, str]):
    """
    Clear CUDA cache for a specific GPU.
    """
    if not torch.cuda.is_available():
        raise HTTPException(400, "CUDA not available")

    gpu_id = request.get("gpu_id", "gpu_0")

    try:
        # Extract device index from gpu_id
        device_idx = int(gpu_id.split("_")[-1])

        # Ensure valid device index
        if device_idx < 0 or device_idx >= torch.cuda.device_count():
            raise HTTPException(400, f"Invalid GPU ID: {gpu_id}")

        # Set the device
        torch.cuda.set_device(device_idx)

        # Get memory before clearing
        before_allocated = torch.cuda.memory_allocated(device_idx)
        before_reserved = torch.cuda.memory_reserved(device_idx)

        # Clear cache
        torch.cuda.empty_cache()

        # Get memory after clearing
        after_allocated = torch.cuda.memory_allocated(device_idx)
        after_reserved = torch.cuda.memory_reserved(device_idx)

        return {
            "status": "success",
            "message": f"Cache cleared for GPU {device_idx}",
            "memory_before": {
                "allocated_gb": before_allocated / 1e9,
                "reserved_gb": before_reserved / 1e9
            },
            "memory_after": {
                "allocated_gb": after_allocated / 1e9,
                "reserved_gb": after_reserved / 1e9
            },
            "freed_memory_gb": (before_reserved - after_reserved) / 1e9
        }
    except Exception as e:
        raise HTTPException(500, f"Error clearing GPU cache: {str(e)}")


@router.post("/flush-queue")
async def flush_queue(request: Dict[str, str], redis_client: redis.Redis = Depends(get_redis_client)):
    """
    Flush a specific message queue.
    """
    queue = request.get("queue")
    if not queue:
        raise HTTPException(400, "Queue name is required")

    try:
        # Get queue length before flushing
        queue_key = f"dramatiq:{queue}:msgs"
        before_count = redis_client.llen(queue_key)

        # Flush the queue
        redis_client.delete(queue_key)

        return {
            "status": "success",
            "message": f"Queue {queue} flushed successfully",
            "removed_messages": before_count
        }
    except Exception as e:
        raise HTTPException(500, f"Error flushing queue: {str(e)}")


@router.post("/terminate-task")
async def terminate_task(request: Dict[str, str], redis_client: redis.Redis = Depends(get_redis_client)):
    """
    Terminate a running task.
    """
    task_id = request.get("task_id")
    if not task_id:
        raise HTTPException(400, "Task ID is required")

    try:
        # Check if task is active
        active_task_json = redis_client.get("active_gpu_task")

        if not active_task_json:
            return {
                "status": "error",
                "message": "No active task found"
            }

        active_task = json.loads(active_task_json)

        if active_task.get("task_id") != task_id:
            return {
                "status": "error",
                "message": f"Task {task_id} is not currently active"
            }

        # Remove the active task
        redis_client.delete("active_gpu_task")

        # Update the job status if job_id is available
        job_id = active_task.get("job_id")
        if job_id:
            job_tracker.update_job_status(
                job_id,
                "failed",
                error="Task terminated by user request"
            )

        return {
            "status": "success",
            "message": f"Task {task_id} terminated successfully",
            "task_data": active_task
        }
    except Exception as e:
        raise HTTPException(500, f"Error terminating task: {str(e)}")


@router.post("/prioritize-job", dependencies=[Depends(get_token_header)])
async def prioritize_job(request: Dict[str, str]):
    """
    Boost priority of a specific job in the priority queue.
    """
    job_id = request.get("job_id")
    if not job_id:
        raise HTTPException(400, "Job ID is required")

    redis_client = get_redis_client()

    try:
        # Import priority queue
        from src.core.background.priority_queue import priority_queue

        # Call the boost_job_priority method
        result = priority_queue.boost_job_priority(job_id)

        return {
            "status": "success",
            "message": f"Job {job_id} priority boosted",
            "details": result
        }
    except Exception as e:
        logger.error(f"Error prioritizing job: {str(e)}")
        raise HTTPException(500, f"Error prioritizing job: {str(e)}")


# Helper functions
def get_directory_size(path: str) -> int:
    """Calculate the total size of a directory."""
    total_size = 0

    for dirpath, dirnames, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if os.path.exists(fp) and os.path.isfile(fp):
                total_size += os.path.getsize(fp)

    return total_size