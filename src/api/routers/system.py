# src/api/routers/system.py (Simplified for Job Chain)

import os
import time
import torch
import psutil
import logging
from fastapi import APIRouter, Depends, HTTPException
from typing import Dict, Any, List, Optional
import redis

from src.core.background.job_tracker import job_tracker
from src.core.background.job_chain import job_chain
from src.api.dependencies import get_redis_client, get_token_header
from src.core.worker_status import get_worker_status_for_ui
from src.config.settings import settings

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/health/detailed")
async def detailed_health_check():
    """
    Get detailed health information about the system and job chains.
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
    worker_info = get_worker_status_for_ui(redis_client)

    # Job chain queue status
    queue_status = job_chain.get_queue_status()

    # Job statistics
    job_stats = job_tracker.count_jobs_by_status()

    # GPU health
    gpu_health = {}
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            total_memory = torch.cuda.get_device_properties(i).total_memory
            allocated_memory = torch.cuda.memory_allocated(i)
            reserved_memory = torch.cuda.memory_reserved(i)
            free_memory = total_memory - reserved_memory

            gpu_health[f"gpu_{i}"] = {
                "device_name": torch.cuda.get_device_name(i),
                "total_memory_gb": total_memory / 1e9,
                "allocated_memory_gb": allocated_memory / 1e9,
                "reserved_memory_gb": reserved_memory / 1e9,
                "free_memory_gb": free_memory / 1e9,
                "free_percentage": (free_memory / total_memory) * 100
            }

    # Assemble the complete health data
    health_data = {
        "system": system_info,
        "workers": worker_info,
        "job_chains": queue_status,
        "jobs": job_stats,
        "gpu_health": gpu_health,
        "status": "healthy"
    }

    return health_data


@router.get("/workers")
async def get_workers_status(redis_client: redis.Redis = Depends(get_redis_client)):
    """
    Get simplified worker status information.
    """
    return get_worker_status_for_ui(redis_client)


@router.get("/queue-stats")
async def get_queue_stats():
    """
    Get job chain queue statistics.
    """
    try:
        return job_chain.get_queue_status()
    except Exception as e:
        logger.error(f"Error getting queue stats: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error getting queue stats: {str(e)}"
        )


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

        # Retrieval settings
        "retriever_top_k": getattr(settings, "retriever_top_k", 30),
        "reranker_top_k": getattr(settings, "reranker_top_k", 10),

        # Chunking settings
        "chunk_size": getattr(settings, "chunk_size", 1000),
        "chunk_overlap": getattr(settings, "chunk_overlap", 200),

        # Job chain settings
        "job_chain_enabled": True,
        "simplified_workers": True
    }

    return config


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
    log_files = {
        "system": "logs/system.log",
        "worker": "logs/worker.log",
        "api": "logs/api.log",
        "error": "logs/error.log"
    }

    if log_type not in log_files:
        raise HTTPException(400, "Invalid log type")

    log_path = log_files[log_type]

    if not os.path.exists(log_path):
        return {
            "status": "error",
            "message": f"Log file not found: {log_path}",
            "content": f"Log file {log_path} does not exist or is not accessible."
        }

    try:
        with open(log_path, "r") as f:
            lines = f.readlines()[-100:]  # Last 100 lines
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
        device_idx = int(gpu_id.split("_")[-1])

        if device_idx < 0 or device_idx >= torch.cuda.device_count():
            raise HTTPException(400, f"Invalid GPU ID: {gpu_id}")

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


@router.get("/job-chains")
async def get_all_job_chains():
    """
    Get overview of all active job chains.
    """
    try:
        # Get all jobs that might have active chains
        all_jobs = job_tracker.get_all_jobs(limit=100)
        active_chains = []

        for job in all_jobs:
            if job.get("status") == "processing":
                chain_status = job_chain.get_job_chain_status(job["job_id"])
                if chain_status:
                    active_chains.append(chain_status)

        return {
            "active_chains": active_chains,
            "total_active": len(active_chains),
            "queue_status": job_chain.get_queue_status()
        }
    except Exception as e:
        logger.error(f"Error getting job chains: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error getting job chains: {str(e)}"
        )


@router.post("/restart-workers", dependencies=[Depends(get_token_header)])
async def restart_workers():
    """
    Signal workers to restart gracefully.
    """
    redis_client = get_redis_client()

    try:
        # Set restart flags for all worker types
        worker_types = ["gpu", "cpu"]
        for worker_type in worker_types:
            restart_key = f"worker:restart:{worker_type}"
            redis_client.set(restart_key, "1", ex=300)  # 5 minutes

        return {
            "status": "success",
            "message": "Restart signals sent to all workers",
            "worker_types": worker_types
        }
    except Exception as e:
        logger.error(f"Error restarting workers: {str(e)}")
        raise HTTPException(500, f"Error restarting workers: {str(e)}")


def get_directory_size(path: str) -> int:
    """Calculate the total size of a directory."""
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if os.path.exists(fp) and os.path.isfile(fp):
                total_size += os.path.getsize(fp)
    return total_size