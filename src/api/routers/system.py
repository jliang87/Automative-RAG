# src/api/routers/system.py - Updated with improved worker status

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
from src.core.worker_status import get_worker_status_for_ui, get_worker_summary, debug_redis_keys
from src.config.settings import settings

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/health/detailed")
async def detailed_health_check():
    """
    Enhanced health check with improved worker detection.
    """
    redis_client = get_redis_client()

    try:
        # Basic system info
        system_info = {
            "timestamp": time.time(),
            "hostname": os.uname().nodename if hasattr(os, "uname") else "unknown",
            "cpu_usage": psutil.cpu_percent(),
            "memory_usage": psutil.virtual_memory().percent,
            "status": "healthy"
        }

        # Get improved worker status
        worker_status_result = get_worker_status_for_ui(redis_client)
        worker_info = worker_status_result.get("summary", {})
        workers_detail = worker_status_result.get("workers", {})
        debug_info = worker_status_result.get("debug_info")

        # GPU health (formatted for UI consumption)
        gpu_health = get_clean_gpu_status()

        # Job statistics (for both pages)
        job_stats = job_tracker.count_jobs_by_status()

        # Queue status (clean format)
        queue_status = get_clean_queue_status()

        # Determine overall system status
        overall_status = "healthy"
        if worker_info.get("healthy_workers", 0) == 0:
            overall_status = "down"
        elif worker_info.get("healthy_workers", 0) < worker_info.get("total_workers", 1) * 0.8:
            overall_status = "degraded"

        response = {
            "status": overall_status,
            "system": system_info,
            "workers": workers_detail,
            "worker_summary": worker_info,
            "gpu_health": gpu_health,
            "jobs": job_stats,
            "queue_status": queue_status
        }

        # Include debug info if no workers found
        if worker_info.get("total_workers", 0) == 0 and debug_info:
            response["debug_info"] = debug_info
            logger.warning(f"No workers detected. Debug info: {debug_info}")

        return response

    except Exception as e:
        logger.error(f"Error in detailed health check: {str(e)}")
        return {
            "status": "error",
            "error": str(e),
            "system": {"status": "error"},
            "workers": {},
            "gpu_health": {},
            "jobs": {},
            "queue_status": {}
        }


@router.get("/workers")
async def get_workers_status(redis_client: redis.Redis = Depends(get_redis_client)):
    """Get improved worker status for UI"""
    return get_worker_status_for_ui(redis_client)


@router.get("/workers/debug")
async def debug_worker_status(redis_client: redis.Redis = Depends(get_redis_client)):
    """Debug endpoint to help diagnose worker detection issues"""
    try:
        debug_info = debug_redis_keys(redis_client)
        worker_status = get_worker_status_for_ui(redis_client)

        return {
            "redis_debug": debug_info,
            "worker_status": worker_status,
            "redis_info": {
                "host": os.environ.get("REDIS_HOST", "localhost"),
                "port": os.environ.get("REDIS_PORT", "6379"),
                "connected": True
            }
        }
    except Exception as e:
        logger.error(f"Error in debug endpoint: {e}")
        return {
            "error": str(e),
            "redis_info": {
                "host": os.environ.get("REDIS_HOST", "localhost"),
                "port": os.environ.get("REDIS_PORT", "6379"),
                "connected": False
            }
        }


def get_clean_gpu_status() -> Dict[str, Any]:
    """Get GPU status in clean format for UI"""
    gpu_health = {}

    if torch.cuda.is_available():
        try:
            for i in range(torch.cuda.device_count()):
                device_props = torch.cuda.get_device_properties(i)
                total_memory = device_props.total_memory
                allocated_memory = torch.cuda.memory_allocated(i)
                reserved_memory = torch.cuda.memory_reserved(i)

                gpu_health[f"gpu_{i}"] = {
                    "device_name": torch.cuda.get_device_name(i),
                    "total_memory_gb": total_memory / 1e9,
                    "allocated_memory_gb": allocated_memory / 1e9,
                    "reserved_memory_gb": reserved_memory / 1e9,
                    "free_memory_gb": (total_memory - reserved_memory) / 1e9,
                    "usage_percent": (allocated_memory / total_memory) * 100 if total_memory > 0 else 0
                }
        except Exception as e:
            logger.error(f"Error getting GPU status: {str(e)}")

    return gpu_health


def get_clean_queue_status() -> Dict[str, Any]:
    """Get queue status in clean format for UI"""
    try:
        return job_chain.get_queue_status()
    except Exception as e:
        logger.error(f"Error getting queue status: {str(e)}")
        return {}


@router.get("/queue-stats")
async def get_queue_stats():
    """Get job chain queue statistics for UI"""
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
    """Get current system configuration settings"""
    config = {
        # Basic settings
        "host": getattr(settings, "host", "0.0.0.0"),
        "port": getattr(settings, "port", 8000),
        "api_auth_enabled": getattr(settings, "api_auth_enabled", True),

        # Model settings
        "default_embedding_model": getattr(settings, "default_embedding_model", "bge-m3"),
        "default_colbert_model": getattr(settings, "default_colbert_model", "colbertv2.0"),
        "default_llm_model": getattr(settings, "default_llm_model", "DeepSeek-R1-Distill-Qwen-7B"),
        "default_whisper_model": getattr(settings, "default_whisper_model", "medium"),

        # GPU settings
        "device": getattr(settings, "device", "cuda:0"),
        "use_fp16": getattr(settings, "use_fp16", True),
        "batch_size": getattr(settings, "batch_size", 16),

        # Architecture info
        "job_chain_enabled": True,
        "worker_architecture": "dedicated_gpu_workers",
        "auto_queue_management": True
    }

    return config


@router.post("/clear-gpu-cache")
async def clear_gpu_cache(request: Dict[str, str]):
    """Clear CUDA cache for GPU management"""
    if not torch.cuda.is_available():
        raise HTTPException(400, "CUDA not available")

    gpu_id = request.get("gpu_id", "gpu_0")

    try:
        device_idx = int(gpu_id.split("_")[-1]) if "_" in gpu_id else 0

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


@router.post("/restart-workers", dependencies=[Depends(get_token_header)])
async def restart_workers():
    """Signal workers to restart gracefully"""
    redis_client = get_redis_client()

    try:
        # Set restart flags for all worker types
        worker_types = ["gpu-inference", "gpu-embedding", "gpu-whisper", "cpu"]
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


@router.get("/disk-usage")
async def get_disk_usage():
    """Get system disk usage information"""
    try:
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
            ("logs", "logs")
        ]

        for name, path in important_dirs:
            if os.path.exists(path):
                try:
                    dir_size = get_directory_size(path)
                    data_dirs[name] = {
                        "path": path,
                        "size": dir_size
                    }
                except Exception as e:
                    data_dirs[name] = {
                        "path": path,
                        "size": 0,
                        "error": str(e)
                    }

        return {
            "partitions": partitions,
            "data_dirs": data_dirs
        }

    except Exception as e:
        logger.error(f"Error getting disk usage: {str(e)}")
        raise HTTPException(500, f"Error getting disk usage: {str(e)}")


def get_directory_size(path: str) -> int:
    """Calculate the total size of a directory"""
    total_size = 0
    try:
        for dirpath, dirnames, filenames in os.walk(path):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                if os.path.exists(fp) and os.path.isfile(fp):
                    total_size += os.path.getsize(fp)
    except Exception as e:
        logger.error(f"Error calculating directory size for {path}: {str(e)}")
    return total_size