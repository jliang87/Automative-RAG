"""
Model management router for the API.

This module provides endpoints for managing AI models,
including loading, unloading, and getting status information.
"""

import os
import time
import json
import logging
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, Depends, HTTPException, Body

from src.api.dependencies import get_redis_client, get_token_header
from src.config.utils import update_config, read_config

# Create a separate router for model management
router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/config")
async def get_model_config():
    """
    Get current model configuration settings.
    """
    from src.config.settings import settings

    config = {
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

        # Whisper settings
        "use_youtube_captions": getattr(settings, "use_youtube_captions", False),
        "use_whisper_as_fallback": getattr(settings, "use_whisper_as_fallback", False),
        "force_whisper": getattr(settings, "force_whisper", True),
    }

    return config


@router.post("/update-config", dependencies=[Depends(get_token_header)])
async def update_model_config(config_updates: Dict[str, Any] = Body(...)):
    """
    Update model configuration settings.
    For demonstration purposes only - in a real system, this would actually modify settings.
    """
    try:
        # In a production system, this would update a configuration file or database
        # For this demo, we'll just acknowledge the request
        from src.config.utils import update_config
        success = update_config(config_updates)

        return {
            "status": "success" if success else "error",
            "message": "Config updates applied" if success else "Failed to update config",
            "updated_fields": list(config_updates.keys())
        }
    except Exception as e:
        logger.error(f"Error updating config: {str(e)}")
        raise HTTPException(500, f"Error updating configuration: {str(e)}")


@router.get("/model-info")
async def get_model_info(redis_client = Depends(get_redis_client)):
    """
    Get detailed information about loaded models.
    """
    try:
        # Get model loading status from Redis
        model_info = {
            "llm": get_llm_info(redis_client),
            "embedding": get_embedding_info(redis_client),
            "colbert": get_colbert_info(redis_client),
            "whisper": get_whisper_info(redis_client)
        }

        return model_info
    except Exception as e:
        logger.error(f"Error getting model info: {str(e)}")
        raise HTTPException(500, f"Error getting model info: {str(e)}")


@router.post("/reload-model", dependencies=[Depends(get_token_header)])
async def reload_model(request: Dict[str, str] = Body(...), redis_client = Depends(get_redis_client)):
    """
    Send a command to reload a specific model.
    """
    model_type = request.get("model_type")
    if not model_type or model_type not in ["embedding", "llm", "colbert", "whisper"]:
        raise HTTPException(400, "Invalid model type")

    try:
        # Set a reload flag in Redis
        reload_key = f"model:reload:{model_type}"
        redis_client.set(reload_key, "1", ex=300)  # Expires in 5 minutes

        # Record the reload request time
        redis_client.set(f"model:reload:requested_at:{model_type}", str(time.time()), ex=300)

        return {
            "status": "success",
            "message": f"Reload signal sent for {model_type} model"
        }
    except Exception as e:
        logger.error(f"Error sending reload signal: {str(e)}")
        raise HTTPException(500, f"Error sending reload signal: {str(e)}")


def get_llm_info(redis_client) -> Dict[str, Any]:
    """
    Get information about the LLM model.
    """
    from src.config.settings import settings

    # Check if model is loaded
    is_loaded = redis_client.get("model_loaded:llm") == b"1"
    loading_time = redis_client.get("model_loading_time:llm")
    loading_time = float(loading_time) if loading_time else 0

    # Get VRAM usage if available
    vram_usage = "Unknown"
    if torch_available():
        try:
            device = getattr(settings, "device", "cuda:0")
            if device.startswith("cuda"):
                device_idx = int(device.split(":")[-1])
                allocated = torch.cuda.memory_allocated(device_idx) / (1024**3)
                vram_usage = f"{allocated:.2f} GB"
        except:
            pass

    return {
        "model_name": getattr(settings, "default_llm_model", "Unknown"),
        "loaded": is_loaded,
        "loading_time": loading_time,
        "device": getattr(settings, "device", "Unknown"),
        "quantization": ("4-bit" if getattr(settings, "llm_use_4bit", False) else
                         ("8-bit" if getattr(settings, "llm_use_8bit", False) else "None")),
        "vram_usage": vram_usage
    }


def get_embedding_info(redis_client) -> Dict[str, Any]:
    """
    Get information about the embedding model.
    """
    from src.config.settings import settings

    # Check if model is loaded
    is_loaded = redis_client.get("model_loaded:embedding") == b"1"
    loading_time = redis_client.get("model_loading_time:embedding")
    loading_time = float(loading_time) if loading_time else 0

    return {
        "model_name": getattr(settings, "default_embedding_model", "Unknown"),
        "loaded": is_loaded,
        "loading_time": loading_time,
        "device": getattr(settings, "device", "Unknown"),
        "batch_size": getattr(settings, "batch_size", 16)
    }


def get_colbert_info(redis_client) -> Dict[str, Any]:
    """
    Get information about the reranking models.
    """
    from src.config.settings import settings

    # Check if model is loaded
    is_loaded = redis_client.get("model_loaded:colbert") == b"1"
    loading_time = redis_client.get("model_loading_time:colbert")
    loading_time = float(loading_time) if loading_time else 0

    return {
        "colbert_model": getattr(settings, "default_colbert_model", "Unknown"),
        "bge_model": getattr(settings, "default_bge_reranker_model", "Unknown"),
        "loaded": is_loaded,
        "loading_time": loading_time,
        "device": getattr(settings, "device", "Unknown"),
        "colbert_weight": getattr(settings, "colbert_weight", 0.8),
        "bge_weight": getattr(settings, "bge_weight", 0.2)
    }


def get_whisper_info(redis_client) -> Dict[str, Any]:
    """
    Get information about the Whisper model.
    """
    from src.config.settings import settings

    # Check if model is loaded
    is_loaded = redis_client.get("model_loaded:whisper") == b"1"
    loading_time = redis_client.get("model_loading_time:whisper")
    loading_time = float(loading_time) if loading_time else 0

    return {
        "model_size": getattr(settings, "default_whisper_model", "Unknown"),
        "loaded": is_loaded,
        "loading_time": loading_time,
        "device": getattr(settings, "device", "Unknown"),
        "use_youtube_captions": getattr(settings, "use_youtube_captions", False)
    }


def torch_available():
    """Check if PyTorch is available."""
    try:
        import torch
        return True
    except ImportError:
        return False