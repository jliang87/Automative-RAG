"""
Utility functions for model path handling.
"""

import os
from typing import Dict, Optional

# Default model names - read from environment or use defaults
DEFAULT_EMBEDDING_MODEL = os.environ.get("DEFAULT_EMBEDDING_MODEL", "bge-small-en-v1.5")
DEFAULT_COLBERT_MODEL = os.environ.get("DEFAULT_COLBERT_MODEL", "colbertv2.0")
DEFAULT_LLM_MODEL = os.environ.get("DEFAULT_LLM_MODEL", "DeepSeek-R1-Distill-Qwen-7B")
DEFAULT_WHISPER_MODEL = os.environ.get("DEFAULT_WHISPER_MODEL", "medium")

def get_model_path(base_dir: str, model_name: str) -> str:
    """
    Get the full path to a model by combining base directory and model name.

    Args:
        base_dir: Base directory for models of this type
        model_name: Name of the specific model

    Returns:
        Full path to the model directory
    """
    return os.path.join(base_dir, model_name)

def get_embedding_model_path(base_dir: Optional[str] = None) -> str:
    """Get the path to the embedding model."""
    base = base_dir or os.environ.get("EMBEDDING_MODEL", "models/embeddings")
    return get_model_path(base, DEFAULT_EMBEDDING_MODEL)

def get_colbert_model_path(base_dir: Optional[str] = None) -> str:
    """Get the path to the ColBERT model."""
    base = base_dir or os.environ.get("COLBERT_MODEL", "models/colbert")
    return get_model_path(base, DEFAULT_COLBERT_MODEL)

def get_llm_model_path(base_dir: Optional[str] = None) -> str:
    """Get the path to the LLM model."""
    base = base_dir or os.environ.get("DEEPSEEK_MODEL", "models/llm")
    return get_model_path(base, DEFAULT_LLM_MODEL)

def get_whisper_model_path(base_dir: Optional[str] = None) -> str:
    """Get the path to the Whisper model."""
    base = base_dir or os.environ.get("WHISPER_CACHE_DIR", "models/whisper")
    model_size = os.environ.get("WHISPER_MODEL_SIZE", DEFAULT_WHISPER_MODEL)
    return get_model_path(base, model_size)

def get_model_paths() -> Dict[str, str]:
    """
    Get paths for all models using the standard directory structure.

    Returns:
        Dictionary with paths for each model type
    """
    return {
        "embedding": get_embedding_model_path(),
        "colbert": get_colbert_model_path(),
        "llm": get_llm_model_path(),
        "whisper": get_whisper_model_path()
    }