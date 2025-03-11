import os
from typing import Callable, Dict, List, Optional, Union

import torch
from langchain_community.embeddings import HuggingFaceEmbeddings
from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv

# Import utility for model paths
from src.utils.model_paths import get_embedding_model_path

# Load environment variables from .env file
load_dotenv()


class Settings(BaseSettings):
    # API settings
    api_key: str = os.getenv("API_KEY", "default-api-key")
    api_auth_enabled: bool = os.getenv("API_AUTH_ENABLED", "true").lower() == "true"

    # Server settings
    host: str = os.getenv("HOST", "0.0.0.0")
    port: int = int(os.getenv("PORT", "8000"))

    # Qdrant settings
    qdrant_host: str = os.getenv("QDRANT_HOST", "localhost")
    qdrant_port: int = int(os.getenv("QDRANT_PORT", "6333"))
    qdrant_collection: str = os.getenv("QDRANT_COLLECTION", "automotive_specs")

    # GPU settings
    device: str = os.getenv("DEVICE", "cuda:0" if torch.cuda.is_available() else "cpu")
    use_fp16: bool = os.getenv("USE_FP16", "true").lower() == "true"
    batch_size: int = int(os.getenv("BATCH_SIZE", "16"))

    # Model settings - these can be HuggingFace IDs or local paths
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "/app/models/embeddings")
    colbert_model: str = os.getenv("COLBERT_MODEL", "/app/models/colbert")

    # LLM settings (local DeepSeek model)
    deepseek_model: str = os.getenv("DEEPSEEK_MODEL", "/app/models/llm")
    llm_use_4bit: bool = os.getenv("LLM_USE_4BIT", "true").lower() == "true"
    llm_use_8bit: bool = os.getenv("LLM_USE_8BIT", "false").lower() == "true"
    llm_temperature: float = float(os.getenv("LLM_TEMPERATURE", "0.1"))
    llm_max_tokens: int = int(os.getenv("LLM_MAX_TOKENS", "512"))

    # Whisper settings for YouTube transcription
    whisper_model_size: str = os.getenv("WHISPER_MODEL_SIZE", "medium")
    whisper_model_path: Optional[str] = os.getenv("WHISPER_MODEL_PATH", None)
    use_youtube_captions: bool = os.getenv("USE_YOUTUBE_CAPTIONS", "true").lower() == "true"
    use_whisper_as_fallback: bool = os.getenv("USE_WHISPER_AS_FALLBACK", "true").lower() == "true"

    # PDF OCR settings
    use_pdf_ocr: bool = os.getenv("USE_PDF_OCR", "true").lower() == "true"
    ocr_languages: str = os.getenv("OCR_LANGUAGES", "eng")

    # Retrieval settings
    retriever_top_k: int = int(os.getenv("RETRIEVER_TOP_K", "20"))
    reranker_top_k: int = int(os.getenv("RERANKER_TOP_K", "5"))
    colbert_batch_size: int = int(os.getenv("COLBERT_BATCH_SIZE", "16"))

    # Chunking settings
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "1000"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "200"))

    # Data and model paths
    data_dir: str = os.getenv("DATA_DIR", "data")
    upload_dir: str = os.getenv("UPLOAD_DIR", "data/uploads")
    models_dir: str = os.getenv("MODELS_DIR", "models")
    embedding_cache_dir: str = os.getenv("EMBEDDING_CACHE_DIR", "models/embeddings")
    llm_cache_dir: str = os.getenv("LLM_CACHE_DIR", "models/llm")
    whisper_cache_dir: str = os.getenv("WHISPER_CACHE_DIR", "models/whisper")

    # Embedding function
    @property
    def embedding_function(self) -> Callable:
        # Get the complete embedding model path
        embedding_model_path = get_embedding_model_path(self.embedding_model)

        # Check if embedding_model is a local path
        if os.path.exists(embedding_model_path):
            # Use local model path
            return HuggingFaceEmbeddings(
                model_name=embedding_model_path,
                model_kwargs={"device": self.device},
                encode_kwargs={"batch_size": self.batch_size, "normalize_embeddings": True},
                cache_folder=self.embedding_cache_dir
            )
        else:
            # Use HuggingFace model ID
            return HuggingFaceEmbeddings(
                model_name=self.embedding_model,
                model_kwargs={"device": self.device},
                encode_kwargs={"batch_size": self.batch_size, "normalize_embeddings": True},
            )

    # Ensure required directories exist
    def initialize_directories(self) -> None:
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.upload_dir, exist_ok=True)
        os.makedirs(os.path.join(self.data_dir, "youtube"), exist_ok=True)
        os.makedirs(os.path.join(self.data_dir, "bilibili"), exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.embedding_cache_dir, exist_ok=True)
        os.makedirs(self.llm_cache_dir, exist_ok=True)
        os.makedirs(self.whisper_cache_dir, exist_ok=True)

        # Set environment variables to control model caching
        os.environ["TRANSFORMERS_CACHE"] = os.getenv("TRANSFORMERS_CACHE", os.path.join(self.models_dir, "cache"))
        os.environ["HF_HOME"] = os.getenv("HF_HOME", os.path.join(self.models_dir, "hub"))

    # GPU configuration
    def get_gpu_info(self) -> Dict[str, any]:
        """Get GPU information if available."""
        gpu_info = {
            "device": self.device,
            "available": torch.cuda.is_available()
        }

        if torch.cuda.is_available():
            gpu_info.update({
                "device_count": torch.cuda.device_count(),
                "current_device": torch.cuda.current_device(),
                "device_name": torch.cuda.get_device_name(0),
                "memory_allocated": f"{torch.cuda.memory_allocated(0) / 1024 ** 3:.2f} GB",
                "memory_cached": f"{torch.cuda.memory_reserved(0) / 1024 ** 3:.2f} GB",
                "max_memory": f"{torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.2f} GB",
            })

        return gpu_info

    # Model config
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


# Create settings instance
settings = Settings()
settings.initialize_directories()

# Print GPU information on startup
if settings.device.startswith("cuda"):
    gpu_info = settings.get_gpu_info()
    print(f"Using GPU: {gpu_info['device_name']}")
    print(f"Total GPU memory: {gpu_info['max_memory']}")
    print(f"Using mixed precision (FP16): {settings.use_fp16}")
    print(f"LLM quantization: {'4-bit' if settings.llm_use_4bit else '8-bit' if settings.llm_use_8bit else 'none'}")
else:
    print(f"Running on CPU - GPU not available")