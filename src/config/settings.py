import os
from typing import Callable, Dict, List, Optional, Union

import torch
from langchain_huggingface import HuggingFaceEmbeddings  # ✅ Correct
from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv

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

    # GPU settings - TESLA T4 OPTIMIZED
    use_gpu: bool = os.getenv("USE_GPU", "true").lower() == "true"
    device: str = os.getenv("DEVICE", "cuda:0" if torch.cuda.is_available() else "cpu")
    use_fp16: bool = os.getenv("USE_FP16", "true").lower() == "true"

    # Batch sizes - Tesla T4 GPU memory optimized
    batch_size: int = int(os.getenv("BATCH_SIZE", "8"))  # For embeddings
    llm_batch_size: int = int(os.getenv("LLM_BATCH_SIZE", "1"))
    embedding_batch_size: int = int(os.getenv("EMBEDDING_BATCH_SIZE", "8"))
    whisper_batch_size: int = int(os.getenv("WHISPER_BATCH_SIZE", "1"))

    # Tesla T4 Memory Management (14.6GB total)
    llm_max_memory_gb: float = float(os.getenv("LLM_MAX_MEMORY_GB", "12.0"))
    gpu_memory_fraction_whisper: float = float(os.getenv("GPU_MEMORY_FRACTION_WHISPER", "0.15"))
    gpu_memory_fraction_embedding: float = float(os.getenv("GPU_MEMORY_FRACTION_EMBEDDING", "0.25"))
    gpu_memory_fraction_inference: float = float(os.getenv("GPU_MEMORY_FRACTION_INFERENCE", "0.60"))

    # Worker Configuration - 8-core CPU optimization
    max_concurrent_queries: int = int(os.getenv("MAX_CONCURRENT_QUERIES", "2"))
    query_timeout: int = int(os.getenv("QUERY_TIMEOUT", "300"))

    # Base directories for models
    # When running in Docker, we use the container paths
    # When running the download scripts, we use the host paths
    host_models_dir: str = os.getenv("HOST_MODELS_DIR", "models")
    container_models_dir: str = os.getenv("CONTAINER_MODELS_DIR", "/app/models")

    # Determine if we're running in a container
    # If CONTAINER_RUNTIME env var is set or /proc/1/cgroup contains 'docker' or 'kubepods'
    @property
    def is_container(self) -> bool:
        if os.getenv("CONTAINER_RUNTIME", ""):
            return True
        try:
            with open('/proc/1/cgroup', 'r') as f:
                return any(e in f.read() for e in ['docker', 'kubepods'])
        except:
            # If we can't determine, assume not a container
            return False

    # Model base directories depend on whether we're in a container or not
    @property
    def models_dir(self) -> str:
        return self.container_models_dir if self.is_container else self.host_models_dir

    # Paths for specific model types
    embedding_model_path: str = os.getenv("EMBEDDING_MODEL_PATH", "embeddings")
    colbert_model_path: str = os.getenv("COLBERT_MODEL_PATH", "colbert")
    llm_model_path: str = os.getenv("LLM_MODEL_PATH", "llm")
    whisper_model_path: str = os.getenv("WHISPER_MODEL_PATH", "whisper")
    bge_reranker_model_path: str = os.getenv("BGE_RERANKER_MODEL_PATH", "bge")

    # Default model names
    default_embedding_model: str = os.getenv("DEFAULT_EMBEDDING_MODEL", "bge-m3")
    default_colbert_model: str = os.getenv("DEFAULT_COLBERT_MODEL", "colbertv2.0")
    default_llm_model: str = os.getenv("DEFAULT_LLM_MODEL", "DeepSeek-R1-Distill-Qwen-7B")
    default_whisper_model: str = os.getenv("DEFAULT_WHISPER_MODEL", "medium")
    default_bge_reranker_model: str = os.getenv("DEFAULT_BGE_RERANKER_MODEL", "bge-reranker-base")

    # BGE reranker settings
    use_bge_reranker: bool = os.getenv("USE_BGE_RERANKER", "true").lower() == "true"
    colbert_weight: float = float(os.getenv("COLBERT_WEIGHT", "0.8"))
    bge_weight: float = float(os.getenv("BGE_WEIGHT", "0.2"))

    # Full paths to models
    @property
    def embedding_model_full_path(self) -> str:
        return os.path.join(self.models_dir, self.embedding_model_path, self.default_embedding_model)

    @property
    def colbert_model_full_path(self) -> str:
        return os.path.join(self.models_dir, self.colbert_model_path, self.default_colbert_model)

    @property
    def llm_model_full_path(self) -> str:
        return os.path.join(self.models_dir, self.llm_model_path, self.default_llm_model)

    @property
    def whisper_model_full_path(self) -> str:
        return os.path.join(self.models_dir, self.whisper_model_path, self.default_whisper_model)

    @property
    def bge_reranker_model_full_path(self) -> str:
        return os.path.join(self.models_dir, self.bge_reranker_model_path, self.default_bge_reranker_model)

    # LLM settings - TESLA T4 OPTIMIZED
    llm_use_4bit: bool = os.getenv("LLM_USE_4BIT", "false").lower() == "true"  # CRITICAL: Default false for Tesla T4
    llm_use_8bit: bool = os.getenv("LLM_USE_8BIT", "false").lower() == "true"
    llm_torch_dtype_str: str = os.getenv("LLM_TORCH_DTYPE", "float16")
    llm_temperature: float = float(os.getenv("LLM_TEMPERATURE", "0.1"))
    llm_max_tokens: int = int(os.getenv("LLM_MAX_TOKENS", "512"))
    llm_repetition_penalty: float = float(os.getenv("LLM_REPETITION_PENALTY", "1.1"))

    # Convert torch dtype string to torch.dtype
    @property
    def llm_torch_dtype(self) -> torch.dtype:
        dtype_map = {
            "float16": torch.float16,
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
            "auto": torch.float16 if self.device.startswith("cuda") else torch.float32
        }
        return dtype_map.get(self.llm_torch_dtype_str, torch.float16)

    # Whisper settings
    whisper_model_size: str = os.getenv("WHISPER_MODEL_SIZE", "medium")
    use_youtube_captions: bool = os.getenv("USE_YOUTUBE_CAPTIONS", "false").lower() == "true"
    use_whisper_as_fallback: bool = os.getenv("USE_WHISPER_AS_FALLBACK", "false").lower() == "true"
    force_whisper: bool = os.getenv("FORCE_WHISPER", "true").lower() == "true"

    # PDF OCR settings
    use_pdf_ocr: bool = os.getenv("USE_PDF_OCR", "true").lower() == "true"
    ocr_languages: str = os.getenv("OCR_LANGUAGES", "en+ch_doc")

    # Retrieval settings
    retriever_top_k: int = int(os.getenv("RETRIEVER_TOP_K", "20"))
    reranker_top_k: int = int(os.getenv("RERANKER_TOP_K", "5"))
    colbert_batch_size: int = int(os.getenv("COLBERT_BATCH_SIZE", "8"))  # Tesla T4 optimized

    # Chunking settings
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "1000"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "200"))

    # Data and upload directories
    host_data_dir: str = os.getenv("HOST_DATA_DIR", "data")
    container_data_dir: str = os.getenv("CONTAINER_DATA_DIR", "/app/data")
    upload_dir: str = os.getenv("UPLOAD_DIR", "uploads")

    @property
    def data_dir(self) -> str:
        base_dir = self.container_data_dir if self.is_container else self.host_data_dir
        return base_dir

    @property
    def upload_path(self) -> str:
        return os.path.join(self.data_dir, self.upload_dir)

    # Cache directories
    transformers_cache: str = os.getenv("TRANSFORMERS_CACHE", "models/cache")
    hf_home: str = os.getenv("HF_HOME", "models/hub")

    # Development/Debug
    debug_mode: bool = os.getenv("DEBUG_MODE", "false").lower() == "true"
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    environment: str = os.getenv("ENVIRONMENT", "production")

    # Tesla T4 Memory Management Methods
    def get_worker_memory_fraction(self) -> float:
        """Get memory fraction for current worker type."""
        worker_type = os.environ.get("WORKER_TYPE", "")

        if worker_type == "gpu-whisper":
            return self.gpu_memory_fraction_whisper
        elif worker_type == "gpu-embedding":
            return self.gpu_memory_fraction_embedding
        elif worker_type == "gpu-inference":
            return self.gpu_memory_fraction_inference
        else:
            return 1.0  # Use all available for unknown worker types

    def should_use_fp16(self) -> bool:
        """Check if FP16 should be used based on configuration."""
        return self.use_fp16 and self.device.startswith("cuda")

    def get_quantization_config(self):
        """Get quantization configuration for LLM loading."""
        from transformers import BitsAndBytesConfig

        if self.llm_use_4bit:
            print("Using 4-bit quantization")
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=self.llm_torch_dtype,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True
            )
        elif self.llm_use_8bit:
            print("Using 8-bit quantization")
            return BitsAndBytesConfig(
                load_in_8bit=True,
            )
        else:
            print(f"No quantization - using {self.llm_torch_dtype}")
            return None

    def get_model_kwargs(self) -> dict:
        """Get model loading kwargs based on configuration."""
        kwargs = {
            "torch_dtype": self.llm_torch_dtype,
            "device_map": "auto" if self.device.startswith("cuda") else None,
            "trust_remote_code": True,
            "local_files_only": True,
            "low_cpu_mem_usage": True
        }

        # Add quantization config if enabled
        quantization_config = self.get_quantization_config()
        if quantization_config:
            kwargs["quantization_config"] = quantization_config

        return kwargs

    # Embedding function
    @property
    def embedding_function(self) -> HuggingFaceEmbeddings:
        try:
            # Use the appropriate model path and environment-configured batch size
            return HuggingFaceEmbeddings(
                model_name=self.embedding_model_full_path,
                model_kwargs={"device": self.device},
                encode_kwargs={
                    "batch_size": self.embedding_batch_size,
                    "normalize_embeddings": True
                },
                cache_folder=os.path.join(self.models_dir, self.embedding_model_path)
            )
        except Exception as e:
            raise ValueError(f"Failed to load embedding model from {self.embedding_model_full_path}. Error: {str(e)}\n"
                             "Please run './download_models.sh'")

    # Ensure required directories exist
    def initialize_directories(self) -> None:
        # Use the appropriate base dirs based on container detection
        data_dir = self.data_dir
        models_dir = self.models_dir

        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(os.path.join(data_dir, self.upload_dir), exist_ok=True)
        os.makedirs(os.path.join(data_dir, "youtube"), exist_ok=True)
        os.makedirs(os.path.join(data_dir, "bilibili"), exist_ok=True)

        os.makedirs(models_dir, exist_ok=True)
        os.makedirs(os.path.join(models_dir, self.embedding_model_path), exist_ok=True)
        os.makedirs(os.path.join(models_dir, self.colbert_model_path), exist_ok=True)
        os.makedirs(os.path.join(models_dir, self.llm_model_path), exist_ok=True)
        os.makedirs(os.path.join(models_dir, self.whisper_model_path), exist_ok=True)
        os.makedirs(os.path.join(models_dir, self.bge_reranker_model_path), exist_ok=True)
        os.makedirs(os.path.join(models_dir, "cache"), exist_ok=True)
        os.makedirs(os.path.join(models_dir, "hub"), exist_ok=True)

        # Set environment variables to control model caching
        os.environ["TRANSFORMERS_CACHE"] = os.path.join(models_dir, "cache")
        os.environ["HF_HOME"] = os.path.join(models_dir, "hub")

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

    def log_configuration(self):
        """Log current configuration for debugging."""
        if self.debug_mode:
            print("=== SETTINGS CONFIGURATION ===")
            print(f"Environment: {self.environment}")
            print(f"Device: {self.device}")
            print(f"Use GPU: {self.use_gpu}")
            print(f"LLM Use 4-bit: {self.llm_use_4bit}")
            print(f"LLM Use 8-bit: {self.llm_use_8bit}")
            print(f"Use FP16: {self.use_fp16}")
            print(f"LLM Torch dtype: {self.llm_torch_dtype}")
            print(f"LLM Max Memory: {self.llm_max_memory_gb}GB")
            print(f"LLM Batch Size: {self.llm_batch_size}")
            print(f"Embedding Batch Size: {self.embedding_batch_size}")
            print(f"ColBERT Batch Size: {self.colbert_batch_size}")
            print(f"Worker Type: {os.environ.get('WORKER_TYPE', 'Not set')}")
            print(f"Memory Fraction: {self.get_worker_memory_fraction()}")
            print("=" * 30)

    # Model config
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


# Create settings instance
settings = Settings()
settings.initialize_directories()

# Log configuration if in debug mode
settings.log_configuration()

# Print environment and paths information on startup
print(f"Running in {'container' if settings.is_container else 'host'} environment")
print(f"Models directory: {settings.models_dir}")
print(f"Data directory: {settings.data_dir}")

# Print GPU information if available
if settings.device.startswith("cuda"):
    gpu_info = settings.get_gpu_info()
    print(f"Using GPU: {gpu_info['device_name']}")
    print(f"Total GPU memory: {gpu_info['max_memory']}")
    print(f"Using mixed precision (FP16): {settings.use_fp16}")

    # CRITICAL: Show quantization status
    if settings.llm_use_4bit:
        print("LLM quantization: 4-bit (⚠️ May cause issues on Tesla T4)")
    elif settings.llm_use_8bit:
        print("LLM quantization: 8-bit")
    else:
        print("LLM quantization: none (FP16 - ✅ Tesla T4 optimized)")
else:
    print(f"Running on CPU - GPU not available")