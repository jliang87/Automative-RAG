import os
from typing import Callable, Dict, List, Optional, Union

import torch
from langchain_community.embeddings import HuggingFaceEmbeddings
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # API settings
    api_key: str = "default-api-key"
    api_auth_enabled: bool = True
    
    # Server settings
    host: str = "0.0.0.0"
    port: int = 8000
    
    # Qdrant settings
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection: str = "automotive_specs"
    
    # GPU settings
    device: str = "cuda:0" if torch.cuda.is_available() else "cpu"
    use_fp16: bool = True
    batch_size: int = 16  # Batch size for GPU operations
    
    # Model settings
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    colbert_model: str = "colbert-ir/colbertv2.0"
    
    # LLM settings (local DeepSeek model)
    deepseek_model: str = "deepseek-ai/deepseek-coder-6.7b-instruct"  # Path or HF model name
    llm_use_4bit: bool = True  # 4-bit quantization (saves VRAM)
    llm_use_8bit: bool = False  # 8-bit quantization (alternative)
    llm_temperature: float = 0.1
    llm_max_tokens: int = 512
    
    # Whisper settings for YouTube transcription
    whisper_model_size: str = "medium"  # tiny, base, small, medium, large
    use_youtube_captions: bool = True
    use_whisper_as_fallback: bool = True
    
    # PDF OCR settings
    use_pdf_ocr: bool = True
    ocr_languages: str = "eng"
    
    # Retrieval settings
    retriever_top_k: int = 20
    reranker_top_k: int = 5
    colbert_batch_size: int = 16
    
    # Chunking settings
    chunk_size: int = 1000
    chunk_overlap: int = 200
    
    # Embedding function
    @property
    def embedding_function(self) -> Callable:
        return HuggingFaceEmbeddings(
            model_name=self.embedding_model,
            model_kwargs={"device": self.device},
            encode_kwargs={"batch_size": self.batch_size, "normalize_embeddings": True},
        )
    
    # Data paths
    data_dir: str = "data"
    upload_dir: str = "data/uploads"
    models_dir: str = "models"  # Directory for downloaded models
    
    # Ensure required directories exist
    def initialize_directories(self) -> None:
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.upload_dir, exist_ok=True)
        os.makedirs(os.path.join(self.data_dir, "youtube"), exist_ok=True)
        os.makedirs(os.path.join(self.data_dir, "bilibili"), exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)
    
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