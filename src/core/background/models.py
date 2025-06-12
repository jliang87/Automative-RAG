"""
Model management for dedicated GPU workers.
All configuration now driven by environment variables via settings.
Tesla T4 optimized with proper memory management.
"""

import os
import torch
import logging
from typing import Optional

from src.config.settings import settings

logger = logging.getLogger(__name__)

# Global variables to hold the preloaded models
_PRELOADED_EMBEDDING_MODEL = None
_PRELOADED_LLM_MODEL = None
_PRELOADED_COLBERT_RERANKER = None
_PRELOADED_WHISPER_MODEL = None


def get_worker_type() -> str:
    """Get the current worker type from environment."""
    return os.environ.get("WORKER_TYPE", "")


def should_load_model(model_env_var: str) -> bool:
    """Check if this worker should load a specific model."""
    return os.environ.get(model_env_var, "false").lower() == "true"


def set_memory_fraction_for_worker():
    """Set appropriate GPU memory fraction based on environment configuration."""
    if not torch.cuda.is_available():
        return

    # Use environment-driven memory fraction from settings
    memory_fraction = settings.get_worker_memory_fraction()

    if memory_fraction < 1.0:
        torch.cuda.set_per_process_memory_fraction(memory_fraction)
        worker_type = get_worker_type()
        logger.info(f"Set GPU memory fraction to {memory_fraction} for {worker_type} (Tesla T4 optimized)")
        logger.info(f"Configuration source: Environment (.env)")


def preload_embedding_model():
    """Preload the embedding model using environment configuration."""
    global _PRELOADED_EMBEDDING_MODEL

    # Skip if already loaded
    if _PRELOADED_EMBEDDING_MODEL is not None:
        return

    # Check worker type
    worker_type = get_worker_type()
    if worker_type != "gpu-embedding":
        logger.info(f"Skipping embedding model preload on {worker_type} worker")
        return

    logger.info(f"Preloading embedding model {settings.default_embedding_model} on {settings.device}")
    logger.info(f"Using environment batch size: {settings.embedding_batch_size}")

    try:
        # Set memory fraction from environment
        set_memory_fraction_for_worker()

        # Clear CUDA cache before loading
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            for i in range(torch.cuda.device_count()):
                torch.cuda.synchronize(i)

        # Load the model using environment configuration
        _PRELOADED_EMBEDDING_MODEL = settings.embedding_function

        # Test the model
        test_embedding = _PRELOADED_EMBEDDING_MODEL.embed_query("Test sentence for embedding model")
        embedding_dim = len(test_embedding)

        logger.info(f"✅ Successfully preloaded embedding model with dimension {embedding_dim}")

        # Log GPU memory if available
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated(0) / (1024 ** 3)
            memory_reserved = torch.cuda.memory_reserved(0) / (1024 ** 3)
            total_memory = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            logger.info(f"GPU memory after embedding: {memory_allocated:.2f}GB / {total_memory:.2f}GB")

    except Exception as e:
        logger.error(f"Failed to preload embedding model: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())


def preload_llm_model():
    """Preload the LLM model using environment configuration."""
    global _PRELOADED_LLM_MODEL

    # Skip if already loaded
    if _PRELOADED_LLM_MODEL is not None:
        return

    # Check worker type
    worker_type = get_worker_type()
    if worker_type != "gpu-inference":
        logger.info(f"Skipping LLM model preload on {worker_type} worker")
        return

    logger.info(f"Preloading LLM model {settings.default_llm_model} on {settings.device}")
    logger.info(f"Tesla T4 Environment configuration:")
    logger.info(f"  Use 4-bit: {settings.llm_use_4bit}")
    logger.info(f"  Use 8-bit: {settings.llm_use_8bit}")
    logger.info(f"  Use FP16: {settings.use_fp16}")
    logger.info(f"  Torch dtype: {settings.llm_torch_dtype}")
    logger.info(f"  Memory fraction: {settings.get_worker_memory_fraction()}")

    try:
        # Set memory fraction from environment
        set_memory_fraction_for_worker()

        # Clear CUDA cache before loading
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            for i in range(torch.cuda.device_count()):
                torch.cuda.synchronize(i)

        # Import the LLM - NO PARAMETERS, use environment config
        from src.core.llm import LocalLLM

        # Load the model using ONLY environment configuration
        _PRELOADED_LLM_MODEL = LocalLLM()  # No parameters - all from environment!

        logger.info(f"✅ Successfully preloaded LLM model using environment configuration")

        # Log memory usage
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated(0) / (1024 ** 3)
            memory_reserved = torch.cuda.memory_reserved(0) / (1024 ** 3)
            total_memory = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            logger.info(f"GPU memory after LLM: {memory_allocated:.2f}GB / {total_memory:.2f}GB")

            # Check if we're using too much memory
            memory_usage_percent = (memory_allocated / total_memory) * 100
            if memory_usage_percent > 85:
                logger.warning(f"⚠️ High memory usage: {memory_usage_percent:.1f}%")
                logger.warning("Consider reducing LLM_MAX_MEMORY_GB or using quantization")

    except Exception as e:
        logger.error(f"❌ Failed to preload LLM model: {str(e)}")

        # Enhanced Tesla T4 error reporting
        if "CUDA driver error: invalid argument" in str(e):
            logger.error("🚨 Tesla T4 Compatibility Issue Detected!")
            logger.error("This error is typically caused by 4-bit quantization on Tesla T4.")
            logger.error("Current quantization settings:")
            logger.error(f"  LLM_USE_4BIT: {settings.llm_use_4bit}")
            logger.error(f"  LLM_USE_8BIT: {settings.llm_use_8bit}")
            logger.error("Solution: Set LLM_USE_4BIT=false in your .env file")

        import traceback
        logger.error(traceback.format_exc())


def preload_colbert_reranker():
    """Preload the ColBERT and BGE reranking models using environment config."""
    global _PRELOADED_COLBERT_RERANKER

    # Skip if already loaded
    if _PRELOADED_COLBERT_RERANKER is not None:
        return

    # Check worker type
    worker_type = get_worker_type()
    if worker_type != "gpu-inference":
        logger.info(f"Skipping reranker preload on {worker_type} worker")
        return

    logger.info(f"Preloading ColBERT and BGE reranking models on {settings.device}")
    logger.info(f"Environment configuration:")
    logger.info(f"  ColBERT batch size: {settings.colbert_batch_size}")
    logger.info(f"  Use BGE reranker: {settings.use_bge_reranker}")
    logger.info(f"  ColBERT weight: {settings.colbert_weight}")
    logger.info(f"  BGE weight: {settings.bge_weight}")

    try:
        # Clear CUDA cache before loading
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            for i in range(torch.cuda.device_count()):
                torch.cuda.synchronize(i)

        # Import reranker
        from src.core.colbert_reranker import ColBERTReranker

        # Load the reranking models using environment configuration
        _PRELOADED_COLBERT_RERANKER = ColBERTReranker(
            model_name=settings.default_colbert_model,
            device=settings.device,
            batch_size=settings.colbert_batch_size,  # Environment-driven batch size
            use_fp16=settings.use_fp16,
            use_bge_reranker=settings.use_bge_reranker,
            colbert_weight=settings.colbert_weight,
            bge_weight=settings.bge_weight,
            bge_model_name=settings.default_bge_reranker_model
        )

        # Test the reranker
        from langchain_core.documents import Document
        test_doc = Document(page_content="This is a test document.", metadata={})
        test_results = _PRELOADED_COLBERT_RERANKER.rerank("test query", [test_doc], 1)

        logger.info(f"✅ Successfully preloaded ColBERT and BGE reranking models")

        # Log memory usage
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated(0) / (1024 ** 3)
            memory_reserved = torch.cuda.memory_reserved(0) / (1024 ** 3)
            total_memory = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            logger.info(f"GPU memory after reranker: {memory_allocated:.2f}GB / {total_memory:.2f}GB")

    except Exception as e:
        logger.error(f"Failed to preload reranking models: {str(e)}")
        _PRELOADED_COLBERT_RERANKER = None


def preload_whisper_model():
    """Preload the Whisper model using environment configuration."""
    global _PRELOADED_WHISPER_MODEL

    # Skip if already loaded
    if _PRELOADED_WHISPER_MODEL is not None:
        return

    # Check worker type
    worker_type = get_worker_type()
    if worker_type != "gpu-whisper":
        logger.info(f"Skipping Whisper model preload on {worker_type} worker")
        return

    logger.info(f"Preloading Whisper model {settings.whisper_model_size} on {settings.device}")
    logger.info(f"Environment configuration:")
    logger.info(f"  Whisper batch size: {settings.whisper_batch_size}")
    logger.info(f"  Use FP16: {settings.use_fp16}")

    try:
        # Set memory fraction from environment
        set_memory_fraction_for_worker()

        # Clear CUDA cache before loading
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            for i in range(torch.cuda.device_count()):
                torch.cuda.synchronize(i)

        # Import the WhisperModel
        from faster_whisper import WhisperModel

        # Use environment-configured model path
        model_path = settings.whisper_model_full_path

        # Load the Whisper model with environment configuration
        compute_type = "float16" if settings.use_fp16 and settings.device.startswith("cuda") else "float32"

        _PRELOADED_WHISPER_MODEL = WhisperModel(
            model_path,
            device=settings.device,
            compute_type=compute_type,
            cpu_threads=4,
            num_workers=2
        )

        logger.info(f"✅ Successfully preloaded Whisper model with compute_type: {compute_type}")

        # Log memory usage
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated(0) / (1024 ** 3)
            memory_reserved = torch.cuda.memory_reserved(0) / (1024 ** 3)
            total_memory = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            logger.info(f"GPU memory after Whisper: {memory_allocated:.2f}GB / {total_memory:.2f}GB")

    except Exception as e:
        logger.error(f"Failed to preload Whisper model: {str(e)}")


def get_vector_store():
    """Get a vector store instance using environment configuration."""
    global _PRELOADED_EMBEDDING_MODEL

    from src.core.vectorstore import QdrantStore
    from qdrant_client import QdrantClient

    # Initialize qdrant client
    qdrant_client = QdrantClient(
        host=settings.qdrant_host,
        port=settings.qdrant_port,
    )

    # If we have a preloaded model and we're on the embedding worker, use it
    worker_type = get_worker_type()
    if _PRELOADED_EMBEDDING_MODEL is not None and worker_type == "gpu-embedding":
        logger.info("Using preloaded embedding model for vector store")
        return QdrantStore(
            client=qdrant_client,
            collection_name=settings.qdrant_collection,
            embedding_function=_PRELOADED_EMBEDDING_MODEL,
        )

    # Create a new embedding model instance with environment config
    logger.info("Creating new embedding model instance for vector store")

    # Use environment-configured settings for device and batch size
    device = settings.device if worker_type == "gpu-embedding" else "cpu"
    batch_size = settings.embedding_batch_size if worker_type == "gpu-embedding" else 4

    from langchain_huggingface import HuggingFaceEmbeddings

    embedding_function = HuggingFaceEmbeddings(
        model_name=settings.embedding_model_full_path,
        model_kwargs={"device": device},
        encode_kwargs={
            "batch_size": batch_size,
            "normalize_embeddings": True
        },
        cache_folder=os.path.join(settings.models_dir, settings.embedding_model_path)
    )

    return QdrantStore(
        client=qdrant_client,
        collection_name=settings.qdrant_collection,
        embedding_function=embedding_function,
    )


def get_llm_model():
    """Get the preloaded LLM model or create a new one using environment config."""
    global _PRELOADED_LLM_MODEL

    # If we have a preloaded model and we're on the inference worker, use it
    worker_type = get_worker_type()
    if _PRELOADED_LLM_MODEL is not None and worker_type == "gpu-inference":
        return _PRELOADED_LLM_MODEL

    # Otherwise, create a new model instance using environment configuration
    from src.core.llm import LocalLLM

    # Create LLM using ONLY environment configuration - no overrides!
    logger.info("Creating LLM instance using environment configuration")
    return LocalLLM()  # All settings come from environment


def get_colbert_reranker():
    """Get the preloaded reranker or create a new one using environment config."""
    global _PRELOADED_COLBERT_RERANKER

    # If we have a preloaded model and we're on the inference worker, use it
    worker_type = get_worker_type()
    if _PRELOADED_COLBERT_RERANKER is not None and worker_type == "gpu-inference":
        return _PRELOADED_COLBERT_RERANKER

    # Otherwise, create a new reranker instance using environment config
    from src.core.colbert_reranker import ColBERTReranker

    # Use environment configuration
    device = settings.device if worker_type == "gpu-inference" else "cpu"
    batch_size = settings.colbert_batch_size if worker_type == "gpu-inference" else 4

    logger.info(f"Creating ColBERT reranker with batch_size={batch_size} on {device}")

    return ColBERTReranker(
        model_name=settings.default_colbert_model,
        device=device,
        batch_size=batch_size,
        use_fp16=settings.use_fp16 and device.startswith("cuda"),
        use_bge_reranker=settings.use_bge_reranker,
        colbert_weight=settings.colbert_weight,
        bge_weight=settings.bge_weight,
        bge_model_name=settings.default_bge_reranker_model
    )


def get_whisper_model():
    """Get the preloaded Whisper model or create a new one using environment config."""
    global _PRELOADED_WHISPER_MODEL

    # If we have a preloaded model and we're on the whisper worker, use it
    worker_type = get_worker_type()
    if _PRELOADED_WHISPER_MODEL is not None and worker_type == "gpu-whisper":
        return _PRELOADED_WHISPER_MODEL

    # Otherwise, create a new Whisper instance using environment config
    from faster_whisper import WhisperModel

    # Use environment configuration
    device = settings.device if worker_type == "gpu-whisper" else "cpu"
    model_path = settings.whisper_model_full_path
    compute_type = "float16" if settings.use_fp16 and device.startswith("cuda") else "float32"

    logger.info(f"Creating Whisper model on {device} with compute_type={compute_type}")

    return WhisperModel(
        model_path,
        device=device,
        compute_type=compute_type,
        cpu_threads=4,
        num_workers=2
    )


def preload_models():
    """Preload only the models this worker needs based on environment variables."""
    worker_type = get_worker_type()
    logger.info(f"Preloading models for worker type: {worker_type}")
    logger.info(f"Tesla T4 environment configuration active")

    # Load models based on environment flags
    if should_load_model("LOAD_WHISPER_MODEL"):
        logger.info("Preloading Whisper model...")
        preload_whisper_model()

    if should_load_model("LOAD_EMBEDDING_MODEL"):
        logger.info("Preloading embedding model...")
        preload_embedding_model()

    if should_load_model("LOAD_LLM_MODEL"):
        logger.info("Preloading LLM model...")
        preload_llm_model()

    if should_load_model("LOAD_COLBERT_MODEL"):
        logger.info("Preloading ColBERT model...")
        preload_colbert_reranker()

    # Log total GPU memory usage
    if torch.cuda.is_available():
        total_allocated = torch.cuda.memory_allocated(0) / (1024**3)
        total_available = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        usage_percent = (total_allocated / total_available) * 100
        logger.info(f"Total GPU memory: {total_allocated:.2f}GB / {total_available:.2f}GB ({usage_percent:.1f}%)")

        if usage_percent > 90:
            logger.warning("⚠️ Very high GPU memory usage! Consider reducing batch sizes or using quantization.")


def reload_models():
    """Reload all models using current environment configuration."""
    global _PRELOADED_EMBEDDING_MODEL, _PRELOADED_LLM_MODEL, _PRELOADED_COLBERT_RERANKER, _PRELOADED_WHISPER_MODEL

    worker_type = get_worker_type()
    logger.info(f"Reloading models for {worker_type} worker using environment configuration")

    # Clear CUDA cache first
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        for i in range(torch.cuda.device_count()):
            torch.cuda.synchronize(i)

    # Force garbage collection
    import gc
    gc.collect()

    # Clear the preloaded models
    _PRELOADED_EMBEDDING_MODEL = None
    _PRELOADED_LLM_MODEL = None
    _PRELOADED_COLBERT_RERANKER = None
    _PRELOADED_WHISPER_MODEL = None

    # Reload models based on worker type using environment config
    preload_models()

    # Log GPU memory after reload
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / (1024 ** 3)
            reserved = torch.cuda.memory_reserved(i) / (1024 ** 3)
            total = torch.cuda.get_device_properties(i).total_memory / (1024 ** 3)
            logger.info(f"GPU {i} after reload: {allocated:.2f}GB / {total:.2f}GB allocated")