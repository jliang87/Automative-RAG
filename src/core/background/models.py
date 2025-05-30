"""
Updated model management for dedicated GPU workers.
This replaces your existing src/core/background/models.py
"""

import os
import torch
import logging
from typing import Optional

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
    """Set appropriate GPU memory fraction based on worker type."""
    if not torch.cuda.is_available():
        return

    worker_type = get_worker_type()

    # Optimized for 16GB GPU (14.58GB usable)
    memory_fractions = {
        "gpu-whisper": 0.15,  # 2.2GB - Whisper medium model
        "gpu-embedding": 0.25,  # 3.6GB - BGE-M3 embedding model
        "gpu-inference": 0.70,  # 10.2GB - LLM + ColBERT + BGE reranker
    }

    if worker_type in memory_fractions:
        fraction = memory_fractions[worker_type]
        torch.cuda.set_per_process_memory_fraction(fraction)
        logger.info(f"Set GPU memory fraction to {fraction} for {worker_type} (16GB GPU optimized)")


def preload_embedding_model():
    """
    Preload the embedding model at worker startup to avoid loading it for each job.
    """
    global _PRELOADED_EMBEDDING_MODEL

    # Skip if already loaded
    if _PRELOADED_EMBEDDING_MODEL is not None:
        return

    # Check worker type
    worker_type = os.environ.get("WORKER_TYPE", "")
    if worker_type != "gpu-embedding":
        logger.info(f"Skipping embedding model preload on {worker_type} worker")
        return

    # Import settings
    from src.config.settings import settings

    logger.info(f"Preloading embedding model {settings.default_embedding_model} on {settings.device}")

    try:
        # Set memory fraction
        set_memory_fraction_for_worker()

        # Clear CUDA cache before loading
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            for i in range(torch.cuda.device_count()):
                torch.cuda.synchronize(i)

            # Log GPU memory status before loading
            for i in range(torch.cuda.device_count()):
                device_name = torch.cuda.get_device_name(i)
                allocated = torch.cuda.memory_allocated(i) / (1024 ** 3)
                reserved = torch.cuda.memory_reserved(i) / (1024 ** 3)
                logger.info(
                    f"GPU {i} ({device_name}) before embedding model loading: Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")

        # Import embedding model
        from langchain_huggingface import HuggingFaceEmbeddings

        # Load the model
        _PRELOADED_EMBEDDING_MODEL = HuggingFaceEmbeddings(
            model_name=settings.embedding_model_full_path,
            model_kwargs={"device": settings.device},
            encode_kwargs={"batch_size": settings.batch_size, "normalize_embeddings": True},
            cache_folder=os.path.join(settings.models_dir, settings.embedding_model_path)
        )

        # Test the model with a simple embedding to ensure it works
        test_embedding = _PRELOADED_EMBEDDING_MODEL.embed_query("Test sentence for embedding model")
        embedding_dim = len(test_embedding)

        # Log GPU memory status after loading
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                allocated = torch.cuda.memory_allocated(i) / (1024 ** 3)
                reserved = torch.cuda.memory_reserved(i) / (1024 ** 3)
                logger.info(
                    f"GPU {i} after embedding model loading: Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")

        logger.info(f"Successfully preloaded embedding model with dimension {embedding_dim}")
    except Exception as e:
        logger.error(f"Failed to preload embedding model: {str(e)}")


def preload_llm_model():
    """
    Preload the LLM model at worker startup.
    """
    global _PRELOADED_LLM_MODEL

    # Skip if already loaded
    if _PRELOADED_LLM_MODEL is not None:
        return

    # Check worker type
    worker_type = os.environ.get("WORKER_TYPE", "")
    if worker_type != "gpu-inference":
        logger.info(f"Skipping LLM model preload on {worker_type} worker")
        return

    # Import settings
    from src.config.settings import settings

    logger.info(f"Preloading LLM model {settings.default_llm_model} on {settings.device}")

    try:
        # Set memory fraction
        set_memory_fraction_for_worker()

        # Clear CUDA cache before loading
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            for i in range(torch.cuda.device_count()):
                torch.cuda.synchronize(i)

            # Log GPU memory status before loading
            for i in range(torch.cuda.device_count()):
                device_name = torch.cuda.get_device_name(i)
                allocated = torch.cuda.memory_allocated(i) / (1024 ** 3)
                reserved = torch.cuda.memory_reserved(i) / (1024 ** 3)
                logger.info(
                    f"GPU {i} ({device_name}) before LLM model loading: Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")

        # Import the LLM
        from src.core.llm import LocalLLM

        # Load the model
        _PRELOADED_LLM_MODEL = LocalLLM(
            model_name=settings.default_llm_model,
            device=settings.device,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
            use_4bit=settings.llm_use_4bit,
            use_8bit=settings.llm_use_8bit,
            torch_dtype=torch.float16 if settings.use_fp16 and settings.device.startswith("cuda") else None
        )

        # Test the LLM with a simple prompt to ensure it works
        test_response = _PRELOADED_LLM_MODEL.answer_query(
            query="What is 2+2?",
            documents=[]
        )

        # Log GPU memory status after loading
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                allocated = torch.cuda.memory_allocated(i) / (1024 ** 3)
                reserved = torch.cuda.memory_reserved(i) / (1024 ** 3)
                logger.info(
                    f"GPU {i} after LLM model loading: Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")

        logger.info(f"Successfully preloaded LLM model")
    except Exception as e:
        logger.error(f"Failed to preload LLM model: {str(e)}")


def preload_colbert_reranker():
    """
    Preload the ColBERT and BGE reranking models.
    """
    global _PRELOADED_COLBERT_RERANKER

    # Skip if already loaded
    if _PRELOADED_COLBERT_RERANKER is not None:
        return

    # Check worker type
    worker_type = os.environ.get("WORKER_TYPE", "")
    if worker_type != "gpu-inference":
        logger.info(f"Skipping reranker preload on {worker_type} worker")
        return

    # Import settings
    from src.config.settings import settings

    logger.info(f"Preloading ColBERT and BGE reranking models on {settings.device}")

    try:
        # ADDED: Clear CUDA cache before loading to ensure maximum available memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            for i in range(torch.cuda.device_count()):
                torch.cuda.synchronize(i)

        # Log GPU memory status before loading
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                device_name = torch.cuda.get_device_name(i)
                allocated = torch.cuda.memory_allocated(i) / (1024 ** 3)
                reserved = torch.cuda.memory_reserved(i) / (1024 ** 3)
                total_memory = torch.cuda.get_device_properties(i).total_memory / (1024 ** 3)
                free_memory = total_memory - reserved
                logger.info(
                    f"GPU {i} ({device_name}) before reranker loading: "
                    f"Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB, "
                    f"Free: {free_memory:.2f} GB of {total_memory:.2f} GB total"
                )

        # Import reranker
        from src.core.colbert_reranker import ColBERTReranker

        # Load the reranking models
        _PRELOADED_COLBERT_RERANKER = ColBERTReranker(
            model_name=settings.default_colbert_model,
            device=settings.device,
            batch_size=settings.colbert_batch_size,
            use_fp16=settings.use_fp16,
            use_bge_reranker=settings.use_bge_reranker,
            colbert_weight=settings.colbert_weight,
            bge_weight=settings.bge_weight,
            bge_model_name=settings.default_bge_reranker_model
        )

        # Test the reranker with a simple query and document to ensure it works
        from langchain_core.documents import Document
        test_doc = Document(page_content="This is a test document.", metadata={})
        test_results = _PRELOADED_COLBERT_RERANKER.rerank("test query", [test_doc], 1)

        # Log GPU memory status after loading
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                allocated = torch.cuda.memory_allocated(i) / (1024 ** 3)
                reserved = torch.cuda.memory_reserved(i) / (1024 ** 3)
                total_memory = torch.cuda.get_device_properties(i).total_memory / (1024 ** 3)
                free_memory = total_memory - reserved
                logger.info(
                    f"GPU {i} after reranker loading: "
                    f"Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB, "
                    f"Free: {free_memory:.2f} GB of {total_memory:.2f} GB total"
                )

        logger.info(f"Successfully preloaded ColBERT and BGE reranking models")
    except Exception as e:
        logger.error(f"Failed to preload reranking models: {str(e)}")
        # ADDED: In case of failure, log more details and continue without reranker
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                allocated = torch.cuda.memory_allocated(i) / (1024 ** 3)
                reserved = torch.cuda.memory_reserved(i) / (1024 ** 3)
                total_memory = torch.cuda.get_device_properties(i).total_memory / (1024 ** 3)
                logger.error(
                    f"GPU {i} memory at failure: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved, {total_memory:.2f}GB total")

        # Set to None so the system can continue without reranking
        _PRELOADED_COLBERT_RERANKER = None


def preload_whisper_model():
    """
    Preload the Whisper model for transcription.
    """
    global _PRELOADED_WHISPER_MODEL

    # Skip if already loaded
    if _PRELOADED_WHISPER_MODEL is not None:
        return

    # Check worker type
    worker_type = os.environ.get("WORKER_TYPE", "")
    if worker_type != "gpu-whisper":
        logger.info(f"Skipping Whisper model preload on {worker_type} worker")
        return

    # Import settings
    from src.config.settings import settings

    logger.info(f"Preloading Whisper model {settings.whisper_model_size} on {settings.device}")

    try:
        # Set memory fraction
        set_memory_fraction_for_worker()

        # Clear CUDA cache before loading
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            for i in range(torch.cuda.device_count()):
                torch.cuda.synchronize(i)

            # Log GPU memory status before loading
            for i in range(torch.cuda.device_count()):
                device_name = torch.cuda.get_device_name(i)
                allocated = torch.cuda.memory_allocated(i) / (1024 ** 3)
                reserved = torch.cuda.memory_reserved(i) / (1024 ** 3)
                logger.info(
                    f"GPU {i} ({device_name}) before Whisper model loading: Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")

        # Import the WhisperModel
        from faster_whisper import WhisperModel

        # Get the model path
        model_path = settings.whisper_model_full_path if hasattr(settings,
                                                                 'whisper_model_full_path') else settings.whisper_model_size

        # Load the Whisper model
        _PRELOADED_WHISPER_MODEL = WhisperModel(
            model_path,
            device=settings.device,
            compute_type="float16" if settings.use_fp16 else "float32",
            cpu_threads=4,  # Use multiple CPU threads for pre/post-processing
            num_workers=2  # Number of workers for parallel processing
        )

        # Log GPU memory status after loading
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                allocated = torch.cuda.memory_allocated(i) / (1024 ** 3)
                reserved = torch.cuda.memory_reserved(i) / (1024 ** 3)
                logger.info(
                    f"GPU {i} after Whisper model loading: Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")

        logger.info(f"Successfully preloaded Whisper model")
    except Exception as e:
        logger.error(f"Failed to preload Whisper model: {str(e)}")


def get_vector_store():
    """
    Get a vector store instance for background tasks with support for preloaded models.
    """
    global _PRELOADED_EMBEDDING_MODEL

    from src.core.vectorstore import QdrantStore
    from src.config.settings import settings
    import torch

    # Initialize qdrant client
    from qdrant_client import QdrantClient
    qdrant_client = QdrantClient(
        host=settings.qdrant_host,
        port=settings.qdrant_port,
    )

    # If we have a preloaded model and we're on the embedding worker, use it
    worker_type = os.environ.get("WORKER_TYPE", "")
    if _PRELOADED_EMBEDDING_MODEL is not None and worker_type == "gpu-embedding":
        logger.info("Using preloaded embedding model for vector store")
        return QdrantStore(
            client=qdrant_client,
            collection_name=settings.qdrant_collection,
            embedding_function=_PRELOADED_EMBEDDING_MODEL,
        )

    # Create a new embedding model instance
    logger.info("Creating new embedding model instance for vector store")
    from langchain_huggingface import HuggingFaceEmbeddings

    # Determine the device (default to CPU for non-embedding workers)
    device = settings.device if worker_type == "gpu-embedding" else "cpu"

    embedding_function = HuggingFaceEmbeddings(
        model_name=settings.embedding_model_full_path,
        model_kwargs={"device": device},
        encode_kwargs={"batch_size": settings.batch_size, "normalize_embeddings": True},
        cache_folder=os.path.join(settings.models_dir, settings.embedding_model_path)
    )

    return QdrantStore(
        client=qdrant_client,
        collection_name=settings.qdrant_collection,
        embedding_function=embedding_function,
    )


def get_llm_model():
    """Get the preloaded LLM model or create a new one."""
    global _PRELOADED_LLM_MODEL

    # If we have a preloaded model and we're on the inference worker, use it
    worker_type = os.environ.get("WORKER_TYPE", "")
    if _PRELOADED_LLM_MODEL is not None and worker_type == "gpu-inference":
        return _PRELOADED_LLM_MODEL

    # Otherwise, create a new model instance
    from src.core.llm import LocalLLM
    from src.config.settings import settings
    import torch

    # Determine the device
    device = settings.device if worker_type == "gpu-inference" else "cpu"

    # Create a new LLM instance
    return LocalLLM(
        model_name=settings.default_llm_model,
        device=device,
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
        use_4bit=settings.llm_use_4bit and device.startswith("cuda"),
        use_8bit=settings.llm_use_8bit and device.startswith("cuda"),
        torch_dtype=torch.float16 if settings.use_fp16 and device.startswith("cuda") else None
    )


def get_colbert_reranker():
    """Get the preloaded reranker or create a new one."""
    global _PRELOADED_COLBERT_RERANKER

    # If we have a preloaded model and we're on the inference worker, use it
    worker_type = os.environ.get("WORKER_TYPE", "")
    if _PRELOADED_COLBERT_RERANKER is not None and worker_type == "gpu-inference":
        return _PRELOADED_COLBERT_RERANKER

    # Otherwise, create a new reranker instance
    from src.core.colbert_reranker import ColBERTReranker
    from src.config.settings import settings

    # Determine the device
    device = settings.device if worker_type == "gpu-inference" else "cpu"

    # Create a new reranker instance
    return ColBERTReranker(
        model_name=settings.default_colbert_model,
        device=device,
        batch_size=settings.colbert_batch_size,
        use_fp16=settings.use_fp16 and device.startswith("cuda"),
        use_bge_reranker=settings.use_bge_reranker,
        colbert_weight=settings.colbert_weight,
        bge_weight=settings.bge_weight,
        bge_model_name=settings.default_bge_reranker_model
    )


def get_whisper_model():
    """Get the preloaded Whisper model or create a new one."""
    global _PRELOADED_WHISPER_MODEL

    # If we have a preloaded model and we're on the whisper worker, use it
    worker_type = os.environ.get("WORKER_TYPE", "")
    if _PRELOADED_WHISPER_MODEL is not None and worker_type == "gpu-whisper":
        return _PRELOADED_WHISPER_MODEL

    # Otherwise, create a new Whisper instance
    from faster_whisper import WhisperModel
    from src.config.settings import settings

    # Determine the device
    device = settings.device if worker_type == "gpu-whisper" else "cpu"

    # Get the model path
    model_path = settings.whisper_model_full_path if hasattr(settings,
                                                             'whisper_model_full_path') else settings.whisper_model_size

    # Create a new Whisper instance
    return WhisperModel(
        model_path,
        device=device,
        compute_type="float16" if settings.use_fp16 and device.startswith("cuda") else "float32",
        cpu_threads=4,  # Use multiple CPU threads for pre/post-processing
        num_workers=2  # Number of workers for parallel processing
    )


# Add preload_models function that's called at worker startup
def preload_models():
    """Preload only the models this worker needs based on environment variables."""
    worker_type = get_worker_type()
    logger.info(f"Preloading models for worker type: {worker_type}")

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
        logger.info(f"Total GPU memory: {total_allocated:.2f}GB / {total_available:.2f}GB")


# Backward compatibility for reload_models function
def reload_models():
    """Reload all models to free up memory and avoid memory leaks."""
    global _PRELOADED_EMBEDDING_MODEL, _PRELOADED_LLM_MODEL, _PRELOADED_COLBERT_RERANKER, _PRELOADED_WHISPER_MODEL

    worker_type = os.environ.get("WORKER_TYPE", "")
    logger.info(f"Reloading models for {worker_type} worker")

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

    # Reload models based on worker type
    preload_models()

    # Log GPU memory after reload
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / (1024 ** 3)
            reserved = torch.cuda.memory_reserved(i) / (1024 ** 3)
            logger.info(
                f"GPU {i} after model reload: Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")