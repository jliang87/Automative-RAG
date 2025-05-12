"""
Model management for background workers.

This module handles preloading and managing ML models for workers,
optimizing GPU memory usage and ensuring models are loaded efficiently.
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
                    f"GPU {i} ({device_name}) before reranker loading: Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")

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
                logger.info(
                    f"GPU {i} after reranker loading: Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")

        logger.info(f"Successfully preloaded ColBERT and BGE reranking models")
    except Exception as e:
        logger.error(f"Failed to preload reranking models: {str(e)}")


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

    # Reload appropriate models based on worker type
    if worker_type == "gpu-inference":
        # Free memory
        del _PRELOADED_LLM_MODEL
        del _PRELOADED_COLBERT_RERANKER
        _PRELOADED_LLM_MODEL = None
        _PRELOADED_COLBERT_RERANKER = None

        # Force garbage collection again
        gc.collect()

        # Reload models
        preload_llm_model()
        preload_colbert_reranker()

    elif worker_type == "gpu-embedding":
        del _PRELOADED_EMBEDDING_MODEL
        _PRELOADED_EMBEDDING_MODEL = None
        gc.collect()
        preload_embedding_model()

    elif worker_type == "gpu-whisper":
        del _PRELOADED_WHISPER_MODEL
        _PRELOADED_WHISPER_MODEL = None
        gc.collect()
        preload_whisper_model()

    # Log GPU memory after reload
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / (1024 ** 3)
            reserved = torch.cuda.memory_reserved(i) / (1024 ** 3)
            logger.info(
                f"GPU {i} after model reload: Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")