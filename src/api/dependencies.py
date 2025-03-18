from typing import Annotated, Any, Dict, List, Optional, Tuple

from fastapi import Depends, HTTPException, Header, status
from qdrant_client import QdrantClient
import torch

from src.config.settings import settings
from src.core.colbert_reranker import ColBERTReranker
from src.core.llm import LocalLLM
from src.core.retriever import HybridRetriever
from src.core.vectorstore import QdrantStore
from src.core.youtube_transcriber import YouTubeTranscriber, BilibiliTranscriber
from src.core.youku_transcriber import YoukuTranscriber
from src.core.pdf_loader import PDFLoader

llm_model = None  # Global variable for the LLM instance
colbert_model = None  # Global variable for the ColBERT instance
youtube_transcriber = None
bilibili_transcriber = None
youku_transcriber = None

def load_transcribers():
    """Load transcriber models once at startup."""
    global youtube_transcriber, bilibili_transcriber, youku_transcriber
    if youtube_transcriber is None:
        print("🚀 Loading YouTube Transcriber...")
        youtube_transcriber = YouTubeTranscriber(
            whisper_model_size=settings.whisper_model_size,
            device=settings.device,
            use_youtube_captions=settings.use_youtube_captions,
            use_whisper_as_fallback=settings.use_whisper_as_fallback,
            force_whisper=settings.force_whisper
        )
        print("✅ YouTube Transcriber Loaded!")

    if bilibili_transcriber is None:
        print("🚀 Loading Bilibili Transcriber...")
        bilibili_transcriber = BilibiliTranscriber(
            whisper_model_size=settings.whisper_model_size,
            device=settings.device,
            force_whisper=settings.force_whisper
        )
        print("✅ Bilibili Transcriber Loaded!")

    if youku_transcriber is None:
        print("🚀 Loading Youku Transcriber...")
        youku_transcriber = YoukuTranscriber(
            whisper_model_size=settings.whisper_model_size,
            device=settings.device,
            force_whisper=settings.force_whisper
        )
        print("✅ Youku Transcriber Loaded!")

def load_llm():
    """Load the LLM model once when the app starts."""
    global llm_model
    if llm_model is None:
        print("🚀 Loading LLM model...")
        llm_model = LocalLLM(
            model_name=settings.default_llm_model,
            device=settings.device,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
            use_4bit=settings.llm_use_4bit,
            use_8bit=settings.llm_use_8bit,
            torch_dtype=torch.float16 if settings.use_fp16 and settings.device.startswith(
                "cuda") else None  # Load once and keep in memory
        )
        print("✅ LLM Model Loaded!")

def load_colbert():
    """Load the ColBERT model once when the app starts."""
    global colbert_model
    if colbert_model is None:
        print("🚀 Loading ColBERT model...")
        colbert_model = ColBERTReranker(
            model_name=settings.default_colbert_model,
            device=settings.device,
            batch_size=settings.colbert_batch_size,
            use_fp16=settings.use_fp16
        )
        print("✅ ColBERT Model Loaded!")

# Authentication dependency
async def get_token_header(x_token: str = Header(...)):
    if x_token != settings.api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token",
        )
    return x_token

# Qdrant client dependency
def get_qdrant_client() -> QdrantClient:
    return QdrantClient(
        host=settings.qdrant_host,
        port=settings.qdrant_port,
    )

# Vector store dependency
def get_vector_store(
    qdrant_client: QdrantClient = Depends(get_qdrant_client),
) -> QdrantStore:
    return QdrantStore(
        client=qdrant_client,
        collection_name=settings.qdrant_collection,
        embedding_function=settings.embedding_function,
    )

# ColBERT reranker dependency with GPU support
def get_colbert_reranker() -> ColBERTReranker:
    """Get the cached ColBERT instance, ensuring it is preloaded."""
    if colbert_model is None:
        raise HTTPException(status_code=500, detail="ColBERT not initialized yet.")
    return colbert_model

# Retriever dependency
def get_retriever(
    vector_store: QdrantStore = Depends(get_vector_store),
    reranker: ColBERTReranker = Depends(get_colbert_reranker),
) -> HybridRetriever:
    return HybridRetriever(
        vector_store=vector_store,
        reranker=reranker,
        top_k=settings.retriever_top_k,
        rerank_top_k=settings.reranker_top_k,
    )

# Local LLM dependency
def get_llm() -> LocalLLM:
    """Get the cached LLM instance, ensuring it is preloaded."""
    if llm_model is None:
        raise HTTPException(status_code=500, detail="LLM not initialized yet.")
    return llm_model

# YouTube transcriber dependency with GPU support
def get_youtube_transcriber() -> YouTubeTranscriber:
    """Retrieve the preloaded YouTube transcriber."""
    if youtube_transcriber is None:
        raise RuntimeError("YouTube Transcriber is not initialized. Call `load_transcribers()` first.")
    return youtube_transcriber

# Bilibili transcriber dependency with GPU support
def get_bilibili_transcriber() -> BilibiliTranscriber:
    """Retrieve the preloaded Bilibili transcriber."""
    if bilibili_transcriber is None:
        raise RuntimeError("Bilibili Transcriber is not initialized. Call `load_transcribers()` first.")
    return bilibili_transcriber

# Youku transcriber dependency with GPU support
def get_youku_transcriber() -> YoukuTranscriber:
    """Retrieve the preloaded Youku transcriber."""
    if youku_transcriber is None:
        raise RuntimeError("Youku Transcriber is not initialized. Call `load_transcribers()` first.")
    return youku_transcriber

# PDF loader dependency with GPU support
def get_pdf_loader() -> PDFLoader:
    return PDFLoader(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        device=settings.device,
        use_ocr=settings.use_pdf_ocr,
        ocr_languages=settings.ocr_languages
    )