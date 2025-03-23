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

# Global instances that will be initialized during app startup
llm_model = None  # LLM instance
colbert_model = None  # ColBERT instance
youtube_transcriber = None
bilibili_transcriber = None
youku_transcriber = None
pdf_loader = None
qdrant_client = None
vector_store = None
retriever = None

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

def load_colbert_and_bge_reranker():
    """Load the ColBERT model once when the app starts."""
    global colbert_model
    if colbert_model is None:
        print("🚀 Loading ColBERT model...")
        colbert_model = ColBERTReranker(
            model_name=settings.default_colbert_model,
            device=settings.device,
            batch_size=settings.colbert_batch_size,
            use_fp16=settings.use_fp16,
            use_bge_reranker=settings.use_bge_reranker,
            colbert_weight=settings.colbert_weight,
            bge_weight=settings.bge_weight,
            bge_model_name=settings.default_bge_reranker_model
        )
        print("✅ ColBERT and BGE Reranker Model Loaded!")

def load_pdf_loader():
    """Load the PDF loader once when the app starts."""
    global pdf_loader
    if pdf_loader is None:
        print("🚀 Loading PDF Loader...")
        pdf_loader = PDFLoader(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            device=settings.device,
            use_ocr=settings.use_pdf_ocr,
            ocr_languages=settings.ocr_languages
        )
        print("✅ PDF Loader Loaded!")

def init_vector_store():
    """Initialize Qdrant client and vector store once at app startup."""
    global qdrant_client, vector_store
    if qdrant_client is None:
        print("🚀 Initializing Qdrant client...")
        qdrant_client = QdrantClient(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
        )
        print("✅ Qdrant Client Initialized!")

    if vector_store is None:
        print("🚀 Initializing Vector Store...")
        vector_store = QdrantStore(
            client=qdrant_client,
            collection_name=settings.qdrant_collection,
            embedding_function=settings.embedding_function,
        )
        print("✅ Vector Store Initialized!")

def init_retriever():
    """Initialize retriever once at app startup."""
    global retriever, vector_store, colbert_model
    if retriever is None and vector_store is not None and colbert_model is not None:
        print("🚀 Initializing Hybrid Retriever...")
        retriever = HybridRetriever(
            vector_store=vector_store,
            reranker=colbert_model,
            top_k=settings.retriever_top_k,
            rerank_top_k=settings.reranker_top_k,
        )
        print("✅ Hybrid Retriever Initialized!")
    elif vector_store is None or colbert_model is None:
        print("⚠️ Cannot initialize retriever: vector_store or colbert_model not loaded")

def load_all_components():
    """Initialize all components at application startup."""
    load_transcribers()
    load_llm()
    load_colbert_and_bge_reranker()
    load_pdf_loader()
    init_vector_store()
    init_retriever()

# Authentication dependency
async def get_token_header(x_token: str = Header(...)):
    if x_token != settings.api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token",
        )
    return x_token

# Qdrant client dependency - reuses the global instance
def get_qdrant_client() -> QdrantClient:
    if qdrant_client is None:
        raise HTTPException(status_code=500, detail="Qdrant client not initialized yet.")
    return qdrant_client

# Vector store dependency - reuses the global instance
def get_vector_store() -> QdrantStore:
    if vector_store is None:
        raise HTTPException(status_code=500, detail="Vector store not initialized yet.")
    return vector_store

# ColBERT reranker dependency with GPU support
def get_colbert_reranker() -> ColBERTReranker:
    """Get the cached ColBERT instance, ensuring it is preloaded."""
    if colbert_model is None:
        raise HTTPException(status_code=500, detail="ColBERT not initialized yet.")
    return colbert_model

# Retriever dependency - reuses the global instance
def get_retriever() -> HybridRetriever:
    if retriever is None:
        raise HTTPException(status_code=500, detail="Retriever not initialized yet.")
    return retriever

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
        raise HTTPException(status_code=500, detail="YouTube Transcriber is not initialized. Call `load_transcribers()` first.")
    return youtube_transcriber

# Bilibili transcriber dependency with GPU support
def get_bilibili_transcriber() -> BilibiliTranscriber:
    """Retrieve the preloaded Bilibili transcriber."""
    if bilibili_transcriber is None:
        raise HTTPException(status_code=500, detail="Bilibili Transcriber is not initialized. Call `load_transcribers()` first.")
    return bilibili_transcriber

# Youku transcriber dependency with GPU support
def get_youku_transcriber() -> YoukuTranscriber:
    """Retrieve the preloaded Youku transcriber."""
    if youku_transcriber is None:
        raise HTTPException(status_code=500, detail="Youku Transcriber is not initialized. Call `load_transcribers()` first.")
    return youku_transcriber

# PDF loader dependency with GPU support
def get_pdf_loader() -> PDFLoader:
    """Retrieve the preloaded PDF loader."""
    if pdf_loader is None:
        raise HTTPException(status_code=500, detail="PDF Loader is not initialized. Call `load_pdf_loader()` first.")
    return pdf_loader