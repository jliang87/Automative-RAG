from typing import Annotated, Any, Dict, List, Optional, Tuple

from fastapi import Depends, HTTPException, Header, status
from qdrant_client import QdrantClient
import torch

from src.config.settings import settings
from src.core.colbert_reranker import ColBERTReranker
from src.core.llm import LocalLLM
from src.core.retriever import HybridRetriever
from src.core.vectorstore import QdrantStore
from src.core.video_transcriber import VideoTranscriber
from src.core.pdf_loader import PDFLoader
from src.core.document_processor import DocumentProcessor

# Global instances that will be initialized during app startup
llm_model = None  # LLM instance
colbert_model = None  # ColBERT instance
video_transcriber = None  # Unified video transcriber
pdf_loader = None
qdrant_client = None
vector_store = None
retriever = None
document_processor = None

def load_video_transcriber():
    """Load video transcriber model once at startup."""
    global video_transcriber
    if video_transcriber is None:
        print("🚀 Loading Video Transcriber...")
        video_transcriber = VideoTranscriber(
            whisper_model_size=settings.whisper_model_size,
            device=settings.device
        )
        print("✅ Video Transcriber Loaded!")


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

def init_document_processor():
    """Initialize document processor once at app startup."""
    global document_processor, vector_store, video_transcriber, pdf_loader
    if document_processor is None:
        print("🚀 Initializing Document Processor...")
        document_processor = DocumentProcessor(
            vector_store=vector_store,
            video_transcriber=video_transcriber,
            pdf_loader=pdf_loader
        )
        print("✅ Document Processor Initialized!")

def load_all_components():
    """Initialize all components at application startup."""
    load_video_transcriber()
    load_llm()
    load_colbert_and_bge_reranker()
    load_pdf_loader()
    init_vector_store()
    init_retriever()
    init_document_processor()

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

# Video transcriber dependency with GPU support
def get_video_transcriber() -> VideoTranscriber:
    """Retrieve the preloaded video transcriber."""
    if video_transcriber is None:
        raise HTTPException(status_code=500, detail="Video Transcriber is not initialized. Call `load_video_transcriber()` first.")
    return video_transcriber

# PDF loader dependency with GPU support
def get_pdf_loader() -> PDFLoader:
    """Retrieve the preloaded PDF loader."""
    if pdf_loader is None:
        raise HTTPException(status_code=500, detail="PDF Loader is not initialized. Call `load_pdf_loader()` first.")
    return pdf_loader

def get_document_processor() -> DocumentProcessor:
    """Get the cached document processor instance."""
    if document_processor is None:
        raise HTTPException(status_code=500, detail="Document Processor not initialized yet.")
    return document_processor