from typing import Annotated, Any, Dict, List, Optional, Tuple

from fastapi import Depends, HTTPException, Header, status
from qdrant_client import QdrantClient

from src.config.settings import settings
from src.core.colbert_reranker import ColBERTReranker
from src.core.local_llm import LocalDeepSeekLLM
from src.core.retriever import HybridRetriever
from src.core.vectorstore import QdrantStore
from src.core.youtube_transcriber import YouTubeTranscriber, BilibiliTranscriber
from src.core.youku_transcriber import YoukuTranscriber
from src.core.pdf_loader import PDFLoaderr

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
    return ColBERTReranker(
        model_name=settings.colbert_model,
        device=settings.device,
        batch_size=settings.colbert_batch_size,
        use_fp16=settings.use_fp16
    )

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
def get_local_llm() -> LocalDeepSeekLLM:
    return LocalDeepSeekLLM(
        model_name=settings.deepseek_model,
        device=settings.device,
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
        use_4bit=settings.llm_use_4bit,
        use_8bit=settings.llm_use_8bit,
        torch_dtype=torch.float16 if settings.use_fp16 and settings.device.startswith("cuda") else None
    )

# YouTube transcriber dependency with GPU support
def get_youtube_transcriber() -> YouTubeTranscriber:
    return YouTubeTranscriber(
        whisper_model_size=settings.whisper_model_size,
        device=settings.device,
        use_youtube_captions=settings.use_youtube_captions,
        use_whisper_as_fallback=settings.use_whisper_as_fallback,
        force_whisper=settings.force_whisper
    )

# Bilibili transcriber dependency with GPU support
def get_bilibili_transcriber() -> BilibiliTranscriber:
    return BilibiliTranscriber(
        whisper_model_size=settings.whisper_model_size,
        device=settings.device,
        force_whisper=settings.force_whisper
    )

# Youku transcriber dependency with GPU support
def get_youku_transcriber() -> YoukuTranscriber:
    return YoukuTranscriber(
        whisper_model_size=settings.whisper_model_size,
        device=settings.device,
        force_whisper=settings.force_whisper
    )

# PDF loader dependency with GPU support
def get_pdf_loader() -> PDFLoader:
    return PDFLoader(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        device=settings.device,
        use_ocr=settings.use_pdf_ocr,
        ocr_languages=settings.ocr_languages
    )