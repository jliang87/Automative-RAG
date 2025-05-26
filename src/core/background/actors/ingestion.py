# src/core/background/actors/ingestion.py (SIMPLIFIED - Remove All Priority Queue Logic)

"""
Simplified ingestion actors that contain only the core work logic.
All coordination is handled by the job chain system.
"""

import os
import logging
from typing import Dict, List, Optional, Any
import dramatiq
import torch

logger = logging.getLogger(__name__)

# Keep only the core work functions - remove all priority queue coordination


def extract_audio_from_video(url: str) -> str:
    """Extract audio from video URL. Pure work function."""
    from src.core.video_transcriber import VideoTranscriber

    transcriber = VideoTranscriber()
    media_path = transcriber.extract_audio(url)
    return media_path


def get_video_metadata(url: str) -> Dict[str, Any]:
    """Get video metadata. Pure work function."""
    from src.core.video_transcriber import VideoTranscriber

    transcriber = VideoTranscriber()
    return transcriber.get_video_metadata(url)


def transcribe_audio_file(media_path: str) -> tuple:
    """Transcribe audio file using Whisper. Pure work function."""
    from src.core.background.models import get_whisper_model

    whisper_model = get_whisper_model()
    segments, info = whisper_model.transcribe(
        media_path,
        beam_size=5,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=500)
    )

    # Collect all segments
    all_text = [segment.text for segment in segments]
    transcript = " ".join(all_text)

    # Apply Chinese conversion if needed
    if info.language == "zh":
        try:
            import opencc
            converter = opencc.OpenCC('t2s')
            transcript = converter.convert(transcript)
        except ImportError:
            logger.warning("opencc not found. Chinese conversion skipped.")

    return transcript, info


def process_pdf_file(file_path: str, custom_metadata: Optional[Dict] = None) -> List:
    """Process PDF file. Pure work function."""
    from src.core.pdf_loader import PDFLoader
    from src.config.settings import settings

    pdf_loader = PDFLoader(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        device="cpu",
        use_ocr=settings.use_pdf_ocr,
        ocr_languages=settings.ocr_languages
    )

    documents = pdf_loader.process_pdf(
        file_path=file_path,
        custom_metadata=custom_metadata,
    )

    return documents


def process_text_content(text: str, metadata: Optional[Dict] = None) -> List:
    """Process text content into chunks. Pure work function."""
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_core.documents import Document
    from src.config.settings import settings

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )

    chunks = text_splitter.split_text(text)

    documents = []
    for i, chunk_text in enumerate(chunks):
        doc = Document(
            page_content=chunk_text,
            metadata={
                "chunk_id": i,
                "source": "manual",
                "total_chunks": len(chunks),
                **(metadata or {})
            }
        )
        documents.append(doc)

    return documents


def generate_embeddings_for_documents(documents: List) -> List[str]:
    """Generate embeddings for documents. Pure work function."""
    from src.core.background.models import get_vector_store
    from langchain_core.documents import Document
    import time

    # Convert back to Document objects if needed
    doc_objects = []
    for doc in documents:
        if isinstance(doc, dict):
            doc_obj = Document(
                page_content=doc["content"],
                metadata=doc["metadata"]
            )
        else:
            doc_obj = doc
        doc_objects.append(doc_obj)

    # Add ingestion timestamp
    current_time = time.time()
    for doc in doc_objects:
        doc.metadata["ingestion_time"] = current_time

    # Add to vector store
    vector_store = get_vector_store()
    doc_ids = vector_store.add_documents(doc_objects)

    return doc_ids


# Legacy actors that might still be called directly (keep minimal versions)
# These are kept for backward compatibility but should be phased out

@dramatiq.actor(queue_name="embedding_tasks", store_results=True, max_retries=2)
def delete_document_by_id(doc_id: str):
    """Delete a document by ID. Simplified version."""
    try:
        from src.core.background.models import get_vector_store

        vector_store = get_vector_store()
        vector_store.delete_by_ids([doc_id])

        logger.info(f"Successfully deleted document: {doc_id}")
        return {"document_id": doc_id, "status": "deleted"}

    except Exception as e:
        logger.error(f"Error deleting document {doc_id}: {str(e)}")
        raise