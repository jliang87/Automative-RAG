import os
from typing import Dict, List, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, Query
from pydantic import HttpUrl, BaseModel
from typing import Dict, Any  # ✅ Import Any from typing
import torch
import logging
import traceback

# Create a logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # Set to DEBUG level

from src.api.dependencies import get_vector_store, get_youtube_transcriber, get_pdf_loader, get_document_processor
from src.core.document_processor import DocumentProcessor
from src.core.vectorstore import QdrantStore
from src.core.base_video_transcriber import YouTubeTranscriber
from src.core.pdf_loader import PDFLoader
from src.models.schema import IngestResponse, ManualIngestRequest, YouTubeIngestRequest

router = APIRouter()


class BilibiliIngestRequest(BaseModel):
    """Request model for Bilibili video ingestion."""
    url: HttpUrl
    metadata: Optional[Dict[str, str]] = None


class BatchVideoIngestRequest(BaseModel):
    """Request model for batch video ingestion."""
    urls: List[HttpUrl]
    platform: str = "youtube"  # "youtube", "bilibili"
    metadata: Optional[List[Dict[str, str]]] = None


@router.post("/youtube", response_model=IngestResponse)
async def ingest_youtube(
    youtube_request: YouTubeIngestRequest,
    force_whisper: bool = Query(True, description="Force using Whisper for transcription"),
    processor: DocumentProcessor = Depends(get_document_processor),
) -> IngestResponse:
    """
    Ingest a YouTube video with GPU-accelerated transcription.

    Args:
        youtube_request: YouTube ingest request with URL and optional metadata
        force_whisper: Force using Whisper even if captions are available

    Returns:
        Ingest response with document IDs
    """
    try:
        # Process the YouTube video using pre-initialized processor
        document_ids = processor.process_youtube_video(
            url=str(youtube_request.url),
            custom_metadata=youtube_request.metadata,
            force_whisper=force_whisper
        )

        return IngestResponse(
            message=f"Successfully ingested YouTube video: {youtube_request.url}",
            document_count=len(document_ids),
            document_ids=document_ids,
        )
    except Exception as e:
        # Log the full traceback
        error_detail = f"Error ingesting YouTube video: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_detail)

        # Return detailed error information
        raise HTTPException(
            status_code=500,
            detail=error_detail,
        )


@router.post("/bilibili", response_model=IngestResponse)
async def ingest_bilibili(
        bilibili_request: BilibiliIngestRequest,
        force_whisper: bool = Query(True, description="Force using Whisper for transcription"),
        processor: DocumentProcessor = Depends(get_document_processor),
) -> IngestResponse:
    """
    Ingest a Bilibili video with GPU-accelerated transcription.
    """
    try:
        # Process the Bilibili video using pre-initialized processor
        document_ids = processor.process_bilibili_video(
            url=str(bilibili_request.url),
            custom_metadata=bilibili_request.metadata,
            force_whisper=force_whisper
        )

        return IngestResponse(
            message=f"Successfully ingested Bilibili video: {bilibili_request.url}",
            document_count=len(document_ids),
            document_ids=document_ids,
        )
    except Exception as e:
        # Log the full traceback
        error_detail = f"Error ingesting Bilibili video: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_detail)

        # Return detailed error information
        raise HTTPException(
            status_code=500,
            detail=error_detail,
        )


@router.post("/batch-videos", response_model=Dict[str, Any])
async def ingest_batch_videos(
        batch_request: BatchVideoIngestRequest,
        processor: DocumentProcessor = Depends(get_document_processor),
) -> Dict[str, Any]:
    """
    Ingest multiple videos in batch with GPU acceleration.
    """
    try:
        # Convert URLs to strings
        urls = [str(url) for url in batch_request.urls]

        # Process the videos in batch using pre-initialized processor
        results = processor.batch_process_videos(
            urls=urls,
            platform=batch_request.platform,
            custom_metadata=batch_request.metadata
        )

        # Count successful ingestions
        success_count = sum(1 for result in results.values() if isinstance(result, list))

        return {
            "message": f"Batch processing completed. {success_count}/{len(urls)} videos ingested successfully.",
            "results": results,
        }
    except Exception as e:
        # Log the full traceback
        error_detail = f"Error in batch processing: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_detail)

        # Return detailed error information
        raise HTTPException(
            status_code=500,
            detail=error_detail,
        )


@router.post("/pdf", response_model=IngestResponse)
async def ingest_pdf(
        file: UploadFile = File(...),
        metadata: Optional[str] = Form(None),
        use_ocr: Optional[bool] = Form(None),
        extract_tables: bool = Form(True),
        processor: DocumentProcessor = Depends(get_document_processor),
) -> IngestResponse:
    """
    Ingest a PDF file with GPU-accelerated OCR and table extraction.
    """
    try:
        # Parse metadata
        custom_metadata = {}
        if metadata:
            import json
            custom_metadata = json.loads(metadata)

        # Save the uploaded file
        upload_dir = "data/uploads"
        os.makedirs(upload_dir, exist_ok=True)

        file_path = os.path.join(upload_dir, file.filename)
        with open(file_path, "wb") as f:
            contents = await file.read()
            f.write(contents)

        # Process the PDF with pre-initialized processor
        document_ids = processor.process_pdf(
            file_path=file_path,
            custom_metadata=custom_metadata,
        )

        return IngestResponse(
            message=f"Successfully ingested PDF: {file.filename}",
            document_count=len(document_ids),
            document_ids=document_ids,
        )
    except Exception as e:
        # Log the full traceback
        error_detail = f"Error ingesting PDF: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_detail)

        # Return detailed error information
        raise HTTPException(
            status_code=500,
            detail=error_detail,
        )


@router.post("/text", response_model=IngestResponse)
async def ingest_text(
        manual_request: ManualIngestRequest,
        processor: DocumentProcessor = Depends(get_document_processor),
) -> IngestResponse:
    """
    Ingest manually entered text.
    """
    try:
        # Process the text using pre-initialized processor
        document_ids = processor.process_text(manual_request)

        return IngestResponse(
            message="Successfully ingested manual text",
            document_count=len(document_ids),
            document_ids=document_ids,
        )
    except Exception as e:
        # Log the full traceback
        error_detail = f"Error ingesting text: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_detail)

        # Return detailed error information
        raise HTTPException(
            status_code=500,
            detail=error_detail,
        )


@router.delete("/documents/{doc_id}", response_model=Dict[str, str])
async def delete_document(
        doc_id: str,
        processor: DocumentProcessor = Depends(get_document_processor),
) -> Dict[str, str]:
    """
    Delete a document by ID.
    """
    try:
        # Delete the document using pre-initialized processor
        processor.delete_documents([doc_id])

        return {"message": f"Successfully deleted document: {doc_id}"}
    except Exception as e:
        # Log the full traceback
        error_detail = f"Error deleting document: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_detail)

        # Return detailed error information
        raise HTTPException(
            status_code=500,
            detail=error_detail,
        )


@router.get("/status", response_model=Dict[str, Any])
async def get_ingest_status(
    vector_store: QdrantStore = Depends(get_vector_store),
    youtube_transcriber: YouTubeTranscriber = Depends(get_youtube_transcriber),
    pdf_loader: PDFLoader = Depends(get_pdf_loader),
) -> Dict[str, Any]:
    """
    Get ingestion status and statistics with GPU information.
    
    Returns:
        Dictionary with status information
    """
    try:
        stats = vector_store.get_stats()

        # Get GPU info if available
        gpu_info = {}
        if torch.cuda.is_available():
            gpu_info = {
                "device": youtube_transcriber.device,
                "device_name": torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else "N/A",
                "device_count": torch.cuda.device_count(),
                "memory_allocated": f"{torch.cuda.memory_allocated() / (1024**3):.2f} GB",
                "memory_reserved": f"{torch.cuda.memory_reserved() / (1024**3):.2f} GB",
                "fp16_enabled": youtube_transcriber.amp_enabled if hasattr(youtube_transcriber, "amp_enabled") else None,
                "whisper_model": youtube_transcriber.whisper_model_size,
                "ocr_enabled": pdf_loader.use_ocr
            }
        
        return {
            "status": "healthy",
            "collection": stats.get("name"),
            "document_count": stats.get("vectors_count", 0),
            "collection_size": stats.get("disk_data_size", 0),
            "gpu_info": gpu_info
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error getting ingestion status: {str(e)}",
        )


@router.post("/reset", response_model=Dict[str, str])
async def reset_vector_store(
    vector_store: QdrantStore = Depends(get_vector_store),
) -> Dict[str, str]:
    """
    Reset the vector store (dangerous operation).
    
    Returns:
        Success message
    """
    try:
        # Delete the collection
        vector_store.client.delete_collection(vector_store.collection_name)
        
        # Re-create the collection
        vector_store._ensure_collection()
        
        return {"message": f"Successfully reset vector store: {vector_store.collection_name}"}
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error resetting vector store: {str(e)}",
        )

@router.post("/video", response_model=IngestResponse)
async def ingest_video(
        video_request: Dict,
        processor: DocumentProcessor = Depends(get_document_processor),
) -> IngestResponse:
    """
    Ingest a video from any supported platform.
    """
    try:
        url = video_request.get("url")
        if not url:
            raise HTTPException(status_code=400, detail="URL is required")

        # Use the generic processor method
        document_ids = processor.process_video_generic(
            url=url,
            custom_metadata=video_request.get("metadata"),
            force_whisper=video_request.get("force_whisper", True)
        )

        # Determine platform for message
        platform = "video"
        if "youtube" in url:
            platform = "YouTube"
        elif "bilibili" in url:
            platform = "Bilibili"

        return IngestResponse(
            message=f"Successfully ingested {platform} video: {url}",
            document_count=len(document_ids),
            document_ids=document_ids,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error ingesting video: {str(e)}")