import os
from typing import Dict, List, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, Query
from pydantic import HttpUrl, BaseModel
from typing import Dict, Any
import torch
import logging
import traceback

# Create a logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # Set to DEBUG level

from src.api.dependencies import get_vector_store, get_document_processor, get_pdf_loader
from src.core.document_processor import DocumentProcessor
from src.core.vectorstore import QdrantStore
from src.core.pdf_loader import PDFLoader
from src.models.schema import IngestResponse, ManualIngestRequest

router = APIRouter()


class VideoIngestRequest(BaseModel):
    """Request model for video ingestion from any platform."""
    url: HttpUrl
    metadata: Optional[Dict[str, str]] = None


class BatchVideoIngestRequest(BaseModel):
    """Request model for batch video ingestion."""
    urls: List[HttpUrl]
    metadata: Optional[List[Dict[str, str]]] = None


@router.post("/video", response_model=IngestResponse)
async def ingest_video(
        video_request: VideoIngestRequest,
        processor: DocumentProcessor = Depends(get_document_processor),
) -> IngestResponse:
    """
    Ingest a video from any supported platform (YouTube, Bilibili, etc.) with GPU-accelerated Whisper transcription.

    Args:
        video_request: Video ingest request with URL and optional metadata
        processor: Document processor dependency

    Returns:
        Ingest response with document IDs
    """
    try:
        # Process the video using unified processor
        document_ids = processor.process_video(
            url=str(video_request.url),
            custom_metadata=video_request.metadata
        )

        # Determine platform for more helpful user feedback
        platform = "video"
        url_str = str(video_request.url)
        if "youtube" in url_str:
            platform = "YouTube"
        elif "bilibili" in url_str:
            platform = "Bilibili"

        return IngestResponse(
            message=f"Successfully ingested {platform} video: {video_request.url}",
            document_count=len(document_ids),
            document_ids=document_ids,
        )
    except Exception as e:
        # Log the full traceback
        error_detail = f"Error ingesting video: {str(e)}\n{traceback.format_exc()}"
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

    Args:
        batch_request: Batch video request with URLs and optional metadata
        processor: Document processor dependency

    Returns:
        Dictionary with results for each URL
    """
    try:
        # Convert URLs to strings
        urls = [str(url) for url in batch_request.urls]

        # Process the videos in batch
        results = processor.batch_process_videos(
            urls=urls,
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
        pdf_loader: PDFLoader = Depends(get_pdf_loader),
        processor: DocumentProcessor = Depends(get_document_processor),
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
                "device": processor.video_transcriber.device,
                "device_name": torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else "N/A",
                "device_count": torch.cuda.device_count(),
                "memory_allocated": f"{torch.cuda.memory_allocated() / (1024 ** 3):.2f} GB",
                "memory_reserved": f"{torch.cuda.memory_reserved() / (1024 ** 3):.2f} GB",
                "fp16_enabled": hasattr(processor.video_transcriber,
                                        "amp_enabled") and processor.video_transcriber.amp_enabled,
                "whisper_model": processor.video_transcriber.whisper_model_size,
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