import os
import uuid
from typing import Dict, List, Optional, Union

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
from src.models.schema import IngestResponse, ManualIngestRequest, BackgroundJobResponse
from src.core.background_tasks import job_tracker, batch_process_videos, process_video_gpu, process_pdf_cpu

router = APIRouter()


class VideoIngestRequest(BaseModel):
    """Request model for video ingestion from any platform."""
    url: HttpUrl
    metadata: Optional[Dict[str, str]] = None


class BatchVideoIngestRequest(BaseModel):
    """Request model for batch video ingestion."""
    urls: List[HttpUrl]
    metadata: Optional[List[Dict[str, str]]] = None


@router.post("/video", response_model=BackgroundJobResponse)
async def ingest_video(
        video_request: VideoIngestRequest,
        processor: DocumentProcessor = Depends(get_document_processor),
) -> BackgroundJobResponse:
    """
    Ingest a video from any supported platform (YouTube, Bilibili, etc.) with GPU-accelerated Whisper transcription.
    All processing happens asynchronously in the background.

    Args:
        video_request: Video ingest request with URL and optional metadata
        processor: Document processor dependency

    Returns:
        Background job response with job ID
    """
    try:
        url_str = str(video_request.url)

        # Determine platform for more helpful user feedback
        platform = "video"
        if "youtube" in url_str:
            platform = "YouTube"
        elif "bilibili" in url_str:
            platform = "Bilibili"

        # Generate a unique job ID
        job_id = str(uuid.uuid4())

        # Create a job record
        job_tracker.create_job(
            job_id=job_id,
            job_type="video_processing",
            metadata={
                "url": url_str,
                "platform": platform,
                "custom_metadata": video_request.metadata
            }
        )

        # Start the background job
        process_video_gpu.send(job_id, url_str, video_request.metadata)

        # Return the job ID immediately
        return BackgroundJobResponse(
            message=f"Processing {platform} video in the background",
            job_id=job_id,
            job_type="video_processing",
            status="pending",
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


@router.post("/batch-videos", response_model=BackgroundJobResponse)
async def ingest_batch_videos(
        batch_request: BatchVideoIngestRequest,
) -> BackgroundJobResponse:
    """
    Ingest multiple videos in batch with GPU acceleration.
    All processing happens asynchronously in the background.
    """
    try:
        # Convert URLs to strings
        urls = [str(url) for url in batch_request.urls]

        # Generate a unique job ID
        job_id = str(uuid.uuid4())

        # Create a job record
        job_tracker.create_job(
            job_id=job_id,
            job_type="batch_video_processing",
            metadata={
                "urls": urls,
                "custom_metadata": batch_request.metadata
            }
        )

        # Call the batch processing function directly
        # Import the regular function (not an actor)
        from src.core.background_tasks import batch_process_videos

        # Call it directly - this will internally queue each video to the GPU worker
        batch_process_videos(job_id, urls, batch_request.metadata)

        # Return the job ID immediately
        return BackgroundJobResponse(
            message=f"Processing {len(urls)} videos in the background",
            job_id=job_id,
            job_type="batch_video_processing",
            status="processing",
        )
    except Exception as e:
        # Log the full traceback
        error_detail = f"Error in batch processing: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_detail)

        # Return detailed error information
        raise HTTPException(
            status_code=500,
            detail=error_detail,
        )


@router.post("/pdf", response_model=BackgroundJobResponse)
async def ingest_pdf(
        file: UploadFile = File(...),
        metadata: Optional[str] = Form(None),
        use_ocr: Optional[bool] = Form(None),
        extract_tables: bool = Form(True),
        processor: DocumentProcessor = Depends(get_document_processor),
) -> BackgroundJobResponse:
    """
    Ingest a PDF file with GPU-accelerated OCR and table extraction.
    All processing happens asynchronously in the background.
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

        # Generate a unique job ID
        job_id = str(uuid.uuid4())

        # Create a job record
        job_tracker.create_job(
            job_id=job_id,
            job_type="pdf_processing",
            metadata={
                "filename": file.filename,
                "filepath": file_path,
                "custom_metadata": custom_metadata,
                "use_ocr": use_ocr,
                "extract_tables": extract_tables
            }
        )

        # Start the background job
        process_pdf_cpu().send(job_id, file_path, custom_metadata)

        # Return the job ID immediately
        return BackgroundJobResponse(
            message=f"Processing PDF in the background: {file.filename}",
            job_id=job_id,
            job_type="pdf_processing",
            status="pending",
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


@router.post("/text", response_model=BackgroundJobResponse)
async def ingest_text(
        manual_request: ManualIngestRequest,
        processor: DocumentProcessor = Depends(get_document_processor),
) -> BackgroundJobResponse:
    """
    Ingest manually entered text.
    All processing happens asynchronously in the background.
    """
    try:
        # Generate a unique job ID
        job_id = str(uuid.uuid4())

        # Create a job record
        job_tracker.create_job(
            job_id=job_id,
            job_type="manual_text",
            metadata={
                "title": manual_request.metadata.title if manual_request.metadata.title else "Manual Text Input",
                "content_length": len(manual_request.content)
            }
        )

        # Start the background job
        from src.core.background_tasks import process_text
        process_text.send(job_id, manual_request.content, manual_request.metadata.dict())

        # Return the job ID immediately
        return BackgroundJobResponse(
            message="Processing text input in the background",
            job_id=job_id,
            job_type="manual_text",
            status="pending",
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


@router.get("/jobs/{job_id}", response_model=Dict[str, Any])
async def get_job_status(job_id: str) -> Dict[str, Any]:
    """
    Get the status of a background job.

    Args:
        job_id: ID of the job to check

    Returns:
        Job information including status, result, and error if any
    """
    job_data = job_tracker.get_job(job_id)

    if not job_data:
        raise HTTPException(
            status_code=404,
            detail=f"Job with ID {job_id} not found"
        )

    return job_data


@router.get("/jobs", response_model=List[Dict[str, Any]])
async def get_all_jobs(
        limit: int = Query(50, ge=1, le=100),
        job_type: Optional[str] = Query(None)
) -> List[Dict[str, Any]]:
    """
    Get all background jobs, optionally filtered by type.

    Args:
        limit: Maximum number of jobs to return (default: 50)
        job_type: Filter jobs by type (e.g., 'video_processing', 'pdf_processing')

    Returns:
        List of job information
    """
    return job_tracker.get_all_jobs(limit=limit, job_type=job_type)


@router.delete("/jobs/{job_id}", response_model=Dict[str, str])
async def delete_job(job_id: str) -> Dict[str, str]:
    """
    Delete a job by ID.

    Args:
        job_id: ID of the job to delete

    Returns:
        Success message
    """
    deleted = job_tracker.delete_job(job_id)

    if not deleted:
        raise HTTPException(
            status_code=404,
            detail=f"Job with ID {job_id} not found"
        )

    return {"message": f"Successfully deleted job: {job_id}"}


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

        # Get background job stats
        job_stats = {
            "pending_jobs": len([j for j in job_tracker.get_all_jobs(limit=1000) if j.get("status") == "pending"]),
            "processing_jobs": len(
                [j for j in job_tracker.get_all_jobs(limit=1000) if j.get("status") == "processing"]),
            "completed_jobs": len([j for j in job_tracker.get_all_jobs(limit=1000) if j.get("status") == "completed"]),
            "failed_jobs": len([j for j in job_tracker.get_all_jobs(limit=1000) if
                                j.get("status") == "failed" or j.get("status") == "timeout"])
        }

        return {
            "status": "healthy",
            "collection": stats.get("name"),
            "document_count": stats.get("vectors_count", 0),
            "collection_size": stats.get("disk_data_size", 0),
            "gpu_info": gpu_info,
            "job_stats": job_stats
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