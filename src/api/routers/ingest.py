import os
import uuid
from typing import Dict, List, Optional, Union

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, Query
from pydantic import HttpUrl, BaseModel
from typing import Dict, Any
import torch
import logging
import traceback
import redis
from src.core.background.job_tracker import JobTracker
from src.api.dependencies import get_vector_store, get_redis_client, get_job_tracker
from src.core.worker_status import get_worker_status_for_ui

# Create a logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # Set to DEBUG level

from src.core.vectorstore import QdrantStore
from src.models.schema import IngestResponse, ManualIngestRequest, BackgroundJobResponse

router = APIRouter()


class VideoIngestRequest(BaseModel):
    """Request model for video ingestion from any platform."""
    url: HttpUrl
    metadata: Optional[Dict[str, str]] = None


class BatchVideoIngestRequest(BaseModel):
    """Request model for batch video ingestion."""
    urls: List[HttpUrl]
    metadata: Optional[List[Dict[str, str]]] = None


from src.core.background import (
    process_video_gpu,
    process_pdf_cpu,
    process_text,
    batch_process_videos,
    job_tracker
)
@router.post("/video", response_model=BackgroundJobResponse)
async def ingest_video(
        video_request: VideoIngestRequest,
) -> BackgroundJobResponse:
    """
    Ingest a video from any supported platform with GPU-accelerated Whisper transcription.
    All processing happens asynchronously in the background.
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
        # Return detailed error information
        raise HTTPException(
            status_code=500,
            detail=f"Error ingesting video: {str(e)}",
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
        from src.core.background import batch_process_videos

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
        extract_tables: bool = Form(True)
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
async def ingest_text(manual_request: ManualIngestRequest) -> BackgroundJobResponse:
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
        from src.core.background import process_text
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


@router.delete("/documents/{doc_id}", response_model=BackgroundJobResponse)
async def delete_document(
        doc_id: str,
        job_tracker: JobTracker = Depends(get_job_tracker),
) -> BackgroundJobResponse:
    """
    Delete a document by ID as a background task.
    """
    try:
        # Generate a unique job ID
        job_id = str(uuid.uuid4())

        # Create a job record
        job_tracker.create_job(
            job_id=job_id,
            job_type="document_deletion",
            metadata={
                "document_id": doc_id
            }
        )

        # Start the background job - use the embedding_tasks queue for GPU access
        from src.core.background import delete_document_gpu
        delete_document_gpu.send(job_id, doc_id)

        # Return the job ID immediately
        return BackgroundJobResponse(
            message=f"Document deletion scheduled in the background",
            job_id=job_id,
            job_type="document_deletion",
            status="pending",
        )
    except Exception as e:
        # Log the full traceback
        error_detail = f"Error scheduling document deletion: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_detail)

        # Return detailed error information
        raise HTTPException(
            status_code=500,
            detail=error_detail,
        )


@router.get("/status", response_model=Dict[str, Any])
async def get_ingest_status(
        vector_store: QdrantStore = Depends(get_vector_store),
        redis: redis.Redis = Depends(get_redis_client),
        job_tracker: JobTracker = Depends(get_job_tracker),
) -> Dict[str, Any]:
    # Get basic collection info
    stats = vector_store.get_stats()

    # Get job stats
    job_stats = {
        "pending_jobs": len([j for j in job_tracker.get_all_jobs(limit=1000) if j.get("status") == "pending"]),
        "processing_jobs": len([j for j in job_tracker.get_all_jobs(limit=1000) if j.get("status") == "processing"]),
        "completed_jobs": len([j for j in job_tracker.get_all_jobs(limit=1000) if j.get("status") == "completed"]),
        "failed_jobs": len([j for j in job_tracker.get_all_jobs(limit=1000) if j.get("status") == "failed" or j.get("status") == "timeout"])
    }

    # Get worker status using the centralized function
    worker_status = get_worker_status_for_ui(redis)
    active_workers = worker_status.get("active_workers", {})

    return {
        "status": "healthy",
        "mode": "api",
        "collection": stats.get("name"),
        "document_count": stats.get("vectors_count", 0),
        "collection_size": stats.get("disk_data_size", 0),
        "job_stats": job_stats,
        "active_workers": active_workers
    }


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