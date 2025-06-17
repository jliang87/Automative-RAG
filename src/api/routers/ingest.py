import os
import uuid
from typing import Dict, List, Optional, Union, Any

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, Query
from pydantic import HttpUrl, BaseModel
import logging

from src.core.background.job_chain import job_chain, JobType
from src.core.background.job_tracker import job_tracker
from src.models.schema import IngestResponse, ManualIngestRequest, BackgroundJobResponse

logger = logging.getLogger(__name__)

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
) -> BackgroundJobResponse:
    """
    Ingest a video from any supported platform with GPU-accelerated Whisper transcription.
    All processing happens asynchronously using the job chain system.
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

        # Start the job chain - this executes immediately
        job_chain.start_job_chain(
            job_id=job_id,
            job_type=JobType.VIDEO_PROCESSING,
            initial_data={
                "url": url_str,
                "metadata": video_request.metadata
            }
        )

        # Return the job ID immediately
        return BackgroundJobResponse(
            message=f"Processing {platform} video in the background",
            job_id=job_id,
            job_type="video_processing",
            status="processing",
        )
    except Exception as e:
        logger.error(f"Error starting video ingestion: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error ingesting video: {str(e)}",
        )


@router.post("/batch-videos", response_model=BackgroundJobResponse)
async def ingest_batch_videos(
        batch_request: BatchVideoIngestRequest,
) -> BackgroundJobResponse:
    """
    Ingest multiple videos in batch.
    Each video gets its own job chain for parallel processing.
    """
    try:
        urls = [str(url) for url in batch_request.urls]
        batch_job_id = str(uuid.uuid4())

        # Create a batch job record
        job_tracker.create_job(
            job_id=batch_job_id,
            job_type="batch_video_processing",
            metadata={
                "urls": urls,
                "video_count": len(urls),
                "custom_metadata": batch_request.metadata
            }
        )

        # Start individual job chains for each video
        individual_jobs = []
        for i, url in enumerate(urls):
            video_job_id = f"{batch_job_id}_video_{i}"
            metadata = batch_request.metadata[i] if batch_request.metadata and i < len(batch_request.metadata) else None

            # Create individual job
            job_tracker.create_job(
                job_id=video_job_id,
                job_type="video_processing",
                metadata={
                    "url": url,
                    "batch_job_id": batch_job_id,
                    "video_index": i,
                    "custom_metadata": metadata
                }
            )

            # Start job chain
            job_chain.start_job_chain(
                job_id=video_job_id,
                job_type=JobType.VIDEO_PROCESSING,
                initial_data={
                    "url": url,
                    "metadata": metadata
                }
            )

            individual_jobs.append(video_job_id)

        # Update batch job with individual job IDs
        job_tracker.update_job_status(
            batch_job_id,
            "processing",
            result={
                "message": f"Started processing {len(urls)} videos",
                "individual_jobs": individual_jobs
            }
        )

        return BackgroundJobResponse(
            message=f"Processing {len(urls)} videos in the background",
            job_id=batch_job_id,
            job_type="batch_video_processing",
            status="processing",
        )
    except Exception as e:
        logger.error(f"Error in batch video processing: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error in batch processing: {str(e)}",
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
    All processing happens asynchronously using the job chain system.
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

        # Start the job chain
        job_chain.start_job_chain(
            job_id=job_id,
            job_type=JobType.PDF_PROCESSING,
            initial_data={
                "file_path": file_path,
                "metadata": custom_metadata
            }
        )

        return BackgroundJobResponse(
            message=f"Processing PDF in the background: {file.filename}",
            job_id=job_id,
            job_type="pdf_processing",
            status="processing",
        )
    except Exception as e:
        logger.error(f"Error starting PDF processing: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error ingesting PDF: {str(e)}",
        )


@router.post("/text", response_model=BackgroundJobResponse)
async def ingest_text(manual_request: ManualIngestRequest) -> BackgroundJobResponse:
    """
    Ingest manually entered text.
    All processing happens asynchronously using the job chain system.
    """
    try:
        # Generate a unique job ID
        job_id = str(uuid.uuid4())

        # Create a job record
        job_tracker.create_job(
            job_id=job_id,
            job_type="text_processing",
            metadata={
                "title": manual_request.metadata.title if manual_request.metadata.title else "Manual Text Input",
                "content_length": len(manual_request.content)
            }
        )

        # Start the job chain
        job_chain.start_job_chain(
            job_id=job_id,
            job_type=JobType.TEXT_PROCESSING,
            initial_data={
                "text": manual_request.content,
                "metadata": manual_request.metadata.dict()
            }
        )

        return BackgroundJobResponse(
            message="Processing text input in the background",
            job_id=job_id,
            job_type="text_processing",
            status="processing",
        )
    except Exception as e:
        logger.error(f"Error starting text processing: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error ingesting text: {str(e)}",
        )


@router.get("/jobs/{job_id}", response_model=Dict[str, Any])
async def get_job_status(job_id: str) -> Dict[str, Any]:
    """
    Get the status of a background job, including job chain information.
    """
    # Get basic job data
    job_data = job_tracker.get_job(job_id)
    if not job_data:
        raise HTTPException(
            status_code=404,
            detail=f"Job with ID {job_id} not found"
        )

    # Get job chain status if available
    chain_status = job_chain.get_job_chain_status(job_id)
    if chain_status:
        job_data["job_chain"] = chain_status

    return job_data


@router.get("/jobs/{job_id}/chain", response_model=Dict[str, Any])
async def get_job_chain_status(job_id: str) -> Dict[str, Any]:
    """
    Get detailed job chain status for a specific job.
    """
    chain_status = job_chain.get_job_chain_status(job_id)
    if not chain_status:
        raise HTTPException(
            status_code=404,
            detail=f"Job chain for ID {job_id} not found"
        )

    return chain_status


@router.get("/jobs", response_model=List[Dict[str, Any]])
async def get_all_jobs(
        limit: int = Query(50, ge=1, le=100),
        job_type: Optional[str] = Query(None)
) -> List[Dict[str, Any]]:
    """
    Get all background jobs, optionally filtered by type.
    """
    return job_tracker.get_all_jobs(limit=limit, job_type=job_type)


@router.delete("/jobs/{job_id}", response_model=Dict[str, str])
async def delete_job(job_id: str) -> Dict[str, str]:
    """
    Delete a job by ID.
    """
    deleted = job_tracker.delete_job(job_id)

    if not deleted:
        raise HTTPException(
            status_code=404,
            detail=f"Job with ID {job_id} not found"
        )

    return {"message": f"Successfully deleted job: {job_id}"}


@router.get("/status", response_model=Dict[str, Any])
async def get_ingest_status() -> Dict[str, Any]:
    """
    Get ingestion system status including job chain queue information.
    """
    try:
        # Get job stats
        job_stats = job_tracker.count_jobs_by_status()

        # Get queue status from job chain
        queue_status = job_chain.get_queue_status()

        return {
            "status": "healthy",
            "mode": "job_chain",
            "job_stats": job_stats,
            "queue_status": queue_status,
            "total_jobs": job_stats.get("total", 0)
        }
    except Exception as e:
        logger.error(f"Error getting ingest status: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error getting status: {str(e)}",
        )


@router.post("/reset", response_model=Dict[str, str])
async def reset_vector_store() -> Dict[str, str]:
    """
    Reset the vector store (dangerous operation).
    """
    try:
        # Import here to avoid circular imports
        from src.core.background.models import get_vector_store

        vector_store = get_vector_store()

        # Delete the collection
        vector_store.client.delete_collection(vector_store.collection_name)

        # Re-create the collection
        vector_store._ensure_collection()

        return {"message": f"Successfully reset vector store: {vector_store.collection_name}"}
    except Exception as e:
        logger.error(f"Error resetting vector store: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error resetting vector store: {str(e)}",
        )