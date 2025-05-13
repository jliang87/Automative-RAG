"""
Dramatiq actors for ingestion tasks.

This module defines actors for handling ingestion tasks like processing videos,
PDFs, and manual text entries. These tasks are executed by CPU and GPU workers.
"""

import os
import json
import time
import logging
from typing import Dict, List, Optional, Union, Any

import dramatiq
import torch

from ..common import JobStatus
from ..job_tracker import job_tracker
from ..priority_queue import priority_queue
from ..models import get_vector_store, get_whisper_model

logger = logging.getLogger(__name__)


# Specialized actor for generating embeddings with priority handling
@dramatiq.actor(
    queue_name="embedding_tasks",
    max_retries=3,
    time_limit=600000,  # 10 minutes
    min_backoff=10000,
    max_backoff=300000,
    store_results=True
)
def generate_embeddings_gpu(job_id: str, chunks: List[Dict], metadata: Optional[Dict] = None):
    """Generate embeddings with priority handling."""
    try:
        # Register task with the priority system
        task_id = f"embedding_{job_id}"
        priority_queue.register_task("embedding_tasks", task_id, {"job_id": job_id})

        # Update job status to show we're queued
        job_tracker.update_job_status(
            job_id,
            JobStatus.PROCESSING,
            result={"message": "In priority queue for GPU resources"},
            stage="embedding_tasks"
        )

        # Wait for priority system to allow this task to run
        wait_start = time.time()
        while not priority_queue.can_run_task("embedding_tasks", task_id):
            # Log every 30 seconds of waiting
            if int(time.time() - wait_start) % 30 == 0:
                logger.info(f"Embedding task {task_id} waiting in priority queue")
            time.sleep(1)

        # Mark this task as now active on GPU
        priority_queue.mark_task_active({
            "task_id": task_id,
            "queue_name": "embedding_tasks",
            "priority": 3,
            "job_id": job_id,
            "registered_at": time.time()
        })

        logger.info(f"Starting embedding generation for job {job_id} with priority handling")

        try:
            # Update job status to processing
            job_tracker.update_job_status(
                job_id,
                JobStatus.PROCESSING,
                result={"message": "Generating embeddings"},
                stage="embedding"
            )

            # Convert chunk dictionaries to Document objects
            from langchain_core.documents import Document
            documents = []
            for chunk in chunks:
                doc = Document(
                    page_content=chunk["content"],
                    metadata=chunk.get("metadata", {})
                )
                documents.append(doc)

            # Get vector store with preloaded embeddings
            vector_store = get_vector_store()

            # Generate embeddings and add to vector store
            start_time = time.time()
            doc_ids = vector_store.add_documents(documents)
            embedding_time = time.time() - start_time

            logger.info(f"Generated embeddings for {len(documents)} documents in {embedding_time:.2f}s")

            # Update job status upon completion
            job_tracker.update_job_status(
                job_id,
                JobStatus.COMPLETED,
                result={
                    "message": f"Generated embeddings for {len(documents)} documents",
                    "document_count": len(documents),
                    "document_ids": doc_ids,
                    "execution_time": embedding_time
                }
            )

            return {
                "document_count": len(documents),
                "document_ids": doc_ids,
                "execution_time": embedding_time
            }
        finally:
            # Always mark task as completed, even if it failed
            priority_queue.mark_task_completed(task_id)

            # Clear cache if needed
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    except Exception as e:
        import traceback
        error_detail = f"Error generating embeddings: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_detail)

        # Update job with error
        job_tracker.update_job_status(
            job_id,
            JobStatus.FAILED,
            error=error_detail
        )

        # Clean up GPU memory on error
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Make sure to mark task as completed even in error case
        priority_queue.mark_task_completed(f"embedding_{job_id}")

        # Re-raise for dramatiq retry mechanism
        raise


# Specialized actor for Whisper transcription with priority handling
@dramatiq.actor(
    queue_name="transcription_tasks",
    max_retries=3,
    time_limit=3600000,  # 1 hour for long videos
    min_backoff=10000,
    max_backoff=300000,
    store_results=True
)
def transcribe_video_gpu(job_id: str, media_path: str):
    """Process a video using GPU-accelerated Whisper transcription with priority handling."""
    try:
        # Register task with the priority system
        task_id = f"transcription_{job_id}"
        priority_queue.register_task("transcription_tasks", task_id, {"job_id": job_id, "media_path": media_path})

        # Update job status to show we're queued
        job_tracker.update_job_status(
            job_id,
            JobStatus.PROCESSING,
            result={"message": "In priority queue for GPU transcription resources"},
            stage="transcription_tasks"
        )

        # Wait for priority system to allow this task to run
        wait_start = time.time()
        while not priority_queue.can_run_task("transcription_tasks", task_id):
            # Log every 60 seconds of waiting
            if int(time.time() - wait_start) % 60 == 0:
                logger.info(f"Transcription task {task_id} waiting in priority queue")
            time.sleep(1)

        # Mark this task as now active on GPU
        priority_queue.mark_task_active({
            "task_id": task_id,
            "queue_name": "transcription_tasks",
            "priority": 4,
            "job_id": job_id,
            "registered_at": time.time()
        })

        logger.info(f"Starting transcription for job {job_id} with priority handling")

        try:
            # Update job status to processing
            job_tracker.update_job_status(
                job_id,
                JobStatus.PROCESSING,
                result={"message": "Transcribing video"},
                stage="transcription"
            )

            # Get the Whisper model
            whisper_model = get_whisper_model()

            # Perform transcription
            start_time = time.time()
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

            transcription_time = time.time() - start_time

            logger.info(f"Transcription completed in {transcription_time:.2f}s, detected language: {info.language}")

            # Update job with success result
            job_tracker.update_job_status(
                job_id,
                JobStatus.COMPLETED,
                result={
                    "transcript": transcript,
                    "language": info.language,
                    "duration": info.duration,
                    "processing_time": transcription_time
                }
            )

            # Chain to process_transcript actor for embedding
            process_transcript.send(job_id, transcript, info.language)

            return transcript

        finally:
            # Always mark task as completed, even if it failed
            priority_queue.mark_task_completed(task_id)

            # Clear cache if needed
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    except Exception as e:
        import traceback
        error_detail = f"Error performing transcription: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_detail)

        # Update job with error
        job_tracker.update_job_status(
            job_id,
            JobStatus.FAILED,
            error=error_detail
        )

        # Clean up GPU memory on error
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Make sure to mark task as completed even in error case
        priority_queue.mark_task_completed(f"transcription_{task_id}")

        # Re-raise for dramatiq retry mechanism
        raise


# Actor for processing text chunks (CPU task)
@dramatiq.actor(
    queue_name="cpu_tasks",
    store_results=True
)
def process_text(job_id: str, text: str, metadata: Dict[str, Any] = None):
    """Process text and prepare for embedding."""
    try:
        logger.info(f"Processing text for job {job_id}")

        # Update job status
        job_tracker.update_job_status(
            job_id,
            JobStatus.PROCESSING,
            result={"message": "Processing text"},
            stage="text_processing"
        )

        # Split text into chunks
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        from src.config.settings import settings

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )

        chunks = text_splitter.split_text(text)
        logger.info(f"Split text into {len(chunks)} chunks")

        # Convert chunks to documents with metadata
        from langchain_core.documents import Document
        documents = []
        for i, chunk_text in enumerate(chunks):
            # Create a document with metadata
            doc = Document(
                page_content=chunk_text,
                metadata={
                    "chunk_id": i,
                    "source": "manual",
                    "source_id": job_id,
                    **metadata
                }
            )
            documents.append(doc)

        # Convert documents to format for embedding
        document_dicts = []
        for doc in documents:
            document_dicts.append({
                "content": doc.page_content,
                "metadata": doc.metadata
            })

        # Chain to embedding task
        embedding_job_id = f"{job_id}_embed"
        job_tracker.create_job(
            job_id=embedding_job_id,
            job_type="embedding",
            metadata={
                "parent_job_id": job_id,
                "chunk_count": len(document_dicts)
            }
        )

        # Send to embedding worker
        generate_embeddings_gpu.send(embedding_job_id, document_dicts, metadata)

        # Update original job status
        job_tracker.update_job_status(
            job_id,
            JobStatus.PROCESSING,
            result={
                "message": "Text processed, embedding in progress",
                "chunk_count": len(chunks),
                "embedding_job_id": embedding_job_id
            }
        )

        return {
            "chunk_count": len(chunks),
            "embedding_job_id": embedding_job_id
        }
    except Exception as e:
        import traceback
        error_detail = f"Error processing text: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_detail)

        # Update job with error
        job_tracker.update_job_status(
            job_id,
            JobStatus.FAILED,
            error=error_detail
        )

        # Re-raise for dramatiq retry mechanism
        raise


# Actor for processing transcript and adding to vector store
@dramatiq.actor(
    queue_name="cpu_tasks",
    store_results=True
)
def process_transcript(job_id: str, transcript: str, language: str):
    """Process transcript and prepare for embedding."""
    try:
        logger.info(f"Processing transcript for job {job_id}")

        # Update job status
        job_tracker.update_job_status(
            job_id,
            JobStatus.PROCESSING,
            result={"message": "Processing transcript"},
            stage="transcript_processing"
        )

        # Get job metadata
        job_data = job_tracker.get_job(job_id)
        if not job_data:
            raise ValueError(f"Job {job_id} not found")

        metadata = job_data.get("metadata", {})

        # Split transcript into chunks
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        from src.config.settings import settings

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )

        chunks = text_splitter.split_text(transcript)
        logger.info(f"Split transcript into {len(chunks)} chunks")

        # Convert chunks to documents with metadata
        from langchain_core.documents import Document
        documents = []
        for i, chunk_text in enumerate(chunks):
            # Create document metadata
            doc_metadata = {
                "chunk_id": i,
                "language": language,
                "source": "video",
                "source_id": job_id,
            }

            # Add any metadata from the job
            if "url" in metadata:
                doc_metadata["url"] = metadata["url"]
            if "platform" in metadata:
                doc_metadata["platform"] = metadata["platform"]

            # Create document
            doc = Document(
                page_content=chunk_text,
                metadata=doc_metadata
            )
            documents.append(doc)

        # Convert documents to format for embedding
        document_dicts = []
        for doc in documents:
            document_dicts.append({
                "content": doc.page_content,
                "metadata": doc.metadata
            })

        # Chain to embedding task
        embedding_job_id = f"{job_id}_embed"
        job_tracker.create_job(
            job_id=embedding_job_id,
            job_type="embedding",
            metadata={
                "parent_job_id": job_id,
                "chunk_count": len(document_dicts)
            }
        )

        # Send to embedding worker
        generate_embeddings_gpu.send(embedding_job_id, document_dicts, metadata)

        # Update original job status
        job_tracker.update_job_status(
            job_id,
            JobStatus.PROCESSING,
            result={
                "message": "Transcript processed, embedding in progress",
                "chunk_count": len(chunks),
                "embedding_job_id": embedding_job_id
            }
        )

        return {
            "chunk_count": len(chunks),
            "embedding_job_id": embedding_job_id
        }
    except Exception as e:
        import traceback
        error_detail = f"Error processing transcript: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_detail)

        # Update job with error
        job_tracker.update_job_status(
            job_id,
            JobStatus.FAILED,
            error=error_detail
        )

        # Re-raise for dramatiq retry mechanism
        raise


# Actor for processing PDFs with OCR on CPU
@dramatiq.actor(
    queue_name="cpu_tasks",
    store_results=True
)
def process_pdf_cpu(job_id: str, file_path: str, custom_metadata: Optional[Dict[str, Any]] = None):
    """Process a PDF file using CPU for OCR and text extraction."""
    try:
        logger.info(f"Processing PDF {file_path} for job {job_id}")

        # Update job status
        job_tracker.update_job_status(
            job_id,
            JobStatus.PROCESSING,
            result={"message": "Processing PDF"},
            stage="pdf_processing"
        )

        # Import PDF loader
        from src.core.pdf_loader import PDFLoader
        from src.config.settings import settings

        # Create PDF loader (on CPU only)
        pdf_loader = PDFLoader(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            device="cpu",
            use_ocr=settings.use_pdf_ocr,
            ocr_languages=settings.ocr_languages
        )

        # Process PDF
        start_time = time.time()
        documents = pdf_loader.process_pdf(
            file_path=file_path,
            custom_metadata=custom_metadata,
        )
        processing_time = time.time() - start_time

        logger.info(f"Processed PDF into {len(documents)} documents in {processing_time:.2f}s")

        # Convert documents to format for embedding
        document_dicts = []
        for doc in documents:
            document_dicts.append({
                "content": doc.page_content,
                "metadata": doc.metadata
            })

        # Chain to embedding task
        embedding_job_id = f"{job_id}_embed"
        job_tracker.create_job(
            job_id=embedding_job_id,
            job_type="embedding",
            metadata={
                "parent_job_id": job_id,
                "file_path": file_path,
                "chunk_count": len(document_dicts)
            }
        )

        # Send to embedding worker
        generate_embeddings_gpu.send(embedding_job_id, document_dicts, custom_metadata)

        # Update original job status
        job_tracker.update_job_status(
            job_id,
            JobStatus.PROCESSING,
            result={
                "message": "PDF processed, embedding in progress",
                "chunk_count": len(documents),
                "processing_time": processing_time,
                "embedding_job_id": embedding_job_id
            }
        )

        return {
            "chunk_count": len(documents),
            "processing_time": processing_time,
            "embedding_job_id": embedding_job_id
        }
    except Exception as e:
        import traceback
        error_detail = f"Error processing PDF: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_detail)

        # Update job with error
        job_tracker.update_job_status(
            job_id,
            JobStatus.FAILED,
            error=error_detail
        )

        # Re-raise for dramatiq retry mechanism
        raise


# Function for processing a batch of videos
def batch_process_videos(job_id: str, urls: List[str], custom_metadata: Optional[List[Dict[str, str]]] = None):
    """Process multiple videos in batch."""
    logger.info(f"Starting batch video processing for job {job_id} with {len(urls)} videos")

    # Initialize result data
    results = {}

    # Update job status
    job_tracker.update_job_status(
        job_id,
        JobStatus.PROCESSING,
        result={
            "message": f"Processing {len(urls)} videos",
            "processed": 0,
            "total": len(urls)
        }
    )

    # Process each video
    for i, url in enumerate(urls):
        # Get metadata for this video
        video_metadata = None
        if custom_metadata and i < len(custom_metadata):
            video_metadata = custom_metadata[i]

        # Create child job for this video
        video_job_id = f"{job_id}_video_{i}"
        job_tracker.create_job(
            job_id=video_job_id,
            job_type="video_processing",
            metadata={
                "url": url,
                "parent_job_id": job_id,
                "custom_metadata": video_metadata
            }
        )

        # Start video processing
        try:
            # Call the video processing task
            from .ingestion import process_video_gpu
            process_video_gpu.send(video_job_id, url, video_metadata)

            # Record in results
            results[url] = {"status": "submitted", "job_id": video_job_id}

            # Update batch job
            job_tracker.update_job_status(
                job_id,
                JobStatus.PROCESSING,
                result={
                    "message": f"Processing {len(urls)} videos",
                    "processed": i + 1,
                    "total": len(urls),
                    "results": results
                }
            )

            logger.info(f"Started processing video {i + 1}/{len(urls)}: {url}")
        except Exception as e:
            logger.error(f"Error scheduling video {i + 1}/{len(urls)}: {url} - {str(e)}")
            results[url] = {"status": "error", "message": str(e)}

    # Update final status
    job_tracker.update_job_status(
        job_id,
        JobStatus.COMPLETED,
        result={
            "message": f"Batch processing completed. Started {len(urls)} video processing jobs.",
            "processed": len(urls),
            "total": len(urls),
            "results": results
        }
    )

    return results


# Actor for GPU video processing
@dramatiq.actor(
    queue_name="embedding_tasks",
    max_retries=3,
    time_limit=3600000,  # 1 hour
    min_backoff=10000,
    max_backoff=300000,
    store_results=True
)
def process_video_gpu(job_id: str, url: str, custom_metadata: Optional[Dict[str, str]] = None):
    """Process a video with GPU acceleration."""
    try:
        logger.info(f"Processing video {url} for job {job_id}")

        # Update job status
        job_tracker.update_job_status(
            job_id,
            JobStatus.PROCESSING,
            result={"message": "Downloading video"},
            stage="download"
        )

        # Import the video transcriber
        from src.core.video_transcriber import VideoTranscriber
        from src.config.settings import settings

        # Create video transcriber
        transcriber = VideoTranscriber(
            whisper_model_size=settings.whisper_model_size,
            device="cpu"  # Start with CPU for download
        )

        # Extract platform and video ID
        platform = transcriber.detect_platform(url)
        video_id = transcriber.extract_video_id(url)

        # Get video metadata
        video_metadata = transcriber.get_video_metadata(url)

        # Download the video/audio
        media_path = transcriber.extract_audio(url)

        # Update job status
        job_tracker.update_job_status(
            job_id,
            JobStatus.PROCESSING,
            result={
                "message": "Video downloaded, waiting for transcription",
                "platform": platform,
                "video_id": video_id,
                "media_path": media_path
            },
            stage="download_complete"
        )

        # Create transcription job
        transcription_job_id = f"{job_id}_transcribe"
        job_tracker.create_job(
            job_id=transcription_job_id,
            job_type="transcription",
            metadata={
                "url": url,
                "parent_job_id": job_id,
                "platform": platform,
                "video_id": video_id,
                "custom_metadata": custom_metadata
            }
        )

        # Send to GPU transcription
        transcribe_video_gpu.send(transcription_job_id, media_path)

        # Update job status
        job_tracker.update_job_status(
            job_id,
            JobStatus.PROCESSING,
            result={
                "message": "Video processing in progress",
                "transcription_job_id": transcription_job_id
            }
        )

        return {
            "status": "processing",
            "transcription_job_id": transcription_job_id,
            "platform": platform,
            "video_id": video_id
        }

    except Exception as e:
        import traceback
        error_detail = f"Error processing video: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_detail)

        # Update job with error
        job_tracker.update_job_status(
            job_id,
            JobStatus.FAILED,
            error=error_detail
        )

        # Re-raise for dramatiq retry mechanism
        raise


@dramatiq.actor(
    queue_name="embedding_tasks",  # Use embedding_tasks queue for GPU access
    max_retries=2,
    store_results=True
)
def delete_document_gpu(job_id: str, doc_id: str):
    """
    Delete a document from the vector store with access to the GPU embedding model.

    Using the embedding_tasks queue ensures this task runs on a worker with GPU access.
    """
    try:
        # Register task with the priority system
        task_id = f"delete_{job_id}"
        priority_queue.register_task("embedding_tasks", task_id, {"job_id": job_id, "doc_id": doc_id})

        # Update job status to show we're queued
        job_tracker.update_job_status(
            job_id,
            JobStatus.PROCESSING,
            result={"message": "In priority queue for GPU resources"},
            stage="embedding_tasks"
        )

        # Wait for priority system to allow this task to run
        wait_start = time.time()
        while not priority_queue.can_run_task("embedding_tasks", task_id):
            # Log every 30 seconds of waiting
            if int(time.time() - wait_start) % 30 == 0:
                logger.info(f"Document deletion task {task_id} waiting in priority queue")
            time.sleep(1)

        # Mark this task as now active on GPU
        priority_queue.mark_task_active({
            "task_id": task_id,
            "queue_name": "embedding_tasks",
            "priority": 3,  # Same priority as other embedding tasks
            "job_id": job_id,
            "registered_at": time.time()
        })

        try:
            # Update job status to processing
            job_tracker.update_job_status(
                job_id,
                JobStatus.PROCESSING,
                result={"message": "Deleting document"},
                stage="deletion"
            )

            # Get vector store with GPU access
            vector_store = get_vector_store()

            # Delete the document
            start_time = time.time()
            vector_store.delete_by_ids([doc_id])
            deletion_time = time.time() - start_time

            logger.info(f"Deleted document {doc_id} in {deletion_time:.2f}s")

            # Update job status upon completion
            job_tracker.update_job_status(
                job_id,
                JobStatus.COMPLETED,
                result={
                    "message": f"Successfully deleted document: {doc_id}",
                    "execution_time": deletion_time
                }
            )

            return {
                "document_id": doc_id,
                "execution_time": deletion_time
            }
        finally:
            # Always mark task as completed, even if it failed
            priority_queue.mark_task_completed(task_id)

            # Clear cache if needed
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    except Exception as e:
        import traceback
        error_detail = f"Error deleting document: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_detail)

        # Update job with error
        job_tracker.update_job_status(
            job_id,
            JobStatus.FAILED,
            error=error_detail
        )

        # Clean up GPU memory on error
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Make sure to mark task as completed even in error case
        priority_queue.mark_task_completed(f"delete_{job_id}")

        # Re-raise for dramatiq retry mechanism
        raise