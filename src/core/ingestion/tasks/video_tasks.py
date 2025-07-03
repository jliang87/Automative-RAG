"""
Video ingestion tasks - Extracted from JobChain
Handles video download and transcription workflows
"""

import time
import logging
from typing import Dict, Optional
import dramatiq

from core.orchestration.job_tracker import job_tracker
from core.orchestration.job_chain import job_chain

logger = logging.getLogger(__name__)


@dramatiq.actor(queue_name="cpu_tasks", store_results=True, max_retries=2)
def download_video_task(job_id: str, url: str, metadata: Optional[Dict] = None):
    """Download video - Unicode cleaning happens automatically!"""
    try:
        logger.info(f"Downloading video for job {job_id}: {url}")

        from core.ingestion.loaders.video_transcriber import VideoTranscriber

        transcriber = VideoTranscriber()

        # Extract audio
        media_path = transcriber.extract_audio(url)

        # Get video metadata
        try:
            video_metadata = transcriber.get_video_metadata(url)
            logger.info(f"Successfully retrieved video metadata for job {job_id}")
            logger.info(f"Title: {video_metadata.get('title', 'NO_TITLE')}")
            logger.info(f"Author: {video_metadata.get('author', 'NO_AUTHOR')}")

        except Exception as e:
            error_msg = f"Failed to extract video metadata: {str(e)}"
            logger.error(error_msg)
            job_chain.task_failed(job_id, error_msg)
            return

        # Validate metadata completeness
        if not video_metadata.get("title") or video_metadata.get("title") == "Unknown Video":
            error_msg = f"Extracted metadata is incomplete or invalid for {url}"
            logger.error(error_msg)
            job_chain.task_failed(job_id, error_msg)
            return

        logger.info(f"Video download completed for job {job_id}: {video_metadata['title']}")

        download_result = {
            "media_path": media_path,
            "video_metadata": video_metadata,
            "download_completed_at": time.time(),
            "url": url,
            "custom_metadata": metadata or {}
        }

        # Store the download result in job tracker
        job_tracker.update_job_status(
            job_id,
            "processing",
            result=download_result,
            stage="download_completed"
        )

        # Trigger the next task
        job_chain.task_completed(job_id, download_result)

    except Exception as e:
        error_msg = f"Video download failed for job {job_id}: {str(e)}"
        logger.error(error_msg)
        job_chain.task_failed(job_id, error_msg)


@dramatiq.actor(queue_name="transcription_tasks", store_results=True, max_retries=2)
def transcribe_video_task(job_id: str, media_path: str):
    """Transcribe video - ALREADY uses EnhancedTranscriptProcessor optimally"""
    try:
        logger.info(f"Transcribing video for job {job_id}: {media_path}")

        from core.background.models import get_whisper_model
        from core.ingestion.loaders.enhanced_transcript_processor import EnhancedTranscriptProcessor
        from src.config.settings import settings
        import json

        # Get the preloaded Whisper model
        whisper_model = get_whisper_model()

        # Perform transcription
        segments, info = whisper_model.transcribe(
            media_path,
            beam_size=5,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500)
        )

        # Collect all segments
        all_text = [segment.text for segment in segments]
        transcript = " ".join(all_text)

        if not transcript.strip():
            error_msg = f"Transcription failed - no text extracted from {media_path}"
            logger.error(error_msg)
            job_chain.task_failed(job_id, error_msg)
            return

        # Apply Chinese conversion if needed
        if info.language == "zh":
            try:
                import opencc
                converter = opencc.OpenCC('t2s')
                transcript = converter.convert(transcript)
                logger.info(f"Applied Chinese character conversion for job {job_id}")
            except ImportError:
                logger.warning("opencc not found. Chinese conversion skipped.")

        # Get existing job data and validate video_metadata exists
        current_job = job_tracker.get_job(job_id, include_progress=False)
        if not current_job:
            error_msg = f"Job {job_id} not found in tracker"
            logger.error(error_msg)
            job_chain.task_failed(job_id, error_msg)
            return

        existing_result = current_job.get("result", {})
        if isinstance(existing_result, str):
            try:
                existing_result = json.loads(existing_result)
            except:
                existing_result = {}

        # Get and validate video_metadata
        video_metadata = existing_result.get("video_metadata", {})
        if not video_metadata or not isinstance(video_metadata, dict):
            error_msg = f"video_metadata missing from previous step for job {job_id}"
            logger.error(error_msg)
            job_chain.task_failed(job_id, error_msg)
            return

        # Validate essential fields exist
        title = video_metadata.get("title", "")
        author = video_metadata.get("uploader", "")

        if not title or title in ["Unknown Video", ""]:
            error_msg = f"Title is empty or invalid for job {job_id}"
            logger.error(error_msg)
            job_chain.task_failed(job_id, error_msg)
            return

        logger.info(f"Validated metadata for job {job_id}")
        logger.info(f"  Title: {title}")
        logger.info(f"  Author: {author}")

        # ✅ USE ENHANCED TRANSCRIPT PROCESSOR
        logger.info(f"🔧 Using enhanced transcript processor for job {job_id}")

        processor = EnhancedTranscriptProcessor()

        # Process transcript with enhanced metadata injection
        enhanced_documents = processor.process_transcript_chunks(
            transcript=transcript,
            video_metadata=video_metadata,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap
        )

        logger.info(f"✅ Enhanced processing completed for job {job_id}: {len(enhanced_documents)} documents")

        # ✅ VERIFY METADATA INJECTION WORKED
        if enhanced_documents:
            sample_doc = enhanced_documents[0]
            sample_content = sample_doc.page_content

            # Check for embedded metadata patterns
            import re
            embedded_patterns = re.findall(r'【[^】]+】', sample_content)

            logger.info(f"🏷️ Metadata injection verification for job {job_id}:")
            logger.info(f"  Embedded patterns found: {len(embedded_patterns)}")
            logger.info(f"  Sample patterns: {embedded_patterns[:3] if embedded_patterns else 'NONE'}")
            logger.info(f"  Vehicle detected: {sample_doc.metadata.get('vehicleDetected', False)}")
            logger.info(f"  Metadata injected: {sample_doc.metadata.get('metadataInjected', False)}")

        # Convert enhanced documents to format for next task
        document_dicts = []
        for doc in enhanced_documents:
            document_dicts.append({
                "content": doc.page_content,  # ✅ NOW CONTAINS EMBEDDED METADATA!
                "metadata": doc.metadata  # ✅ NOW CONTAINS ENHANCED METADATA!
            })

        logger.info(
            f"Transcription completed for job {job_id}: {len(enhanced_documents)} enhanced chunks, language: {info.language}")

        # Create transcription result while preserving ALL existing data
        transcription_result = {}
        transcription_result.update(existing_result)

        # Add new transcription data with enhanced processing
        transcription_result.update({
            "documents": document_dicts,  # ✅ NOW WITH EMBEDDED METADATA
            "transcript": transcript,
            "language": info.language,
            "duration": info.duration,
            "chunk_count": len(enhanced_documents),
            "transcription_completed_at": time.time(),
            "detected_source": "bilibili" if 'bilibili.com' in video_metadata.get('url', '') else 'youtube',
            # ✅ ADD ENHANCED PROCESSING MARKERS
            "enhanced_processing_used": True,
            "metadata_injection_applied": True,
            "processing_method": "enhanced_transcript_processor",
            "unified_ingestion_system": True
        })

        logger.info(f"Transcription completed for job {job_id}")

        # Trigger next task
        job_chain.task_completed(job_id, transcription_result)

    except Exception as e:
        error_msg = f"Video transcription failed for job {job_id}: {str(e)}"
        logger.error(error_msg)
        job_chain.task_failed(job_id, error_msg)


def start_video_processing(job_id: str, data: Dict):
    """
    Start video processing workflow

    Args:
        job_id: Job identifier
        data: Job data containing url and metadata
    """
    logger.info(f"Starting video processing workflow for job {job_id}")

    # Validate required data
    if "url" not in data:
        error_msg = "URL required for video processing"
        logger.error(error_msg)
        job_chain.task_failed(job_id, error_msg)
        return

    # Start the video download task
    download_video_task.send(job_id, data["url"], data.get("metadata"))