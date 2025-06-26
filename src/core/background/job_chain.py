import json
import time
import logging
from typing import List, Tuple, Dict, Optional, Any
from langchain_core.documents import Document
from enum import Enum
import dramatiq
import numpy as np

from .job_tracker import job_tracker, JobStatus
from .common import get_redis_client
from src.core.mode_config import mode_config, QueryMode, estimate_token_count
from src.config.settings import settings

logger = logging.getLogger(__name__)


def minimal_effective_filter(
        results: List[Tuple[Document, float]],
        token_budget: int,
        debug_mode: bool = False
) -> List[Tuple[Document, float]]:
    """
    MINIMAL effective filtering - ChatGPT's recommended approach.

    NO REDUNDANT SIMILARITY FILTERING: Vector store already sorted by similarity!
    FOCUS: Only filter obviously bad content + token management + diversity.

    Args:
        results: List of (document, similarity_score) tuples (already sorted by vector store)
        token_budget: Maximum tokens for context
        debug_mode: Whether to log filtering details

    Returns:
        Clean documents ready for LLM within token budget
    """
    if not results:
        return results

    from src.utils.quality_utils import has_garbled_content

    # STEP 1: Only filter OBVIOUSLY bad content (trust vector store ranking)
    valid_docs = []
    excluded_count = 0

    for doc, similarity_score in results:
        content = doc.page_content.strip()

        # Only filter truly problematic content
        if (len(content) > 30 and  # Not too short
                len(content) < 5000 and  # Not too long
                content and  # Not empty
                not has_garbled_content(content)):  # Not garbled OCR

            valid_docs.append((doc, similarity_score))
        else:
            excluded_count += 1

    if debug_mode:
        logger.info(f"Minimal filtering: {len(results)} -> {len(valid_docs)} docs")
        logger.info(f"Excluded obviously bad content: {excluded_count}")

    # STEP 2: Apply diversity filter (prevent source domination)
    diverse_docs = apply_document_diversity(valid_docs, max_per_source=3)

    # STEP 3: Apply token budget management
    final_docs = apply_token_budget_management(diverse_docs, token_budget)

    if debug_mode:
        logger.info(f"Final result: {len(final_docs)} documents within token budget")

    return final_docs


def apply_document_diversity(
        documents: List[Tuple[Document, float]],
        max_per_source: int = 3
) -> List[Tuple[Document, float]]:
    """
    Apply basic source diversity to prevent domination by one source.

    Args:
        documents: List of (document, similarity_score) tuples
        max_per_source: Maximum documents per source

    Returns:
        Diversified list of documents
    """
    if not documents:
        return documents

    source_counts = {}
    diversified_docs = []

    # Keep order (highest similarity first)
    for doc, score in documents:
        source_id = doc.metadata.get("source_id", "unknown")

        # Check if we've hit the limit for this source
        if source_counts.get(source_id, 0) < max_per_source:
            diversified_docs.append((doc, score))
            source_counts[source_id] = source_counts.get(source_id, 0) + 1

    logger.info(f"Diversity filtering: {len(documents)} -> {len(diversified_docs)} documents")
    logger.info(f"Sources: {len(source_counts)} unique sources")
    return diversified_docs


def apply_token_budget_management(
        documents: List[Tuple[Document, float]],
        token_budget: int
) -> List[Tuple[Document, float]]:
    """
    Apply token budget limits while preserving highest similarity documents.

    Args:
        documents: List of (document, similarity_score) tuples (sorted by similarity)
        token_budget: Maximum tokens allowed

    Returns:
        Documents within token budget
    """
    if not documents:
        return documents

    selected_docs = []
    total_tokens = 0

    for doc, score in documents:
        # Estimate tokens for this document
        content_tokens = estimate_token_count(doc.page_content)

        # Check if adding this doc would exceed budget
        if total_tokens + content_tokens > token_budget and selected_docs:
            break  # Stop if we'd exceed budget (but keep at least one doc)

        # Add the document
        selected_docs.append((doc, score))
        total_tokens += content_tokens

        # Safety limit
        if len(selected_docs) >= 12:  # Reasonable upper limit
            break

    logger.info(f"Token budget: {total_tokens}/{token_budget} tokens used")

    return selected_docs


def apply_token_budget_and_diversity(
        documents: List[Tuple[Document, float]],
        token_budget: int,
        max_per_source: int = 3
) -> List[Tuple[Document, float]]:
    """
    DEPRECATED: Use separate functions for clarity.

    This function is kept for backwards compatibility but internally
    calls the new separated functions.
    """
    # Apply diversity first
    diverse_docs = apply_document_diversity(documents, max_per_source)

    # Then apply token budget
    return apply_token_budget_management(diverse_docs, token_budget)


class JobType(Enum):
    VIDEO_PROCESSING = "video_processing"
    PDF_PROCESSING = "pdf_processing"
    TEXT_PROCESSING = "text_processing"
    LLM_INFERENCE = "llm_inference"


class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class JobChain:
    def __init__(self):
        self.redis = get_redis_client()

        # Define job workflows - each job type has a sequence of tasks
        self.workflows = {
            JobType.VIDEO_PROCESSING: [
                ("download_video", "cpu_tasks"),
                ("transcribe_video", "transcription_tasks"),
                ("generate_embeddings", "embedding_tasks")
            ],
            JobType.PDF_PROCESSING: [
                ("process_pdf", "cpu_tasks"),
                ("generate_embeddings", "embedding_tasks")
            ],
            JobType.TEXT_PROCESSING: [
                ("process_text", "cpu_tasks"),
                ("generate_embeddings", "embedding_tasks")
            ],
            JobType.LLM_INFERENCE: [
                ("retrieve_documents", "embedding_tasks"),
                ("llm_inference", "inference_tasks")
            ]
        }

    def start_job_chain(self, job_id: str, job_type: JobType, initial_data: Dict[str, Any]) -> None:
        """Start a job chain with mode support."""
        workflow = self.workflows.get(job_type)
        if not workflow:
            raise ValueError(f"Unknown job type: {job_type}")

        # Extract query mode from initial data
        query_mode = initial_data.get("query_mode", "facts")

        # Store the job chain state with mode information
        chain_state = {
            "job_id": job_id,
            "job_type": job_type.value,
            "workflow": [(task, queue) for task, queue in workflow],
            "current_step": 0,
            "total_steps": len(workflow),
            "data": initial_data,
            "query_mode": query_mode,
            "started_at": time.time(),
            "status": TaskStatus.RUNNING.value,
            "step_timings": {}
        }

        self._save_chain_state(job_id, chain_state)

        # Update job tracker with mode information
        job_tracker.update_job_status(
            job_id,
            JobStatus.PROCESSING,
            result={
                "message": f"Starting {job_type.value} workflow in '{query_mode}' mode",
                "step": 1,
                "total_steps": len(workflow),
                "query_mode": query_mode
            },
            stage="chain_started"
        )

        # Execute the first task
        self._execute_next_task(job_id)

    def _execute_next_task(self, job_id: str) -> None:
        """Execute the next task in the chain."""
        chain_state = self._get_chain_state(job_id)
        if not chain_state:
            logger.error(f"No chain state found for job {job_id}")
            return

        current_step = chain_state["current_step"]
        workflow = chain_state["workflow"]

        # Check if we've completed all steps
        if current_step >= len(workflow):
            self._complete_job_chain(job_id)
            return

        # Get the current task
        task_name, queue_name = workflow[current_step]

        logger.info(f"Executing step {current_step + 1}/{len(workflow)} for job {job_id}: {task_name}")

        # Update job status
        job_tracker.update_job_status(
            job_id,
            "processing",
            result={
                "message": f"Executing {task_name}",
                "step": current_step + 1,
                "total_steps": len(workflow)
            },
            stage=task_name
        )

        # Update progress
        progress = ((current_step + 0.5) / len(workflow)) * 100
        job_tracker.update_job_progress(job_id, progress, f"Executing {task_name}")

        # Record step start time
        chain_state["step_timings"][task_name] = {"started_at": time.time()}
        self._save_chain_state(job_id, chain_state)

        # Get the complete current data from job tracker
        current_job = job_tracker.get_job(job_id, include_progress=False)
        current_job_result = current_job.get("result", {}) if current_job else {}

        # Parse if string
        if isinstance(current_job_result, str):
            try:
                current_job_result = json.loads(current_job_result)
            except:
                current_job_result = {}

        # Merge chain data with current job data
        complete_data = {}
        complete_data.update(chain_state["data"])
        complete_data.update(current_job_result)

        logger.info(f"Executing {task_name} with data keys: {list(complete_data.keys())}")

        # Check if there's already a running task for this queue type
        if self._is_queue_busy(queue_name):
            logger.info(f"Queue {queue_name} is busy, task {task_name} will wait")
            self._queue_task(job_id, task_name, queue_name, complete_data)
        else:
            # Execute immediately
            self._execute_task_immediately(job_id, task_name, queue_name, complete_data)

    def _execute_task_immediately(self, job_id: str, task_name: str, queue_name: str, data: Dict[str, Any]) -> None:
        """Execute a task immediately with mode support."""
        # Mark queue as busy
        self._mark_queue_busy(queue_name, job_id, task_name)

        # Extract query mode for mode-aware tasks
        query_mode = data.get("query_mode", "facts")

        # Execute the appropriate task based on task_name with mode support
        try:
            if task_name == "download_video":
                download_video_task.send(job_id, data.get("url"), data.get("metadata"))
            elif task_name == "transcribe_video":
                transcribe_video_task.send(job_id, data.get("media_path"))
            elif task_name == "process_pdf":
                process_pdf_task.send(job_id, data.get("file_path"), data.get("metadata"))
            elif task_name == "process_text":
                process_text_task.send(job_id, data.get("text"), data.get("metadata"))
            elif task_name == "generate_embeddings":
                generate_embeddings_task.send(job_id, data.get("documents"))
            elif task_name == "retrieve_documents":
                retrieve_documents_task.send(
                    job_id,
                    data.get("query"),
                    data.get("metadata_filter"),
                    query_mode
                )
            elif task_name == "llm_inference":
                llm_inference_task.send(
                    job_id,
                    data.get("query"),
                    data.get("documents"),
                    query_mode
                )
            else:
                logger.error(f"Unknown task: {task_name}")
                self.task_failed(job_id, f"Unknown task: {task_name}")
        except Exception as e:
            logger.error(f"Error executing task {task_name}: {str(e)}")
            self.task_failed(job_id, f"Error executing task {task_name}: {str(e)}")

    def _queue_task(self, job_id: str, task_name: str, queue_name: str, data: Dict[str, Any]) -> None:
        """Queue a task to wait for the queue to become available."""
        queued_task = {
            "job_id": job_id,
            "task_name": task_name,
            "queue_name": queue_name,
            "data": data,
            "queued_at": time.time()
        }

        # Add to waiting queue
        self.redis.lpush(f"waiting_tasks:{queue_name}", json.dumps(queued_task, ensure_ascii=False))
        logger.info(f"Queued task {task_name} for job {job_id} in {queue_name}")

        # Update job progress to show waiting
        job_tracker.update_job_progress(job_id, None, f"Waiting for {queue_name} to become available")

    def task_completed(self, job_id: str, result: Dict[str, Any]) -> None:
        """Called when a task completes successfully."""
        logger.info(f"Task completed for job {job_id}, triggering next task")

        # Update chain state
        chain_state = self._get_chain_state(job_id)
        if not chain_state:
            logger.error(f"No chain state found for job {job_id}")
            return

        # Record step completion time
        current_step = chain_state["current_step"]
        if current_step < len(chain_state["workflow"]):
            task_name, queue_name = chain_state["workflow"][current_step]

            # Update timing information
            if task_name in chain_state["step_timings"]:
                chain_state["step_timings"][task_name]["completed_at"] = time.time()
                chain_state["step_timings"][task_name]["duration"] = (
                        chain_state["step_timings"][task_name]["completed_at"] -
                        chain_state["step_timings"][task_name]["started_at"]
                )

            # Mark queue as free and process waiting tasks
            self._mark_queue_free(queue_name)
            self._process_waiting_tasks(queue_name)

        # Get current job data from job tracker (source of truth)
        current_job = job_tracker.get_job(job_id, include_progress=False)
        existing_job_result = current_job.get("result", {}) if current_job else {}

        # Parse existing result if it's a string
        if isinstance(existing_job_result, str):
            try:
                existing_job_result = json.loads(existing_job_result)
            except:
                existing_job_result = {}

        # Merge task result with existing job data
        combined_result = {}
        combined_result.update(existing_job_result)
        combined_result.update(result)

        # Update chain state data with the combined result
        chain_state["data"].update(combined_result)

        # Move to next step
        chain_state["current_step"] += 1

        # Update progress
        progress = (chain_state["current_step"] / len(chain_state["workflow"])) * 100
        job_tracker.update_job_progress(job_id, progress,
                                        f"Completed step {chain_state['current_step']}/{len(chain_state['workflow'])}")

        # Update job tracker with the combined result
        job_tracker.update_job_status(
            job_id,
            "processing",
            result=combined_result,
            stage=f"completed_step_{current_step + 1}",
            replace_result=True
        )

        # Save updated chain state
        self._save_chain_state(job_id, chain_state)

        # Execute next task
        self._execute_next_task(job_id)

    def task_failed(self, job_id: str, error: str) -> None:
        """Called when a task fails."""
        logger.error(f"Task failed for job {job_id}: {error}")

        # Get chain state to free up the queue
        chain_state = self._get_chain_state(job_id)
        if chain_state:
            current_step = chain_state["current_step"]
            if current_step < len(chain_state["workflow"]):
                task_name, queue_name = chain_state["workflow"][current_step]

                # Record failure timing
                if task_name in chain_state["step_timings"]:
                    chain_state["step_timings"][task_name]["failed_at"] = time.time()
                    chain_state["step_timings"][task_name]["duration"] = (
                            chain_state["step_timings"][task_name]["failed_at"] -
                            chain_state["step_timings"][task_name]["started_at"]
                    )

                # Free up the queue
                self._mark_queue_free(queue_name)
                self._process_waiting_tasks(queue_name)

        # Update job status
        job_tracker.update_job_status(
            job_id,
            JobStatus.FAILED,
            error=error,
            replace_result=False
        )

        # Update progress to show failure
        job_tracker.update_job_progress(job_id, 0, f"Failed: {error[:50]}...")

        # Clean up chain state
        self._delete_chain_state(job_id)

    def _complete_job_chain(self, job_id: str) -> None:
        """Complete the entire job chain while preserving ALL job results."""
        logger.info(f"Job chain completed for job {job_id}")

        # Get final chain state for timing information
        chain_state = self._get_chain_state(job_id)
        total_duration = time.time() - chain_state["started_at"] if chain_state else 0

        # Get the CURRENT job data, not chain data
        current_job = job_tracker.get_job(job_id, include_progress=False)

        if not current_job:
            logger.error(f"No job data found for completed job {job_id}")
            self._delete_chain_state(job_id)
            return

        existing_result = current_job.get("result", {})

        # Parse existing result if it's a string
        if isinstance(existing_result, str):
            try:
                existing_result = json.loads(existing_result)
            except:
                existing_result = {}

        # Completion info
        completion_info = {
            "job_chain_completion": {
                "message": "Job chain completed successfully",
                "total_duration": total_duration,
                "step_timings": chain_state.get("step_timings", {}) if chain_state else {},
                "completed_at": time.time()
            }
        }

        # Preserve ALL existing result data
        if existing_result and isinstance(existing_result, dict):
            final_result = {}
            final_result.update(existing_result)
            final_result.update(completion_info)

            logger.info(f"Preserving all job data for {job_id} with keys: {list(existing_result.keys())}")
        else:
            final_result = completion_info
            logger.info(f"No existing result to preserve for job {job_id}")

        # Update job status with the final result
        job_tracker.update_job_status(
            job_id,
            "completed",
            result=final_result,
            replace_result=True
        )

        # Update progress to 100%
        job_tracker.update_job_progress(job_id, 100, "Job completed successfully")

        # Clean up chain state
        self._delete_chain_state(job_id)

    def _is_queue_busy(self, queue_name: str) -> bool:
        """Check if a queue is currently busy."""
        return self.redis.exists(f"queue_busy:{queue_name}")

    def _mark_queue_busy(self, queue_name: str, job_id: str, task_name: str) -> None:
        """Mark a queue as busy."""
        busy_info = {
            "job_id": job_id,
            "task_name": task_name,
            "started_at": time.time()
        }
        self.redis.set(f"queue_busy:{queue_name}", json.dumps(busy_info, ensure_ascii=False), ex=3600)
        logger.info(f"Marked queue {queue_name} as busy for job {job_id}")

    def _mark_queue_free(self, queue_name: str) -> None:
        """Mark a queue as free."""
        self.redis.delete(f"queue_busy:{queue_name}")
        logger.info(f"Marked queue {queue_name} as free")

    def _process_waiting_tasks(self, queue_name: str) -> None:
        """Process any tasks waiting for this queue to become free."""
        waiting_task_json = self.redis.rpop(f"waiting_tasks:{queue_name}")
        if waiting_task_json:
            waiting_task = json.loads(waiting_task_json)
            logger.info(f"Processing waiting task for queue {queue_name}: {waiting_task['task_name']}")

            self._execute_task_immediately(
                waiting_task["job_id"],
                waiting_task["task_name"],
                waiting_task["queue_name"],
                waiting_task["data"]
            )

    def get_job_chain_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get the current status of a job chain."""
        chain_state = self._get_chain_state(job_id)
        if not chain_state:
            return None

        current_step = chain_state["current_step"]
        workflow = chain_state["workflow"]

        return {
            "job_id": job_id,
            "job_type": chain_state["job_type"],
            "status": chain_state["status"],
            "current_step": current_step,
            "total_steps": len(workflow),
            "current_task": workflow[current_step][0] if current_step < len(workflow) else None,
            "progress_percentage": (current_step / len(workflow)) * 100,
            "started_at": chain_state["started_at"],
            "step_timings": chain_state.get("step_timings", {}),
            "data_keys": list(chain_state["data"].keys())
        }

    def get_queue_status(self) -> Dict[str, Any]:
        """Get the status of all queues."""
        queue_names = ["cpu_tasks", "transcription_tasks", "embedding_tasks", "inference_tasks"]
        queue_status = {}

        for queue_name in queue_names:
            busy_info_json = self.redis.get(f"queue_busy:{queue_name}")
            waiting_count = self.redis.llen(f"waiting_tasks:{queue_name}")

            if busy_info_json:
                busy_info = json.loads(busy_info_json)
                queue_status[queue_name] = {
                    "status": "busy",
                    "current_job": busy_info["job_id"],
                    "current_task": busy_info["task_name"],
                    "busy_since": busy_info["started_at"],
                    "waiting_tasks": waiting_count
                }
            else:
                queue_status[queue_name] = {
                    "status": "free",
                    "waiting_tasks": waiting_count
                }

        return queue_status

    def _save_chain_state(self, job_id: str, chain_state: Dict[str, Any]) -> None:
        """Save chain state with proper UTF-8 encoding."""
        state_json = json.dumps(chain_state, ensure_ascii=False)
        self.redis.set(f"job_chain:{job_id}", state_json, ex=86400)

    def _get_chain_state(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get chain state from Redis."""
        state_json = self.redis.get(f"job_chain:{job_id}")
        if state_json:
            return json.loads(state_json)
        return None

    def _delete_chain_state(self, job_id: str) -> None:
        """Delete chain state from Redis."""
        self.redis.delete(f"job_chain:{job_id}")


# Global job chain instance
job_chain = JobChain()


# ==============================================================================
# SIMPLIFIED TASK DEFINITIONS - STREAMLINED FOR EFFECTIVENESS
# ==============================================================================

@dramatiq.actor(queue_name="cpu_tasks", store_results=True, max_retries=2)
def download_video_task(job_id: str, url: str, metadata: Optional[Dict] = None):
    """Download video - Unicode cleaning happens automatically!"""
    try:
        logger.info(f"Downloading video for job {job_id}: {url}")

        from src.core.video_transcriber import VideoTranscriber

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
    """Transcribe video - NOW USING ENHANCED PROCESSOR WITH METADATA INJECTION!"""
    try:
        logger.info(f"Transcribing video for job {job_id}: {media_path}")

        from .models import get_whisper_model
        # ✅ IMPORT ENHANCED PROCESSOR
        from src.core.enhanced_transcript_processor import EnhancedTranscriptProcessor

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

        # ❌ REMOVE OLD BASIC TEXT SPLITTING:
        # text_splitter = RecursiveCharacterTextSplitter(...)
        # chunks = text_splitter.split_text(transcript)

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

        # ✅ NEW: USE ENHANCED TRANSCRIPT PROCESSOR
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
            "processing_method": "enhanced_transcript_processor"
        })

        logger.info(f"Transcription completed for job {job_id}")

        # Trigger next task
        job_chain.task_completed(job_id, transcription_result)

    except Exception as e:
        error_msg = f"Video transcription failed for job {job_id}: {str(e)}"
        logger.error(error_msg)
        job_chain.task_failed(job_id, error_msg)


@dramatiq.actor(queue_name="cpu_tasks", store_results=True, max_retries=2)
def process_pdf_task(job_id: str, file_path: str, metadata: Optional[Dict] = None):
    """Process PDF - Unicode cleaning happens automatically!"""
    try:
        logger.info(f"Processing PDF for job {job_id}: {file_path}")

        from src.core.pdf_loader import PDFLoader

        # Create PDF loader
        pdf_loader = PDFLoader(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            device="cpu",
            use_ocr=settings.use_pdf_ocr,
            ocr_languages=settings.ocr_languages
        )

        # Process PDF
        documents = pdf_loader.process_pdf(
            file_path=file_path,
            custom_metadata=metadata,
        )

        logger.info(f"PDF processing completed for job {job_id}: {len(documents)} documents")

        # Convert documents to format for next task
        document_dicts = []
        for doc in documents:
            document_dicts.append({
                "content": doc.page_content,
                "metadata": doc.metadata
            })

        pdf_result = {
            "documents": document_dicts,
            "document_count": len(documents),
            "pdf_processing_completed_at": time.time(),
            "file_path": file_path,
            "custom_metadata": metadata
        }

        # Store result in job tracker
        job_tracker.update_job_status(
            job_id,
            "processing",
            result=pdf_result,
            stage="pdf_processing_completed",
            replace_result=True
        )

        # Trigger next task
        job_chain.task_completed(job_id, pdf_result)

    except Exception as e:
        logger.error(f"PDF processing failed for job {job_id}: {str(e)}")
        job_chain.task_failed(job_id, f"PDF processing failed: {str(e)}")


@dramatiq.actor(queue_name="cpu_tasks", store_results=True, max_retries=2)
def process_text_task(job_id: str, text: str, metadata: Optional[Dict] = None):
    """Process text - Unicode cleaning happens automatically!"""
    try:
        logger.info(f"Processing text for job {job_id}")

        from langchain_text_splitters import RecursiveCharacterTextSplitter
        from langchain_core.documents import Document

        # Validate text input
        if not text or not text.strip():
            error_msg = f"Text input is empty for job {job_id}"
            logger.error(error_msg)
            job_chain.task_failed(job_id, error_msg)
            return

        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )

        chunks = text_splitter.split_text(text)

        # Create documents
        documents = []
        for i, chunk_text in enumerate(chunks):
            # Apply metadata extraction to the full text (not just chunks)
            if i == 0:  # Only extract metadata once from full text
                from src.utils.helpers import extract_metadata_from_text
                extracted_metadata = extract_metadata_from_text(text)
            else:
                extracted_metadata = {}

            # Combine extracted metadata with provided metadata
            doc_metadata = {
                "chunk_id": i,
                "source": "manual",
                "source_id": job_id,
                "total_chunks": len(chunks),
                **extracted_metadata,
                **(metadata or {})
            }

            doc = Document(
                page_content=chunk_text,
                metadata=doc_metadata
            )
            documents.append(doc)

        logger.info(f"Text processing completed for job {job_id}: {len(chunks)} chunks")

        # Convert documents to format for next task
        document_dicts = []
        for doc in documents:
            document_dicts.append({
                "content": doc.page_content,
                "metadata": doc.metadata
            })

        text_result = {
            "documents": document_dicts,
            "chunk_count": len(chunks),
            "text_processing_completed_at": time.time(),
            "original_text": text,
            "custom_metadata": metadata
        }

        # Store result in job tracker
        job_tracker.update_job_status(
            job_id,
            "processing",
            result=text_result,
            stage="text_processing_completed",
            replace_result=True
        )

        # Trigger next task
        job_chain.task_completed(job_id, text_result)

    except Exception as e:
        logger.error(f"Text processing failed for job {job_id}: {str(e)}")
        job_chain.task_failed(job_id, f"Text processing failed: {str(e)}")


@dramatiq.actor(queue_name="embedding_tasks", store_results=True, max_retries=2)
def generate_embeddings_task(job_id: str, documents: List[Dict]):
    """Generate embeddings - Unicode cleaning happens automatically!"""
    try:
        logger.info(f"Generating embeddings for job {job_id}: {len(documents)} documents")

        from .models import get_vector_store
        from langchain_core.documents import Document

        # Validate documents exist
        if not documents:
            error_msg = f"No documents provided for embedding generation in job {job_id}"
            logger.error(error_msg)
            job_chain.task_failed(job_id, error_msg)
            return

        # Convert back to Document objects
        doc_objects = []
        for doc_dict in documents:
            if not doc_dict.get("content") or not doc_dict.get("metadata"):
                error_msg = f"Invalid document structure in job {job_id} - missing content or metadata"
                logger.error(error_msg)
                job_chain.task_failed(job_id, error_msg)
                return

            doc = Document(
                page_content=doc_dict["content"],
                metadata=doc_dict["metadata"]
            )
            doc_objects.append(doc)

        # Log sample for verification
        if doc_objects:
            sample_doc = doc_objects[0]
            logger.info(f"Sample document for job {job_id}")
            logger.info(f"  Sample title: {sample_doc.metadata.get('title', 'NO_TITLE')}")
            logger.info(f"  Sample author: {sample_doc.metadata.get('author', 'NO_AUTHOR')}")
            logger.info(f"  Total documents: {len(doc_objects)}")

        # Get existing job data for context
        current_job = job_tracker.get_job(job_id, include_progress=False)
        existing_result = current_job.get("result", {}) if current_job else {}

        if isinstance(existing_result, str):
            try:
                existing_result = json.loads(existing_result)
            except:
                existing_result = {}

        # Add ingestion timestamp and job ID to ALL documents
        current_time = time.time()
        for doc in doc_objects:
            # Ensure job_id is always present
            doc.metadata["job_id"] = job_id

            # Add ingestion timestamp if not present
            if "ingestion_time" not in doc.metadata:
                doc.metadata["ingestion_time"] = current_time

            # Ensure document has an ID for proper indexing
            if "id" not in doc.metadata or not doc.metadata["id"]:
                doc.metadata["id"] = f"doc-{job_id}-{len(doc_objects)}-{int(current_time)}"

        # Add to vector store using preloaded embedding model
        vector_store = get_vector_store()
        doc_ids = vector_store.add_documents(doc_objects)

        if not doc_ids:
            error_msg = f"Vector store failed to generate document IDs for job {job_id}"
            logger.error(error_msg)
            job_chain.task_failed(job_id, error_msg)
            return

        logger.info(f"Successfully added {len(doc_ids)} documents to vector store")

        # Create final result while PRESERVING ALL existing data
        final_result = {}
        final_result.update(existing_result)

        # Add the new embedding data
        final_result.update({
            "document_ids": doc_ids,
            "document_count": len(doc_ids),
            "embedding_completed_at": time.time(),
            "ingestion_completed": True
        })

        logger.info(f"Embedding generation completed for job {job_id}")

        # Complete the job (this is the final step for most processing jobs)
        job_chain.task_completed(job_id, final_result)

    except Exception as e:
        error_msg = f"Embedding generation failed for job {job_id}: {str(e)}"
        logger.error(error_msg)
        job_chain.task_failed(job_id, error_msg)


@dramatiq.actor(queue_name="embedding_tasks", store_results=True, max_retries=2)
def retrieve_documents_task(job_id: str, query: str, metadata_filter: Optional[Dict] = None, query_mode: str = "facts"):
    """
    Document retrieval with ChatGPT's recommended post-reranking fact validation.

    STRATEGY:
    1. Trust vector store semantic ranking
    2. Apply minimal filtering (garbage removal)
    3. POST-RERANKING: Apply automotive domain fact validation
    4. Add validation warnings to metadata (don't remove docs)
    """
    try:
        from .models import get_vector_store
        from src.utils.quality_utils import automotive_fact_check_documents
        import numpy as np

        logger.info(f"Retrieval + automotive fact validation for job {job_id} in '{query_mode}' mode: {query}")

        vector_store = get_vector_store()

        # Parse query mode
        try:
            mode_enum = QueryMode(query_mode)
        except ValueError:
            logger.warning(f"Invalid query mode '{query_mode}', falling back to facts")
            mode_enum = QueryMode.FACTS

        # Get mode-specific parameters (simplified)
        retrieval_params = mode_config.get_retrieval_params(mode_enum)
        context_params = mode_config.get_context_params(mode_enum)

        # Phase 1: Initial broad retrieval (trust vector store ranking)
        initial_k = retrieval_params["retrieval_k"]
        results = vector_store.similarity_search_with_score(
            query=query,
            k=initial_k,
            metadata_filter=metadata_filter
        )

        logger.info(f"Initial retrieval returned {len(results)} results")

        if not results:
            logger.warning(f"No documents found for query: {query}")
            job_chain.task_completed(job_id, {
                "documents": [],
                "document_count": 0,
                "retrieval_completed_at": time.time(),
                "retrieval_method": f"automotive_validated_{query_mode}",
                "query_used": query,
                "query_mode": query_mode
            })
            return

        # Phase 2: MINIMAL filtering (trust semantic ranking, remove only garbage)
        debug_mode = getattr(settings, 'debug_mode', False)
        token_budget = context_params["max_context_tokens"]

        filtered_results = minimal_effective_filter(
            results,
            token_budget,
            debug_mode
        )

        # Phase 3: POST-RERANKING automotive domain fact validation (ChatGPT's approach)
        logger.info("Applying automotive domain fact validation...")
        fact_checked_results = automotive_fact_check_documents(filtered_results)

        # Count validation results
        docs_with_warnings = sum(1 for doc, _ in fact_checked_results
                                 if doc.metadata.get("automotive_warnings"))

        logger.info(f"Automotive fact validation: {len(fact_checked_results)} docs processed, "
                    f"{docs_with_warnings} with warnings")

        # Calculate statistics
        if fact_checked_results:
            relevance_scores = [score for _, score in fact_checked_results]
            avg_relevance_score = sum(relevance_scores) / len(relevance_scores)
        else:
            avg_relevance_score = 0.0

        # Format results for transfer to inference worker
        serialized_docs = []
        for doc, score in fact_checked_results:
            json_safe_score = float(score) if isinstance(score, (np.floating, np.float32, np.float64)) else score

            cleaned_metadata = {}
            for key, value in doc.metadata.items():
                if isinstance(value, (np.floating, np.float32, np.float64)):
                    cleaned_metadata[key] = float(value)
                elif isinstance(value, (np.integer, np.int32, np.int64)):
                    cleaned_metadata[key] = int(value)
                elif isinstance(value, np.ndarray):
                    cleaned_metadata[key] = value.tolist()
                elif isinstance(value, list):
                    cleaned_metadata[key] = value  # Keep lists (warnings, etc.)
                else:
                    cleaned_metadata[key] = value

            serialized_docs.append({
                "content": doc.page_content,
                "metadata": cleaned_metadata,
                "relevance_score": json_safe_score
            })

        logger.info(f"Automotive-validated retrieval completed for job {job_id}: {len(fact_checked_results)} documents")
        logger.info(f"Average relevance score: {avg_relevance_score:.3f}")

        # Trigger LLM inference with fact-checked documents
        job_chain.task_completed(job_id, {
            "documents": serialized_docs,
            "document_count": len(serialized_docs),
            "retrieval_completed_at": time.time(),
            "retrieval_method": f"automotive_validated_{query_mode}",
            "query_mode": query_mode,
            "avg_relevance_score": avg_relevance_score,
            "token_budget_used": token_budget,
            "automotive_validation": {
                "total_documents": len(fact_checked_results),
                "documents_with_warnings": docs_with_warnings,
                "validation_applied": True,
                "domain_specific": "automotive"
            },
            "enhanced_with_fact_checking": True
        })

    except Exception as e:
        logger.error(f"Automotive-validated retrieval failed for job {job_id}: {str(e)}")
        job_chain.task_failed(job_id, f"Automotive-validated retrieval failed: {str(e)}")


@dramatiq.actor(queue_name="inference_tasks", store_results=True, max_retries=2)
def llm_inference_task(job_id: str, query: str, documents: List[Dict], query_mode: str = "facts"):
    """
    LLM inference with automotive domain fact validation and user trust features.

    STRATEGY (ChatGPT's approach):
    1. Generate answer with mode-specific parameters
    2. Apply automotive domain fact validation to final answer
    3. Add validation warnings as footnotes for user trust
    """
    try:
        from .models import get_llm_model
        from langchain_core.documents import Document
        from src.utils.quality_utils import (
            automotive_fact_check_answer,
            format_automotive_warnings_for_user,
            get_automotive_validation_summary
        )

        logger.info(f"LLM inference + automotive validation for job {job_id} in '{query_mode}' mode")

        # Get the preloaded LLM model
        llm_model = get_llm_model()

        # Convert serialized documents back to Document objects
        doc_objects = []
        relevance_scores = []

        for doc_dict in documents:
            doc = Document(
                page_content=doc_dict["content"],
                metadata=doc_dict["metadata"]
            )
            doc_objects.append(doc)

            # Extract relevance score
            relevance_score = doc_dict.get("relevance_score", 0.0)
            relevance_scores.append(relevance_score)

        # Create documents with scores for LLM processing
        documents_with_scores = list(zip(doc_objects, relevance_scores))

        # Calculate average relevance for confidence assessment
        avg_relevance_score = sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0.0

        logger.info(f"Processing {len(documents)} documents with avg relevance: {avg_relevance_score:.3f}")

        # Parse query mode for mode-specific parameters
        try:
            mode_enum = QueryMode(query_mode)
        except ValueError:
            mode_enum = QueryMode.FACTS

        # Get mode-specific LLM parameters
        llm_params = mode_config.get_llm_params(mode_enum)

        # Generate answer with mode-specific parameters
        base_answer = llm_model.answer_query_with_mode_specific_params(
            query=query,
            documents=documents_with_scores,
            query_mode=query_mode,
            temperature=llm_params["temperature"],
            max_tokens=llm_params["max_tokens"],
            top_p=llm_params.get("top_p", 0.85),
            repetition_penalty=llm_params.get("repetition_penalty", 1.1)
        )

        # AUTOMOTIVE DOMAIN FACT VALIDATION (ChatGPT's approach)
        logger.info("Applying automotive domain fact validation to answer...")

        # Validate answer against automotive domain knowledge
        validation_results = automotive_fact_check_answer(base_answer, doc_objects)

        # Format validation warnings for user display (increases trust)
        warning_footnotes = format_automotive_warnings_for_user(validation_results)

        # Create final answer with validation footnotes
        final_answer = base_answer + warning_footnotes

        # Get document validation summary
        doc_validation_summary = get_automotive_validation_summary(doc_objects)

        # Simple confidence assessment combining relevance and automotive validation
        automotive_confidence = validation_results.get("automotive_confidence", "unknown")
        simple_confidence = min(100, avg_relevance_score * 100 +
                                (20 if automotive_confidence == "high" else
                                 10 if automotive_confidence == "medium" else 0))

        # Get current job data
        current_job = job_tracker.get_job(job_id, include_progress=False)
        existing_result = current_job.get("result", {}) if current_job else {}

        if isinstance(existing_result, str):
            try:
                existing_result = json.loads(existing_result)
            except:
                existing_result = {}

        # Create enhanced inference result with automotive validation
        inference_result = {
            "answer": final_answer,
            "base_answer": base_answer,  # Answer without validation footnotes
            "query": query,
            "query_mode": query_mode,
            "inference_completed_at": time.time(),
            "llm_parameters_used": llm_params,

            # Simple confidence metrics
            "simple_confidence": simple_confidence,
            "avg_relevance_score": avg_relevance_score,
            "documents_used": len(documents),

            # AUTOMOTIVE DOMAIN VALIDATION (the key addition)
            "automotive_validation": {
                "answer_validation": validation_results,
                "document_validation_summary": doc_validation_summary,
                "confidence_level": automotive_confidence,
                "has_warnings": bool(
                    validation_results.get("answer_warnings") or validation_results.get("source_warnings")),
                "user_trust_features": {
                    "validation_footnotes_added": bool(warning_footnotes),
                    "transparent_fact_checking": True,
                    "domain_specific_validation": "automotive"
                }
            },

            "enhanced_with_automotive_validation": True
        }

        # Preserve existing data and add inference result
        final_result = {}
        final_result.update(existing_result)
        final_result.update(inference_result)

        logger.info(f"Automotive-validated LLM inference completed for job {job_id}")
        logger.info(f"Simple confidence: {simple_confidence:.1f}%, Automotive confidence: {automotive_confidence}")

        if validation_results.get("answer_warnings"):
            logger.info(f"Answer warnings: {len(validation_results['answer_warnings'])}")
        if validation_results.get("source_warnings"):
            logger.info(f"Source warnings: {len(set(validation_results['source_warnings']))}")

        # Complete the job
        job_chain.task_completed(job_id, final_result)

    except Exception as e:
        error_msg = f"Automotive-validated LLM inference failed for job {job_id}: {str(e)}"
        logger.error(error_msg)
        job_chain.task_failed(job_id, error_msg)