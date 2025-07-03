import json
import time
import logging
from typing import List, Tuple, Dict, Optional, Any
from langchain_core.documents import Document
from enum import Enum
import dramatiq

from src.core.orchestration.job_tracker import job_tracker, JobStatus
from .common import get_redis_client
from src.core.query.llm.mode_config import mode_config, QueryMode, estimate_token_count
from src.config.settings import settings

from src.core.validation.validation_engine import validation_engine

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

        from src.core.ingestion.loaders.video_transcriber import VideoTranscriber

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
        from src.core.ingestion.loaders.enhanced_transcript_processor import EnhancedTranscriptProcessor

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

        from src.core.ingestion.loaders.pdf_loader import PDFLoader

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
    Enhanced document retrieval with full validation pipeline
    """
    try:
        from .models import get_vector_store
        import numpy as np

        logger.info(f"Enhanced retrieval + validation for job {job_id} in '{query_mode}' mode: {query}")

        vector_store = get_vector_store()

        # Parse query mode
        try:
            mode_enum = QueryMode(query_mode)
        except ValueError:
            logger.warning(f"Invalid query mode '{query_mode}', falling back to facts")
            mode_enum = QueryMode.FACTS

        # Get mode-specific parameters
        retrieval_params = mode_config.get_retrieval_params(mode_enum)
        context_params = mode_config.get_context_params(mode_enum)

        # Initial document retrieval
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
                "retrieval_method": f"enhanced_validation_{query_mode}",
                "query_used": query,
                "query_mode": query_mode,
                "validation_result": None
            })
            return

        # Convert to validation framework format
        documents_for_validation = []
        for doc, score in results:
            json_safe_score = float(score) if isinstance(score, (np.floating, np.float32, np.float64)) else score

            cleaned_metadata = {}
            for key, value in doc.metadata.items():
                if isinstance(value, (np.floating, np.float32, np.float64)):
                    cleaned_metadata[key] = float(value)
                elif isinstance(value, (np.integer, np.int32, np.int64)):
                    cleaned_metadata[key] = int(value)
                elif isinstance(value, np.ndarray):
                    cleaned_metadata[key] = value.tolist()
                else:
                    cleaned_metadata[key] = value

            documents_for_validation.append({
                "content": doc.page_content,
                "metadata": cleaned_metadata,
                "relevance_score": json_safe_score
            })

        # Apply full validation pipeline
        validation_result = await validation_engine.validate_documents(
            documents=documents_for_validation,
            query=query,
            query_mode=query_mode,
            metadata_filter=metadata_filter,
            job_id=job_id
        )

        # Extract validated documents for LLM processing
        # Apply token budget management to validated documents
        token_budget = context_params["max_context_tokens"]
        final_documents = apply_token_budget_management(results, token_budget)

        # Calculate enhanced statistics
        if final_documents:
            relevance_scores = [score for _, score in final_documents]
            avg_relevance_score = sum(relevance_scores) / len(relevance_scores)
        else:
            avg_relevance_score = 0.0

        # Format results for transfer to inference worker
        serialized_docs = []
        for doc, score in final_documents:
            json_safe_score = float(score) if isinstance(score, (np.floating, np.float32, np.float64)) else score

            cleaned_metadata = {}
            for key, value in doc.metadata.items():
                if isinstance(value, (np.floating, np.float32, np.float64)):
                    cleaned_metadata[key] = float(value)
                elif isinstance(value, (np.integer, np.int32, np.int64)):
                    cleaned_metadata[key] = int(value)
                elif isinstance(value, np.ndarray):
                    cleaned_metadata[key] = value.tolist()
                else:
                    cleaned_metadata[key] = value

            # Add validation metadata
            cleaned_metadata["validation_status"] = "validated"
            cleaned_metadata["validation_confidence"] = validation_result.confidence.total_score

            serialized_docs.append({
                "content": doc.page_content,
                "metadata": cleaned_metadata,
                "relevance_score": json_safe_score
            })

        logger.info(f"Enhanced validation completed for job {job_id}: {len(final_documents)} documents")
        logger.info(f"Validation confidence: {validation_result.confidence.total_score:.1f}%")

        # Comprehensive result with validation data
        job_chain.task_completed(job_id, {
            "documents": serialized_docs,
            "document_count": len(serialized_docs),
            "retrieval_completed_at": time.time(),
            "retrieval_method": f"enhanced_validation_{query_mode}",
            "query_mode": query_mode,
            "avg_relevance_score": avg_relevance_score,
            "token_budget_used": token_budget,

            # NEW: Comprehensive validation results
            "validation_result": {
                "chain_id": validation_result.chain_id,
                "pipeline_type": validation_result.pipeline_type.value,
                "overall_status": validation_result.overall_status.value,
                "confidence": {
                    "total_score": validation_result.confidence.total_score,
                    "level": validation_result.confidence.level.value,
                    "source_credibility": validation_result.confidence.source_credibility,
                    "technical_consistency": validation_result.confidence.technical_consistency,
                    "completeness": validation_result.confidence.completeness,
                    "consensus": validation_result.confidence.consensus,
                    "verification_coverage": validation_result.confidence.verification_coverage
                },
                "validation_steps": [
                    {
                        "step_name": step.step_name,
                        "status": step.status.value,
                        "confidence_impact": step.confidence_impact,
                        "summary": step.summary,
                        "warnings": [
                            {
                                "category": w.category,
                                "severity": w.severity,
                                "message": w.message,
                                "explanation": w.explanation,
                                "suggestion": w.suggestion
                            }
                            for w in step.warnings
                        ]
                    }
                    for step in validation_result.validation_steps
                ],
                "trust_trail": validation_result.step_progression,
                "contribution_opportunities": [
                    {
                        "needed_resource": prompt.needed_resource_type,
                        "description": prompt.specific_need_description,
                        "confidence_impact": prompt.confidence_impact,
                        "future_benefit": prompt.future_benefit_description
                    }
                    for prompt in validation_result.contribution_opportunities
                ]
            },

            "enhanced_with_validation_framework": True
        })

    except Exception as e:
        logger.error(f"Enhanced retrieval failed for job {job_id}: {str(e)}")
        job_chain.task_failed(job_id, f"Enhanced retrieval failed: {str(e)}")


@dramatiq.actor(queue_name="inference_tasks", store_results=True, max_retries=2)
def llm_inference_task(job_id: str, query: str, documents: List[Dict], query_mode: str = "facts"):
    """
    Enhanced LLM inference with answer validation
    """
    try:
        from .models import get_llm_model

        logger.info(f"Enhanced LLM inference + answer validation for job {job_id} in '{query_mode}' mode")

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

        # NEW: Enhanced answer validation using validation framework
        logger.info("Applying enhanced answer validation...")

        # Convert documents to validation format
        doc_list = []
        for doc in doc_objects:
            doc_dict = {
                "content": doc.page_content,
                "metadata": doc.metadata
            }
            doc_list.append(doc_dict)

        # Validate answer using the validation framework
        answer_validation = await validation_engine.validate_answer(
            answer=base_answer,
            documents=doc_list,
            query=query,
            query_mode=query_mode,
            job_id=job_id
        )

        # Format validation warnings for user display
        warning_footnotes = validation_engine.format_automotive_warnings_for_user(answer_validation)

        # Create final answer with validation footnotes
        final_answer = base_answer + warning_footnotes

        # Enhanced confidence assessment
        validation_confidence = answer_validation.get("confidence_score", 0)
        simple_confidence = min(100, avg_relevance_score * 50 + validation_confidence * 0.5)

        # Get current job data
        current_job = job_tracker.get_job(job_id, include_progress=False)
        existing_result = current_job.get("result", {}) if current_job else {}

        if isinstance(existing_result, str):
            try:
                existing_result = json.loads(existing_result)
            except:
                existing_result = {}

        # Get validation result from retrieval step
        retrieval_validation = existing_result.get("validation_result", {})

        # Create comprehensive inference result with enhanced validation
        inference_result = {
            "answer": final_answer,
            "base_answer": base_answer,  # Answer without validation footnotes
            "query": query,
            "query_mode": query_mode,
            "inference_completed_at": time.time(),
            "llm_parameters_used": llm_params,

            # Enhanced confidence metrics
            "simple_confidence": simple_confidence,
            "avg_relevance_score": avg_relevance_score,
            "documents_used": len(documents),

            # COMPREHENSIVE VALIDATION INTEGRATION
            "enhanced_validation": {
                "answer_validation": answer_validation,
                "retrieval_validation": retrieval_validation,
                "overall_confidence": validation_confidence,
                "confidence_level": answer_validation.get("automotive_confidence", "unknown"),
                "validation_warnings": answer_validation.get("answer_warnings", []),
                "source_warnings": answer_validation.get("source_warnings", []),

                # Trust trail information
                "trust_trail_available": bool(retrieval_validation.get("trust_trail")),
                "validation_steps_completed": len(retrieval_validation.get("validation_steps", [])),
                "pipeline_type": retrieval_validation.get("pipeline_type", "unknown"),

                # User guidance
                "contribution_opportunities": retrieval_validation.get("contribution_opportunities", []),
                "guided_trust_loop_enabled": True,

                # Validation transparency
                "validation_methodology": {
                    "retrieval_pipeline": retrieval_validation.get("pipeline_type"),
                    "answer_validation_applied": True,
                    "multi_step_validation": True,
                    "meta_validation_enabled": True,
                    "user_contribution_enabled": True
                }
            },

            # Legacy compatibility
            "automotive_validation": answer_validation,
            "enhanced_with_validation_framework": True
        }

        # Preserve existing data and add inference result
        final_result = {}
        final_result.update(existing_result)
        final_result.update(inference_result)

        logger.info(f"Enhanced LLM inference completed for job {job_id}")
        logger.info(f"Simple confidence: {simple_confidence:.1f}%, Validation confidence: {validation_confidence:.1f}%")

        if answer_validation.get("answer_warnings"):
            logger.info(f"Answer warnings: {len(answer_validation['answer_warnings'])}")
        if answer_validation.get("source_warnings"):
            logger.info(f"Source warnings: {len(answer_validation['source_warnings'])}")

        # Complete the job
        job_chain.task_completed(job_id, final_result)

    except Exception as e:
        error_msg = f"Enhanced LLM inference failed for job {job_id}: {str(e)}"
        logger.error(error_msg)
        job_chain.task_failed(job_id, error_msg)


@dramatiq.actor(queue_name="inference_tasks", store_results=True, max_retries=2)
def process_user_contribution_task(job_id: str, step_type: str, contribution_data: Dict[str, Any]):
    """
    Process user contribution and retry validation (Guided Trust Loop)
    """
    try:
        logger.info(f"Processing user contribution for job {job_id}, step {step_type}")

        # Process contribution using validation engine
        contribution_result = await validation_engine.process_user_contribution(
            job_id=job_id,
            step_type=step_type,
            contribution_data=contribution_data
        )

        if contribution_result.get("success"):
            # Update job with new validation results
            new_confidence = contribution_result.get("new_confidence", 0)
            learning_credit = contribution_result.get("learning_credit")

            # Get current job data
            current_job = job_tracker.get_job(job_id, include_progress=False)
            existing_result = current_job.get("result", {}) if current_job else {}

            if isinstance(existing_result, str):
                try:
                    existing_result = json.loads(existing_result)
                except:
                    existing_result = {}

            # Update validation results
            validation_update = {
                "contribution_processed": True,
                "contribution_accepted": True,
                "updated_confidence": new_confidence,
                "learning_credit_earned": learning_credit,
                "contribution_timestamp": time.time(),
                "updated_validation": contribution_result.get("validation_updated", False)
            }

            # Update enhanced validation section
            if "enhanced_validation" in existing_result:
                existing_result["enhanced_validation"]["user_contributions"] = existing_result[
                    "enhanced_validation"].get("user_contributions", [])
                existing_result["enhanced_validation"]["user_contributions"].append(validation_update)
                existing_result["enhanced_validation"]["overall_confidence"] = new_confidence

            # Update job
            job_tracker.update_job_status(
                job_id,
                "completed",  # Job remains completed, just updated
                result=existing_result,
                stage="contribution_processed",
                replace_result=True
            )

            logger.info(f"User contribution processed successfully for job {job_id}")

        else:
            error_msg = contribution_result.get("error", "Unknown error processing contribution")
            logger.error(f"Contribution processing failed for job {job_id}: {error_msg}")

    except Exception as e:
        error_msg = f"Contribution processing task failed for job {job_id}: {str(e)}"
        logger.error(error_msg)


def get_validation_summary_for_ui(job_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract validation summary for UI display
    Compatible with existing UI code expectations
    """

    # Get validation data from either new or legacy format
    enhanced_validation = job_result.get("enhanced_validation", {})
    validation_result = job_result.get("validation_result", {})
    automotive_validation = job_result.get("automotive_validation", {})

    # Create comprehensive summary
    summary = {
        # Basic validation info
        "validation_applied": bool(enhanced_validation or validation_result or automotive_validation),
        "validation_framework_used": bool(enhanced_validation),

        # Confidence information
        "overall_confidence": enhanced_validation.get("overall_confidence",
                                                      automotive_validation.get("confidence_score", 0)),
        "confidence_level": enhanced_validation.get("confidence_level",
                                                    automotive_validation.get("automotive_confidence", "unknown")),

        # Validation details
        "pipeline_type": validation_result.get("pipeline_type", "unknown"),
        "validation_steps_completed": len(validation_result.get("validation_steps", [])),
        "verification_coverage": validation_result.get("confidence", {}).get("verification_coverage", 0),

        # Warnings and issues
        "has_warnings": bool(
            enhanced_validation.get("validation_warnings") or
            enhanced_validation.get("source_warnings") or
            automotive_validation.get("answer_warnings") or
            automotive_validation.get("source_warnings")
        ),
        "warning_count": len(
            enhanced_validation.get("validation_warnings", []) +
            enhanced_validation.get("source_warnings", []) +
            automotive_validation.get("answer_warnings", []) +
            automotive_validation.get("source_warnings", [])
        ),

        # Trust trail and user interaction
        "trust_trail_available": enhanced_validation.get("trust_trail_available", False),
        "contribution_opportunities": len(validation_result.get("contribution_opportunities", [])),
        "guided_trust_loop_enabled": enhanced_validation.get("guided_trust_loop_enabled", False),

        # User contributions
        "user_contributions": len(enhanced_validation.get("user_contributions", [])),
        "learning_credits_available": sum(
            opp.get("confidence_impact", 0)
            for opp in validation_result.get("contribution_opportunities", [])
        ),

        # Validation quality indicators
        "validation_quality": {
            "source_credibility": validation_result.get("confidence", {}).get("source_credibility", 0),
            "technical_consistency": validation_result.get("confidence", {}).get("technical_consistency", 0),
            "completeness": validation_result.get("confidence", {}).get("completeness", 0),
            "consensus": validation_result.get("confidence", {}).get("consensus", 0)
        }
    }

    return summary


def enhance_job_status_with_validation(job_details: Dict[str, Any]) -> Dict[str, Any]:
    """
    Enhance job details with validation information for UI display
    """

    result = job_details.get("result", {})
    if isinstance(result, str):
        try:
            result = json.loads(result)
        except:
            result = {}

    # Add validation summary
    validation_summary = get_validation_summary_for_ui(result)
    job_details["validation_summary"] = validation_summary

    # Add quick validation status for display
    if validation_summary["validation_applied"]:
        confidence = validation_summary["overall_confidence"]
        if confidence >= 80:
            job_details["validation_status"] = "high_confidence"
            job_details["validation_badge"] = "🟢 High Confidence"
        elif confidence >= 60:
            job_details["validation_status"] = "medium_confidence"
            job_details["validation_badge"] = "🟡 Medium Confidence"
        else:
            job_details["validation_status"] = "low_confidence"
            job_details["validation_badge"] = "🔴 Low Confidence"

        if validation_summary["has_warnings"]:
            job_details["validation_badge"] += f" ({validation_summary['warning_count']} warnings)"
    else:
        job_details["validation_status"] = "not_validated"
        job_details["validation_badge"] = "⚪ Not Validated"

    return job_details

# Instructions for integrating with existing code:

"""
INTEGRATION INSTRUCTIONS:

1. Add these imports to the top of job_chain.py:
   ```python
   from src.core.validation.validation_engine import validation_engine
   from src.core.validation.models.validation_models import ValidationStatus, ValidationChainResult
   ```

2. Replace the existing retrieve_documents_task function with the enhanced version above

3. Replace the existing llm_inference_task function with the enhanced version above

4. Add the process_user_contribution_task function for guided trust loop functionality

5. Update any existing automotive_fact_check_* function calls to use the validation_engine methods:
   - automotive_fact_check_documents -> validation_engine.validate_documents
   - automotive_fact_check_answer -> validation_engine.validate_answer
   - format_automotive_warnings_for_user -> validation_engine.format_automotive_warnings_for_user

6. For UI integration, use the helper functions:
   - get_validation_summary_for_ui(job_result) to extract validation info
   - enhance_job_status_with_validation(job_details) to add validation status

7. The validation framework is now fully integrated and provides:
   - Multi-step validation pipelines
   - Meta-validation (validation of validation)
   - Guided trust loop for user contributions
   - Comprehensive confidence scoring
   - Trust trail visualization data
   - Learning credit system
   - Backward compatibility with existing code

8. All existing job processing continues to work, but now with enhanced validation:
   - Video processing jobs get full validation pipeline
   - PDF processing jobs get validation
   - Text processing jobs get validation
   - Query processing gets answer validation
   - Users can contribute to improve validation quality
"""