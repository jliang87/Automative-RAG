# The job_chain.py file stays mostly the same, but here are the key updates needed:

import json
import time
import logging
from typing import Dict, List, Optional, Any
from enum import Enum
import dramatiq

from .job_tracker import job_tracker, JobStatus
from .common import get_redis_client

logger = logging.getLogger(__name__)

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
    """
    Event-driven job chain with dedicated workers.
    """

    def __init__(self):
        self.redis = get_redis_client()

        # Define job workflows - each job type has a sequence of tasks
        # UPDATED: Ensure correct queue routing for dedicated workers
        self.workflows = {
            JobType.VIDEO_PROCESSING: [
                ("download_video", "cpu_tasks"),           # CPU worker
                ("transcribe_video", "transcription_tasks"), # Whisper worker
                ("generate_embeddings", "embedding_tasks")   # Embedding worker
            ],
            JobType.PDF_PROCESSING: [
                ("process_pdf", "cpu_tasks"),              # CPU worker
                ("generate_embeddings", "embedding_tasks")   # Embedding worker
            ],
            JobType.TEXT_PROCESSING: [
                ("process_text", "cpu_tasks"),             # CPU worker
                ("generate_embeddings", "embedding_tasks")   # Embedding worker
            ],
            JobType.LLM_INFERENCE: [
                ("retrieve_documents", "embedding_tasks"),   # Embedding worker
                ("llm_inference", "inference_tasks")         # Inference worker
            ]
        }

    def start_job_chain(self, job_id: str, job_type: JobType, initial_data: Dict[str, Any]) -> None:
        """
        Start a job chain. This immediately executes the first task.

        Args:
            job_id: Unique job identifier
            job_type: Type of job to execute
            initial_data: Data needed for the first task
        """
        workflow = self.workflows.get(job_type)
        if not workflow:
            raise ValueError(f"Unknown job type: {job_type}")

        # Store the job chain state
        chain_state = {
            "job_id": job_id,
            "job_type": job_type.value,
            "workflow": [(task, queue) for task, queue in workflow],
            "current_step": 0,
            "total_steps": len(workflow),
            "data": initial_data,
            "started_at": time.time(),
            "status": TaskStatus.RUNNING.value,
            "step_timings": {}
        }

        self._save_chain_state(job_id, chain_state)

        # Update job tracker
        job_tracker.update_job_status(
            job_id,
            JobStatus.PROCESSING,
            result={"message": f"Starting {job_type.value} workflow", "step": 1, "total_steps": len(workflow)},
            stage="chain_started"
        )

        # Immediately execute the first task
        self._execute_next_task(job_id)

    def _execute_next_task(self, job_id: str) -> None:
        """
        Execute the next task in the chain for a given job.
        FIXED: Pass complete data to next task.
        """
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
        progress = ((current_step + 0.5) / len(workflow)) * 100  # 0.5 for "in progress"
        job_tracker.update_job_progress(job_id, progress, f"Executing {task_name}")

        # Record step start time
        chain_state["step_timings"][task_name] = {"started_at": time.time()}
        self._save_chain_state(job_id, chain_state)

        # CRITICAL FIX: Get the complete current data for the task
        # Get from job tracker, not just chain state
        current_job = job_tracker.get_job(job_id, include_progress=False)
        current_job_result = current_job.get("result", {}) if current_job else {}

        # Parse if string
        if isinstance(current_job_result, str):
            try:
                import json
                current_job_result = json.loads(current_job_result)
            except:
                current_job_result = {}

        # Merge chain data with current job data
        complete_data = {}
        complete_data.update(chain_state["data"])  # Start with chain data
        complete_data.update(current_job_result)  # Merge with current job result

        logger.info(f"Executing {task_name} with data keys: {list(complete_data.keys())}")
        if "video_metadata" in complete_data:
            vm = complete_data["video_metadata"]
            logger.info(f"Passing video_metadata to {task_name} with URL: {vm.get('url', 'NO_URL')}")

        # Check if there's already a running task for this queue type
        if self._is_queue_busy(queue_name):
            logger.info(f"Queue {queue_name} is busy, task {task_name} will wait")
            # Queue the task - it will be picked up when the queue is free
            self._queue_task(job_id, task_name, queue_name, complete_data)
        else:
            # Execute immediately
            self._execute_task_immediately(job_id, task_name, queue_name, complete_data)

    def _execute_task_immediately(self, job_id: str, task_name: str, queue_name: str, data: Dict[str, Any]) -> None:
        """Execute a task immediately and mark the queue as busy."""
        # Mark queue as busy
        self._mark_queue_busy(queue_name, job_id, task_name)

        # Execute the appropriate task based on task_name
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
                retrieve_documents_task.send(job_id, data.get("query"), data.get("metadata_filter"))
            elif task_name == "llm_inference":
                llm_inference_task.send(job_id, data.get("query"), data.get("documents"))
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
        self.redis.lpush(f"waiting_tasks:{queue_name}", json.dumps(queued_task))
        logger.info(f"Queued task {task_name} for job {job_id} in {queue_name}")

        # Update job progress to show waiting
        job_tracker.update_job_progress(job_id, None, f"Waiting for {queue_name} to become available")

    def task_completed(self, job_id: str, result: Dict[str, Any]) -> None:
        """
        Called when a task completes successfully.
        This automatically triggers the next task in the chain.
        FIXED: Properly preserve and pass data between tasks.
        """
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

        # CRITICAL FIX: Properly merge task result with chain data
        # Get existing job tracker data FIRST
        current_job = job_tracker.get_job(job_id, include_progress=False)
        existing_job_result = current_job.get("result", {}) if current_job else {}

        # Parse existing result if it's a string
        if isinstance(existing_job_result, str):
            try:
                import json
                existing_job_result = json.loads(existing_job_result)
            except:
                existing_job_result = {}

        # CRITICAL: Merge task result with existing job data
        combined_result = {}
        combined_result.update(existing_job_result)  # Preserve existing data first
        combined_result.update(result)  # Add new task result

        # VERIFICATION: Log the merge
        logger.info(f"Task completion merge for job {job_id}:")
        logger.info(f"  Existing job result keys: {list(existing_job_result.keys())}")
        logger.info(f"  New task result keys: {list(result.keys())}")
        logger.info(f"  Combined result keys: {list(combined_result.keys())}")

        # Check if video_metadata is preserved
        if "video_metadata" in combined_result:
            vm = combined_result["video_metadata"]
            logger.info(f"  video_metadata preserved with URL: {vm.get('url', 'NO_URL')}")
        elif "video_metadata" in result:
            logger.info(f"  video_metadata found in task result")
        elif "video_metadata" in existing_job_result:
            logger.info(f"  video_metadata found in existing job result")
        else:
            logger.warning(f"  video_metadata NOT found in either result!")

        # Update chain state data with the combined result
        chain_state["data"].update(combined_result)

        # Move to next step
        chain_state["current_step"] += 1

        # Update progress
        progress = (chain_state["current_step"] / len(chain_state["workflow"])) * 100
        job_tracker.update_job_progress(job_id, progress,
                                        f"Completed step {chain_state['current_step']}/{len(chain_state['workflow'])}")

        # CRITICAL: Update job tracker with the combined result
        job_tracker.update_job_status(
            job_id,
            "processing",
            result=combined_result,  # Store the combined result, not just the task result
            stage=f"completed_step_{current_step + 1}"
        )

        # Save updated state
        self._save_chain_state(job_id, chain_state)

        # Execute next task
        self._execute_next_task(job_id)

    def task_failed(self, job_id: str, error: str) -> None:
        """
        Called when a task fails.
        This stops the chain and marks the job as failed.
        """
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
            error=error
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

        # CRITICAL FIX: Get the CURRENT job data, not chain data
        current_job = job_tracker.get_job(job_id, include_progress=False)

        if not current_job:
            logger.error(f"No job data found for completed job {job_id}")
            # Clean up and return
            self._delete_chain_state(job_id)
            return

        existing_result = current_job.get("result", {})

        # Parse existing result if it's a string
        if isinstance(existing_result, str):
            try:
                import json
                existing_result = json.loads(existing_result)
            except:
                existing_result = {}

        # Default completion info
        completion_info = {
            "job_chain_completion": {
                "message": "Job chain completed successfully",
                "total_duration": total_duration,
                "step_timings": chain_state.get("step_timings", {}) if chain_state else {},
                "completed_at": time.time()
            }
        }

        # CRITICAL FIX: Preserve ALL existing result data
        if existing_result and isinstance(existing_result, dict):
            # Preserve everything and add completion info
            final_result = {}
            final_result.update(existing_result)  # Preserve all existing data first
            final_result.update(completion_info)  # Add completion info

            logger.info(f"Preserving all job data for {job_id} with keys: {list(existing_result.keys())}")

            # VERIFICATION: Check video_metadata preservation
            if "video_metadata" in final_result:
                vm = final_result["video_metadata"]
                logger.info(f"Final job completion: video_metadata preserved with URL: {vm.get('url', 'NO_URL')}")
            else:
                logger.warning(f"Final job completion: video_metadata NOT found in final result")
        else:
            # No meaningful existing result
            final_result = completion_info
            logger.info(f"No existing result to preserve for job {job_id}")

        # Update job status with the final result
        job_tracker.update_job_status(
            job_id,
            "completed",
            result=final_result
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
        self.redis.set(f"queue_busy:{queue_name}", json.dumps(busy_info), ex=3600)  # 1 hour timeout
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
            "data_keys": list(chain_state["data"].keys())  # Don't expose all data
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
        """Save chain state to Redis."""
        self.redis.set(f"job_chain:{job_id}", json.dumps(chain_state), ex=86400)  # 24 hour expiry

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


# Task actor definitions - each one calls job_chain.task_completed() or job_chain.task_failed()

# ==============================================================================
# UPDATED TASK DEFINITIONS WITH DEDICATED WORKER SUPPORT
# ==============================================================================

@dramatiq.actor(queue_name="cpu_tasks", store_results=True, max_retries=2)
def download_video_task(job_id: str, url: str, metadata: Optional[Dict] = None):
    """Download video and trigger next task - FAIL if metadata extraction fails."""
    try:
        logger.info(f"Downloading video for job {job_id}: {url}")

        from src.core.video_transcriber import VideoTranscriber
        transcriber = VideoTranscriber()

        # Extract audio
        media_path = transcriber.extract_audio(url)

        # Get video metadata - THIS WILL RAISE EXCEPTION IF IT FAILS
        try:
            video_metadata = transcriber.get_video_metadata(url)
            logger.info(f"Successfully retrieved video metadata for job {job_id}")
        except Exception as e:
            # NO FALLBACK - fail the entire job
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

        job_chain.task_completed(job_id, download_result)

    except Exception as e:
        error_msg = f"Video download failed for job {job_id}: {str(e)}"
        logger.error(error_msg)
        job_chain.task_failed(job_id, error_msg)

    except Exception as e:
        logger.error(f"Video download failed for job {job_id}: {str(e)}")
        job_chain.task_failed(job_id, f"Video download failed: {str(e)}")


@dramatiq.actor(queue_name="transcription_tasks", store_results=True, max_retries=2)
def transcribe_video_task(job_id: str, media_path: str):
    """Transcribe video using preloaded Whisper model - FAIL if video_metadata is missing."""
    try:
        logger.info(f"Transcribing video for job {job_id}: {media_path}")

        from .models import get_whisper_model
        from langchain_core.documents import Document
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        from src.config.settings import settings

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

        # Split transcript into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )
        chunks = text_splitter.split_text(transcript)

        # CRITICAL: Get existing job data and VALIDATE video_metadata exists
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

        # FAIL if video_metadata is missing or invalid
        video_metadata = existing_result.get("video_metadata", {})
        if not video_metadata or not isinstance(video_metadata, dict):
            error_msg = f"video_metadata missing from previous step for job {job_id}"
            logger.error(error_msg)
            job_chain.task_failed(job_id, error_msg)
            return

        # FAIL if video_metadata is just fallback data
        if (video_metadata.get("title") in ["Unknown Video", ""] or
                video_metadata.get("author") in ["Unknown", "Unknown Author", ""]):
            error_msg = f"video_metadata contains fallback/invalid data for job {job_id}"
            logger.error(error_msg)
            job_chain.task_failed(job_id, error_msg)
            return

        # Get the original URL - MUST exist
        original_url = video_metadata.get("url")
        if not original_url:
            error_msg = f"No URL found in video_metadata for job {job_id}"
            logger.error(error_msg)
            job_chain.task_failed(job_id, error_msg)
            return

        # Get video ID - MUST exist
        video_id = video_metadata.get("video_id")
        if not video_id:
            error_msg = f"No video_id found in video_metadata for job {job_id}"
            logger.error(error_msg)
            job_chain.task_failed(job_id, error_msg)
            return

        logger.info(f"Using valid video metadata: {video_metadata.get('title')} by {video_metadata.get('author')}")

        # Determine source type
        source_type = "video"
        job_metadata = current_job.get("metadata", {})
        if isinstance(job_metadata, str):
            try:
                job_metadata = json.loads(job_metadata)
            except:
                job_metadata = {}

        platform = job_metadata.get("platform", "").lower()
        if platform == "youtube":
            source_type = "youtube"
        elif platform == "bilibili":
            source_type = "bilibili"
        elif "youtube.com" in original_url or "youtu.be" in original_url:
            source_type = "youtube"
        elif "bilibili.com" in original_url:
            source_type = "bilibili"

        logger.info(f"Detected source type: {source_type} for job {job_id}")

        # Create documents with VALIDATED metadata
        documents = []
        for i, chunk in enumerate(chunks):
            doc = Document(
                page_content=chunk,
                metadata={
                    "chunk_id": i,
                    "source": source_type,
                    "source_id": video_id,
                    "language": info.language,
                    "total_chunks": len(chunks),

                    # Use VALIDATED video metadata
                    "title": video_metadata["title"],
                    "author": video_metadata["author"],
                    "url": original_url,
                    "video_id": video_id,
                    "published_date": video_metadata.get("published_date"),
                    "description": video_metadata.get("description", ""),
                    "length": video_metadata.get("length", 0),
                    "views": video_metadata.get("views", 0),

                    # Add custom metadata from job if any
                    "custom_metadata": job_metadata.get("custom_metadata", {})
                }
            )
            documents.append(doc)

        logger.info(f"Transcription completed for job {job_id}: {len(chunks)} chunks, language: {info.language}")

        # Create transcription result while preserving ALL existing data
        transcription_result = {}
        transcription_result.update(existing_result)  # Preserve everything

        # Add new transcription data
        transcription_result.update({
            "documents": [{"content": doc.page_content, "metadata": doc.metadata} for doc in documents],
            "transcript": transcript,
            "language": info.language,
            "duration": info.duration,
            "chunk_count": len(chunks),
            "transcription_completed_at": time.time(),
            "detected_source": source_type,
        })

        # FINAL VALIDATION: Ensure video_metadata is still preserved
        if "video_metadata" not in transcription_result:
            error_msg = f"CRITICAL: video_metadata lost during transcription processing for job {job_id}"
            logger.error(error_msg)
            job_chain.task_failed(job_id, error_msg)
            return

        logger.info(f"SUCCESS: Transcription completed with valid metadata for job {job_id}")

        # On success, trigger next task
        job_chain.task_completed(job_id, transcription_result)

    except Exception as e:
        error_msg = f"Video transcription failed for job {job_id}: {str(e)}"
        logger.error(error_msg)
        job_chain.task_failed(job_id, error_msg)


@dramatiq.actor(queue_name="cpu_tasks", store_results=True, max_retries=2)
def process_pdf_task(job_id: str, file_path: str, metadata: Optional[Dict] = None):
    """Process PDF and trigger next task."""
    try:
        logger.info(f"Processing PDF for job {job_id}: {file_path}")

        # Import here to avoid circular imports
        from src.core.pdf_loader import PDFLoader
        from src.config.settings import settings

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

        # CRITICAL FIX: Store PDF processing result
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
            stage="pdf_processing_completed"
        )

        # On success, trigger next task
        job_chain.task_completed(job_id, pdf_result)

    except Exception as e:
        logger.error(f"PDF processing failed for job {job_id}: {str(e)}")
        job_chain.task_failed(job_id, f"PDF processing failed: {str(e)}")


@dramatiq.actor(queue_name="cpu_tasks", store_results=True, max_retries=2)
def process_text_task(job_id: str, text: str, metadata: Optional[Dict] = None):
    """Process text and trigger next task."""
    try:
        logger.info(f"Processing text for job {job_id}")

        # Import here to avoid circular imports
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        from langchain_core.documents import Document
        from src.config.settings import settings

        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )

        chunks = text_splitter.split_text(text)

        # Convert chunks to documents with metadata
        documents = []
        for i, chunk_text in enumerate(chunks):
            doc = Document(
                page_content=chunk_text,
                metadata={
                    "chunk_id": i,
                    "source": "manual",
                    "source_id": job_id,
                    "total_chunks": len(chunks),
                    **(metadata or {})
                }
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

        # CRITICAL FIX: Store text processing result
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
            stage="text_processing_completed"
        )

        # On success, trigger next task
        job_chain.task_completed(job_id, text_result)

    except Exception as e:
        logger.error(f"Text processing failed for job {job_id}: {str(e)}")
        job_chain.task_failed(job_id, f"Text processing failed: {str(e)}")


@dramatiq.actor(queue_name="embedding_tasks", store_results=True, max_retries=2)
def generate_embeddings_task(job_id: str, documents: List[Dict]):
    """Generate embeddings using preloaded embedding model - FAIL if video_metadata is missing."""
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

        # CRITICAL: Get existing job data and VALIDATE video_metadata exists
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

        # FAIL if video_metadata is missing or invalid
        video_metadata = existing_result.get("video_metadata", {})
        if not video_metadata or not isinstance(video_metadata, dict):
            error_msg = f"video_metadata missing from previous steps for job {job_id}"
            logger.error(error_msg)
            job_chain.task_failed(job_id, error_msg)
            return

        # FAIL if video_metadata contains fallback/invalid data
        if (video_metadata.get("title") in ["Unknown Video", ""] or
                video_metadata.get("author") in ["Unknown", "Unknown Author", ""] or
                not video_metadata.get("url") or
                not video_metadata.get("video_id")):
            error_msg = f"video_metadata contains invalid/fallback data for job {job_id}"
            logger.error(error_msg)
            job_chain.task_failed(job_id, error_msg)
            return

        logger.info(f"Using validated video_metadata: {video_metadata['title']} by {video_metadata['author']}")

        # Get detected source type from previous step
        detected_source = existing_result.get("detected_source")
        if not detected_source:
            error_msg = f"No detected_source from transcription step for job {job_id}"
            logger.error(error_msg)
            job_chain.task_failed(job_id, error_msg)
            return

        # Add ingestion timestamp and job ID to ALL documents
        current_time = time.time()
        for doc in doc_objects:
            # Validate document has required metadata
            if not doc.metadata.get("title") or not doc.metadata.get("author"):
                error_msg = f"Document missing required metadata (title/author) in job {job_id}"
                logger.error(error_msg)
                job_chain.task_failed(job_id, error_msg)
                return

            # Ensure job_id is always present
            doc.metadata["job_id"] = job_id

            # Add ingestion timestamp if not present
            if "ingestion_time" not in doc.metadata:
                doc.metadata["ingestion_time"] = current_time

            # Ensure document has an ID for proper indexing
            if "id" not in doc.metadata or not doc.metadata["id"]:
                doc.metadata["id"] = f"doc-{job_id}-{len(doc_objects)}-{int(current_time)}"

        # Validate that video_metadata is properly embedded in documents
        for doc in doc_objects:
            if (doc.metadata.get("title") in ["Unknown Video", ""] or
                    doc.metadata.get("author") in ["Unknown", "Unknown Author", ""] or
                    not doc.metadata.get("url")):
                error_msg = f"Document metadata contains invalid data for job {job_id}"
                logger.error(error_msg)
                job_chain.task_failed(job_id, error_msg)
                return

        logger.info(f"All documents validated with proper metadata for job {job_id}")

        # Check if we're in metadata-only mode
        if not hasattr(job_tracker, '_metadata_only_mode'):
            # Add to vector store using preloaded embedding model
            vector_store = get_vector_store()
            doc_ids = vector_store.add_documents(doc_objects)

            if not doc_ids:
                error_msg = f"Vector store failed to generate document IDs for job {job_id}"
                logger.error(error_msg)
                job_chain.task_failed(job_id, error_msg)
                return

            logger.info(f"Embedding generation completed for job {job_id}: {len(doc_ids)} document IDs")
        else:
            # In metadata-only mode, simulate doc IDs
            doc_ids = [f"sim-{job_id}-{i}" for i in range(len(doc_objects))]
            logger.info(
                f"Simulated embedding generation for job {job_id}: {len(doc_ids)} document IDs (metadata-only mode)")

        # Create final result while PRESERVING ALL existing data
        final_result = {}
        final_result.update(existing_result)  # Preserve everything from previous steps

        # Add the new embedding data
        final_result.update({
            "document_ids": doc_ids,
            "document_count": len(doc_ids),
            "embedding_completed_at": time.time(),
            "ingestion_completed": True
        })

        # FINAL VALIDATION: Ensure video_metadata is still preserved
        if "video_metadata" not in final_result:
            error_msg = f"CRITICAL: video_metadata lost during embedding generation for job {job_id}"
            logger.error(error_msg)
            job_chain.task_failed(job_id, error_msg)
            return

        # VALIDATE final video_metadata quality
        final_vm = final_result["video_metadata"]
        if (not final_vm.get("title") or final_vm.get("title") == "Unknown Video" or
                not final_vm.get("author") or final_vm.get("author") in ["Unknown", "Unknown Author"] or
                not final_vm.get("url")):
            error_msg = f"CRITICAL: Final video_metadata is invalid for job {job_id}"
            logger.error(error_msg)
            job_chain.task_failed(job_id, error_msg)
            return

        logger.info(f"SUCCESS: Embedding generation completed with validated metadata for job {job_id}")
        logger.info(f"Final video_metadata: {final_vm['title']} by {final_vm['author']}")

        # On success, complete the job (this is the final step for video processing)
        job_chain.task_completed(job_id, final_result)

    except Exception as e:
        error_msg = f"Embedding generation failed for job {job_id}: {str(e)}"
        logger.error(error_msg)
        job_chain.task_failed(job_id, error_msg)


@dramatiq.actor(queue_name="embedding_tasks", store_results=True, max_retries=2)
def retrieve_documents_task(job_id: str, query: str, metadata_filter: Optional[Dict] = None):
    """Retrieve documents using preloaded embedding model."""
    try:
        logger.info(f"Retrieving documents for job {job_id}: {query}")

        # Import here to avoid circular imports
        from .models import get_vector_store
        from src.config.settings import settings
        import numpy as np

        # Get vector store with preloaded embedding model
        vector_store = get_vector_store()

        # Perform retrieval
        results = vector_store.similarity_search_with_score(
            query=query,
            k=settings.retriever_top_k,
            metadata_filter=metadata_filter
        )

        # Format results for transfer to inference worker - FIXED VERSION
        serialized_docs = []
        for doc, score in results:
            # CRITICAL FIX: Convert numpy.float32 to Python float
            json_safe_score = float(score) if isinstance(score, (np.floating, np.float32, np.float64)) else score

            # Clean metadata to ensure JSON serialization compatibility
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

            serialized_docs.append({
                "content": doc.page_content,
                "metadata": cleaned_metadata,
                "relevance_score": json_safe_score
            })

        logger.info(f"Document retrieval completed for job {job_id}: {len(serialized_docs)} documents")

        # On success, trigger LLM inference
        job_chain.task_completed(job_id, {
            "documents": serialized_docs,
            "document_count": len(serialized_docs),
            "retrieval_completed_at": time.time()
        })

    except Exception as e:
        logger.error(f"Document retrieval failed for job {job_id}: {str(e)}")
        job_chain.task_failed(job_id, f"Document retrieval failed: {str(e)}")


@dramatiq.actor(queue_name="inference_tasks", store_results=True, max_retries=2)
def llm_inference_task(job_id: str, query: str, documents: List[Dict]):
    """Perform LLM inference using preloaded models."""
    try:
        logger.info(f"Performing LLM inference for job {job_id}: {query}")

        # Import here to avoid circular imports
        from .models import get_llm_model, get_colbert_reranker
        from langchain_core.documents import Document
        from src.config.settings import settings
        import numpy as np

        # Convert documents back to Document objects with scores
        doc_objects = []
        for doc_dict in documents:
            doc = Document(
                page_content=doc_dict["content"],
                metadata=doc_dict.get("metadata", {})
            )
            # CRITICAL FIX: Convert numpy.float32 to Python float
            score = doc_dict.get("relevance_score", 0)
            if isinstance(score, (np.floating, np.float32, np.float64)):
                score = float(score)
            doc_objects.append((doc, score))

        # Perform reranking using preloaded ColBERT model
        reranker = get_colbert_reranker()
        if reranker is not None:
            logger.info(f"Reranking {len(doc_objects)} documents for job {job_id}")
            reranked_docs = reranker.rerank(query, [doc for doc, _ in doc_objects], settings.reranker_top_k)
        else:
            logger.warning("Reranker not available, using original document order")
            reranked_docs = doc_objects[:settings.reranker_top_k]

        # Get preloaded LLM model and perform inference
        llm = get_llm_model()
        answer = llm.answer_query(
            query=query,
            documents=reranked_docs,
            metadata_filter=None
        )

        # Prepare formatted documents for response with JSON-safe scores
        formatted_documents = []
        for doc, score in reranked_docs:
            # CRITICAL FIX: Ensure all numeric values are JSON-serializable
            json_safe_score = float(score) if isinstance(score, (np.floating, np.float32, np.float64)) else score

            # Clean metadata to ensure all values are JSON-serializable
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

            formatted_doc = {
                "id": cleaned_metadata.get("id", ""),
                "content": doc.page_content,
                "metadata": cleaned_metadata,
                "relevance_score": json_safe_score,
            }
            formatted_documents.append(formatted_doc)

        logger.info(f"LLM inference completed for job {job_id}")

        # CRITICAL FIX: Create the complete inference result with JSON-safe data
        inference_result = {
            "query": query,
            "answer": answer,
            "documents": formatted_documents,
            "document_count": len(formatted_documents),
            "inference_completed_at": time.time()
        }

        # CRITICAL FIX: Store the complete result in job tracker BEFORE calling task_completed
        job_tracker.update_job_status(
            job_id,
            "processing",  # Keep as processing until job chain completes
            result=inference_result,  # Store the actual inference result
            stage="llm_inference_completed"
        )

        # Pass the inference result to job chain (this will trigger completion)
        job_chain.task_completed(job_id, inference_result)

    except Exception as e:
        logger.error(f"LLM inference failed for job {job_id}: {str(e)}")
        job_chain.task_failed(job_id, f"LLM inference failed: {str(e)}")