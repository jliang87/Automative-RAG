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
        This is called both initially and after each task completion.
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
            JobStatus.PROCESSING,
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

        # Check if there's already a running task for this queue type
        if self._is_queue_busy(queue_name):
            logger.info(f"Queue {queue_name} is busy, task {task_name} will wait")
            # Queue the task - it will be picked up when the queue is free
            self._queue_task(job_id, task_name, queue_name, chain_state["data"])
        else:
            # Execute immediately
            self._execute_task_immediately(job_id, task_name, queue_name, chain_state["data"])

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

        # Update data with result
        chain_state["data"].update(result)

        # Move to next step
        chain_state["current_step"] += 1

        # Update progress
        progress = (chain_state["current_step"] / len(chain_state["workflow"])) * 100
        job_tracker.update_job_progress(job_id, progress,
                                        f"Completed step {chain_state['current_step']}/{len(chain_state['workflow'])}")

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
        """Complete the entire job chain."""
        logger.info(f"Job chain completed for job {job_id}")

        # Get final chain state for timing information
        chain_state = self._get_chain_state(job_id)
        total_duration = time.time() - chain_state["started_at"] if chain_state else 0

        # Update job status
        job_tracker.update_job_status(
            job_id,
            JobStatus.COMPLETED,
            result={
                "message": "Job chain completed successfully",
                "total_duration": total_duration,
                "step_timings": chain_state.get("step_timings", {}) if chain_state else {}
            }
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
    """Download video and trigger next task."""
    try:
        logger.info(f"Downloading video for job {job_id}: {url}")

        # Import here to avoid circular imports
        from src.core.video_transcriber import VideoTranscriber

        transcriber = VideoTranscriber()

        # Extract audio (this includes download)
        media_path = transcriber.extract_audio(url)

        # Get video metadata
        video_metadata = transcriber.get_video_metadata(url)

        logger.info(f"Video download completed for job {job_id}, media saved to: {media_path}")

        # On success, trigger next task
        job_chain.task_completed(job_id, {
            "media_path": media_path,
            "video_metadata": video_metadata,
            "download_completed_at": time.time()
        })

    except Exception as e:
        logger.error(f"Video download failed for job {job_id}: {str(e)}")
        job_chain.task_failed(job_id, f"Video download failed: {str(e)}")


@dramatiq.actor(queue_name="transcription_tasks", store_results=True, max_retries=2)
def transcribe_video_task(job_id: str, media_path: str):
    """Transcribe video using preloaded Whisper model."""
    try:
        logger.info(f"Transcribing video for job {job_id}: {media_path}")

        # Import here to avoid circular imports
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

        # Create documents with metadata
        documents = []
        for i, chunk in enumerate(chunks):
            doc = Document(
                page_content=chunk,
                metadata={
                    "chunk_id": i,
                    "source": "video",
                    "source_id": job_id,
                    "language": info.language,
                    "total_chunks": len(chunks)
                }
            )
            documents.append(doc)

        logger.info(f"Transcription completed for job {job_id}: {len(chunks)} chunks, language: {info.language}")

        # On success, trigger next task
        job_chain.task_completed(job_id, {
            "documents": [{"content": doc.page_content, "metadata": doc.metadata} for doc in documents],
            "transcript": transcript,
            "language": info.language,
            "duration": info.duration,
            "chunk_count": len(chunks),
            "transcription_completed_at": time.time()
        })

    except Exception as e:
        logger.error(f"Video transcription failed for job {job_id}: {str(e)}")
        job_chain.task_failed(job_id, f"Video transcription failed: {str(e)}")


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

        # On success, trigger next task
        job_chain.task_completed(job_id, {
            "documents": document_dicts,
            "document_count": len(documents),
            "pdf_processing_completed_at": time.time()
        })

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

        # On success, trigger next task
        job_chain.task_completed(job_id, {
            "documents": document_dicts,
            "chunk_count": len(chunks),
            "text_processing_completed_at": time.time()
        })

    except Exception as e:
        logger.error(f"Text processing failed for job {job_id}: {str(e)}")
        job_chain.task_failed(job_id, f"Text processing failed: {str(e)}")


@dramatiq.actor(queue_name="embedding_tasks", store_results=True, max_retries=2)
def generate_embeddings_task(job_id: str, documents: List[Dict]):
    """Generate embeddings using preloaded embedding model."""
    try:
        logger.info(f"Generating embeddings for job {job_id}: {len(documents)} documents")

        # Import here to avoid circular imports
        from .models import get_vector_store
        from langchain_core.documents import Document

        # Convert back to Document objects
        doc_objects = []
        for doc_dict in documents:
            doc = Document(
                page_content=doc_dict["content"],
                metadata=doc_dict["metadata"]
            )
            doc_objects.append(doc)

        # Add ingestion timestamp
        current_time = time.time()
        for doc in doc_objects:
            doc.metadata["ingestion_time"] = current_time
            doc.metadata["job_id"] = job_id

        # Add to vector store using preloaded embedding model
        vector_store = get_vector_store()
        doc_ids = vector_store.add_documents(doc_objects)

        logger.info(f"Embedding generation completed for job {job_id}: {len(doc_ids)} document IDs")

        # On success, complete the job (no next task for ingestion workflows)
        job_chain.task_completed(job_id, {
            "document_ids": doc_ids,
            "document_count": len(doc_ids),
            "embedding_completed_at": time.time()
        })

    except Exception as e:
        logger.error(f"Embedding generation failed for job {job_id}: {str(e)}")
        job_chain.task_failed(job_id, f"Embedding generation failed: {str(e)}")


@dramatiq.actor(queue_name="embedding_tasks", store_results=True, max_retries=2)
def retrieve_documents_task(job_id: str, query: str, metadata_filter: Optional[Dict] = None):
    """Retrieve documents using preloaded embedding model."""
    try:
        logger.info(f"Retrieving documents for job {job_id}: {query}")

        # Import here to avoid circular imports
        from .models import get_vector_store
        from src.config.settings import settings

        # Get vector store with preloaded embedding model
        vector_store = get_vector_store()

        # Perform retrieval
        results = vector_store.similarity_search_with_score(
            query=query,
            k=settings.retriever_top_k,
            metadata_filter=metadata_filter
        )

        # Format results for transfer to inference worker
        serialized_docs = []
        for doc, score in results:
            serialized_docs.append({
                "content": doc.page_content,
                "metadata": doc.metadata,
                "relevance_score": score
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

        # Convert documents back to Document objects with scores
        doc_objects = []
        for doc_dict in documents:
            doc = Document(
                page_content=doc_dict["content"],
                metadata=doc_dict.get("metadata", {})
            )
            score = doc_dict.get("relevance_score", 0)
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

        # Prepare formatted documents for response
        formatted_documents = []
        for doc, score in reranked_docs:
            formatted_doc = {
                "id": doc.metadata.get("id", ""),
                "content": doc.page_content,
                "metadata": doc.metadata,
                "relevance_score": score,
            }
            formatted_documents.append(formatted_doc)

        logger.info(f"LLM inference completed for job {job_id}")

        # On success, complete the job
        job_chain.task_completed(job_id, {
            "query": query,
            "answer": answer,
            "documents": formatted_documents,
            "inference_completed_at": time.time()
        })

    except Exception as e:
        logger.error(f"LLM inference failed for job {job_id}: {str(e)}")
        job_chain.task_failed(job_id, f"LLM inference failed: {str(e)}")