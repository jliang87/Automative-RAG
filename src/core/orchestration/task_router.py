"""
Task Router - Routes job types to appropriate task handlers
Replaces the massive task execution logic in JobChain
"""

import logging
from typing import Dict, Any
from enum import Enum

logger = logging.getLogger(__name__)


class JobType(Enum):
    VIDEO_PROCESSING = "video_processing"
    PDF_PROCESSING = "pdf_processing"
    TEXT_PROCESSING = "text_processing"
    LLM_INFERENCE = "llm_inference"


class TaskRouter:
    """
    Routes job types to their appropriate task handlers.

    This replaces the massive _execute_task_immediately method in JobChain
    with a clean, modular approach.
    """

    def __init__(self):
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

    def route_task(self, job_id: str, task_name: str, queue_name: str, data: Dict[str, Any]) -> None:
        """
        Route a task to its appropriate handler.

        Args:
            job_id: Job identifier
            task_name: Name of the task to execute
            queue_name: Queue name for the task
            data: Task data
        """
        try:
            logger.info(f"Routing task {task_name} for job {job_id}")

            # Route to appropriate task handler
            if task_name == "download_video":
                self._route_video_download(job_id, data)

            elif task_name == "transcribe_video":
                self._route_video_transcription(job_id, data)

            elif task_name == "process_pdf":
                self._route_pdf_processing(job_id, data)

            elif task_name == "process_text":
                self._route_text_processing(job_id, data)

            elif task_name == "generate_embeddings":
                self._route_embedding_generation(job_id, data)

            elif task_name == "retrieve_documents":
                self._route_document_retrieval(job_id, data)

            elif task_name == "llm_inference":
                self._route_llm_inference(job_id, data)

            else:
                error_msg = f"Unknown task type: {task_name}"
                logger.error(error_msg)
                from src.core.orchestration.job_chain import job_chain
                job_chain.task_failed(job_id, error_msg)

        except Exception as e:
            error_msg = f"Error routing task {task_name} for job {job_id}: {str(e)}"
            logger.error(error_msg)
            from src.core.orchestration.job_chain import job_chain
            job_chain.task_failed(job_id, error_msg)

    def start_job_workflow(self, job_id: str, job_type: JobType, data: Dict[str, Any]) -> None:
        """
        Start a complete job workflow.

        Args:
            job_id: Job identifier
            job_type: Type of job workflow to start
            data: Initial job data
        """
        try:
            logger.info(f"Starting {job_type.value} workflow for job {job_id}")

            if job_type == JobType.VIDEO_PROCESSING:
                from src.core.ingestion.tasks.video_tasks import start_video_processing
                start_video_processing(job_id, data)

            elif job_type == JobType.PDF_PROCESSING:
                from src.core.ingestion.tasks.pdf_tasks import start_pdf_processing
                start_pdf_processing(job_id, data)

            elif job_type == JobType.TEXT_PROCESSING:
                from src.core.ingestion.tasks.text_tasks import start_text_processing
                start_text_processing(job_id, data)

            elif job_type == JobType.LLM_INFERENCE:
                from src.core.query.tasks.retrieval_tasks import start_document_retrieval
                start_document_retrieval(job_id, data)

            else:
                error_msg = f"Unknown job type: {job_type}"
                logger.error(error_msg)
                from src.core.orchestration.job_chain import job_chain
                job_chain.task_failed(job_id, error_msg)

        except Exception as e:
            error_msg = f"Error starting {job_type.value} workflow for job {job_id}: {str(e)}"
            logger.error(error_msg)
            from src.core.orchestration.job_chain import job_chain
            job_chain.task_failed(job_id, error_msg)

    def _route_video_download(self, job_id: str, data: Dict[str, Any]) -> None:
        """Route video download task."""
        from src.core.ingestion.tasks.video_tasks import download_video_task
        download_video_task.send(job_id, data.get("url"), data.get("metadata"))

    def _route_video_transcription(self, job_id: str, data: Dict[str, Any]) -> None:
        """Route video transcription task."""
        from src.core.ingestion.tasks.video_tasks import transcribe_video_task
        transcribe_video_task.send(job_id, data.get("media_path"))

    def _route_pdf_processing(self, job_id: str, data: Dict[str, Any]) -> None:
        """Route PDF processing task."""
        from src.core.ingestion.tasks.pdf_tasks import process_pdf_task
        process_pdf_task.send(job_id, data.get("file_path"), data.get("metadata"))

    def _route_text_processing(self, job_id: str, data: Dict[str, Any]) -> None:
        """Route text processing task."""
        from src.core.ingestion.tasks.text_tasks import process_text_task
        process_text_task.send(job_id, data.get("text"), data.get("metadata"))

    def _route_embedding_generation(self, job_id: str, data: Dict[str, Any]) -> None:
        """Route embedding generation task."""
        from src.core.ingestion.tasks.embedding_tasks import generate_embeddings_task
        generate_embeddings_task.send(job_id, data.get("documents"))

    def _route_document_retrieval(self, job_id: str, data: Dict[str, Any]) -> None:
        """Route document retrieval task."""
        from src.core.query.tasks.retrieval_tasks import retrieve_documents_task
        query_mode = data.get("query_mode", "facts")
        retrieve_documents_task.send(
            job_id,
            data.get("query"),
            data.get("metadata_filter"),
            query_mode
        )

    def _route_llm_inference(self, job_id: str, data: Dict[str, Any]) -> None:
        """Route LLM inference task."""
        from src.core.query.tasks.inference_tasks import llm_inference_task
        query_mode = data.get("query_mode", "facts")
        llm_inference_task.send(
            job_id,
            data.get("query"),
            data.get("documents"),
            query_mode
        )

    def get_workflow_for_job_type(self, job_type: JobType) -> list:
        """Get the workflow definition for a job type."""
        return self.workflows.get(job_type, [])

    def get_supported_job_types(self) -> list:
        """Get list of supported job types."""
        return list(self.workflows.keys())


# Global task router instance
task_router = TaskRouter()