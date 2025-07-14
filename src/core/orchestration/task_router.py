"""
Task Router - Routes job types to appropriate task handlers
Enhanced with comprehensive validation pipeline support
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
    COMPREHENSIVE_VALIDATION = "comprehensive_validation"  # NEW


class TaskRouter:
    """
    Routes job types to their appropriate task handlers.
    Enhanced with comprehensive validation pipeline support.
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
            ],
            # NEW: Comprehensive validation workflow
            JobType.COMPREHENSIVE_VALIDATION: [
                ("knowledge_validation", "cpu_tasks"),
                ("pre_llm_validation", "inference_tasks"),
                ("main_llm_inference", "inference_tasks"),
                ("post_llm_validation", "inference_tasks"),
                ("final_validation", "inference_tasks")
            ]
        }

    def route_task(self, job_id: str, task_name: str, queue_name: str, data: Dict[str, Any]) -> None:
        """
        Route a task to its appropriate handler.
        Enhanced with validation task routing.
        """
        try:
            logger.info(f"Routing task {task_name} for job {job_id}")

            # ✅ KEEP all existing task routing
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

            # 🆕 NEW: Comprehensive validation routing
            elif task_name == "knowledge_validation":
                self._route_knowledge_validation(job_id, data)
            elif task_name == "pre_llm_validation":
                self._route_pre_llm_validation(job_id, data)
            elif task_name == "main_llm_inference":
                self._route_main_llm_inference(job_id, data)
            elif task_name == "post_llm_validation":
                self._route_post_llm_validation(job_id, data)
            elif task_name == "final_validation":
                self._route_final_validation(job_id, data)
            elif task_name == "meta_validation":
                self._route_meta_validation(job_id, data)
            elif task_name == "auto_fetch":
                self._route_auto_fetch(job_id, data)
            elif task_name == "validation_cache":
                self._route_validation_cache(job_id, data)

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

    # ✅ KEEP all existing routing methods
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

    # 🆕 NEW: Validation task routing methods
    def _route_knowledge_validation(self, job_id: str, data: Dict[str, Any]) -> None:
        """Route knowledge-based validation task."""
        from src.core.validation.tasks.knowledge_validation_task import knowledge_validation_task
        knowledge_validation_task.send(job_id, data)

    def _route_pre_llm_validation(self, job_id: str, data: Dict[str, Any]) -> None:
        """Route pre-LLM validation task."""
        from src.core.validation.tasks.llm_phase_validation_task import llm_phase_validation_task
        # Add phase type to data
        data["phase_type"] = "pre_validation"
        llm_phase_validation_task.send(job_id, data)

    def _route_main_llm_inference(self, job_id: str, data: Dict[str, Any]) -> None:
        """Route main LLM inference with validation context."""
        # Use existing LLM inference but with validation enhancement
        data["validation_enhanced"] = True
        self._route_llm_inference(job_id, data)

    def _route_post_llm_validation(self, job_id: str, data: Dict[str, Any]) -> None:
        """Route post-LLM validation task."""
        from src.core.validation.tasks.llm_phase_validation_task import llm_phase_validation_task
        # Add phase type to data
        data["phase_type"] = "post_validation"
        llm_phase_validation_task.send(job_id, data)

    def _route_final_validation(self, job_id: str, data: Dict[str, Any]) -> None:
        """Route final validation assessment task."""
        from src.core.validation.tasks.llm_phase_validation_task import llm_phase_validation_task
        # Add phase type to data
        data["phase_type"] = "final_assessment"
        llm_phase_validation_task.send(job_id, data)

    def _route_meta_validation(self, job_id: str, data: Dict[str, Any]) -> None:
        """Route meta-validation coordination task."""
        from src.core.validation.tasks.meta_validation_task import meta_validation_task
        meta_validation_task.send(job_id, data)

    def _route_auto_fetch(self, job_id: str, data: Dict[str, Any]) -> None:
        """Route auto-fetch operations task."""
        from src.core.validation.tasks.auto_fetch_task import auto_fetch_task
        auto_fetch_task.send(job_id, data)

    def _route_validation_cache(self, job_id: str, data: Dict[str, Any]) -> None:
        """Route validation caching task."""
        from src.core.validation.tasks.validation_cache_task import validation_cache_task
        validation_cache_task.send(job_id, data)

    def start_job_workflow(self, job_id: str, job_type: JobType, data: Dict[str, Any]) -> None:
        """
        Start a complete job workflow.
        Enhanced with comprehensive validation support.
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

            # 🆕 NEW: Comprehensive validation workflow
            elif job_type == JobType.COMPREHENSIVE_VALIDATION:
                self._start_comprehensive_validation_workflow(job_id, data)

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

    def _start_comprehensive_validation_workflow(self, job_id: str, data: Dict[str, Any]) -> None:
        """Start comprehensive validation workflow."""
        from src.core.validation.tasks.knowledge_validation_task import knowledge_validation_task

        # Initialize validation context
        validation_data = {
            "query": data.get("query"),
            "query_mode": data.get("query_mode", "facts"),
            "documents": data.get("documents", []),
            "metadata_filter": data.get("metadata_filter"),
            "validation_workflow": "comprehensive"
        }

        # Start with knowledge validation
        knowledge_validation_task.send(job_id, validation_data)

    def get_workflow_for_job_type(self, job_type: JobType) -> list:
        """Get the workflow definition for a job type."""
        return self.workflows.get(job_type, [])

    def get_supported_job_types(self) -> list:
        """Get list of supported job types."""
        return list(self.workflows.keys())


# Global task router instance
task_router = TaskRouter()