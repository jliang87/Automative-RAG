"""
Updated Task Router - Pure task routing with NO business logic
Business logic moved to WorkflowController and models
Core layer executes any specification provided without business knowledge
"""

import logging
from typing import Dict, Any
from enum import Enum

from src.core.orchestration.queue_manager import QueueNames

logger = logging.getLogger(__name__)


class JobType(Enum):
    """Legacy job types for core infrastructure compatibility"""
    VIDEO_PROCESSING = "video_processing"
    PDF_PROCESSING = "pdf_processing"
    TEXT_PROCESSING = "text_processing"
    LLM_INFERENCE = "llm_inference"
    COMPREHENSIVE_VALIDATION = "comprehensive_validation"  # Will be removed


class TaskRouter:
    """
    Pure task routing - NO business logic
    Infrastructure-only: routes tasks to appropriate queues without knowing workflow details
    All workflow logic now handled upstream in controller/model layers
    """

    def __init__(self):
        # Only pure task-to-queue routing mapping (no workflow definitions)
        self.task_to_queue_mapping = {
            # Video processing tasks
            "download_video": QueueNames.CPU_TASKS.value,
            "transcribe_video": QueueNames.TRANSCRIPTION_TASKS.value,

            # Document processing tasks
            "process_pdf": QueueNames.CPU_TASKS.value,
            "process_text": QueueNames.CPU_TASKS.value,
            "document_parsing": QueueNames.CPU_TASKS.value,
            "content_extraction": QueueNames.CPU_TASKS.value,

            # Embedding and indexing tasks
            "generate_embeddings": QueueNames.EMBEDDING_TASKS.value,
            "document_indexing": QueueNames.EMBEDDING_TASKS.value,
            "retrieve_documents": QueueNames.EMBEDDING_TASKS.value,

            # LLM tasks (all LLM work uses same queue)
            "llm_inference": QueueNames.LLM_TASKS.value,
            "response_formatting": QueueNames.CPU_TASKS.value,

            # Validation tasks (legacy - will be removed)
            "knowledge_validation": QueueNames.CPU_TASKS.value,
            "pre_llm_validation": QueueNames.LLM_TASKS.value,
            "post_llm_validation": QueueNames.LLM_TASKS.value,
            "final_validation": QueueNames.LLM_TASKS.value,
            "meta_validation": QueueNames.CPU_TASKS.value,

            # Causation tasks (future)
            "data_preparation": QueueNames.CPU_TASKS.value,
            "causation_detection": QueueNames.LLM_TASKS.value,
            "relationship_mapping": QueueNames.CPU_TASKS.value,

            # General tasks
            "data_validation": QueueNames.CPU_TASKS.value,
            "notification": QueueNames.CPU_TASKS.value
        }

    def route_task(self, job_id: str, task_name: str, queue_name: str, data: Dict[str, Any]) -> None:
        """
        Pure task routing - delegates to appropriate task execution without business logic
        NO workflow knowledge - just routes based on task type
        """
        try:
            logger.info(f"Routing task {task_name} for job {job_id} to queue {queue_name}")

            # Route to appropriate task executor based on task type
            if task_name == "download_video":
                self._route_to_video_download_executor(job_id, data)
            elif task_name == "transcribe_video":
                self._route_to_transcription_executor(job_id, data)
            elif task_name in ["process_pdf", "process_text", "document_parsing"]:
                self._route_to_document_processing_executor(job_id, task_name, data)
            elif task_name in ["content_extraction"]:
                self._route_to_content_extraction_executor(job_id, data)
            elif task_name in ["generate_embeddings", "document_indexing"]:
                self._route_to_embedding_executor(job_id, task_name, data)
            elif task_name == "retrieve_documents":
                self._route_to_retrieval_executor(job_id, data)
            elif task_name == "llm_inference":
                self._route_to_llm_executor(job_id, data)
            elif task_name == "response_formatting":
                self._route_to_formatting_executor(job_id, data)

            # Legacy validation tasks (will be removed)
            elif task_name in ["knowledge_validation", "pre_llm_validation", "post_llm_validation",
                             "final_validation", "meta_validation"]:
                self._route_to_legacy_validation_executor(job_id, task_name, data)

            # Future causation tasks
            elif task_name in ["data_preparation", "causation_detection", "relationship_mapping"]:
                self._route_to_causation_executor(job_id, task_name, data)

            # General tasks
            elif task_name in ["data_validation", "notification"]:
                self._route_to_general_executor(job_id, task_name, data)

            else:
                error_msg = f"Unknown task type: {task_name}"
                logger.error(error_msg)
                self._handle_routing_failure(job_id, error_msg)

        except Exception as e:
            error_msg = f"Error routing task {task_name} for job {job_id}: {str(e)}"
            logger.error(error_msg)
            self._handle_routing_failure(job_id, error_msg)

    # ========================================================================
    # Pure Task Routing Methods (NO business logic)
    # ========================================================================

    def _route_to_video_download_executor(self, job_id: str, data: Dict[str, Any]) -> None:
        """Route to video download task executor"""
        try:
            from src.core.ingestion.tasks.video_tasks import download_video_task
            download_video_task.send(job_id, data.get("url"), data.get("metadata"))
        except ImportError:
            logger.warning(f"Video download task not available for job {job_id}")
            self._handle_missing_task_executor(job_id, "video_download")

    def _route_to_transcription_executor(self, job_id: str, data: Dict[str, Any]) -> None:
        """Route to video transcription task executor"""
        try:
            from src.core.ingestion.tasks.video_tasks import transcribe_video_task
            transcribe_video_task.send(job_id, data.get("media_path"))
        except ImportError:
            logger.warning(f"Video transcription task not available for job {job_id}")
            self._handle_missing_task_executor(job_id, "video_transcription")

    def _route_to_document_processing_executor(self, job_id: str, task_name: str, data: Dict[str, Any]) -> None:
        """Route to document processing task executor"""
        try:
            if task_name == "process_pdf":
                from src.core.ingestion.tasks.pdf_tasks import process_pdf_task
                process_pdf_task.send(job_id, data.get("file_path"), data.get("metadata"))
            elif task_name == "process_text":
                from src.core.ingestion.tasks.text_tasks import process_text_task
                process_text_task.send(job_id, data.get("text"), data.get("metadata"))
            elif task_name == "document_parsing":
                from src.core.ingestion.tasks.document_tasks import parse_document_task
                parse_document_task.send(job_id, data.get("file_path"), data.get("document_type"))
        except ImportError:
            logger.warning(f"Document processing task {task_name} not available for job {job_id}")
            self._handle_missing_task_executor(job_id, task_name)

    def _route_to_content_extraction_executor(self, job_id: str, data: Dict[str, Any]) -> None:
        """Route to content extraction task executor"""
        try:
            from src.core.ingestion.tasks.document_tasks import extract_content_task
            extract_content_task.send(job_id, data.get("text_content"), data.get("extraction_config"))
        except ImportError:
            logger.warning(f"Content extraction task not available for job {job_id}")
            self._handle_missing_task_executor(job_id, "content_extraction")

    def _route_to_embedding_executor(self, job_id: str, task_name: str, data: Dict[str, Any]) -> None:
        """Route to embedding generation task executor"""
        try:
            from src.core.ingestion.tasks.embedding_tasks import generate_embeddings_task
            if task_name == "generate_embeddings":
                generate_embeddings_task.send(job_id, data.get("documents"))
            elif task_name == "document_indexing":
                from src.core.ingestion.tasks.embedding_tasks import index_document_task
                index_document_task.send(job_id, data.get("content"), data.get("metadata"))
        except ImportError:
            logger.warning(f"Embedding task {task_name} not available for job {job_id}")
            self._handle_missing_task_executor(job_id, task_name)

    def _route_to_retrieval_executor(self, job_id: str, data: Dict[str, Any]) -> None:
        """Route to document retrieval task executor"""
        try:
            from src.core.query.tasks.retrieval_tasks import retrieve_documents_task
            retrieve_documents_task.send(
                job_id,
                data.get("query"),
                data.get("metadata_filter"),
                data.get("query_mode", "facts"),
                data.get("top_k", 10)
            )
        except ImportError:
            logger.warning(f"Document retrieval task not available for job {job_id}")
            self._handle_missing_task_executor(job_id, "document_retrieval")

    def _route_to_llm_executor(self, job_id: str, data: Dict[str, Any]) -> None:
        """Route to LLM inference task executor"""
        try:
            from src.core.query.tasks.inference_tasks import llm_inference_task
            llm_inference_task.send(
                job_id,
                data.get("query"),
                data.get("documents"),
                data.get("query_mode", "facts"),
                data.get("prompt_template")
            )
        except ImportError:
            logger.warning(f"LLM inference task not available for job {job_id}")
            self._handle_missing_task_executor(job_id, "llm_inference")

    def _route_to_formatting_executor(self, job_id: str, data: Dict[str, Any]) -> None:
        """Route to response formatting task executor"""
        try:
            from src.core.query.tasks.inference_tasks import format_response_task
            format_response_task.send(
                job_id,
                data.get("answer"),
                data.get("documents"),
                data.get("query_mode", "facts"),
                data.get("response_format", "markdown")
            )
        except ImportError:
            logger.warning(f"Response formatting task not available for job {job_id}")
            self._handle_missing_task_executor(job_id, "response_formatting")

    def _route_to_legacy_validation_executor(self, job_id: str, task_name: str, data: Dict[str, Any]) -> None:
        """Route to legacy validation task executor (will be removed)"""
        logger.warning(f"Legacy validation task {task_name} for job {job_id} - these will be removed")

        try:
            # Route to appropriate validation executor if available
            if task_name == "knowledge_validation":
                from src.core.validation.tasks.knowledge_validation_task import knowledge_validation_task
                knowledge_validation_task.send(job_id, data)
            elif task_name in ["pre_llm_validation", "post_llm_validation", "final_validation"]:
                from src.core.validation.tasks.llm_phase_validation_task import llm_phase_validation_task
                llm_phase_validation_task.send(job_id, data)
            elif task_name == "meta_validation":
                from src.core.validation.tasks.meta_validation_task import meta_validation_task
                meta_validation_task.send(job_id, data)
        except ImportError:
            logger.warning(f"Legacy validation task {task_name} not available for job {job_id}")
            self._handle_missing_task_executor(job_id, task_name)

    def _route_to_causation_executor(self, job_id: str, task_name: str, data: Dict[str, Any]) -> None:
        """Route to causation analysis task executor (future)"""
        logger.info(f"Causation task {task_name} for job {job_id} - future implementation")

        # Placeholder routing for future causation tasks
        try:
            if task_name == "data_preparation":
                from src.core.causation.tasks.preparation_tasks import prepare_data_task
                prepare_data_task.send(job_id, data)
            elif task_name == "causation_detection":
                from src.core.causation.tasks.analysis_tasks import detect_causation_task
                detect_causation_task.send(job_id, data)
            elif task_name == "relationship_mapping":
                from src.core.causation.tasks.mapping_tasks import map_relationships_task
                map_relationships_task.send(job_id, data)
        except ImportError:
            logger.info(f"Causation task {task_name} not yet implemented for job {job_id}")
            self._handle_missing_task_executor(job_id, task_name)

    def _route_to_general_executor(self, job_id: str, task_name: str, data: Dict[str, Any]) -> None:
        """Route to general task executor"""
        try:
            if task_name == "data_validation":
                from src.core.general.tasks.validation_tasks import validate_data_task
                validate_data_task.send(job_id, data)
            elif task_name == "notification":
                from src.core.general.tasks.notification_tasks import send_notification_task
                send_notification_task.send(job_id, data)
        except ImportError:
            logger.warning(f"General task {task_name} not available for job {job_id}")
            self._handle_missing_task_executor(job_id, task_name)

    # ========================================================================
    # Infrastructure Support Methods
    # ========================================================================

    def get_queue_for_task(self, task_name: str) -> str:
        """Get queue name for a task (pure infrastructure function)"""
        return self.task_to_queue_mapping.get(task_name, QueueNames.CPU_TASKS.value)

    def get_supported_tasks(self) -> List[str]:
        """Get list of supported task types"""
        return list(self.task_to_queue_mapping.keys())

    def validate_task_routing(self, task_name: str) -> bool:
        """Validate that a task can be routed"""
        return task_name in self.task_to_queue_mapping

    def get_queue_mapping(self) -> Dict[str, str]:
        """Get complete task-to-queue mapping"""
        return self.task_to_queue_mapping.copy()

    # ========================================================================
    # Error Handling (Infrastructure Level)
    # ========================================================================

    def _handle_routing_failure(self, job_id: str, error_message: str) -> None:
        """Handle task routing failure"""
        logger.error(f"Task routing failed for job {job_id}: {error_message}")

        try:
            from src.core.orchestration.job_chain import job_chain
            job_chain.task_failed(job_id, error_message)
        except Exception as e:
            logger.error(f"Could not notify job chain of routing failure: {str(e)}")

    def _handle_missing_task_executor(self, job_id: str, task_name: str) -> None:
        """Handle missing task executor gracefully"""
        logger.warning(f"Task executor {task_name} not available for job {job_id}")

        # Create placeholder result to keep workflow moving
        placeholder_result = {
            "task_name": task_name,
            "status": "SKIPPED",
            "reason": f"Task executor {task_name} not available",
            "skipped_at": time.time()
        }

        try:
            from src.core.orchestration.job_chain import job_chain
            job_chain.task_completed(job_id, placeholder_result)
        except Exception as e:
            logger.error(f"Could not notify job chain of skipped task: {str(e)}")

    # ========================================================================
    # Legacy Methods (For Backward Compatibility)
    # ========================================================================

    def get_workflow_for_job_type(self, job_type: JobType) -> list:
        """
        Legacy method for backward compatibility with existing job chain
        Returns empty list since workflows are now defined upstream
        """
        logger.warning(f"Legacy method get_workflow_for_job_type called for {job_type}")
        return []

# Global task router instance
task_router = TaskRouter()