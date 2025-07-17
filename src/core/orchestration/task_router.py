import logging
from typing import Dict, Any
from enum import Enum

# ✅ Import from corrected queue definitions (Tesla T4 constrained)
from src.core.orchestration.queue_manager import QueueNames

logger = logging.getLogger(__name__)


class JobType(Enum):
    VIDEO_PROCESSING = "video_processing"
    PDF_PROCESSING = "pdf_processing"
    TEXT_PROCESSING = "text_processing"
    LLM_INFERENCE = "llm_inference"  # ✅ EXISTING - unchanged
    COMPREHENSIVE_VALIDATION = "comprehensive_validation"  # 🆕 NEW - clean branch


class TaskRouter:
    """
    Routes job types to their appropriate task handlers.
    ✅ CORRECTED: Respects Tesla T4 memory constraints - no new GPU queues.
    """

    def __init__(self):
        # ✅ CORRECTED workflows using Tesla T4 constrained queues
        self.workflows = {
            JobType.VIDEO_PROCESSING: [
                ("download_video", QueueNames.CPU_TASKS.value),
                ("transcribe_video", QueueNames.TRANSCRIPTION_TASKS.value),
                ("generate_embeddings", QueueNames.EMBEDDING_TASKS.value)
            ],
            JobType.PDF_PROCESSING: [
                ("process_pdf", QueueNames.CPU_TASKS.value),  # ✅ CORRECTED: CPU queue
                ("generate_embeddings", QueueNames.EMBEDDING_TASKS.value)
            ],
            JobType.TEXT_PROCESSING: [
                ("process_text", QueueNames.CPU_TASKS.value),  # ✅ CORRECTED: CPU queue
                ("generate_embeddings", QueueNames.EMBEDDING_TASKS.value)
            ],
            JobType.LLM_INFERENCE: [  # ✅ EXISTING - unchanged
                ("retrieve_documents", QueueNames.EMBEDDING_TASKS.value),
                ("llm_inference", QueueNames.LLM_TASKS.value)  # ✅ RENAMED queue
            ],
            # 🆕 CORRECTED: Validation workflow using Tesla T4 constrained queues
            JobType.COMPREHENSIVE_VALIDATION: [
                ("knowledge_validation", QueueNames.CPU_TASKS.value),  # ✅ CORRECTED: CPU queue
                ("pre_llm_validation", QueueNames.LLM_TASKS.value),  # ✅ CORRECTED: Same LLM queue
                ("retrieve_documents", QueueNames.EMBEDDING_TASKS.value),  # ✅ EXISTING task
                ("llm_inference", QueueNames.LLM_TASKS.value),  # ✅ EXISTING task - same queue
                ("post_llm_validation", QueueNames.LLM_TASKS.value),  # ✅ CORRECTED: Same LLM queue
                ("final_validation", QueueNames.LLM_TASKS.value)  # ✅ CORRECTED: Same LLM queue
            ]
        }

    def route_task(self, job_id: str, task_name: str, queue_name: str, data: Dict[str, Any]) -> None:
        """
        Route a task to its appropriate handler.
        ✅ CORRECTED: All validation LLM tasks use same queue for memory sharing.
        """
        try:
            logger.info(f"Routing task {task_name} for job {job_id} to queue {queue_name}")

            # ✅ EXISTING task routing - completely unchanged
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

            # 🆕 CORRECTED: Validation task routing using Tesla T4 constrained queues
            elif task_name == "knowledge_validation":
                self._route_knowledge_validation(job_id, data)
            elif task_name == "pre_llm_validation":
                self._route_pre_llm_validation(job_id, data)
            elif task_name == "post_llm_validation":
                self._route_post_llm_validation(job_id, data)
            elif task_name == "final_validation":
                self._route_final_validation(job_id, data)
            elif task_name == "meta_validation":
                self._route_meta_validation(job_id, data)
            elif task_name == "auto_fetch":
                self._route_auto_fetch(job_id, data)

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

    # ✅ EXISTING routing methods - no changes
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
        """
        Route LLM inference task.
        ✅ CRITICAL: Uses same queue as validation LLM tasks for memory sharing.
        """
        from src.core.query.tasks.inference_tasks import llm_inference_task
        query_mode = data.get("query_mode", "facts")
        llm_inference_task.send(
            job_id,
            data.get("query"),
            data.get("documents"),
            query_mode
        )

    # 🆕 CORRECTED: Validation task routing using Tesla T4 constrained queues
    def _route_knowledge_validation(self, job_id: str, data: Dict[str, Any]) -> None:
        """✅ CORRECTED: Route to CPU queue (Tesla T4 constraint)."""
        from src.core.validation.tasks.knowledge_validation_task import knowledge_validation_task
        knowledge_validation_task.send(job_id, data)

    def _route_pre_llm_validation(self, job_id: str, data: Dict[str, Any]) -> None:
        """✅ CORRECTED: Route to LLM queue (shares memory with inference)."""
        from src.core.validation.tasks.llm_phase_validation_task import llm_phase_validation_task
        data["phase_type"] = "pre_validation"
        llm_phase_validation_task.send(job_id, data)

    def _route_post_llm_validation(self, job_id: str, data: Dict[str, Any]) -> None:
        """✅ CORRECTED: Route to LLM queue (shares memory with inference)."""
        from src.core.validation.tasks.llm_phase_validation_task import llm_phase_validation_task
        data["phase_type"] = "post_validation"
        llm_phase_validation_task.send(job_id, data)

    def _route_final_validation(self, job_id: str, data: Dict[str, Any]) -> None:
        """✅ CORRECTED: Route to LLM queue (shares memory with inference)."""
        from src.core.validation.tasks.llm_phase_validation_task import llm_phase_validation_task
        data["phase_type"] = "final_assessment"
        llm_phase_validation_task.send(job_id, data)

    def _route_meta_validation(self, job_id: str, data: Dict[str, Any]) -> None:
        """✅ CORRECTED: Route to CPU queue (Tesla T4 constraint)."""
        from src.core.validation.tasks.meta_validation_task import meta_validation_task
        meta_validation_task.send(job_id, data)

    def _route_auto_fetch(self, job_id: str, data: Dict[str, Any]) -> None:
        """✅ CORRECTED: Route to CPU queue (Tesla T4 constraint)."""
        from src.core.validation.tasks.auto_fetch_task import auto_fetch_task
        auto_fetch_task.send(job_id, data)

    def start_job_workflow(self, job_id: str, job_type: JobType, data: Dict[str, Any]) -> None:
        """
        Start a complete job workflow using existing orchestration.
        ✅ CORRECTED: Validation workflows respect Tesla T4 constraints.
        """
        try:
            logger.info(f"Starting {job_type.value} workflow for job {job_id}")

            # ✅ EXISTING workflows - no changes
            if job_type == JobType.VIDEO_PROCESSING:
                from src.core.ingestion.tasks.video_tasks import start_video_processing
                start_video_processing(job_id, data)

            elif job_type == JobType.PDF_PROCESSING:
                from src.core.ingestion.tasks.pdf_tasks import start_pdf_processing
                start_pdf_processing(job_id, data)

            elif job_type == JobType.TEXT_PROCESSING:
                from src.core.ingestion.tasks.text_tasks import start_text_processing
                start_text_processing(job_id, data)

            elif job_type == JobType.LLM_INFERENCE:  # ✅ EXISTING - unchanged
                from src.core.query.tasks.retrieval_tasks import start_document_retrieval
                start_document_retrieval(job_id, data)

            # 🆕 CORRECTED: Validation workflow using Tesla T4 constrained queues
            elif job_type == JobType.COMPREHENSIVE_VALIDATION:
                self._start_validation_workflow(job_id, data)

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

    def _start_validation_workflow(self, job_id: str, data: Dict[str, Any]) -> None:
        """
        🆕 CORRECTED: Start validation workflow using Tesla T4 constrained queues.
        """

        # Enhance data with validation context
        validation_data = {
            "query": data.get("query"),
            "query_mode": data.get("query_mode", "facts"),
            "metadata_filter": data.get("metadata_filter"),
            "validation_workflow": True,
            "original_request": data
        }

        # Start with first task (CPU-based knowledge validation)
        from src.core.validation.tasks.knowledge_validation_task import start_knowledge_validation
        start_knowledge_validation(job_id, validation_data)

    def get_workflow_for_job_type(self, job_type: JobType) -> list:
        """Get the workflow definition for a job type."""
        return self.workflows.get(job_type, [])

    def get_supported_job_types(self) -> list:
        """Get list of supported job types."""
        return list(JobType)

    def get_queue_mapping(self) -> Dict[str, str]:
        """✅ CORRECTED: Get mapping using Tesla T4 constrained queues."""
        return {
            "cpu_tasks": QueueNames.CPU_TASKS.value,
            "transcription_tasks": QueueNames.TRANSCRIPTION_TASKS.value,
            "embedding_tasks": QueueNames.EMBEDDING_TASKS.value,
            "llm_tasks": QueueNames.LLM_TASKS.value  # ✅ RENAMED from inference_tasks
        }

    def validate_queue_configurations(self) -> bool:
        """Validate that all queue configurations respect Tesla T4 constraints."""
        try:
            from src.core.orchestration.dramatiq_helpers import validate_dramatiq_health
            hardware_info = queue_manager.get_hardware_constraints_info()

            logger.info(f"Validating queues for {hardware_info['gpu_constraints']['model']}")
            logger.info(f"GPU Memory: {hardware_info['gpu_constraints']['memory_gb']}GB")
            logger.info(f"Design Principle: {hardware_info['gpu_constraints']['design_principle']}")

            return validate_dramatiq_health()
        except Exception as e:
            logger.error(f"Queue configuration validation failed: {str(e)}")
            return False

    def get_tesla_t4_constraints_info(self) -> Dict[str, Any]:
        """✅ NEW: Get Tesla T4 constraint information for documentation."""
        from src.core.orchestration.queue_manager import queue_manager
        return queue_manager.get_hardware_constraints_info()


# Global task router instance
task_router = TaskRouter()