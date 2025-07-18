"""
Enhanced Task Router - Better validation workflow integration and routing
"""

import logging
from typing import Dict, Any
from enum import Enum
import time

from src.core.orchestration.queue_manager import QueueNames

logger = logging.getLogger(__name__)


class JobType(Enum):
    VIDEO_PROCESSING = "video_processing"
    PDF_PROCESSING = "pdf_processing"
    TEXT_PROCESSING = "text_processing"
    LLM_INFERENCE = "llm_inference"
    COMPREHENSIVE_VALIDATION = "comprehensive_validation"


class TaskRouter:
    """
    Enhanced task router with better validation workflow support.
    """

    def __init__(self):
        # Enhanced workflows with better validation integration
        self.workflows = {
            JobType.VIDEO_PROCESSING: [
                ("download_video", QueueNames.CPU_TASKS.value),
                ("transcribe_video", QueueNames.TRANSCRIPTION_TASKS.value),
                ("generate_embeddings", QueueNames.EMBEDDING_TASKS.value)
            ],
            JobType.PDF_PROCESSING: [
                ("process_pdf", QueueNames.CPU_TASKS.value),
                ("generate_embeddings", QueueNames.EMBEDDING_TASKS.value)
            ],
            JobType.TEXT_PROCESSING: [
                ("process_text", QueueNames.CPU_TASKS.value),
                ("generate_embeddings", QueueNames.EMBEDDING_TASKS.value)
            ],
            JobType.LLM_INFERENCE: [
                ("retrieve_documents", QueueNames.EMBEDDING_TASKS.value),
                ("llm_inference", QueueNames.LLM_TASKS.value)
            ],
            # ✅ ENHANCED: Better validation workflow
            JobType.COMPREHENSIVE_VALIDATION: [
                ("knowledge_validation", QueueNames.CPU_TASKS.value),
                ("pre_llm_validation", QueueNames.LLM_TASKS.value),
                ("retrieve_documents", QueueNames.EMBEDDING_TASKS.value),
                ("llm_inference", QueueNames.LLM_TASKS.value),
                ("post_llm_validation", QueueNames.LLM_TASKS.value),
                ("final_validation", QueueNames.LLM_TASKS.value)
                # Note: meta_validation, auto_fetch, user_guidance steps are inserted dynamically
            ]
        }

    def route_task(self, job_id: str, task_name: str, queue_name: str, data: Dict[str, Any]) -> None:
        """Enhanced task routing with better validation support."""
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

            # ✅ ENHANCED: Better validation task routing
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
            elif task_name == "process_user_contribution":
                self._route_user_contribution(job_id, data)

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

    # ===== EXISTING ROUTING METHODS - NO CHANGES =====

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

    # ===== ENHANCED VALIDATION ROUTING METHODS =====

    def _route_knowledge_validation(self, job_id: str, data: Dict[str, Any]) -> None:
        """✅ ENHANCED: Route knowledge validation with better error handling."""
        try:
            from src.core.validation.tasks.knowledge_validation_task import knowledge_validation_task

            # Validate required data
            if "query" not in data:
                raise ValueError("query required for knowledge validation")

            # Prepare validation context
            validation_data = self._prepare_validation_context(job_id, data, "knowledge_validation")

            knowledge_validation_task.send(job_id, validation_data)
            logger.info(f"✅ Knowledge validation routed for job {job_id}")

        except ImportError as e:
            logger.error(f"Knowledge validation task not available: {str(e)}")
            self._handle_missing_validation_task(job_id, "knowledge_validation", str(e))
        except Exception as e:
            logger.error(f"Error routing knowledge validation: {str(e)}")
            raise

    def _route_pre_llm_validation(self, job_id: str, data: Dict[str, Any]) -> None:
        """✅ ENHANCED: Route pre-LLM validation."""
        try:
            from src.core.validation.tasks.llm_phase_validation_task import llm_phase_validation_task

            validation_data = self._prepare_validation_context(job_id, data, "pre_llm_validation")
            validation_data["phase_type"] = "pre_validation"

            llm_phase_validation_task.send(job_id, validation_data)
            logger.info(f"✅ Pre-LLM validation routed for job {job_id}")

        except ImportError as e:
            logger.error(f"LLM phase validation task not available: {str(e)}")
            self._handle_missing_validation_task(job_id, "pre_llm_validation", str(e))
        except Exception as e:
            logger.error(f"Error routing pre-LLM validation: {str(e)}")
            raise

    def _route_post_llm_validation(self, job_id: str, data: Dict[str, Any]) -> None:
        """✅ ENHANCED: Route post-LLM validation."""
        try:
            from src.core.validation.tasks.llm_phase_validation_task import llm_phase_validation_task

            validation_data = self._prepare_validation_context(job_id, data, "post_llm_validation")
            validation_data["phase_type"] = "post_validation"

            # Include previous results for post-validation
            validation_data["previous_results"] = data

            llm_phase_validation_task.send(job_id, validation_data)
            logger.info(f"✅ Post-LLM validation routed for job {job_id}")

        except ImportError as e:
            logger.error(f"LLM phase validation task not available: {str(e)}")
            self._handle_missing_validation_task(job_id, "post_llm_validation", str(e))
        except Exception as e:
            logger.error(f"Error routing post-LLM validation: {str(e)}")
            raise

    def _route_final_validation(self, job_id: str, data: Dict[str, Any]) -> None:
        """✅ ENHANCED: Route final validation assessment."""
        try:
            from src.core.validation.tasks.llm_phase_validation_task import llm_phase_validation_task

            validation_data = self._prepare_validation_context(job_id, data, "final_validation")
            validation_data["phase_type"] = "final_assessment"

            # Include all previous results for final assessment
            validation_data["previous_results"] = data

            llm_phase_validation_task.send(job_id, validation_data)
            logger.info(f"✅ Final validation routed for job {job_id}")

        except ImportError as e:
            logger.error(f"LLM phase validation task not available: {str(e)}")
            self._handle_missing_validation_task(job_id, "final_validation", str(e))
        except Exception as e:
            logger.error(f"Error routing final validation: {str(e)}")
            raise

    def _route_meta_validation(self, job_id: str, data: Dict[str, Any]) -> None:
        """✅ ENHANCED: Route meta-validation with failure context."""
        try:
            from src.core.validation.tasks.meta_validation_task import meta_validation_task

            # Prepare meta-validation context with failure information
            meta_data = self._prepare_validation_context(job_id, data, "meta_validation")

            # Extract failure context from previous validation steps
            failure_context = self._extract_failure_context(data)
            meta_data["failure_context"] = failure_context
            meta_data["failed_step"] = failure_context.get("failed_step", "unknown")

            meta_validation_task.send(job_id, meta_data)
            logger.info(f"✅ Meta-validation routed for job {job_id} with failure context")

        except ImportError as e:
            logger.error(f"Meta-validation task not available: {str(e)}")
            self._handle_missing_validation_task(job_id, "meta_validation", str(e))
        except Exception as e:
            logger.error(f"Error routing meta-validation: {str(e)}")
            raise

    def _route_auto_fetch(self, job_id: str, data: Dict[str, Any]) -> None:
        """✅ ENHANCED: Route auto-fetch with target specification."""
        try:
            from src.core.validation.tasks.auto_fetch_task import auto_fetch_task

            # Prepare auto-fetch context
            fetch_data = {
                "fetch_targets": data.get("fetch_targets", ["official_specs", "epa_data"]),
                "query_context": self._extract_query_context(data),
                "failed_step": data.get("failed_step", "unknown"),
                "query": data.get("query", ""),
                "query_mode": data.get("query_mode", "facts")
            }

            auto_fetch_task.send(job_id, fetch_data)
            logger.info(f"✅ Auto-fetch routed for job {job_id}")

        except ImportError as e:
            logger.error(f"Auto-fetch task not available: {str(e)}")
            self._handle_missing_validation_task(job_id, "auto_fetch", str(e))
        except Exception as e:
            logger.error(f"Error routing auto-fetch: {str(e)}")
            raise

    def _route_user_contribution(self, job_id: str, data: Dict[str, Any]) -> None:
        """✅ NEW: Route user contribution processing."""
        try:
            from src.core.query.tasks.inference_tasks import process_user_contribution_task

            # Extract user contribution data
            user_guidance = data.get("user_guidance", {})
            failed_step = data.get("failed_step", "unknown")

            process_user_contribution_task.send(job_id, failed_step, user_guidance)
            logger.info(f"✅ User contribution processing routed for job {job_id}")

        except ImportError as e:
            logger.error(f"User contribution task not available: {str(e)}")
            self._handle_missing_validation_task(job_id, "process_user_contribution", str(e))
        except Exception as e:
            logger.error(f"Error routing user contribution: {str(e)}")
            raise

    # ===== HELPER METHODS FOR VALIDATION ROUTING =====

    def _prepare_validation_context(self, job_id: str, data: Dict[str, Any], validation_type: str) -> Dict[str, Any]:
        """Prepare standardized validation context."""
        return {
            "query": data.get("query", ""),
            "query_mode": data.get("query_mode", "facts"),
            "documents": data.get("documents", []),
            "metadata_filter": data.get("metadata_filter"),
            "validation_workflow": True,
            "validation_type": validation_type,
            "job_id": job_id,
            "original_request": data
        }

    def _extract_failure_context(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract failure context from validation results."""
        failure_context = {
            "primary_reason": "Validation step could not complete",
            "failed_step": "unknown",
            "confidence_score": 0.0,
            "missing_resources": []
        }

        # Extract from knowledge validation results
        if "knowledge_validation_steps" in data:
            knowledge_steps = data["knowledge_validation_steps"]
            failed_steps = [step for step in knowledge_steps if step.get("status") in ["failed", "unverifiable"]]

            if failed_steps:
                failure_context["failed_step"] = failed_steps[0].get("step_name", "unknown")
                failure_context["primary_reason"] = f"Knowledge validation failed: {failed_steps[0].get('step_name')}"

                # Extract warnings as missing resources
                for step in failed_steps:
                    failure_context["missing_resources"].extend(step.get("warnings", []))

        # Extract confidence scores
        if "knowledge_confidence" in data:
            failure_context["confidence_score"] = data["knowledge_confidence"]
        elif "final_confidence" in data:
            failure_context["confidence_score"] = data["final_confidence"]

        return failure_context

    def _extract_query_context(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract query context for auto-fetch."""
        query_context = {
            "query": data.get("query", ""),
            "query_mode": data.get("query_mode", "facts")
        }

        # Try to extract vehicle information from query or metadata
        query = data.get("query", "").lower()

        # Simple vehicle extraction (could be enhanced)
        import re

        # Extract year
        year_match = re.search(r'\b(19|20)\d{2}\b', query)
        if year_match:
            query_context["year"] = year_match.group()

        # Extract common manufacturers
        manufacturers = ["toyota", "honda", "ford", "bmw", "mercedes", "audi", "volkswagen", "nissan", "hyundai"]
        for manufacturer in manufacturers:
            if manufacturer in query:
                query_context["manufacturer"] = manufacturer
                break

        return query_context

    def _handle_missing_validation_task(self, job_id: str, task_name: str, error: str) -> None:
        """Handle missing validation task gracefully."""
        logger.warning(f"Validation task {task_name} not available for job {job_id}, skipping")

        # Create a placeholder result and continue workflow
        placeholder_result = {
            "validation_type": task_name,
            "status": "SKIPPED",
            "reason": f"Validation task not available: {error}",
            "confidence": 0.5,  # Neutral confidence
            "skipped_at": time.time()
        }

        from src.core.orchestration.job_chain import job_chain
        job_chain.task_completed(job_id, placeholder_result)

    # ===== WORKFLOW MANAGEMENT =====

    def start_job_workflow(self, job_id: str, job_type: JobType, data: Dict[str, Any]) -> None:
        """Enhanced workflow start with validation support."""
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

            elif job_type == JobType.LLM_INFERENCE:
                from src.core.query.tasks.retrieval_tasks import start_document_retrieval
                start_document_retrieval(job_id, data)

            # ✅ ENHANCED: Better validation workflow
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
        """✅ ENHANCED: Start validation workflow with better context preparation."""

        # Validate required data for validation workflow
        required_fields = ["query"]
        missing_fields = [field for field in required_fields if field not in data]

        if missing_fields:
            error_msg = f"Missing required fields for validation workflow: {missing_fields}"
            logger.error(error_msg)
            from src.core.orchestration.job_chain import job_chain
            job_chain.task_failed(job_id, error_msg)
            return

        # Enhance data with validation context
        validation_data = {
            "query": data.get("query"),
            "query_mode": data.get("query_mode", "facts"),
            "metadata_filter": data.get("metadata_filter"),
            "validation_workflow": True,
            "validation_started_at": time.time(),
            "original_request": data
        }

        # Start with first task (knowledge validation)
        try:
            from src.core.validation.tasks.knowledge_validation_task import start_knowledge_validation
            start_knowledge_validation(job_id, validation_data)
            logger.info(f"✅ Validation workflow started for job {job_id}")
        except ImportError as e:
            logger.error(f"Knowledge validation not available: {str(e)}")
            self._handle_missing_validation_task(job_id, "knowledge_validation", str(e))

    # ===== EXISTING METHODS (unchanged) =====

    def get_workflow_for_job_type(self, job_type: JobType) -> list:
        """Get the workflow definition for a job type."""
        return self.workflows.get(job_type, [])

    def get_supported_job_types(self) -> list:
        """Get list of supported job types."""
        return list(JobType)

    def get_queue_mapping(self) -> Dict[str, str]:
        """Get mapping using Tesla T4 constrained queues."""
        return {
            "cpu_tasks": QueueNames.CPU_TASKS.value,
            "transcription_tasks": QueueNames.TRANSCRIPTION_TASKS.value,
            "embedding_tasks": QueueNames.EMBEDDING_TASKS.value,
            "llm_tasks": QueueNames.LLM_TASKS.value
        }

    def validate_queue_configurations(self) -> bool:
        """Validate that all queue configurations respect Tesla T4 constraints."""
        try:
            from src.core.orchestration.queue_manager import queue_manager
            return queue_manager.validate_dramatiq_health()
        except Exception as e:
            logger.error(f"Queue configuration validation failed: {str(e)}")
            return False

    def get_tesla_t4_constraints_info(self) -> Dict[str, Any]:
        """Get Tesla T4 constraint information for documentation."""
        from src.core.orchestration.queue_manager import queue_manager
        return queue_manager.get_hardware_constraints_info()


# Global task router instance
task_router = TaskRouter()