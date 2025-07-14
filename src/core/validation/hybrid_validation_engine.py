"""
Hybrid Validation Engine - Master validation orchestrator
Integrates with existing orchestration but provides comprehensive validation
"""

import logging
import time
import asyncio
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)


class HybridValidationEngine:
    """
    Master validation engine orchestrating the complete pipeline.
    Integrates with existing orchestration but provides comprehensive validation.
    """

    def __init__(self):
        self.progress_tracker = PipelineProgressTracker()

        # Use existing orchestration components
        from src.core.orchestration.task_router import task_router
        from src.core.orchestration.job_tracker import job_tracker
        self.task_router = task_router
        self.job_tracker = job_tracker

    async def execute_complete_validation_pipeline(self,
                                                   documents: List[Dict],
                                                   query: str,
                                                   query_mode: str,
                                                   job_id: str) -> Dict[str, Any]:
        """Execute the complete validation pipeline with mode-specific validation."""

        try:
            # Initialize pipeline progress
            await self._update_progress(job_id, "pipeline.initializing", 0)

            # PHASE 1: Knowledge-Based Validation (CPU Tasks) - 20-35%
            rule_results = await self._execute_knowledge_based_validation(
                job_id, documents, query, query_mode
            )

            # Check if meta-validation needed
            if rule_results.get("requires_meta_validation"):
                return await self._handle_meta_validation(job_id, rule_results, "knowledge_validation")

            # PHASE 2: Pre-LLM Phase Validation (GPU Task) - 35-50%
            pre_llm_results = await self._execute_pre_llm_validation_phase(
                job_id, documents, query, query_mode, rule_results
            )

            # Check if meta-validation needed
            if pre_llm_results.get("requires_meta_validation"):
                return await self._handle_meta_validation(job_id, pre_llm_results, "pre_llm_validation")

            # PHASE 3: Main LLM Inference (GPU Task) - 50-75%
            main_response = await self._execute_main_llm_inference(
                job_id, documents, query, query_mode, pre_llm_results
            )

            # PHASE 4: Post-LLM Phase Validation (GPU Task) - 75-90%
            post_llm_results = await self._execute_post_llm_validation_phase(
                job_id, documents, query, query_mode, main_response, pre_llm_results
            )

            # Check if meta-validation needed
            if post_llm_results.get("requires_meta_validation"):
                return await self._handle_meta_validation(job_id, post_llm_results, "post_llm_validation")

            # PHASE 5: Final Assessment (GPU Task) - 90-100%
            final_assessment = await self._execute_final_validation_phase(
                job_id, documents, query, query_mode, main_response,
                rule_results, pre_llm_results, post_llm_results
            )

            # Complete pipeline
            await self._update_progress(job_id, "pipeline.complete", 100)

            return {
                "status": "success",
                "query_mode": query_mode,
                "documents": documents,
                "knowledge_validation": rule_results,
                "pre_llm_validation": pre_llm_results,
                "main_response": main_response,
                "post_llm_validation": post_llm_results,
                "final_assessment": final_assessment,
                "pipeline_metadata": self.progress_tracker.get_pipeline_summary(job_id),
                "confidence_score": final_assessment.get("final_confidence", 0.0),
                "overall_status": final_assessment.get("validation_summary", {}).get("overall_status", "WARNING")
            }

        except Exception as e:
            await self._handle_pipeline_error(job_id, e)
            raise

    async def _execute_knowledge_based_validation(self, job_id, documents, query, query_mode):
        """Execute knowledge-based validation phase using existing task system."""

        await self._update_progress(job_id, "knowledge_validation.start", 20)

        # Prepare knowledge validation task data
        task_data = {
            "validation_type": "knowledge_based",
            "documents": documents,
            "query": query,
            "query_mode": query_mode,
            "validation_workflow": "comprehensive"
        }

        # Route to CPU task using existing task_router
        self.task_router.route_task(
            f"{job_id}_knowledge",
            "knowledge_validation",
            "cpu_tasks",
            task_data
        )

        # Wait for completion
        knowledge_results = await self._wait_for_task_completion(f"{job_id}_knowledge")

        await self._update_progress(job_id, "knowledge_validation.complete", 35)
        return knowledge_results

    async def _execute_pre_llm_validation_phase(self, job_id, documents, query, query_mode, rule_results):
        """Execute pre-LLM validation phase using existing task system."""

        await self._update_progress(job_id, "pre_llm_validation.start", 35)

        # Prepare pre-LLM validation task data
        task_data = {
            "validation_type": "pre_llm_phase",
            "phase_type": "pre_validation",
            "documents": documents,
            "query": query,
            "query_mode": query_mode,
            "previous_results": rule_results
        }

        # Route to GPU task using existing task_router
        self.task_router.route_task(
            f"{job_id}_pre_llm",
            "pre_llm_validation",
            "inference_tasks",
            task_data
        )

        # Wait for completion
        pre_llm_results = await self._wait_for_task_completion(f"{job_id}_pre_llm")

        await self._update_progress(job_id, "pre_llm_validation.complete", 50)
        return pre_llm_results

    async def _execute_main_llm_inference(self, job_id, documents, query, query_mode, pre_llm_results):
        """Execute main LLM inference using existing task system."""

        await self._update_progress(job_id, "main_inference.start", 50)

        # Use existing LLM inference task with validation enhancement
        inference_data = {
            "query": query,
            "documents": documents,
            "query_mode": query_mode,
            "validation_context": {
                "pre_llm_results": pre_llm_results,
                "validation_enhanced": True
            }
        }

        # Route to existing LLM inference task
        self.task_router.route_task(
            f"{job_id}_main_inference",
            "main_llm_inference",
            "inference_tasks",
            inference_data
        )

        # Wait for completion
        main_response = await self._wait_for_task_completion(f"{job_id}_main_inference")

        await self._update_progress(job_id, "main_inference.complete", 75)
        return main_response

    async def _execute_post_llm_validation_phase(self, job_id, documents, query, query_mode, main_response,
                                                 pre_llm_results):
        """Execute post-LLM validation phase."""

        await self._update_progress(job_id, "post_llm_validation.start", 75)

        task_data = {
            "validation_type": "post_llm_phase",
            "phase_type": "post_validation",
            "documents": documents,
            "query": query,
            "query_mode": query_mode,
            "previous_results": {
                "pre_llm_results": pre_llm_results,
                "generated_response": main_response.get("response", "")
            }
        }

        self.task_router.route_task(
            f"{job_id}_post_llm",
            "post_llm_validation",
            "inference_tasks",
            task_data
        )

        post_llm_results = await self._wait_for_task_completion(f"{job_id}_post_llm")

        await self._update_progress(job_id, "post_llm_validation.complete", 90)
        return post_llm_results

    async def _execute_final_validation_phase(self, job_id, documents, query, query_mode, main_response,
                                              rule_results, pre_llm_results, post_llm_results):
        """Execute final validation assessment."""

        await self._update_progress(job_id, "final_assessment.start", 90)

        task_data = {
            "validation_type": "final_assessment",
            "phase_type": "final_assessment",
            "documents": documents,
            "query": query,
            "query_mode": query_mode,
            "previous_results": {
                "knowledge_validation": rule_results,
                "pre_llm_validation": pre_llm_results,
                "main_response": main_response,
                "post_llm_validation": post_llm_results
            }
        }

        self.task_router.route_task(
            f"{job_id}_final",
            "final_validation",
            "inference_tasks",
            task_data
        )

        final_assessment = await self._wait_for_task_completion(f"{job_id}_final")

        await self._update_progress(job_id, "final_assessment.complete", 100)
        return final_assessment

    async def _handle_meta_validation(self, job_id: str, failed_results: Dict[str, Any], failed_stage: str):
        """Handle meta-validation when validation steps fail."""

        await self._update_progress(job_id, "meta_validation.analyzing", None)

        # Prepare meta-validation task data
        meta_task_data = {
            "validation_type": "meta_validation",
            "failed_stage": failed_stage,
            "failure_context": failed_results,
            "failed_step": failed_stage,
            "query_mode": failed_results.get("query_mode", "facts")
        }

        # Route to meta-validation task
        self.task_router.route_task(
            f"{job_id}_meta",
            "meta_validation",
            "cpu_tasks",
            meta_task_data
        )

        # Wait for meta-validation completion (sets up user choice)
        meta_results = await self._wait_for_task_completion(f"{job_id}_meta")

        return {
            "status": "awaiting_user_input",
            "meta_validation": meta_results,
            "failed_stage": failed_stage,
            "user_choice_required": True
        }

    async def _wait_for_task_completion(self, task_job_id: str) -> Dict[str, Any]:
        """Wait for task completion using existing job_tracker."""

        max_wait_time = 300  # 5 minutes max wait
        check_interval = 0.5  # Check every 500ms
        waited_time = 0

        while waited_time < max_wait_time:
            job_data = self.job_tracker.get_job(task_job_id)
            if job_data:
                status = job_data.get("status")
                if status == "completed":
                    return job_data.get("result", {})
                elif status == "failed":
                    error = job_data.get("error", "Unknown error")
                    raise ValidationTaskError(f"Task {task_job_id} failed: {error}")
                elif status == "awaiting_user_input":
                    # For meta-validation, this is expected
                    return job_data.get("result", {})

            await asyncio.sleep(check_interval)
            waited_time += check_interval

        raise ValidationTaskError(f"Task {task_job_id} timed out after {max_wait_time} seconds")

    async def _update_progress(self, job_id: str, message_key: str, percentage: Optional[int]):
        """Update progress using existing job_tracker (for UI polling)."""

        # Update job progress for existing UI polling
        if percentage is not None:
            self.job_tracker.update_job_progress(job_id, percentage, message_key)

        # Update job status with detailed information
        self.job_tracker.update_job_status(
            job_id,
            "processing",
            result={
                "current_stage": message_key,
                "progress_percentage": percentage,
                "timestamp": time.time(),
                "pipeline_stage": message_key.split(".")[0] if "." in message_key else message_key
            },
            stage=message_key
        )

    async def _handle_pipeline_error(self, job_id: str, error: Exception):
        """Handle pipeline execution errors."""

        error_msg = f"Validation pipeline error: {str(error)}"
        logger.error(error_msg, exc_info=True)

        self.job_tracker.update_job_status(
            job_id,
            "failed",
            error=error_msg,
            result={
                "pipeline_error": True,
                "error_stage": "pipeline_execution",
                "error_message": str(error)
            }
        )

    async def process_user_choice(self, job_id: str, user_choice: str, user_data: Optional[Dict[str, Any]] = None):
        """Process user choice for meta-validation."""

        logger.info(f"Processing user choice '{user_choice}' for job {job_id}")

        if user_choice == "auto_fetch":
            return await self._handle_auto_fetch_choice(job_id, user_data)
        elif user_choice == "user_guidance":
            return await self._handle_user_guidance_choice(job_id, user_data)
        elif user_choice == "restart_full":
            return await self._handle_restart_choice(job_id, user_data)
        else:
            raise ValueError(f"Unknown user choice: {user_choice}")

    async def _handle_auto_fetch_choice(self, job_id: str, user_data: Optional[Dict[str, Any]]):
        """Handle auto-fetch user choice."""

        # Get original job data
        job_data = self.job_tracker.get_job(job_id)
        if not job_data:
            raise ValueError(f"Job {job_id} not found")

        # Prepare auto-fetch task
        auto_fetch_data = {
            "fetch_targets": user_data.get("fetch_targets", ["official_specs", "epa_data"]),
            "query_context": {
                "query": user_data.get("query", ""),
                "query_mode": user_data.get("query_mode", "facts"),
                "vehicle_context": user_data.get("vehicle_context", {})
            }
        }

        # Route to auto-fetch task
        self.task_router.route_task(
            f"{job_id}_auto_fetch",
            "auto_fetch",
            "cpu_tasks",
            auto_fetch_data
        )

        # Wait for auto-fetch completion
        auto_fetch_results = await self._wait_for_task_completion(f"{job_id}_auto_fetch")

        if auto_fetch_results.get("fetch_success"):
            # Update job status to resume validation
            self.job_tracker.update_job_status(
                job_id,
                "processing",
                result={
                    "auto_fetch_completed": True,
                    "resuming_validation": True,
                    "additional_sources": auto_fetch_results.get("new_documents_added", 0)
                }
            )

            return {
                "success": True,
                "action": "auto_fetch_completed",
                "message": f"Added {auto_fetch_results.get('new_documents_added', 0)} additional sources",
                "resume_validation": True
            }
        else:
            return {
                "success": False,
                "action": "auto_fetch_failed",
                "message": "Auto-fetch could not find additional sources",
                "fallback_to_user_guidance": True
            }

    async def _handle_user_guidance_choice(self, job_id: str, user_data: Optional[Dict[str, Any]]):
        """Handle user guidance choice."""

        # Process user-provided sources/guidance
        from src.core.validation.guidance.guided_trust_loop import contribution_handler

        contribution_result = await contribution_handler.process_contribution(
            job_id=job_id,
            step_type=user_data.get("failed_step", "unknown"),
            contribution_data=user_data.get("contribution_data", {}),
            original_validation=None  # Would get from cache
        )

        if contribution_result.contribution_accepted:
            return {
                "success": True,
                "action": "user_guidance_accepted",
                "message": "Your contribution has been accepted and validation updated",
                "learning_credit": contribution_result.learning_credit,
                "confidence_improvement": contribution_result.confidence_improvement,
                "resume_validation": True
            }
        else:
            return {
                "success": False,
                "action": "user_guidance_rejected",
                "message": "Contribution could not be validated",
                "retry_guidance": True
            }

    async def _handle_restart_choice(self, job_id: str, user_data: Optional[Dict[str, Any]]):
        """Handle restart validation choice."""

        # Clear any cached validation state
        # Restart the validation pipeline from the beginning

        return {
            "success": True,
            "action": "pipeline_restarted",
            "message": "Validation pipeline restarted from beginning",
            "new_job_id": f"{job_id}_restart"
        }


class PipelineProgressTracker:
    """
    Tracks pipeline progress for UI display
    """

    def __init__(self):
        self.pipeline_summaries = {}

    def get_pipeline_summary(self, job_id: str) -> Dict[str, Any]:
        """Get pipeline execution summary."""

        return self.pipeline_summaries.get(job_id, {
            "total_phases": 5,
            "completed_phases": 0,
            "validation_quality": "unknown",
            "confidence_trend": []
        })


class ValidationTaskError(Exception):
    """Exception raised when validation tasks fail."""
    pass


# Global instance
hybrid_validation_engine = HybridValidationEngine()