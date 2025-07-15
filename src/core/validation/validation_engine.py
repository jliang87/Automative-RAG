"""
Validation Pipeline Engine - Master validation orchestrator
Manages the complete 5-phase validation pipeline with proper task orchestration
"""

import logging
import time
import asyncio
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

from .models.validation_models import (
    ValidationContext, ValidationChainResult, PipelineType, ValidationStepType,
    ValidationStepResult, ValidationStatus, ConfidenceBreakdown, ConfidenceLevel
)
from .confidence_calculator import ConfidenceCalculator
from .steps.steps_readiness_checker import MetaValidator

logger = logging.getLogger(__name__)


class ValidationPipelineEngine:
    """
    Master validation pipeline engine orchestrating the complete 5-phase pipeline.
    Manages: Knowledge → Pre-LLM → Main LLM → Post-LLM → Final Assessment
    """

    def __init__(self):
        self.progress_tracker = PipelineProgressTracker()
        self.pipeline_configurator = PipelineConfigurator()
        self.confidence_calculator = ConfidenceCalculator()
        self.meta_validator = MetaValidator()

        # Use existing orchestration components
        from src.core.orchestration.task_router import task_router
        from src.core.orchestration.job_tracker import job_tracker
        self.task_router = task_router
        self.job_tracker = job_tracker

        # Initialize validation step implementations
        self.validation_step_implementations = self._initialize_validation_steps()

    def _initialize_validation_steps(self) -> Dict[ValidationStepType, Any]:
        """Initialize validation step implementations"""

        from .steps.retrieval_quality import RetrievalQualityValidator
        from .steps.source_credibility_validator import SourceCredibilityValidator
        from .steps.technical_consistency_validator import TechnicalConsistencyValidator
        from .steps.completeness_analysis import CompletenessValidator
        from .steps.consensus_analysis import ConsensusValidator
        from .steps.llm_response_quality import LLMResponseQualityValidator

        return {
            ValidationStepType.RETRIEVAL: RetrievalQualityValidator,
            ValidationStepType.SOURCE_CREDIBILITY: SourceCredibilityValidator,
            ValidationStepType.TECHNICAL_CONSISTENCY: TechnicalConsistencyValidator,
            ValidationStepType.COMPLETENESS: CompletenessValidator,
            ValidationStepType.CONSENSUS: ConsensusValidator,
            ValidationStepType.LLM_INFERENCE: LLMResponseQualityValidator
        }

    async def execute_complete_validation_pipeline(self,
                                                   documents: List[Dict],
                                                   query: str,
                                                   query_mode: str,
                                                   job_id: str) -> Dict[str, Any]:
        """Execute the complete 5-phase validation pipeline"""

        try:
            # Initialize pipeline progress
            await self._update_pipeline_progress(job_id, "pipeline.initializing", 0)

            # PHASE 1: Knowledge-Based Validation (CPU Tasks) - 0-25%
            knowledge_results = await self._execute_knowledge_validation_phase(
                job_id, documents, query, query_mode
            )

            # Check if meta-validation needed
            if knowledge_results.get("requires_meta_validation"):
                return await self._handle_meta_validation_routing(job_id, knowledge_results, "knowledge_validation")

            # PHASE 2: Pre-LLM Validation (GPU Task) - 25-40%
            pre_llm_results = await self._execute_pre_llm_validation_phase(
                job_id, documents, query, query_mode, knowledge_results
            )

            # Check if meta-validation needed
            if pre_llm_results.get("requires_meta_validation"):
                return await self._handle_meta_validation_routing(job_id, pre_llm_results, "pre_llm_validation")

            # PHASE 3: Main LLM Inference (GPU Task) - 40-65%
            main_inference_results = await self._execute_main_inference_phase(
                job_id, documents, query, query_mode, pre_llm_results
            )

            # PHASE 4: Post-LLM Validation (GPU Task) - 65-85%
            post_llm_results = await self._execute_post_llm_validation_phase(
                job_id, documents, query, query_mode, main_inference_results, pre_llm_results
            )

            # Check if meta-validation needed
            if post_llm_results.get("requires_meta_validation"):
                return await self._handle_meta_validation_routing(job_id, post_llm_results, "post_llm_validation")

            # PHASE 5: Final Assessment (GPU Task) - 85-100%
            final_assessment = await self._execute_final_assessment_phase(
                job_id, documents, query, query_mode, main_inference_results,
                knowledge_results, pre_llm_results, post_llm_results
            )

            # Complete pipeline
            await self._update_pipeline_progress(job_id, "pipeline.complete", 100)

            return {
                "status": "success",
                "query_mode": query_mode,
                "documents": documents,
                "knowledge_validation": knowledge_results,
                "pre_llm_validation": pre_llm_results,
                "main_response": main_inference_results,
                "post_llm_validation": post_llm_results,
                "final_assessment": final_assessment,
                "pipeline_metadata": self.progress_tracker.get_pipeline_summary(job_id),
                "confidence_score": final_assessment.get("final_confidence", 0.0),
                "overall_status": final_assessment.get("validation_summary", {}).get("overall_status", "WARNING")
            }

        except Exception as e:
            await self._handle_pipeline_error(job_id, e)
            raise

    async def _execute_knowledge_validation_phase(self, job_id, documents, query, query_mode):
        """Execute knowledge-based validation phase (CPU intensive)"""

        await self._update_pipeline_progress(job_id, "knowledge_validation.start", 5)

        # Create validation context
        context = ValidationContext(
            query_id=job_id,
            query_text=query,
            query_mode=query_mode,
            documents=documents
        )

        # Execute knowledge-based validation steps
        pipeline_config = self.pipeline_configurator.get_pipeline_config(query_mode)
        knowledge_steps = []

        for step_config in pipeline_config.get("knowledge_steps", []):
            step_type = ValidationStepType(step_config["step_type"])

            # Get step implementation
            step_class = self.validation_step_implementations.get(step_type)
            if step_class:
                step_instance = step_class(step_config, self.meta_validator)
                step_result = await step_instance.execute(context)
                knowledge_steps.append(step_result)

                # Update progress incrementally
                progress = 5 + (len(knowledge_steps) / len(pipeline_config.get("knowledge_steps", []))) * 20
                await self._update_pipeline_progress(job_id, f"knowledge_validation.step_{step_type.value}", progress)

        # Calculate confidence for knowledge steps
        knowledge_confidence = self.confidence_calculator.calculate_confidence(
            knowledge_steps, pipeline_config.get("confidence_weights", {})
        )

        # Determine if meta-validation is needed
        requires_meta_validation = any(
            step.status.value in ["unverifiable", "failed"] for step in knowledge_steps
        )

        await self._update_pipeline_progress(job_id, "knowledge_validation.complete", 25)

        return {
            "validation_type": "knowledge_based",
            "query_mode": query_mode,
            "knowledge_validation_steps": [
                {
                    "step_name": step.step_name,
                    "step_type": step.step_type.value,
                    "status": step.status.value,
                    "confidence_impact": step.confidence_impact,
                    "warnings": [w.message for w in step.warnings],
                    "duration_ms": step.duration_ms
                }
                for step in knowledge_steps
            ],
            "knowledge_confidence": knowledge_confidence.total_score,
            "requires_meta_validation": requires_meta_validation,
            "meta_validation_opportunities": [
                step.contribution_prompt.__dict__ if step.contribution_prompt else None
                for step in knowledge_steps
            ]
        }

    async def _execute_pre_llm_validation_phase(self, job_id, documents, query, query_mode, knowledge_results):
        """Execute pre-LLM validation phase (GPU intensive)"""

        await self._update_pipeline_progress(job_id, "pre_llm_validation.start", 25)

        # Prepare pre-LLM validation task data
        task_data = {
            "validation_type": "pre_llm_phase",
            "phase_type": "pre_validation",
            "documents": documents,
            "query": query,
            "query_mode": query_mode,
            "previous_results": knowledge_results
        }

        # Route to GPU task using existing task_router
        task_id = f"{job_id}_pre_llm"
        self.task_router.route_task(
            task_id,
            "llm_phase_validation",
            "inference_tasks",
            task_data
        )

        # Wait for completion
        pre_llm_results = await self._wait_for_task_completion(task_id)

        await self._update_pipeline_progress(job_id, "pre_llm_validation.complete", 40)
        return pre_llm_results

    async def _execute_main_inference_phase(self, job_id, documents, query, query_mode, pre_llm_results):
        """Execute main LLM inference phase (GPU intensive)"""

        await self._update_pipeline_progress(job_id, "main_inference.start", 40)

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
        task_id = f"{job_id}_main_inference"
        self.task_router.route_task(
            task_id,
            "main_llm_inference",
            "inference_tasks",
            inference_data
        )

        # Wait for completion
        main_response = await self._wait_for_task_completion(task_id)

        await self._update_pipeline_progress(job_id, "main_inference.complete", 65)
        return main_response

    async def _execute_post_llm_validation_phase(self, job_id, documents, query, query_mode,
                                                main_response, pre_llm_results):
        """Execute post-LLM validation phase (GPU intensive)"""

        await self._update_pipeline_progress(job_id, "post_llm_validation.start", 65)

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

        task_id = f"{job_id}_post_llm"
        self.task_router.route_task(
            task_id,
            "llm_phase_validation",
            "inference_tasks",
            task_data
        )

        post_llm_results = await self._wait_for_task_completion(task_id)

        await self._update_pipeline_progress(job_id, "post_llm_validation.complete", 85)
        return post_llm_results

    async def _execute_final_assessment_phase(self, job_id, documents, query, query_mode,
                                             main_response, knowledge_results, pre_llm_results, post_llm_results):
        """Execute final validation assessment phase (GPU intensive)"""

        await self._update_pipeline_progress(job_id, "final_assessment.start", 85)

        task_data = {
            "validation_type": "final_assessment",
            "phase_type": "final_assessment",
            "documents": documents,
            "query": query,
            "query_mode": query_mode,
            "previous_results": {
                "knowledge_validation": knowledge_results,
                "pre_llm_validation": pre_llm_results,
                "main_response": main_response,
                "post_llm_validation": post_llm_results
            }
        }

        task_id = f"{job_id}_final"
        self.task_router.route_task(
            task_id,
            "llm_phase_validation",
            "inference_tasks",
            task_data
        )

        final_assessment = await self._wait_for_task_completion(task_id)

        await self._update_pipeline_progress(job_id, "final_assessment.complete", 100)
        return final_assessment

    async def _handle_meta_validation_routing(self, job_id: str, failed_results: Dict[str, Any], failed_stage: str):
        """Handle meta-validation when validation steps fail"""

        await self._update_pipeline_progress(job_id, "meta_validation.analyzing", None)

        # Prepare meta-validation task data
        meta_task_data = {
            "validation_type": "meta_validation",
            "failed_stage": failed_stage,
            "failure_context": failed_results,
            "failed_step": failed_stage,
            "query_mode": failed_results.get("query_mode", "facts")
        }

        # Route to meta-validation task
        task_id = f"{job_id}_meta"
        self.task_router.route_task(
            task_id,
            "meta_validation",
            "cpu_tasks",
            meta_task_data
        )

        # Wait for meta-validation completion (sets up user choice)
        meta_results = await self._wait_for_task_completion(task_id)

        return {
            "status": "awaiting_user_input",
            "meta_validation": meta_results,
            "failed_stage": failed_stage,
            "user_choice_required": True
        }

    async def process_user_choice(self, job_id: str, user_choice: str, user_data: Optional[Dict[str, Any]] = None):
        """Process user choice for meta-validation"""

        logger.info(f"Pipeline Engine: Processing user choice '{user_choice}' for job {job_id}")

        if user_choice == "auto_fetch":
            return await self._handle_auto_fetch_choice(job_id, user_data)
        elif user_choice == "user_guidance":
            return await self._handle_user_guidance_choice(job_id, user_data)
        elif user_choice == "restart_full":
            return await self._handle_restart_choice(job_id, user_data)
        else:
            raise ValueError(f"Unknown user choice: {user_choice}")

    async def _handle_auto_fetch_choice(self, job_id: str, user_data: Optional[Dict[str, Any]]):
        """Handle auto-fetch user choice"""

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
        task_id = f"{job_id}_auto_fetch"
        self.task_router.route_task(
            task_id,
            "auto_fetch",
            "cpu_tasks",
            auto_fetch_data
        )

        # Wait for auto-fetch completion
        auto_fetch_results = await self._wait_for_task_completion(task_id)

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
        """Handle user guidance choice"""

        # Process user-provided sources/guidance
        from .meta_validation.user_guidance.guided_trust_loop import contribution_handler

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
        """Handle restart validation choice"""

        # Clear any cached validation state
        # Restart the validation pipeline from the beginning

        return {
            "success": True,
            "action": "pipeline_restarted",
            "message": "Validation pipeline restarted from beginning",
            "new_job_id": f"{job_id}_restart"
        }

    async def _wait_for_task_completion(self, task_id: str) -> Dict[str, Any]:
        """Wait for task completion using existing job_tracker"""

        max_wait_time = 300  # 5 minutes max wait
        check_interval = 0.5  # Check every 500ms
        waited_time = 0

        while waited_time < max_wait_time:
            job_data = self.job_tracker.get_job(task_id)
            if job_data:
                status = job_data.get("status")
                if status == "completed":
                    return job_data.get("result", {})
                elif status == "failed":
                    error = job_data.get("error", "Unknown error")
                    raise ValidationPipelineError(f"Task {task_id} failed: {error}")
                elif status == "awaiting_user_input":
                    # For meta-validation, this is expected
                    return job_data.get("result", {})

            await asyncio.sleep(check_interval)
            waited_time += check_interval

        raise ValidationPipelineError(f"Task {task_id} timed out after {max_wait_time} seconds")

    async def _update_pipeline_progress(self, job_id: str, stage_key: str, percentage: Optional[int]):
        """Update pipeline progress for tracking"""

        # Update job progress for UI polling
        if percentage is not None:
            self.job_tracker.update_job_progress(job_id, percentage, stage_key)

        # Update job status with detailed information
        self.job_tracker.update_job_status(
            job_id,
            "processing",
            result={
                "current_stage": stage_key,
                "progress_percentage": percentage,
                "timestamp": time.time(),
                "pipeline_phase": stage_key.split(".")[0] if "." in stage_key else stage_key
            },
            stage=stage_key
        )

    async def _handle_pipeline_error(self, job_id: str, error: Exception):
        """Handle pipeline execution errors"""

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

    def get_pipeline_statistics(self) -> Dict[str, Any]:
        """Get pipeline engine statistics"""

        return {
            "pipeline_phases": 5,
            "validation_steps_available": len(self.validation_step_implementations),
            "pipeline_configurations": len(self.pipeline_configurator.get_available_configurations()),
            "engine_status": "operational"
        }


class PipelineConfigurator:
    """
    Configures validation pipelines based on query mode and type
    """

    def __init__(self):
        self.pipeline_configurations = self._initialize_pipeline_configurations()

    def _initialize_pipeline_configurations(self) -> Dict[str, Dict[str, Any]]:
        """Initialize pipeline configurations for different query modes"""

        return {
            "facts": {
                "knowledge_steps": [
                    {"step_type": "retrieval", "weight": 0.2},
                    {"step_type": "source_credibility", "weight": 0.4},
                    {"step_type": "technical_consistency", "weight": 0.4}
                ],
                "confidence_weights": {
                    ValidationStepType.RETRIEVAL: 0.15,
                    ValidationStepType.SOURCE_CREDIBILITY: 0.4,
                    ValidationStepType.TECHNICAL_CONSISTENCY: 0.3,
                    ValidationStepType.COMPLETENESS: 0.1,
                    ValidationStepType.LLM_INFERENCE: 0.05
                }
            },
            "features": {
                "knowledge_steps": [
                    {"step_type": "retrieval", "weight": 0.2},
                    {"step_type": "source_credibility", "weight": 0.3},
                    {"step_type": "completeness", "weight": 0.5}
                ],
                "confidence_weights": {
                    ValidationStepType.RETRIEVAL: 0.15,
                    ValidationStepType.SOURCE_CREDIBILITY: 0.3,
                    ValidationStepType.TECHNICAL_CONSISTENCY: 0.2,
                    ValidationStepType.COMPLETENESS: 0.25,
                    ValidationStepType.LLM_INFERENCE: 0.1
                }
            },
            "tradeoffs": {
                "knowledge_steps": [
                    {"step_type": "retrieval", "weight": 0.2},
                    {"step_type": "source_credibility", "weight": 0.3},
                    {"step_type": "completeness", "weight": 0.3},
                    {"step_type": "consensus", "weight": 0.2}
                ],
                "confidence_weights": {
                    ValidationStepType.RETRIEVAL: 0.1,
                    ValidationStepType.SOURCE_CREDIBILITY: 0.25,
                    ValidationStepType.TECHNICAL_CONSISTENCY: 0.2,
                    ValidationStepType.COMPLETENESS: 0.25,
                    ValidationStepType.CONSENSUS: 0.15,
                    ValidationStepType.LLM_INFERENCE: 0.05
                }
            },
            "scenarios": {
                "knowledge_steps": [
                    {"step_type": "retrieval", "weight": 0.3},
                    {"step_type": "completeness", "weight": 0.4},
                    {"step_type": "consensus", "weight": 0.3}
                ],
                "confidence_weights": {
                    ValidationStepType.RETRIEVAL: 0.2,
                    ValidationStepType.SOURCE_CREDIBILITY: 0.2,
                    ValidationStepType.TECHNICAL_CONSISTENCY: 0.15,
                    ValidationStepType.COMPLETENESS: 0.3,
                    ValidationStepType.CONSENSUS: 0.1,
                    ValidationStepType.LLM_INFERENCE: 0.05
                }
            },
            "debate": {
                "knowledge_steps": [
                    {"step_type": "retrieval", "weight": 0.2},
                    {"step_type": "source_credibility", "weight": 0.3},
                    {"step_type": "consensus", "weight": 0.5}
                ],
                "confidence_weights": {
                    ValidationStepType.RETRIEVAL: 0.1,
                    ValidationStepType.SOURCE_CREDIBILITY: 0.3,
                    ValidationStepType.TECHNICAL_CONSISTENCY: 0.15,
                    ValidationStepType.COMPLETENESS: 0.2,
                    ValidationStepType.CONSENSUS: 0.2,
                    ValidationStepType.LLM_INFERENCE: 0.05
                }
            },
            "quotes": {
                "knowledge_steps": [
                    {"step_type": "retrieval", "weight": 0.3},
                    {"step_type": "source_credibility", "weight": 0.4},
                    {"step_type": "completeness", "weight": 0.3}
                ],
                "confidence_weights": {
                    ValidationStepType.RETRIEVAL: 0.2,
                    ValidationStepType.SOURCE_CREDIBILITY: 0.4,
                    ValidationStepType.TECHNICAL_CONSISTENCY: 0.1,
                    ValidationStepType.COMPLETENESS: 0.2,
                    ValidationStepType.CONSENSUS: 0.05,
                    ValidationStepType.LLM_INFERENCE: 0.05
                }
            }
        }

    def get_pipeline_config(self, query_mode: str) -> Dict[str, Any]:
        """Get pipeline configuration for query mode"""

        return self.pipeline_configurations.get(query_mode, self.pipeline_configurations["facts"])

    def get_available_configurations(self) -> List[str]:
        """Get available pipeline configuration names"""

        return list(self.pipeline_configurations.keys())


class PipelineProgressTracker:
    """
    Tracks pipeline progress for UI display
    """

    def __init__(self):
        self.pipeline_summaries = {}

    def get_pipeline_summary(self, job_id: str) -> Dict[str, Any]:
        """Get pipeline execution summary"""

        return self.pipeline_summaries.get(job_id, {
            "total_phases": 5,
            "completed_phases": 0,
            "validation_quality": "unknown",
            "confidence_trend": []
        })


class ValidationPipelineError(Exception):
    """Exception raised when validation pipeline fails"""
    pass


# Global pipeline engine instance
validation_pipeline_engine = ValidationPipelineEngine()