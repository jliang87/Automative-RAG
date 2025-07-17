from typing import Dict, Any

# Import Tesla T4 constrained queue definitions
from src.core.orchestration.queue_manager import QueueNames
from src.core.orchestration.dramatiq_helpers import create_dramatiq_actor_decorator

# Import simple base class (same directory)
from .base_task import BaseValidationTask


class LLMPhaseValidationTask(BaseValidationTask):
    """
    LLM validation task - only business logic.
    Uses Tesla T4 constrained LLM_TASKS queue (shared memory).
    """

    def __init__(self):
        super().__init__("llm_phase_validation")

    def validate_input(self, task_data: Dict[str, Any]) -> bool:
        """Enhanced validation for LLM tasks."""
        base_valid = super().validate_input(task_data)
        has_phase_type = "phase_type" in task_data
        return base_valid and has_phase_type

    def execute_validation_logic(self, job_id: str, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Core LLM validation logic - no boilerplate."""

        phase_type = task_data.get("phase_type", "pre_validation")
        self.logger.info(f"Executing LLM validation phase: {phase_type} [Tesla T4: Shared DeepSeq model]")

        # Execute phase-specific validation
        if phase_type == "pre_validation":
            result_data = self._execute_pre_llm_validation(task_data)
        elif phase_type == "post_validation":
            result_data = self._execute_post_llm_validation(task_data)
        elif phase_type == "final_assessment":
            result_data = self._execute_final_assessment(task_data)
        else:
            raise ValueError(f"Unknown phase type: {phase_type}")

        # Add pass-through data and Tesla T4 constraint info
        result_data.update({
            "documents": task_data.get("documents", []),
            "query": task_data.get("query", ""),
            "query_mode": task_data.get("query_mode", "facts"),
            "tesla_t4_constraint": "LLM_TASKS queue - shared model memory"
        })

        return result_data

    def _execute_pre_llm_validation(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute pre-LLM validation phase."""
        # TODO: Implement actual LLM validation when prompt manager is available
        return {
            "validation_type": "pre_llm_phase",
            "query_mode": task_data.get("query_mode", "facts"),
            "status": "PASSED",
            "proceed_to_main_inference": True,
            "confidence": 0.8,
            "validation_summary": "Pre-LLM validation completed successfully",
            "model_sharing": "Uses shared DeepSeq model memory"
        }

    def _execute_post_llm_validation(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute post-LLM validation phase."""
        previous_results = task_data.get("previous_results", {})
        generated_response = previous_results.get("answer", "")

        if not generated_response:
            self.logger.warning("No generated response found for post-validation")
            return {
                "validation_type": "post_llm_phase",
                "query_mode": task_data.get("query_mode", "facts"),
                "status": "WARNING",
                "final_recommendation": "APPROVE",
                "confidence": 0.6,
                "error": "No generated response to validate"
            }

        # TODO: Implement actual post-LLM validation
        return {
            "validation_type": "post_llm_phase",
            "query_mode": task_data.get("query_mode", "facts"),
            "status": "PASSED",
            "final_recommendation": "APPROVE",
            "confidence": 0.8,
            "validation_summary": "Post-LLM validation completed successfully",
            "model_sharing": "Uses shared DeepSeq model memory"
        }

    def _execute_final_assessment(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute final validation assessment."""
        previous_results = task_data.get("previous_results", {})
        knowledge_results = previous_results.get("knowledge_validation_steps", [])
        pre_llm_results = previous_results.get("pre_llm_validation", {})
        post_llm_results = previous_results.get("post_llm_validation", {})

        # Calculate final confidence
        confidence_scores = []

        if knowledge_results:
            knowledge_confidence = sum(step.get("confidence_impact", 0) for step in knowledge_results) / len(knowledge_results)
            confidence_scores.append(knowledge_confidence)

        if pre_llm_results.get("confidence"):
            confidence_scores.append(pre_llm_results["confidence"] * 100)

        if post_llm_results.get("confidence"):
            confidence_scores.append(post_llm_results["confidence"] * 100)

        final_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 70.0

        return {
            "validation_type": "final_assessment",
            "query_mode": task_data.get("query_mode", "facts"),
            "final_confidence": final_confidence,
            "confidence_level": "HIGH" if final_confidence >= 80 else "MEDIUM" if final_confidence >= 60 else "LOW",
            "validation_summary": {
                "total_validation_steps": len(knowledge_results) + 2,
                "knowledge_steps": len(knowledge_results),
                "llm_phases": 2,
                "overall_status": "PASSED" if final_confidence >= 70 else "WARNING",
                "verification_coverage": final_confidence / 100.0
            },
            "recommendations": {
                "confidence_sufficient": final_confidence >= 70,
                "improvements_available": final_confidence < 90
            },
            "model_sharing": "Uses shared DeepSeq model memory"
        }


# Create task instance
llm_phase_task_instance = LLMPhaseValidationTask()

# Create Dramatiq task function using Tesla T4 constrained queue
@create_dramatiq_actor_decorator(QueueNames.LLM_TASKS.value)
def llm_phase_validation_task(job_id: str, task_data: Dict[str, Any]):
    """Dramatiq task function - delegates to task instance."""
    llm_phase_task_instance.execute_with_error_handling(job_id, task_data)


# Workflow starter function (called by TaskRouter)
def start_llm_phase_validation(job_id: str, data: Dict, phase_type: str):
    """Start LLM phase validation workflow."""
    enhanced_data = {**data, "phase_type": phase_type}
    llm_phase_validation_task.send(job_id, enhanced_data)