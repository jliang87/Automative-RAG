"""
LLM Phase Validation Task (GPU Task)
Executes LLM-based validation phases
"""

import logging
from typing import Dict, Any
from src.core.background.tasks import gpu_bound_task
from src.core.orchestration.job_chain import job_chain

logger = logging.getLogger(__name__)


@gpu_bound_task
def llm_phase_validation_task(job_id: str, task_data: Dict[str, Any]):
    """Execute LLM phase validation (GPU task)."""

    try:
        logger.info(f"Starting LLM phase validation for job {job_id}")

        # Import LLM validation components
        from src.core.validation.validation_engine import validation_engine
        from src.core.validation.models.validation_models import ValidationContext

        query = task_data.get("query", "")
        query_mode = task_data.get("query_mode", "facts")
        documents = task_data.get("documents", [])
        phase_type = task_data.get("phase_type", "pre_validation")
        previous_results = task_data.get("previous_results", {})

        # Create validation context
        context = ValidationContext(
            query_id=job_id,
            query_text=query,
            query_mode=query_mode,
            documents=documents,
            processing_metadata=previous_results
        )

        # Execute phase-specific validation
        if phase_type == "pre_validation":
            result = await _execute_pre_llm_validation(context, query_mode)
        elif phase_type == "post_validation":
            result = await _execute_post_llm_validation(context, query_mode, previous_results)
        elif phase_type == "final_assessment":
            result = await _execute_final_assessment(context, query_mode, previous_results)
        else:
            raise ValueError(f"Unknown phase type: {phase_type}")

        # Report completion
        job_chain.task_completed(job_id, result)
        logger.info(f"LLM phase validation ({phase_type}) completed for job {job_id}")

    except Exception as e:
        error_msg = f"LLM phase validation failed: {str(e)}"
        logger.error(error_msg)
        job_chain.task_failed(job_id, error_msg)


async def _execute_pre_llm_validation(context, query_mode):
    """Execute pre-LLM validation phase."""

    # Import mode-specific prompt templates
    from src.core.validation.prompts.mode_prompt_manager import get_pre_validation_prompt
    from src.core.query.tasks.inference_tasks import call_llm_with_prompt

    # Get mode-specific prompt
    prompt = get_pre_validation_prompt(query_mode, context)

    # Call LLM for pre-validation
    llm_response = await call_llm_with_prompt(prompt, temperature=0.1)

    # Parse structured response
    import json
    try:
        validation_result = json.loads(llm_response)
    except json.JSONDecodeError:
        # Fallback parsing
        validation_result = {"status": "WARNING", "proceed_to_inference": True}

    return {
        "validation_type": "pre_llm_phase",
        "query_mode": query_mode,
        "llm_validation_result": validation_result,
        "status": validation_result.get("overall_assessment", {}).get("status", "WARNING"),
        "proceed_to_main_inference": validation_result.get("overall_assessment", {}).get("proceed_to_inference", True),
        "confidence": validation_result.get("overall_assessment", {}).get("overall_confidence", 0.7),
        "requires_meta_validation": validation_result.get("overall_assessment", {}).get("meta_validation_needed", {})
    }


async def _execute_post_llm_validation(context, query_mode, previous_results):
    """Execute post-LLM validation phase."""

    from src.core.validation.prompts.mode_prompt_manager import get_post_validation_prompt
    from src.core.query.tasks.inference_tasks import call_llm_with_prompt

    # Get generated response from previous results
    generated_response = previous_results.get("generated_response", "")

    # Get mode-specific prompt
    prompt = get_post_validation_prompt(query_mode, context, generated_response)

    # Call LLM for post-validation
    llm_response = await call_llm_with_prompt(prompt, temperature=0.1)

    # Parse structured response
    import json
    try:
        validation_result = json.loads(llm_response)
    except json.JSONDecodeError:
        validation_result = {"status": "WARNING", "final_recommendation": "APPROVE"}

    return {
        "validation_type": "post_llm_phase",
        "query_mode": query_mode,
        "llm_validation_result": validation_result,
        "status": validation_result.get("overall_assessment", {}).get("status", "WARNING"),
        "final_recommendation": validation_result.get("overall_assessment", {}).get("final_recommendation", "APPROVE"),
        "confidence": validation_result.get("overall_assessment", {}).get("overall_confidence", 0.7),
        "corrections_needed": validation_result.get("overall_assessment", {}).get("corrections_needed", [])
    }


async def _execute_final_assessment(context, query_mode, previous_results):
    """Execute final validation assessment."""

    from src.core.validation.confidence_calculator import ConfidenceCalculator

    # Compile all validation results
    knowledge_results = previous_results.get("knowledge_validation_steps", [])
    pre_llm_results = previous_results.get("pre_llm_validation", {})
    post_llm_results = previous_results.get("post_llm_validation", {})

    # Calculate final confidence
    confidence_calculator = ConfidenceCalculator()

    # Create mock validation steps for final calculation
    from src.core.validation.models.validation_models import ValidationStepResult, ValidationStepType, ValidationStatus
    from datetime import datetime

    final_steps = []

    # Add knowledge validation summary
    if knowledge_results:
        avg_knowledge_confidence = sum(step.get("confidence_impact", 0) for step in knowledge_results) / len(
            knowledge_results)
        final_steps.append(ValidationStepResult(
            step_id="knowledge_summary",
            step_type=ValidationStepType.TECHNICAL_CONSISTENCY,
            step_name="Knowledge Validation Summary",
            status=ValidationStatus.PASSED if avg_knowledge_confidence > 0 else ValidationStatus.WARNING,
            confidence_impact=avg_knowledge_confidence,
            summary=f"Knowledge validation completed with {len(knowledge_results)} checks",
            started_at=datetime.now()
        ))

    # Add LLM validation summaries
    if pre_llm_results:
        final_steps.append(ValidationStepResult(
            step_id="pre_llm_summary",
            step_type=ValidationStepType.COMPLETENESS,
            step_name="Pre-LLM Validation Summary",
            status=ValidationStatus.PASSED if pre_llm_results.get("confidence", 0) > 0.6 else ValidationStatus.WARNING,
            confidence_impact=pre_llm_results.get("confidence", 0.7) * 10,
            summary="Pre-LLM validation completed",
            started_at=datetime.now()
        ))

    if post_llm_results:
        final_steps.append(ValidationStepResult(
            step_id="post_llm_summary",
            step_type=ValidationStepType.LLM_INFERENCE,
            step_name="Post-LLM Validation Summary",
            status=ValidationStatus.PASSED if post_llm_results.get("confidence", 0) > 0.6 else ValidationStatus.WARNING,
            confidence_impact=post_llm_results.get("confidence", 0.7) * 10,
            summary="Post-LLM validation completed",
            started_at=datetime.now()
        ))

    # Calculate final confidence
    final_confidence = confidence_calculator.calculate_confidence(final_steps)

    return {
        "validation_type": "final_assessment",
        "query_mode": query_mode,
        "final_confidence": final_confidence.total_score,
        "confidence_level": final_confidence.level.value,
        "validation_summary": {
            "total_validation_steps": len(final_steps),
            "knowledge_steps": len(knowledge_results),
            "llm_phases": 2 if pre_llm_results and post_llm_results else 1,
            "overall_status": "PASSED" if final_confidence.total_score >= 70 else "WARNING",
            "verification_coverage": final_confidence.verification_coverage
        },
        "recommendations": {
            "confidence_sufficient": final_confidence.total_score >= 70,
            "improvements_available": final_confidence.total_score < 90,
            "user_contribution_opportunities": len(
                [step for step in final_steps if hasattr(step, 'contribution_prompt') and step.contribution_prompt])
        }
    }
