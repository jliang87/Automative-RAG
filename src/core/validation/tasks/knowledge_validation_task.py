"""
Knowledge Pool Powered Validation Task (CPU Task)
Executes rule-based validation using knowledge pool data
"""

import logging
from typing import Dict, Any
from src.core.background.tasks import cpu_bound_task
from src.core.orchestration.job_chain import job_chain

logger = logging.getLogger(__name__)


@cpu_bound_task
def knowledge_validation_task(job_id: str, task_data: Dict[str, Any]):
    """Execute knowledge pool powered validation (CPU task)."""

    try:
        logger.info(f"Starting knowledge validation for job {job_id}")

        # Import validation components
        from src.core.validation.validation_interface import validation_engine
        from src.core.validation.models.validation_models import ValidationContext

        query = task_data.get("query", "")
        query_mode = task_data.get("query_mode", "facts")
        documents = task_data.get("documents", [])

        # Create validation context
        context = ValidationContext(
            query_id=job_id,
            query_text=query,
            query_mode=query_mode,
            documents=documents
        )

        # Get appropriate pipeline manager
        pipeline_type = validation_engine.pipeline_manager.determine_pipeline_type(
            query_mode, query
        )

        # Execute knowledge-based validation steps only
        config = validation_engine.pipeline_manager.get_pipeline_config(pipeline_type)
        knowledge_steps = []

        for step_config in config.steps:
            if step_config.step_type.value in ["retrieval", "source_credibility", "technical_consistency"]:
                # Get step implementation
                step_class = validation_engine.pipeline_manager.validation_steps.get(step_config.step_type)
                if step_class:
                    step_instance = step_class(step_config, validation_engine.pipeline_manager.meta_validator)
                    step_result = await step_instance.execute(context)
                    knowledge_steps.append(step_result)

        # Calculate confidence for knowledge steps
        knowledge_confidence = validation_engine.pipeline_manager.confidence_calculator.calculate_confidence(
            knowledge_steps, config.confidence_weights
        )

        # Determine if meta-validation is needed
        requires_meta_validation = any(
            step.status.value in ["unverifiable", "failed"] for step in knowledge_steps
        )

        # Report completion
        result = {
            "validation_type": "knowledge_based",
            "query_mode": query_mode,
            "knowledge_validation_steps": [
                {
                    "step_name": step.step_name,
                    "status": step.status.value,
                    "confidence_impact": step.confidence_impact,
                    "warnings": [w.message for w in step.warnings]
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

        job_chain.task_completed(job_id, result)
        logger.info(f"Knowledge validation completed for job {job_id}")

    except Exception as e:
        error_msg = f"Knowledge validation failed: {str(e)}"
        logger.error(error_msg)
        job_chain.task_failed(job_id, error_msg)