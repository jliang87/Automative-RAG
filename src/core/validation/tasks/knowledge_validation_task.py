from typing import Dict, Any

# Import Tesla T4 constrained queue definitions
from src.core.orchestration.queue_manager import QueueNames
from src.core.orchestration.dramatiq_helpers import create_dramatiq_actor_decorator

# Import simple base class (same directory)
from .base_task import BaseValidationTask


class KnowledgeValidationTask(BaseValidationTask):
    """
    Knowledge validation task - only business logic.
    Boilerplate handled by BaseValidationTask.
    """

    def __init__(self):
        super().__init__("knowledge_validation")

    def execute_validation_logic(self, job_id: str, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Core business logic - no boilerplate."""

        # Import validation components
        from src.core.validation.steps.retrieval_quality import RetrievalQualityValidator
        from src.core.validation.steps.source_credibility_validator import SourceCredibilityValidator
        from src.core.validation.steps.technical_consistency_validator import TechnicalConsistencyValidator
        from src.core.validation.steps.steps_readiness_checker import MetaValidator
        from src.core.validation.confidence_calculator import ConfidenceCalculator

        # Create validation context (inherited method)
        context = self.create_validation_context(job_id, task_data)

        # Execute validation steps
        meta_validator = MetaValidator()
        confidence_calculator = ConfidenceCalculator()
        knowledge_steps = []

        # Step 1: Retrieval Quality
        try:
            retrieval_validator = RetrievalQualityValidator({}, meta_validator)
            retrieval_result = retrieval_validator.execute(context)
            knowledge_steps.append(retrieval_result)
            self.logger.info("✅ Retrieval quality validation completed")
        except Exception as e:
            self.logger.error(f"Retrieval validation failed: {str(e)}")

        # Step 2: Source Credibility
        try:
            credibility_validator = SourceCredibilityValidator({}, meta_validator)
            credibility_result = credibility_validator.execute(context)
            knowledge_steps.append(credibility_result)
            self.logger.info("✅ Source credibility validation completed")
        except Exception as e:
            self.logger.error(f"Source credibility validation failed: {str(e)}")

        # Step 3: Technical Consistency
        try:
            consistency_validator = TechnicalConsistencyValidator({}, meta_validator)
            consistency_result = consistency_validator.execute(context)
            knowledge_steps.append(consistency_result)
            self.logger.info("✅ Technical consistency validation completed")
        except Exception as e:
            self.logger.error(f"Technical consistency validation failed: {str(e)}")

        # Calculate confidence
        knowledge_confidence = confidence_calculator.calculate_confidence(knowledge_steps)

        # Determine if meta-validation needed
        requires_meta_validation = any(
            step.status.value in ["unverifiable", "failed"] for step in knowledge_steps
        )

        # Return result (base class handles job_chain.task_completed)
        return {
            "validation_type": "knowledge_based",
            "query_mode": task_data.get("query_mode", "facts"),
            "knowledge_validation_steps": [
                {
                    "step_name": step.step_name,
                    "step_type": step.step_type.value,
                    "status": step.status.value,
                    "confidence_impact": step.confidence_impact,
                    "warnings": [w.message for w in step.warnings],
                    "duration_ms": getattr(step, 'duration_ms', 0)
                }
                for step in knowledge_steps
            ],
            "knowledge_confidence": knowledge_confidence.total_score,
            "requires_meta_validation": requires_meta_validation,
            "documents": task_data.get("documents", []),
            "query": task_data.get("query", ""),
            "query_mode": task_data.get("query_mode", "facts"),
            "tesla_t4_constraint": "CPU_TASKS queue used"
        }


# Create task instance
knowledge_task_instance = KnowledgeValidationTask()

# Create Dramatiq task function using Tesla T4 constrained queue
@create_dramatiq_actor_decorator(QueueNames.CPU_TASKS.value)
def knowledge_validation_task(job_id: str, task_data: Dict[str, Any]):
    """Dramatiq task function - delegates to task instance."""
    knowledge_task_instance.execute_with_error_handling(job_id, task_data)


# Workflow starter function (called by TaskRouter)
def start_knowledge_validation(job_id: str, data: Dict):
    """Start knowledge validation workflow."""
    if "query" not in data:
        error_msg = "query required for knowledge validation"
        from src.core.orchestration.job_chain import job_chain
        job_chain.task_failed(job_id, error_msg)
        return

    knowledge_validation_task.send(job_id, data)