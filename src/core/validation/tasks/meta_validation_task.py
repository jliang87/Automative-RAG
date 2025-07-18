from typing import Dict, Any

from .base_validation_task import BaseValidationTask
from src.core.orchestration.queue_manager import queue_manager, QueueNames
from src.core.orchestration.job_chain import job_chain


class MetaValidationTask(BaseValidationTask):
    """Meta-validation task - only business logic."""

    def __init__(self):
        super().__init__("meta_validation")

    def execute_validation_logic(self, job_id: str, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Core meta-validation logic - no boilerplate."""

        failure_context = task_data.get("failure_context", {})
        failed_step = task_data.get("failed_step", "unknown")
        query_mode = task_data.get("query_mode", "facts")

        # Generate user choice options
        user_choice_request = {
            "job_id": job_id,
            "failure_reason": failure_context.get("primary_reason", "Validation step could not complete"),
            "failed_step": failed_step,
            "options": {
                "auto_fetch": {
                    "available": True,
                    "description": "Automatically fetch additional authoritative sources",
                    "estimated_time": "30-60 seconds",
                    "confidence_boost_estimate": 15.0
                },
                "user_guidance": {
                    "available": True,
                    "description": "Provide additional sources or information manually",
                    "estimated_time": "user dependent",
                    "confidence_boost_estimate": 20.0
                },
                "restart_full": {
                    "available": True,
                    "description": "Start validation pipeline from beginning",
                    "estimated_time": "2-3 minutes"
                }
            }
        }

        return {
            "validation_type": "meta_validation",
            "status": "awaiting_user_choice",
            "user_choice_request": user_choice_request,
            "guidance_available": True,
            "tesla_t4_constraint": "CPU_TASKS queue used"
        }

    def execute_with_error_handling(self, job_id: str, task_data: Dict[str, Any]) -> None:
        """Override to use special JobChain method for user input."""
        try:
            self.logger.info(f"Starting {self.task_name} for job {job_id}")

            if not self.validate_input(task_data):
                raise ValueError("Input validation failed")

            result = self.execute_validation_logic(job_id, task_data)

            # TODO: Use special JobChain method for user input when implemented
            # For now, use regular task_completed but mark as awaiting input
            result["requires_user_input"] = True
            job_chain.task_completed(job_id, result)
            self.logger.info(f"Meta-validation prepared user choice for job {job_id}")

        except Exception as e:
            error_msg = f"{self.task_name} failed for job {job_id}: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            job_chain.task_failed(job_id, error_msg)


# Create task instance
meta_task_instance = MetaValidationTask()


# Create Dramatiq task function
@queue_manager.create_task_decorator(QueueNames.CPU_TASKS.value)
def meta_validation_task(job_id: str, task_data: Dict[str, Any]):
    """Dramatiq task function - delegates to task instance."""
    meta_task_instance.execute_with_error_handling(job_id, task_data)


def start_meta_validation(job_id: str, data: Dict):
    """Start meta-validation workflow."""
    meta_validation_task.send(job_id, data)