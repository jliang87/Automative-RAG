from typing import Dict, Any

from src.core.orchestration.queue_manager import QueueNames
from src.core.orchestration.dramatiq_helpers import create_dramatiq_actor_decorator

# Import simple base class (same directory)
from .base_task import BaseValidationTask


class ValidationCacheTask(BaseValidationTask):
    """Validation cache task - only business logic."""

    def __init__(self):
        super().__init__("validation_cache")

    def validate_input(self, task_data: Dict[str, Any]) -> bool:
        """Cache task needs specific validation data."""
        required_fields = ["validation_step", "result"]
        return all(field in task_data for field in required_fields)

    def execute_validation_logic(self, job_id: str, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Core validation caching logic - no boilerplate."""

        validation_step = task_data.get("validation_step")
        result = task_data.get("result")
        documents_hash = task_data.get("documents_hash")
        query_id = task_data.get("query_id", job_id)

        self.logger.info(f"Caching validation result for step: {validation_step}")

        # TODO: Implement actual cache manager when available
        try:
            # Simulate caching operation
            cache_key = f"validation:{query_id}:{validation_step}"
            if documents_hash:
                cache_key += f":{documents_hash}"

            # Mock cache operation
            cache_success = True
            self.logger.info(f"Cached validation result with key: {cache_key}")

        except Exception as e:
            self.logger.error(f"Failed to cache validation result: {str(e)}")
            cache_success = False

        return {
            "validation_type": "cache_operation",
            "cache_success": cache_success,
            "cached_step": validation_step,
            "cache_key": cache_key if cache_success else None,
            "query_id": query_id,
            "tesla_t4_constraint": "CPU_TASKS queue used"
        }


# Create task instance
validation_cache_task_instance = ValidationCacheTask()

# Create Dramatiq task function
@create_dramatiq_actor_decorator(QueueNames.CPU_TASKS.value)
def validation_cache_task(job_id: str, task_data: Dict[str, Any]):
    """Dramatiq task function - delegates to task instance."""
    validation_cache_task_instance.execute_with_error_handling(job_id, task_data)


def start_validation_cache(job_id: str, data: Dict):
    """Start validation cache workflow."""
    validation_cache_task.send(job_id, data)