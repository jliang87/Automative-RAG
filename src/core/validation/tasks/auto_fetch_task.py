import time
from typing import Dict, Any

from .base_validation_task import BaseValidationTask
from src.core.orchestration.queue_manager import queue_manager, QueueNames


class AutoFetchTask(BaseValidationTask):
    """Auto-fetch task - only business logic."""

    def __init__(self):
        super().__init__("auto_fetch")

    def validate_input(self, task_data: Dict[str, Any]) -> bool:
        """Auto-fetch doesn't need query/query_mode."""
        return "fetch_targets" in task_data or "query_context" in task_data

    def execute_validation_logic(self, job_id: str, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Core auto-fetch logic - no boilerplate."""

        fetch_targets = task_data.get("fetch_targets", [])
        query_context = task_data.get("query_context", {})

        self.logger.info(f"Executing auto-fetch for targets: {fetch_targets}")

        # TODO: Implement actual auto-fetch coordinator when available
        # For now, simulate the operation
        time.sleep(2)  # Simulate fetch operation

        # Mock successful fetch result
        fetch_results = {
            "success": True,
            "new_documents_count": len(fetch_targets) * 2,
            "sources_added": fetch_targets,
            "fetch_duration": 2.0
        }

        return {
            "validation_type": "auto_fetch",
            "fetch_results": fetch_results,
            "new_documents_added": fetch_results.get("new_documents_count", 0),
            "fetch_success": fetch_results.get("success", False),
            "retry_validation_ready": fetch_results.get("success", False),
            "tesla_t4_constraint": "CPU_TASKS queue used"
        }


# Create task instance
auto_fetch_task_instance = AutoFetchTask()


# Create Dramatiq task function
@queue_manager.create_task_decorator(QueueNames.CPU_TASKS.value)
def auto_fetch_task(job_id: str, task_data: Dict[str, Any]):
    """Dramatiq task function - delegates to task instance."""
    auto_fetch_task_instance.execute_with_error_handling(job_id, task_data)


def start_auto_fetch(job_id: str, data: Dict):
    """Start auto-fetch workflow."""
    auto_fetch_task.send(job_id, data)