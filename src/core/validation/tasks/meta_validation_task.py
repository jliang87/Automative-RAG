from typing import Dict, Any

from src.core.orchestration.queue_manager import QueueNames
from src.core.orchestration.dramatiq_helpers import create_dramatiq_actor_decorator
from src.core.orchestration.job_chain import job_chain

# Import simple base class (same directory)
from .base_task import BaseValidationTask


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
            
            # Use special JobChain method for user input
            job_chain.task_waiting_for_user_input(job_id, result)
            self.logger.info(f"Meta-validation prepared user choice for job {job_id}")
            
        except Exception as e:
            error_msg = f"{self.task_name} failed for job {job_id}: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            job_chain.task_failed(job_id, error_msg)


# Create task instance
meta_task_instance = MetaValidationTask()

# Create Dramatiq task function
@create_dramatiq_actor_decorator(QueueNames.CPU_TASKS.value)
def meta_validation_task(job_id: str, task_data: Dict[str, Any]):
    """Dramatiq task function - delegates to task instance."""
    meta_task_instance.execute_with_error_handling(job_id, task_data)


def start_meta_validation(job_id: str, data: Dict):
    """Start meta-validation workflow."""
    meta_validation_task.send(job_id, data)


# ===== File: src/core/validation/tasks/auto_fetch_task.py =====

"""
Auto-fetch task using simple base class.
NO orchestration - just business logic.
"""

import time
from typing import Dict, Any

from src.core.orchestration.queue_manager import QueueNames
from src.core.orchestration.dramatiq_helpers import create_dramatiq_actor_decorator

# Import simple base class (same directory)
from .base_task import BaseValidationTask


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
        
        # Simulate processing time
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
@create_dramatiq_actor_decorator(QueueNames.CPU_TASKS.value)
def auto_fetch_task(job_id: str, task_data: Dict[str, Any]):
    """Dramatiq task function - delegates to task instance."""
    auto_fetch_task_instance.execute_with_error_handling(job_id, task_data)


def start_auto_fetch(job_id: str, data: Dict):
    """Start auto-fetch workflow."""
    auto_fetch_task.send(job_id, data)