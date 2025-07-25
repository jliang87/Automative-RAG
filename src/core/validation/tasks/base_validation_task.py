import time
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any

# Import orchestration components (existing)
from src.core.orchestration.job_tracker import job_tracker
from src.core.orchestration.job_chain import job_chain


class BaseValidationTask(ABC):
    """
    Simple base class to reduce boilerplate in validation tasks.
    NO orchestration logic - just common error handling and logging patterns.
    """

    def __init__(self, task_name: str):
        self.task_name = task_name
        self.logger = logging.getLogger(f"validation.tasks.{task_name}")

    @abstractmethod
    def execute_validation_logic(self, job_id: str, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Implement the core validation business logic.
        Must be implemented by each specific validation task.
        """
        pass

    def validate_input(self, task_data: Dict[str, Any]) -> bool:
        """Basic input validation. Override if needed."""
        required_fields = ["query", "query_mode"]
        return all(field in task_data for field in required_fields)

    def create_validation_context(self, job_id: str, task_data: Dict[str, Any]):
        """Helper to create validation context."""
        try:
            from src.models import ValidationContext
            return ValidationContext(
                query_id=job_id,
                query_text=task_data.get("query", ""),
                query_mode=task_data.get("query_mode", "facts"),
                documents=task_data.get("documents", []),
                processing_metadata=task_data.get("previous_results", {})
            )
        except ImportError:
            self.logger.warning("ValidationContext not available, using basic context")
            return {
                "query_id": job_id,
                "query_text": task_data.get("query", ""),
                "query_mode": task_data.get("query_mode", "facts"),
                "documents": task_data.get("documents", [])
            }

    def execute_with_error_handling(self, job_id: str, task_data: Dict[str, Any]) -> None:
        """
        Common execution pattern with error handling.
        This is what the Dramatiq task function calls.
        """
        start_time = time.time()

        try:
            self.logger.info(f"Starting {self.task_name} for job {job_id}")

            # Validate input
            if not self.validate_input(task_data):
                raise ValueError("Input validation failed")

            # Execute business logic (implemented by subclass)
            result = self.execute_validation_logic(job_id, task_data)

            # Add execution time
            result["validation_completed_at"] = time.time()
            result["execution_time"] = time.time() - start_time

            # Report success
            job_chain.task_completed(job_id, result)
            self.logger.info(f"{self.task_name} completed for job {job_id}")

        except Exception as e:
            error_msg = f"{self.task_name} failed for job {job_id}: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            job_chain.task_failed(job_id, error_msg)