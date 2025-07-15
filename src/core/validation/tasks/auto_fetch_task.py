"""
Auto-Fetch Operations Task (CPU Task)
Automatically fetches additional authoritative sources
"""

import logging
from typing import Dict, Any
from src.core.background.tasks import cpu_bound_task
from src.core.orchestration.job_chain import job_chain

logger = logging.getLogger(__name__)


@cpu_bound_task
def auto_fetch_task(job_id: str, task_data: Dict[str, Any]):
    """Execute auto-fetch operations (CPU task)."""

    try:
        logger.info(f"Starting auto-fetch for job {job_id}")

        fetch_targets = task_data.get("fetch_targets", [])
        query_context = task_data.get("query_context", {})

        # Import auto-fetch coordinator
        from src.core.validation.meta_validation import AutoFetchCoordinator

        auto_fetch_coordinator = AutoFetchCoordinator()

        # Execute auto-fetch operations
        fetch_results = await auto_fetch_coordinator.execute_auto_fetch(
            fetch_targets=fetch_targets,
            query_context=query_context
        )

        # Report completion
        result = {
            "validation_type": "auto_fetch",
            "fetch_results": fetch_results,
            "new_documents_added": fetch_results.get("new_documents_count", 0),
            "fetch_success": fetch_results.get("success", False),
            "retry_validation_ready": fetch_results.get("success", False)
        }

        job_chain.task_completed(job_id, result)
        logger.info(f"Auto-fetch completed for job {job_id}")

    except Exception as e:
        error_msg = f"Auto-fetch failed: {str(e)}"
        logger.error(error_msg)
        job_chain.task_failed(job_id, error_msg)