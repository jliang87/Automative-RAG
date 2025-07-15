"""
Validation Result Caching Task (CPU Task)
Caches validation results for smart retry functionality
"""

import logging
from typing import Dict, Any
from src.core.background.tasks import cpu_bound_task
from src.core.orchestration.job_chain import job_chain

logger = logging.getLogger(__name__)


@cpu_bound_task
def validation_cache_task(job_id: str, task_data: Dict[str, Any]):
    """Cache validation results for smart retry (CPU task)."""

    try:
        logger.info(f"Starting validation caching for job {job_id}")

        # Import cache manager
        from src.core.validation.meta_validation import ValidationCacheManager

        cache_manager = ValidationCacheManager()

        # Cache validation result
        await cache_manager.cache_validation_result(
            query_id=task_data.get("query_id", job_id),
            validation_step=task_data.get("validation_step"),
            result=task_data.get("result"),
            documents_hash=task_data.get("documents_hash")
        )

        # Report completion
        result = {
            "validation_type": "cache_operation",
            "cache_success": True,
            "cached_step": task_data.get("validation_step")
        }

        job_chain.task_completed(job_id, result)
        logger.info(f"Validation caching completed for job {job_id}")

    except Exception as e:
        error_msg = f"Validation caching failed: {str(e)}"
        logger.error(error_msg)
        job_chain.task_failed(job_id, error_msg)