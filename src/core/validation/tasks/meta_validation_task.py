"""
Meta-Validation Coordination Task (CPU Task)
Handles validation failures and user choice coordination
"""

import logging
from typing import Dict, Any
from src.core.background.tasks import cpu_bound_task
from src.core.orchestration.job_chain import job_chain

logger = logging.getLogger(__name__)


@cpu_bound_task
def meta_validation_task(job_id: str, task_data: Dict[str, Any]):
    """Execute meta-validation coordination (CPU task)."""

    try:
        logger.info(f"Starting meta-validation for job {job_id}")

        # Import meta-validation components
        from src.core.validation.steps.steps_readiness_checker import MetaValidator
        from src.core.validation.guidance.guided_trust_loop import guidance_engine

        failure_context = task_data.get("failure_context", {})
        failed_step = task_data.get("failed_step", "unknown")
        query_mode = task_data.get("query_mode", "facts")

        # Analyze failure and generate user_guidance
        meta_validator = MetaValidator()

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

        # Update job status with user choice request
        from src.core.orchestration.job_tracker import job_tracker

        job_tracker.update_job_status(
            job_id,
            "awaiting_user_input",
            result={
                "user_choice_request": {
                    "type": "meta_validation_choice",
                    "failure_reason": user_choice_request["failure_reason"],
                    "options": user_choice_request["options"],
                    "message": "Validation needs your help to continue",
                    "user_guidance": guidance_engine.generate_guidance_for_failure(failed_step, failure_context)
                }
            }
        )

        # Report meta-validation ready (awaiting user input)
        result = {
            "validation_type": "meta_validation",
            "status": "awaiting_user_choice",
            "user_choice_request": user_choice_request,
            "guidance_available": True
        }

        job_chain.task_completed(job_id, result)
        logger.info(f"Meta-validation prepared user choice for job {job_id}")

    except Exception as e:
        error_msg = f"Meta-validation failed: {str(e)}"
        logger.error(error_msg)
        job_chain.task_failed(job_id, error_msg)


async def _generate_user_choice_options(failure_context, failed_step, query_mode):
    """Generate user choice options based on failure context."""

    from src.core.validation.meta_validation.user_guidance.failure_guidance import GuidanceEngine

    guidance_engine = GuidanceEngine()

    # Generate specific guidance for the failure
    guidance = guidance_engine.generate_guidance_for_failure(failed_step, failure_context)

    # Create user choice options
    options = {
        "auto_fetch": {
            "available": True,
            "description": "Automatically fetch additional authoritative sources",
            "estimated_time": "30-60 seconds",
            "confidence_boost_estimate": 15.0,
            "targets": _determine_auto_fetch_targets(failed_step, failure_context)
        },
        "user_guidance": {
            "available": True,
            "description": "Provide additional sources or information manually",
            "estimated_time": "user dependent",
            "confidence_boost_estimate": 20.0,
            "guidance": guidance,
            "contribution_types": ["url_link", "file_upload", "text_input"]
        },
        "restart_full": {
            "available": True,
            "description": "Start validation pipeline from beginning",
            "estimated_time": "2-3 minutes",
            "note": "This will clear all progress and start fresh"
        }
    }

    return options


def _determine_auto_fetch_targets(failed_step, failure_context):
    """Determine what auto-fetch targets are appropriate for the failure."""

    targets = []

    if "technical_consistency" in failed_step:
        targets.extend(["official_specs", "epa_data"])

    if "source_credibility" in failed_step:
        targets.extend(["manufacturer_sources", "safety_ratings"])

    if "completeness" in failed_step:
        targets.extend(["official_specs", "manufacturer_sources"])

    # Default targets if none specific
    if not targets:
        targets = ["official_specs", "epa_data"]

    return targets
