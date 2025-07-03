"""
Simplified JobChain - MUCH smaller, focused only on workflow orchestration
Previously 2000+ lines, now ~300 lines with clear separation of concerns
"""

import json
import time
import logging
from typing import Dict, Optional, Any
from enum import Enum

from src.core.orchestration.job_tracker import job_tracker, JobStatus
from src.core.orchestration.task_router import task_router, JobType
from src.core.orchestration.queue_manager import queue_manager

logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class JobChain:
    """
    SIMPLIFIED JobChain - orchestrates workflows, delegates tasks to specialized handlers.

    No longer contains massive task definitions - those are in task modules.
    Focuses purely on workflow state management and task coordination.
    """

    def __init__(self):
        self.redis = queue_manager.redis
        self.task_router = task_router

    def start_job_chain(self, job_id: str, job_type: JobType, initial_data: Dict[str, Any]) -> None:
        """Start a job chain with mode support."""
        workflow = self.task_router.get_workflow_for_job_type(job_type)
        if not workflow:
            raise ValueError(f"Unknown job type: {job_type}")

        # Extract query mode from initial data
        query_mode = initial_data.get("query_mode", "facts")

        # Store the job chain state with mode information
        chain_state = {
            "job_id": job_id,
            "job_type": job_type.value,
            "workflow": [(task, queue) for task, queue in workflow],
            "current_step": 0,
            "total_steps": len(workflow),
            "data": initial_data,
            "query_mode": query_mode,
            "started_at": time.time(),
            "status": TaskStatus.RUNNING.value,
            "step_timings": {}
        }

        self._save_chain_state(job_id, chain_state)

        # Update job tracker with mode information
        job_tracker.update_job_status(
            job_id,
            JobStatus.PROCESSING,
            result={
                "message": f"Starting {job_type.value} workflow in '{query_mode}' mode",
                "step": 1,
                "total_steps": len(workflow),
                "query_mode": query_mode
            },
            stage="chain_started"
        )

        # Execute the first task
        self._execute_next_task(job_id)

    def _execute_next_task(self, job_id: str) -> None:
        """Execute the next task in the chain."""
        chain_state = self._get_chain_state(job_id)
        if not chain_state:
            logger.error(f"No chain state found for job {job_id}")
            return

        current_step = chain_state["current_step"]
        workflow = chain_state["workflow"]

        # Check if we've completed all steps
        if current_step >= len(workflow):
            self._complete_job_chain(job_id)
            return

        # Get the current task
        task_name, queue_name = workflow[current_step]

        logger.info(f"Executing step {current_step + 1}/{len(workflow)} for job {job_id}: {task_name}")

        # Update job status
        job_tracker.update_job_status(
            job_id,
            "processing",
            result={
                "message": f"Executing {task_name}",
                "step": current_step + 1,
                "total_steps": len(workflow)
            },
            stage=task_name
        )

        # Update progress
        progress = ((current_step + 0.5) / len(workflow)) * 100
        job_tracker.update_job_progress(job_id, progress, f"Executing {task_name}")

        # Record step start time
        chain_state["step_timings"][task_name] = {"started_at": time.time()}
        self._save_chain_state(job_id, chain_state)

        # Get the complete current data from job tracker
        current_job = job_tracker.get_job(job_id, include_progress=False)
        current_job_result = current_job.get("result", {}) if current_job else {}

        # Parse if string
        if isinstance(current_job_result, str):
            try:
                current_job_result = json.loads(current_job_result)
            except:
                current_job_result = {}

        # Merge chain data with current job data
        complete_data = {}
        complete_data.update(chain_state["data"])
        complete_data.update(current_job_result)

        logger.info(f"Executing {task_name} with data keys: {list(complete_data.keys())}")

        # Check if there's already a running task for this queue type
        if queue_manager.is_queue_busy(queue_name):
            logger.info(f"Queue {queue_name} is busy, task {task_name} will wait")
            queue_manager.queue_task(job_id, task_name, queue_name, complete_data)
        else:
            # Execute immediately via task router
            self._execute_task_immediately(job_id, task_name, queue_name, complete_data)

    def _execute_task_immediately(self, job_id: str, task_name: str, queue_name: str, data: Dict[str, Any]) -> None:
        """Execute a task immediately via task router."""
        # Mark queue as busy
        queue_manager.mark_queue_busy(queue_name, job_id, task_name)

        # Delegate to task router (much cleaner!)
        try:
            self.task_router.route_task(job_id, task_name, queue_name, data)
        except Exception as e:
            logger.error(f"Error routing task {task_name}: {str(e)}")
            self.task_failed(job_id, f"Error routing task {task_name}: {str(e)}")

    def task_completed(self, job_id: str, result: Dict[str, Any]) -> None:
        """Called when a task completes successfully."""
        logger.info(f"Task completed for job {job_id}, triggering next task")

        # Update chain state
        chain_state = self._get_chain_state(job_id)
        if not chain_state:
            logger.error(f"No chain state found for job {job_id}")
            return

        # Record step completion time
        current_step = chain_state["current_step"]
        if current_step < len(chain_state["workflow"]):
            task_name, queue_name = chain_state["workflow"][current_step]

            # Update timing information
            if task_name in chain_state["step_timings"]:
                chain_state["step_timings"][task_name]["completed_at"] = time.time()
                chain_state["step_timings"][task_name]["duration"] = (
                        chain_state["step_timings"][task_name]["completed_at"] -
                        chain_state["step_timings"][task_name]["started_at"]
                )

            # Mark queue as free and process waiting tasks
            queue_manager.mark_queue_free(queue_name)
            queue_manager.process_waiting_tasks(queue_name)

        # Get current job data from job tracker (source of truth)
        current_job = job_tracker.get_job(job_id, include_progress=False)
        existing_job_result = current_job.get("result", {}) if current_job else {}

        # Parse existing result if it's a string
        if isinstance(existing_job_result, str):
            try:
                existing_job_result = json.loads(existing_job_result)
            except:
                existing_job_result = {}

        # Merge task result with existing job data
        combined_result = {}
        combined_result.update(existing_job_result)
        combined_result.update(result)

        # Update chain state data with the combined result
        chain_state["data"].update(combined_result)

        # Move to next step
        chain_state["current_step"] += 1

        # Update progress
        progress = (chain_state["current_step"] / len(chain_state["workflow"])) * 100
        job_tracker.update_job_progress(job_id, progress,
                                        f"Completed step {chain_state['current_step']}/{len(chain_state['workflow'])}")

        # Update job tracker with the combined result
        job_tracker.update_job_status(
            job_id,
            "processing",
            result=combined_result,
            stage=f"completed_step_{current_step + 1}",
            replace_result=True
        )

        # Save updated chain state
        self._save_chain_state(job_id, chain_state)

        # Execute next task
        self._execute_next_task(job_id)

    def task_failed(self, job_id: str, error: str) -> None:
        """Called when a task fails."""
        logger.error(f"Task failed for job {job_id}: {error}")

        # Get chain state to free up the queue
        chain_state = self._get_chain_state(job_id)
        if chain_state:
            current_step = chain_state["current_step"]
            if current_step < len(chain_state["workflow"]):
                task_name, queue_name = chain_state["workflow"][current_step]

                # Record failure timing
                if task_name in chain_state["step_timings"]:
                    chain_state["step_timings"][task_name]["failed_at"] = time.time()
                    chain_state["step_timings"][task_name]["duration"] = (
                            chain_state["step_timings"][task_name]["failed_at"] -
                            chain_state["step_timings"][task_name]["started_at"]
                    )

                # Free up the queue
                queue_manager.mark_queue_free(queue_name)
                queue_manager.process_waiting_tasks(queue_name)

        # Update job status
        job_tracker.update_job_status(
            job_id,
            JobStatus.FAILED,
            error=error,
            replace_result=False
        )

        # Update progress to show failure
        job_tracker.update_job_progress(job_id, 0, f"Failed: {error[:50]}...")

        # Clean up chain state
        self._delete_chain_state(job_id)

    def _complete_job_chain(self, job_id: str) -> None:
        """Complete the entire job chain while preserving ALL job results."""
        logger.info(f"Job chain completed for job {job_id}")

        # Get final chain state for timing information
        chain_state = self._get_chain_state(job_id)
        total_duration = time.time() - chain_state["started_at"] if chain_state else 0

        # Get the CURRENT job data, not chain data
        current_job = job_tracker.get_job(job_id, include_progress=False)

        if not current_job:
            logger.error(f"No job data found for completed job {job_id}")
            self._delete_chain_state(job_id)
            return

        existing_result = current_job.get("result", {})

        # Parse existing result if it's a string
        if isinstance(existing_result, str):
            try:
                existing_result = json.loads(existing_result)
            except:
                existing_result = {}

        # Completion info
        completion_info = {
            "job_chain_completion": {
                "message": "Job chain completed successfully",
                "total_duration": total_duration,
                "step_timings": chain_state.get("step_timings", {}) if chain_state else {},
                "completed_at": time.time()
            }
        }

        # Preserve ALL existing result data
        if existing_result and isinstance(existing_result, dict):
            final_result = {}
            final_result.update(existing_result)
            final_result.update(completion_info)

            logger.info(f"Preserving all job data for {job_id} with keys: {list(existing_result.keys())}")
        else:
            final_result = completion_info
            logger.info(f"No existing result to preserve for job {job_id}")

        # Update job status with the final result
        job_tracker.update_job_status(
            job_id,
            "completed",
            result=final_result,
            replace_result=True
        )

        # Update progress to 100%
        job_tracker.update_job_progress(job_id, 100, "Job completed successfully")

        # Clean up chain state
        self._delete_chain_state(job_id)

    def get_job_chain_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get the current status of a job chain."""
        chain_state = self._get_chain_state(job_id)
        if not chain_state:
            return None

        current_step = chain_state["current_step"]
        workflow = chain_state["workflow"]

        return {
            "job_id": job_id,
            "job_type": chain_state["job_type"],
            "status": chain_state["status"],
            "current_step": current_step,
            "total_steps": len(workflow),
            "current_task": workflow[current_step][0] if current_step < len(workflow) else None,
            "progress_percentage": (current_step / len(workflow)) * 100,
            "started_at": chain_state["started_at"],
            "step_timings": chain_state.get("step_timings", {}),
            "data_keys": list(chain_state["data"].keys())
        }

    def get_queue_status(self) -> Dict[str, Any]:
        """Get the status of all queues via queue manager."""
        return queue_manager.get_queue_status()

    def _save_chain_state(self, job_id: str, chain_state: Dict[str, Any]) -> None:
        """Save chain state with proper UTF-8 encoding."""
        state_json = json.dumps(chain_state, ensure_ascii=False)
        self.redis.set(f"job_chain:{job_id}", state_json, ex=86400)

    def _get_chain_state(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get chain state from Redis."""
        state_json = self.redis.get(f"job_chain:{job_id}")
        if state_json:
            return json.loads(state_json)
        return None

    def _delete_chain_state(self, job_id: str) -> None:
        """Delete chain state from Redis."""
        self.redis.delete(f"job_chain:{job_id}")


# Global job chain instance
job_chain = JobChain()


# ===============================================================================
# VALIDATION PIPELINE ACCESS FOR ENHANCED INGESTION SYSTEM
# ===============================================================================

def get_enhanced_ingestion_processors():
    """
    Get available enhanced ingestion processors for validation pipeline use.

    This function provides the validation pipeline with access to the unified
    ingestion system processors, enabling meta-validators to call specific
    ingestion types as needed.

    Returns:
        Dict mapping processor types to factory creation functions
    """

    try:
        from core.ingestion.factory import ProcessorFactory
        ENHANCED_PROCESSING_AVAILABLE = True
    except ImportError:
        logger.warning("Enhanced ingestion system not available for validation pipeline")
        return {}

    return {
        "text": lambda: ProcessorFactory.create_processor("text"),
        "pdf": lambda: ProcessorFactory.create_processor("pdf"),
        "video": lambda: ProcessorFactory.create_processor("video"),
        "factory": ProcessorFactory,
        "available_types": ProcessorFactory.get_supported_types(),
        "validation_ready": True
    }


def process_content_for_validation(content_type: str, source_data: Any, metadata: Optional[Dict] = None):
    """
    Process content using enhanced ingestion system for validation pipeline.

    This function allows meta-validators to process different content types
    using the unified ingestion system, ensuring consistent metadata extraction
    and formatting across all validation workflows.

    Args:
        content_type: Type of content ("text", "pdf", "video")
        source_data: Source content (text string, file path, URL, etc.)
        metadata: Optional metadata to include

    Returns:
        List of processed Document objects with enhanced metadata

    Raises:
        ValueError: If content type not supported or processing fails
    """

    try:
        from core.ingestion.factory import ProcessorFactory
    except ImportError:
        raise ValueError("Enhanced ingestion system not available for validation")

    try:
        # Create appropriate processor
        processor = ProcessorFactory.create_processor(content_type)

        # Validate source before processing
        validation_result = processor.validate_source(source_data)
        if not validation_result.get("valid", False):
            error = validation_result.get("error", "Source validation failed")
            raise ValueError(f"Invalid source for {content_type}: {error}")

        # Process content with enhanced metadata extraction
        logger.info(f"Processing {content_type} content for validation pipeline")
        documents = processor.process(source_data, metadata)

        # Add validation context metadata
        for doc in documents:
            doc.metadata.update({
                "processed_for_validation": True,
                "validation_processing_timestamp": time.time(),
                "validation_processor_type": content_type,
                "unified_ingestion_used": True
            })

        logger.info(f"✅ Processed {len(documents)} documents for validation pipeline")
        return documents

    except Exception as e:
        logger.error(f"Error processing {content_type} for validation: {str(e)}")
        raise ValueError(f"Processing failed for {content_type}: {str(e)}")


def get_ingestion_processor_capabilities():
    """
    Get capabilities of enhanced ingestion processors for validation planning.

    Returns:
        Dict with processor capabilities and supported features
    """

    try:
        from core.ingestion.factory import ProcessorFactory
    except ImportError:
        return {"available": False, "error": "Enhanced ingestion system not available"}

    try:
        capabilities = {
            "available": True,
            "unified_ingestion_system": True,
            "enhanced_transcript_processor": True,
            "automotive_metadata_extraction": True,
            "content_injection": True,
            "vehicle_detection": True,
            "supported_types": ProcessorFactory.get_supported_types(),
            "processors": {}
        }

        # Get capabilities for each processor type
        for proc_type in ["text", "pdf", "video"]:
            try:
                processor = ProcessorFactory.create_processor(proc_type)
                capabilities["processors"][proc_type] = processor.get_processing_stats()
            except Exception as e:
                capabilities["processors"][proc_type] = {"error": str(e)}

        return capabilities

    except Exception as e:
        return {
            "available": False,
            "error": f"Error getting capabilities: {str(e)}"
        }


# ===============================================================================
# INTEGRATION INSTRUCTIONS FOR VALIDATION FRAMEWORK
# ===============================================================================

"""
VALIDATION FRAMEWORK INTEGRATION INSTRUCTIONS:

The validation framework can now access the restructured ingestion system through:

1. Enhanced Ingestion Processors:
   ```python
   from src.core.orchestration.job_chain import get_enhanced_ingestion_processors
   processors = get_enhanced_ingestion_processors()
   text_processor = processors["text"]()
   pdf_processor = processors["pdf"]()
   ```

2. Direct Content Processing:
   ```python
   from src.core.orchestration.job_chain import process_content_for_validation
   documents = process_content_for_validation("text", some_text, metadata)
   documents = process_content_for_validation("pdf", "/path/to/file.pdf", metadata)
   ```

3. Capability Discovery:
   ```python
   from src.core.orchestration.job_chain import get_ingestion_processor_capabilities
   capabilities = get_ingestion_processor_capabilities()
   supported_types = capabilities["supported_types"]
   ```

4. Task-Level Integration:
   - Video processing: core.ingestion.tasks.video_tasks
   - PDF processing: core.ingestion.tasks.pdf_tasks
   - Text processing: core.ingestion.tasks.text_tasks
   - Document retrieval: core.query.tasks.retrieval_tasks
   - LLM inference: core.query.tasks.inference_tasks

5. Queue Management:
   ```python
   from core.orchestration.queue_manager import queue_manager
   queue_status = queue_manager.get_queue_status()
   ```

6. Job Tracking:
   ```python
   from src.core.orchestration.job_tracker import job_tracker
   job_details = job_tracker.get_job(job_id)
   ```

This restructure provides the validation framework with clean, modular access
to all ingestion and query components while maintaining complete backward
compatibility with existing job processing workflows.
"""