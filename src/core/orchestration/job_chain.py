"""
Enhanced JobChain - Added validation workflow support and user input handling
Previously missing critical methods for validation workflows
"""

import json
import time
import logging
from typing import Dict, Optional, Any, List
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
    AWAITING_USER_INPUT = "awaiting_user_input"  # ✅ NEW: For user input handling


class ValidationPhase(Enum):
    """Validation workflow phases for progress tracking."""
    KNOWLEDGE_VALIDATION = "knowledge_validation"
    PRE_LLM_VALIDATION = "pre_llm_validation"
    DOCUMENT_RETRIEVAL = "document_retrieval"
    LLM_INFERENCE = "llm_inference"
    POST_LLM_VALIDATION = "post_llm_validation"
    FINAL_ASSESSMENT = "final_assessment"
    META_VALIDATION = "meta_validation"
    AUTO_FETCH = "auto_fetch"
    USER_GUIDANCE = "user_guidance"


class JobChain:
    """
    Enhanced JobChain with validation workflow support and user input handling.
    """

    def __init__(self):
        self.redis = queue_manager.redis
        self.task_router = task_router

    def start_job_chain(self, job_id: str, job_type: JobType, initial_data: Dict[str, Any]) -> None:
        """Start a job chain with enhanced validation workflow support."""
        workflow = self.task_router.get_workflow_for_job_type(job_type)
        if not workflow:
            raise ValueError(f"Unknown job type: {job_type}")

        # Extract query mode from initial data
        query_mode = initial_data.get("query_mode", "facts")

        # Store the job chain state with validation tracking
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
            "step_timings": {},
            # ✅ NEW: Validation workflow tracking
            "validation_workflow": job_type == JobType.COMPREHENSIVE_VALIDATION,
            "validation_phases_completed": [],
            "requires_meta_validation": False,
            "user_input_required": False,
            "validation_confidence": 0.0
        }

        self._save_chain_state(job_id, chain_state)

        # Update job tracker with enhanced validation info
        job_tracker.update_job_status(
            job_id,
            JobStatus.PROCESSING,
            result={
                "message": f"Starting {job_type.value} workflow in '{query_mode}' mode",
                "step": 1,
                "total_steps": len(workflow),
                "query_mode": query_mode,
                "validation_workflow": chain_state["validation_workflow"]
            },
            stage="chain_started"
        )

        # Execute the first task
        self._execute_next_task(job_id)

    def _execute_next_task(self, job_id: str) -> None:
        """Execute the next task in the chain with validation workflow awareness."""
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

        # ✅ NEW: Track validation phases
        if chain_state.get("validation_workflow"):
            validation_phase = self._map_task_to_validation_phase(task_name)
            if validation_phase:
                chain_state["current_validation_phase"] = validation_phase.value
                self._save_chain_state(job_id, chain_state)

        # Update job status with validation phase info
        job_update = {
            "message": f"Executing {task_name}",
            "step": current_step + 1,
            "total_steps": len(workflow)
        }

        if chain_state.get("validation_workflow"):
            job_update["validation_phase"] = chain_state.get("current_validation_phase")
            job_update["validation_progress"] = self._calculate_validation_progress(chain_state)

        job_tracker.update_job_status(job_id, "processing", result=job_update, stage=task_name)

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

        # Delegate to task router
        try:
            self.task_router.route_task(job_id, task_name, queue_name, data)
        except Exception as e:
            logger.error(f"Error routing task {task_name}: {str(e)}")
            self.task_failed(job_id, f"Error routing task {task_name}: {str(e)}")

    def task_completed(self, job_id: str, result: Dict[str, Any]) -> None:
        """Enhanced task completion with validation workflow branching."""
        logger.info(f"Task completed for job {job_id}, checking for validation workflow branching")

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

            # ✅ NEW: Track validation phase completion
            if chain_state.get("validation_workflow"):
                validation_phase = self._map_task_to_validation_phase(task_name)
                if validation_phase and validation_phase.value not in chain_state["validation_phases_completed"]:
                    chain_state["validation_phases_completed"].append(validation_phase.value)

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

        # ✅ NEW: Check for validation workflow branching
        if chain_state.get("validation_workflow"):
            branch_decision = self._check_validation_branching(job_id, task_name, combined_result, chain_state)
            if branch_decision:
                return  # Branching handled, don't continue normal workflow

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

    # ===== NEW METHODS FOR VALIDATION WORKFLOW SUPPORT =====

    def task_waiting_for_user_input(self, job_id: str, user_prompt: Dict[str, Any]) -> None:
        """✅ NEW: Handle tasks that need user input (meta-validation)."""
        logger.info(f"Task waiting for user input for job {job_id}")

        # Get chain state
        chain_state = self._get_chain_state(job_id)
        if chain_state:
            # Pause workflow
            chain_state["status"] = TaskStatus.AWAITING_USER_INPUT.value
            chain_state["user_input_required"] = True
            chain_state["user_prompt"] = user_prompt

            # Update timing for current step
            current_step = chain_state["current_step"]
            if current_step < len(chain_state["workflow"]):
                task_name, queue_name = chain_state["workflow"][current_step]
                if task_name in chain_state["step_timings"]:
                    chain_state["step_timings"][task_name]["waiting_for_input_at"] = time.time()

                # Free up the queue since we're waiting for user input
                queue_manager.mark_queue_free(queue_name)
                queue_manager.process_waiting_tasks(queue_name)

            self._save_chain_state(job_id, chain_state)

        # Update job status
        job_tracker.update_job_status(
            job_id,
            "awaiting_input",
            result=user_prompt,
            stage="awaiting_user_choice"
        )

        # Update progress to show waiting
        job_tracker.update_job_progress(job_id, None, "Waiting for user input")

        logger.info(f"Job {job_id} is now waiting for user input")

    def resume_from_user_input(self, job_id: str, user_choice: Dict[str, Any]) -> None:
        """✅ NEW: Resume workflow after user input."""
        logger.info(f"Resuming job {job_id} from user input with choice: {user_choice.get('choice')}")

        chain_state = self._get_chain_state(job_id)
        if not chain_state:
            logger.error(f"No chain state found for job {job_id}")
            return

        # Process user choice
        choice = user_choice.get("choice")

        if choice == "auto_fetch":
            self._handle_auto_fetch_choice(job_id, user_choice, chain_state)
        elif choice == "user_guidance":
            self._handle_user_guidance_choice(job_id, user_choice, chain_state)
        elif choice == "restart_full":
            self._handle_restart_choice(job_id, user_choice, chain_state)
        else:
            logger.error(f"Unknown user choice: {choice}")
            self.task_failed(job_id, f"Unknown user choice: {choice}")
            return

        # Update chain state
        chain_state["status"] = TaskStatus.RUNNING.value
        chain_state["user_input_required"] = False
        chain_state["user_choice"] = user_choice

        # Record user input timing
        current_step = chain_state["current_step"]
        if current_step < len(chain_state["workflow"]):
            task_name, _ = chain_state["workflow"][current_step]
            if task_name in chain_state["step_timings"]:
                chain_state["step_timings"][task_name]["user_input_completed_at"] = time.time()

        self._save_chain_state(job_id, chain_state)

        # Update job status
        job_tracker.update_job_status(
            job_id,
            "processing",
            result={"message": f"Resuming with user choice: {choice}"},
            stage="resumed_from_user_input"
        )

    def _insert_meta_validation_step(self, job_id: str) -> None:
        """✅ NEW: Insert meta-validation step into workflow."""
        logger.info(f"Inserting meta-validation step for job {job_id}")

        chain_state = self._get_chain_state(job_id)
        if not chain_state:
            return

        # Insert meta-validation after current step
        current_step = chain_state["current_step"]
        workflow = chain_state["workflow"]

        # Add meta-validation step
        meta_validation_step = ("meta_validation", queue_manager.get_recommended_queue_for_task_type("meta_validation"))
        workflow.insert(current_step + 1, meta_validation_step)

        # Update workflow and total steps
        chain_state["workflow"] = workflow
        chain_state["total_steps"] = len(workflow)
        chain_state["meta_validation_inserted"] = True

        self._save_chain_state(job_id, chain_state)

        logger.info(f"Meta-validation step inserted at position {current_step + 1}")

    def _check_validation_branching(self, job_id: str, completed_task: str, result: Dict[str, Any],
                                  chain_state: Dict[str, Any]) -> bool:
        """✅ NEW: Check if validation workflow needs branching."""

        # Check if knowledge validation requires meta-validation
        if completed_task == "knowledge_validation":
            requires_meta = result.get("requires_meta_validation", False)
            if requires_meta:
                logger.info(f"Knowledge validation failed for job {job_id}, inserting meta-validation")
                self._insert_meta_validation_step(job_id)
                return False  # Continue with modified workflow

        # Check if any validation step indicates failure needing user input
        validation_status = result.get("status")
        if validation_status == "awaiting_user_choice":
            logger.info(f"Task {completed_task} requires user input for job {job_id}")
            # This will be handled by task_waiting_for_user_input method
            return False

        # Update validation confidence tracking
        if "confidence" in result:
            chain_state["validation_confidence"] = result["confidence"]
        elif "final_confidence" in result:
            chain_state["validation_confidence"] = result["final_confidence"]

        return False  # No branching, continue normal workflow

    def _handle_auto_fetch_choice(self, job_id: str, user_choice: Dict[str, Any],
                                chain_state: Dict[str, Any]) -> None:
        """Handle auto-fetch user choice."""
        logger.info(f"Handling auto-fetch choice for job {job_id}")

        # Insert auto-fetch step
        auto_fetch_step = ("auto_fetch", queue_manager.get_recommended_queue_for_task_type("auto_fetch"))
        current_step = chain_state["current_step"]
        chain_state["workflow"].insert(current_step + 1, auto_fetch_step)
        chain_state["total_steps"] = len(chain_state["workflow"])

        # Continue with next step (which will be auto-fetch)
        self._execute_next_task(job_id)

    def _handle_user_guidance_choice(self, job_id: str, user_choice: Dict[str, Any],
                                   chain_state: Dict[str, Any]) -> None:
        """Handle user guidance choice."""
        logger.info(f"Handling user guidance choice for job {job_id}")

        # Process user-provided guidance/sources
        guidance_data = user_choice.get("guidance_data", {})

        # Insert user guidance processing step
        user_guidance_step = ("process_user_contribution", queue_manager.get_recommended_queue_for_task_type("meta_validation"))
        current_step = chain_state["current_step"]
        chain_state["workflow"].insert(current_step + 1, user_guidance_step)
        chain_state["total_steps"] = len(chain_state["workflow"])

        # Add guidance data to chain state
        chain_state["data"]["user_guidance"] = guidance_data

        # Continue with next step
        self._execute_next_task(job_id)

    def _handle_restart_choice(self, job_id: str, user_choice: Dict[str, Any],
                             chain_state: Dict[str, Any]) -> None:
        """Handle restart validation choice."""
        logger.info(f"Handling restart choice for job {job_id}")

        # Reset to beginning of validation workflow
        # Find the first validation step
        workflow = chain_state["workflow"]
        for i, (task_name, _) in enumerate(workflow):
            if task_name == "knowledge_validation":
                chain_state["current_step"] = i
                break

        # Clear previous validation results
        chain_state["validation_phases_completed"] = []
        chain_state["validation_confidence"] = 0.0

        # Continue from validation start
        self._execute_next_task(job_id)

    def _map_task_to_validation_phase(self, task_name: str) -> Optional[ValidationPhase]:
        """Map task names to validation phases."""
        mapping = {
            "knowledge_validation": ValidationPhase.KNOWLEDGE_VALIDATION,
            "pre_llm_validation": ValidationPhase.PRE_LLM_VALIDATION,
            "retrieve_documents": ValidationPhase.DOCUMENT_RETRIEVAL,
            "llm_inference": ValidationPhase.LLM_INFERENCE,
            "post_llm_validation": ValidationPhase.POST_LLM_VALIDATION,
            "final_validation": ValidationPhase.FINAL_ASSESSMENT,
            "meta_validation": ValidationPhase.META_VALIDATION,
            "auto_fetch": ValidationPhase.AUTO_FETCH,
            "process_user_contribution": ValidationPhase.USER_GUIDANCE
        }
        return mapping.get(task_name)

    def _calculate_validation_progress(self, chain_state: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate detailed validation progress."""
        total_phases = len(ValidationPhase)
        completed_phases = len(chain_state.get("validation_phases_completed", []))

        return {
            "completed_phases": completed_phases,
            "total_phases": total_phases,
            "progress_percentage": (completed_phases / total_phases) * 100,
            "current_phase": chain_state.get("current_validation_phase"),
            "phases_completed": chain_state.get("validation_phases_completed", []),
            "confidence": chain_state.get("validation_confidence", 0.0)
        }

    # ===== ENHANCED VALIDATION WORKFLOW STATUS METHODS =====

    def get_validation_progress(self, job_id: str) -> Dict[str, Any]:
        """✅ NEW: Get detailed validation progress."""
        chain_state = self._get_chain_state(job_id)
        if not chain_state or not chain_state.get("validation_workflow"):
            return {"error": "Not a validation workflow or job not found"}

        progress = self._calculate_validation_progress(chain_state)

        return {
            "job_id": job_id,
            "validation_workflow": True,
            "status": chain_state["status"],
            "progress": progress,
            "current_step": chain_state["current_step"],
            "total_steps": chain_state["total_steps"],
            "requires_user_input": chain_state.get("user_input_required", False),
            "user_prompt": chain_state.get("user_prompt"),
            "step_timings": chain_state.get("step_timings", {}),
            "tesla_t4_queue_status": queue_manager.get_queue_status()
        }

    def task_failed(self, job_id: str, error: str) -> None:
        """Enhanced task failure handling with validation workflow cleanup."""
        logger.error(f"Task failed for job {job_id}: {error}")

        # Get chain state to free up the queue and handle validation cleanup
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
        """Complete the entire job chain with validation workflow summary."""
        logger.info(f"Job chain completed for job {job_id}")

        # Get final chain state for timing and validation information
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

        # Enhanced completion info with validation details
        completion_info = {
            "job_chain_completion": {
                "message": "Job chain completed successfully",
                "total_duration": total_duration,
                "step_timings": chain_state.get("step_timings", {}) if chain_state else {},
                "completed_at": time.time()
            }
        }

        # Add validation workflow summary if applicable
        if chain_state and chain_state.get("validation_workflow"):
            validation_summary = self._calculate_validation_progress(chain_state)
            completion_info["validation_summary"] = {
                **validation_summary,
                "workflow_completed": True,
                "final_confidence": chain_state.get("validation_confidence", 0.0),
                "meta_validation_used": chain_state.get("meta_validation_inserted", False),
                "user_input_required": chain_state.get("user_input_required", False)
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

    # ===== EXISTING METHODS (unchanged) =====

    def get_job_chain_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get the current status of a job chain with validation details."""
        chain_state = self._get_chain_state(job_id)
        if not chain_state:
            return None

        current_step = chain_state["current_step"]
        workflow = chain_state["workflow"]

        status = {
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

        # Add validation-specific status
        if chain_state.get("validation_workflow"):
            status["validation_status"] = self._calculate_validation_progress(chain_state)
            status["requires_user_input"] = chain_state.get("user_input_required", False)
            status["user_prompt"] = chain_state.get("user_prompt")

        return status

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
# VALIDATION PIPELINE ACCESS FOR ENHANCED INGESTION SYSTEM (unchanged)
# ===============================================================================

def get_enhanced_ingestion_processors():
    """Get available enhanced ingestion processors for validation pipeline use."""
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
    """Process content using enhanced ingestion system for validation pipeline."""
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
    """Get capabilities of enhanced ingestion processors for validation planning."""
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