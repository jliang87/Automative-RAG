import logging
import uuid
import time
from typing import Dict, List, Any, Optional
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class ValidationType(Enum):
    BASIC = "basic"
    COMPREHENSIVE = "comprehensive"
    USER_GUIDED = "user_guided"


class ValidationStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ValidationStepType(Enum):
    CONTENT_CHECK = "content_check"
    FACTUAL_VERIFICATION = "factual_verification"
    USER_CONFIRMATION = "user_confirmation"
    QUALITY_ASSESSMENT = "quality_assessment"


class ValidationStep:
    def __init__(self, step_id: str, step_type: ValidationStepType, name: str, description: str):
        self.step_id = step_id
        self.step_type = step_type
        self.name = name
        self.description = description
        self.status = ValidationStatus.PENDING
        self.confidence = 0.0
        self.result = None
        self.error_message = None
        self.user_input_required = False
        self.completed_at = None


class ValidationWorkflow:
    def __init__(self, validation_id: str, job_id: str, validation_type: ValidationType):
        self.validation_id = validation_id
        self.job_id = job_id
        self.validation_type = validation_type
        self.steps = []
        self.overall_status = ValidationStatus.PENDING
        self.overall_confidence = 0.0
        self.validation_passed = False
        self.awaiting_user_input = False
        self.current_step = None
        self.issues_identified = []
        self.final_recommendations = []
        self.created_at = datetime.utcnow()
        self.updated_at = datetime.utcnow()


class ValidationService:
    """Service for handling validation workflows."""

    def __init__(self, redis_client, job_tracker):
        self.redis_client = redis_client
        self.job_tracker = job_tracker
        self.workflows = {}  # In-memory storage for demo

    async def create_workflow(self, job_id: str, config: Dict[str, Any]) -> ValidationWorkflow:
        """Create a new validation workflow."""
        validation_id = str(uuid.uuid4())

        # Determine validation type from config
        validation_type = ValidationType.BASIC
        if config.get('validation_type'):
            try:
                validation_type = ValidationType(config['validation_type'])
            except ValueError:
                validation_type = ValidationType.BASIC
        elif config.get('comprehensive'):
            validation_type = ValidationType.COMPREHENSIVE
        elif config.get('user_guided'):
            validation_type = ValidationType.USER_GUIDED

        workflow = ValidationWorkflow(
            validation_id=validation_id,
            job_id=job_id,
            validation_type=validation_type
        )

        # Add steps based on validation type
        self._add_validation_steps(workflow)

        self.workflows[validation_id] = workflow
        logger.info(f"Created validation workflow {validation_id} for job {job_id}")
        return workflow

    def _add_validation_steps(self, workflow: ValidationWorkflow):
        """Add appropriate validation steps based on workflow type."""
        if workflow.validation_type == ValidationType.BASIC:
            workflow.steps = [
                ValidationStep("step_1", ValidationStepType.CONTENT_CHECK, "内容检查", "基础内容验证"),
                ValidationStep("step_2", ValidationStepType.QUALITY_ASSESSMENT, "质量评估", "答案质量评估")
            ]
        elif workflow.validation_type == ValidationType.COMPREHENSIVE:
            workflow.steps = [
                ValidationStep("step_1", ValidationStepType.CONTENT_CHECK, "内容检查", "全面内容验证"),
                ValidationStep("step_2", ValidationStepType.FACTUAL_VERIFICATION, "事实验证", "事实准确性验证"),
                ValidationStep("step_3", ValidationStepType.QUALITY_ASSESSMENT, "质量评估", "深度质量评估"),
                ValidationStep("step_4", ValidationStepType.USER_CONFIRMATION, "用户确认", "用户最终确认")
            ]
        elif workflow.validation_type == ValidationType.USER_GUIDED:
            workflow.steps = [
                ValidationStep("step_1", ValidationStepType.CONTENT_CHECK, "内容检查", "用户引导验证"),
                ValidationStep("step_2", ValidationStepType.USER_CONFIRMATION, "用户引导", "用户参与验证过程"),
                ValidationStep("step_3", ValidationStepType.QUALITY_ASSESSMENT, "质量评估", "基于用户输入的评估")
            ]

    async def get_validation_workflow(self, validation_id: str) -> Optional[ValidationWorkflow]:
        """Get validation workflow by ID."""
        return self.workflows.get(validation_id)

    async def get_validation_progress(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get validation progress for a job."""
        # Find workflow by job_id
        for workflow in self.workflows.values():
            if workflow.job_id == job_id:
                completed_steps = len([s for s in workflow.steps if s.status == ValidationStatus.COMPLETED])

                return {
                    "validation_id": workflow.validation_id,
                    "status": workflow.overall_status.value,
                    "progress": completed_steps / len(workflow.steps) if workflow.steps else 0,
                    "current_step": workflow.current_step.value if workflow.current_step else None,
                    "steps_completed": completed_steps,
                    "total_steps": len(workflow.steps),
                    "awaiting_user_input": workflow.awaiting_user_input,
                    "confidence": workflow.overall_confidence
                }
        return None

    async def submit_user_choice(self, job_id: str, choice_data: Dict[str, Any]) -> Dict[str, Any]:
        """Submit user choice for validation."""
        logger.info(f"User choice submitted for job {job_id}: {choice_data.get('decision')}")

        # Find workflow and update it
        for workflow in self.workflows.values():
            if workflow.job_id == job_id:
                workflow.awaiting_user_input = False
                workflow.updated_at = datetime.utcnow()

                # Process the user choice
                choice = choice_data.get('decision', '')
                if choice == 'approve':
                    workflow.validation_passed = True
                    workflow.overall_status = ValidationStatus.COMPLETED
                elif choice == 'reject':
                    workflow.validation_passed = False
                    workflow.overall_status = ValidationStatus.FAILED

                break

        return {"message": "User choice recorded", "job_id": job_id, "choice": choice_data}

    async def restart_validation_workflow(self, job_id: str, restart_data: Optional[Dict[str, Any]] = None) -> Dict[
        str, Any]:
        """Restart validation workflow."""
        # Find and reset workflow
        for workflow in self.workflows.values():
            if workflow.job_id == job_id:
                workflow.overall_status = ValidationStatus.PENDING
                workflow.overall_confidence = 0.0
                workflow.awaiting_user_input = False
                workflow.validation_passed = False
                workflow.updated_at = datetime.utcnow()

                # Reset all steps
                for step in workflow.steps:
                    step.status = ValidationStatus.PENDING
                    step.confidence = 0.0
                    step.completed_at = None
                    step.result = None
                    step.error_message = None

                logger.info(f"Restarted validation workflow for job {job_id}")
                break

        return {"message": "Validation workflow restarted", "job_id": job_id}

    async def cancel_validation_workflow(self, job_id: str) -> Dict[str, str]:
        """Cancel validation workflow."""
        # Find and cancel workflow
        workflows_to_remove = []
        for validation_id, workflow in self.workflows.items():
            if workflow.job_id == job_id:
                workflow.overall_status = ValidationStatus.CANCELLED
                workflow.updated_at = datetime.utcnow()
                workflows_to_remove.append(validation_id)
                logger.info(f"Cancelled validation workflow for job {job_id}")

        # Remove cancelled workflows
        for validation_id in workflows_to_remove:
            del self.workflows[validation_id]

        return {"message": "Validation workflow cancelled", "job_id": job_id}

    async def update_workflow_step(self, validation_id: str, step_id: str, status: ValidationStatus,
                                   confidence: float = None, result: Dict[str, Any] = None,
                                   error_message: str = None) -> bool:
        """Update a specific step in a validation workflow."""
        workflow = self.workflows.get(validation_id)
        if not workflow:
            return False

        # Find and update the step
        for step in workflow.steps:
            if step.step_id == step_id:
                step.status = status
                if confidence is not None:
                    step.confidence = confidence
                if result is not None:
                    step.result = result
                if error_message is not None:
                    step.error_message = error_message
                if status == ValidationStatus.COMPLETED:
                    step.completed_at = datetime.utcnow()
                break

        # Update overall workflow status and confidence
        completed_steps = [s for s in workflow.steps if s.status == ValidationStatus.COMPLETED]
        failed_steps = [s for s in workflow.steps if s.status == ValidationStatus.FAILED]

        if len(completed_steps) == len(workflow.steps):
            workflow.overall_status = ValidationStatus.COMPLETED
            workflow.validation_passed = True
        elif failed_steps:
            workflow.overall_status = ValidationStatus.FAILED
            workflow.validation_passed = False
        else:
            workflow.overall_status = ValidationStatus.IN_PROGRESS

        # Calculate overall confidence
        if workflow.steps:
            total_confidence = sum(step.confidence for step in workflow.steps)
            workflow.overall_confidence = total_confidence / len(workflow.steps)

        workflow.updated_at = datetime.utcnow()
        return True

    async def get_workflow_summary(self, validation_id: str) -> Optional[Dict[str, Any]]:
        """Get a summary of the validation workflow."""
        workflow = self.workflows.get(validation_id)
        if not workflow:
            return None

        step_summaries = []
        for step in workflow.steps:
            step_summaries.append({
                "step_id": step.step_id,
                "name": step.name,
                "status": step.status.value,
                "confidence": step.confidence,
                "completed_at": step.completed_at.isoformat() if step.completed_at else None
            })

        return {
            "validation_id": workflow.validation_id,
            "job_id": workflow.job_id,
            "validation_type": workflow.validation_type.value,
            "overall_status": workflow.overall_status.value,
            "overall_confidence": workflow.overall_confidence,
            "validation_passed": workflow.validation_passed,
            "awaiting_user_input": workflow.awaiting_user_input,
            "steps": step_summaries,
            "issues_identified": workflow.issues_identified,
            "final_recommendations": workflow.final_recommendations,
            "created_at": workflow.created_at.isoformat(),
            "updated_at": workflow.updated_at.isoformat()
        }