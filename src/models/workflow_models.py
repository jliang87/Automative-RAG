"""
Workflow Models - Primary orchestrator for all workflow types
Contains workflow definitions, instances, and coordinates with other models
"""

import uuid
import time
import logging
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field

from .enums import JobType, JobStatus, QueryMode
from .task_models import TaskSpec, TaskInstance
from .document_models import DocumentMetadata

logger = logging.getLogger(__name__)


class WorkflowType(str, Enum):
    """Types of workflows supported by the system"""
    VIDEO_PROCESSING = "video_processing"
    DOCUMENT_PROCESSING = "document_processing"
    QUERY_PROCESSING = "query_processing"
    CAUSATION_ANALYSIS = "causation_analysis"
    VALIDATION_WORKFLOW = "validation_workflow"  # Will be removed in future


class WorkflowStatus(str, Enum):
    """Workflow execution status"""
    PENDING = "pending"
    RUNNING = "running"
    AWAITING_INPUT = "awaiting_input"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class WorkflowDefinition(BaseModel):
    """Definition of a workflow type with its task sequence"""
    workflow_type: WorkflowType
    name: str
    description: str
    task_sequence: List[TaskSpec]
    default_config: Dict[str, Any] = Field(default_factory=dict)
    estimated_duration: int = 30  # seconds
    requires_user_input: bool = False

    class Config:
        use_enum_values = True


class WorkflowInstance(BaseModel):
    """Instance of a running workflow"""
    workflow_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    workflow_type: WorkflowType
    status: WorkflowStatus = WorkflowStatus.PENDING

    # Task management
    task_instances: List[TaskInstance] = Field(default_factory=list)
    current_task_index: int = 0

    # Timing
    created_at: datetime = Field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    # Data flow
    input_data: Dict[str, Any] = Field(default_factory=dict)
    output_data: Dict[str, Any] = Field(default_factory=dict)
    intermediate_data: Dict[str, Any] = Field(default_factory=dict)

    # User interaction
    awaiting_user_input: bool = False
    user_prompt: Optional[Dict[str, Any]] = None

    # Metadata
    metadata: Dict[str, Any] = Field(default_factory=dict)
    error_message: Optional[str] = None

    class Config:
        use_enum_values = True

    def get_current_task(self) -> Optional[TaskInstance]:
        """Get the currently executing task"""
        if 0 <= self.current_task_index < len(self.task_instances):
            return self.task_instances[self.current_task_index]
        return None

    def get_progress_percentage(self) -> float:
        """Calculate workflow progress percentage"""
        if not self.task_instances:
            return 0.0

        completed_tasks = len([t for t in self.task_instances if t.status == "completed"])
        return (completed_tasks / len(self.task_instances)) * 100.0

    def is_complete(self) -> bool:
        """Check if workflow is complete"""
        return self.status in [WorkflowStatus.COMPLETED, WorkflowStatus.FAILED, WorkflowStatus.CANCELLED]


class WorkflowModel:
    """
    Primary orchestrator model that coordinates with other models
    Contains ALL workflow logic and delegates to specialized models
    """

    def __init__(self, task_model, document_model, query_model, causation_model=None):
        self.task_model = task_model
        self.document_model = document_model
        self.query_model = query_model
        self.causation_model = causation_model

        # In-memory storage for demo (would be database in production)
        self.workflow_definitions = {}
        self.workflow_instances = {}

        # Initialize built-in workflow definitions
        self._initialize_workflow_definitions()

    def _initialize_workflow_definitions(self):
        """Initialize built-in workflow definitions"""

        # Video Processing Workflow
        self.workflow_definitions[WorkflowType.VIDEO_PROCESSING] = WorkflowDefinition(
            workflow_type=WorkflowType.VIDEO_PROCESSING,
            name="Video Processing",
            description="Process video content for analysis",
            task_sequence=[
                TaskSpec(task_type="video_download", queue="cpu_tasks"),
                TaskSpec(task_type="video_transcription", queue="transcription_tasks"),
                TaskSpec(task_type="document_indexing", queue="embedding_tasks")
            ],
            estimated_duration=120
        )

        # Document Processing Workflow
        self.workflow_definitions[WorkflowType.DOCUMENT_PROCESSING] = WorkflowDefinition(
            workflow_type=WorkflowType.DOCUMENT_PROCESSING,
            name="Document Processing",
            description="Process documents for indexing",
            task_sequence=[
                TaskSpec(task_type="document_parsing", queue="cpu_tasks"),
                TaskSpec(task_type="content_extraction", queue="cpu_tasks"),
                TaskSpec(task_type="document_indexing", queue="embedding_tasks")
            ],
            estimated_duration=60
        )

        # Query Processing Workflow
        self.workflow_definitions[WorkflowType.QUERY_PROCESSING] = WorkflowDefinition(
            workflow_type=WorkflowType.QUERY_PROCESSING,
            name="Query Processing",
            description="Process user queries and generate responses",
            task_sequence=[
                TaskSpec(task_type="document_retrieval", queue="embedding_tasks"),
                TaskSpec(task_type="llm_inference", queue="llm_tasks"),
                TaskSpec(task_type="response_formatting", queue="cpu_tasks")
            ],
            estimated_duration=30
        )

        # Causation Analysis Workflow (future)
        self.workflow_definitions[WorkflowType.CAUSATION_ANALYSIS] = WorkflowDefinition(
            workflow_type=WorkflowType.CAUSATION_ANALYSIS,
            name="Causation Analysis",
            description="Analyze causal relationships in data",
            task_sequence=[
                TaskSpec(task_type="data_preparation", queue="cpu_tasks"),
                TaskSpec(task_type="causation_detection", queue="llm_tasks"),
                TaskSpec(task_type="relationship_mapping", queue="cpu_tasks")
            ],
            estimated_duration=90
        )

    async def create_workflow(self, workflow_type: WorkflowType, input_data: Dict[str, Any],
                              metadata: Optional[Dict[str, Any]] = None) -> str:
        """Create a new workflow instance"""

        # Get workflow definition
        definition = self.workflow_definitions.get(workflow_type)
        if not definition:
            raise ValueError(f"Unknown workflow type: {workflow_type}")

        # Create workflow instance
        instance = WorkflowInstance(
            workflow_type=workflow_type,
            input_data=input_data,
            metadata=metadata or {}
        )

        # Create task instances from definition
        for task_spec in definition.task_sequence:
            task_instance = await self.task_model.create_task_instance(
                task_spec, instance.workflow_id
            )
            instance.task_instances.append(task_instance)

        # Store instance
        self.workflow_instances[instance.workflow_id] = instance

        logger.info(f"Created workflow {instance.workflow_id} of type {workflow_type}")
        return instance.workflow_id

    async def start_workflow(self, workflow_id: str):
        """Start workflow execution"""
        instance = self.workflow_instances.get(workflow_id)
        if not instance:
            raise ValueError(f"Workflow {workflow_id} not found")

        instance.status = WorkflowStatus.RUNNING
        instance.started_at = datetime.now()

        # Delegate to appropriate model based on workflow type
        if instance.workflow_type == WorkflowType.VIDEO_PROCESSING:
            await self._start_video_processing(instance)
        elif instance.workflow_type == WorkflowType.DOCUMENT_PROCESSING:
            await self._start_document_processing(instance)
        elif instance.workflow_type == WorkflowType.QUERY_PROCESSING:
            await self._start_query_processing(instance)
        elif instance.workflow_type == WorkflowType.CAUSATION_ANALYSIS:
            await self._start_causation_analysis(instance)
        else:
            raise ValueError(f"Unsupported workflow type: {instance.workflow_type}")

        logger.info(f"Started workflow {workflow_id}")

    async def _start_video_processing(self, instance: WorkflowInstance):
        """Start video processing workflow - delegates to DocumentModel"""
        video_url = instance.input_data.get("url")
        if not video_url:
            await self._fail_workflow(instance, "Missing video URL")
            return

        # Call DocumentModel for video processing
        try:
            document_id = await self.document_model.process_video(
                url=video_url,
                workflow_id=instance.workflow_id,
                metadata=instance.input_data.get("metadata", {})
            )
            instance.intermediate_data["document_id"] = document_id
            await self._execute_next_task(instance)
        except Exception as e:
            await self._fail_workflow(instance, str(e))

    async def _start_document_processing(self, instance: WorkflowInstance):
        """Start document processing workflow - delegates to DocumentModel"""
        file_path = instance.input_data.get("file_path")
        content = instance.input_data.get("content")

        if not file_path and not content:
            await self._fail_workflow(instance, "Missing file path or content")
            return

        try:
            if file_path:
                document_id = await self.document_model.process_file(
                    file_path=file_path,
                    workflow_id=instance.workflow_id,
                    metadata=instance.input_data.get("metadata", {})
                )
            else:
                document_id = await self.document_model.process_text(
                    content=content,
                    workflow_id=instance.workflow_id,
                    metadata=instance.input_data.get("metadata", {})
                )

            instance.intermediate_data["document_id"] = document_id
            await self._execute_next_task(instance)
        except Exception as e:
            await self._fail_workflow(instance, str(e))

    async def _start_query_processing(self, instance: WorkflowInstance):
        """Start query processing workflow - delegates to QueryModel"""
        query = instance.input_data.get("query")
        if not query:
            await self._fail_workflow(instance, "Missing query")
            return

        try:
            # Call QueryModel for query processing
            response_data = await self.query_model.process_query(
                query=query,
                query_mode=instance.input_data.get("query_mode", QueryMode.FACTS),
                workflow_id=instance.workflow_id,
                metadata=instance.input_data.get("metadata_filter")
            )
            instance.intermediate_data.update(response_data)
            await self._execute_next_task(instance)
        except Exception as e:
            await self._fail_workflow(instance, str(e))

    async def _start_causation_analysis(self, instance: WorkflowInstance):
        """Start causation analysis workflow - delegates to CausationModel (future)"""
        if not self.causation_model:
            await self._fail_workflow(instance, "Causation analysis not available")
            return

        # Future implementation
        await self._fail_workflow(instance, "Causation analysis not implemented yet")

    async def _execute_next_task(self, instance: WorkflowInstance):
        """Execute the next task in the workflow"""
        current_task = instance.get_current_task()
        if not current_task:
            await self._complete_workflow(instance)
            return

        # Mark current task as running
        current_task.status = "running"
        current_task.started_at = datetime.now()

        # Delegate task execution to TaskModel
        try:
            result = await self.task_model.execute_task(
                current_task,
                instance.intermediate_data
            )

            # Update task and move to next
            current_task.status = "completed"
            current_task.completed_at = datetime.now()
            current_task.output_data = result

            # Move to next task
            instance.current_task_index += 1
            await self._execute_next_task(instance)

        except Exception as e:
            current_task.status = "failed"
            current_task.error_message = str(e)
            await self._fail_workflow(instance, f"Task {current_task.task_type} failed: {str(e)}")

    async def task_completed(self, workflow_id: str, task_result: Dict[str, Any]):
        """Handle task completion callback"""
        instance = self.workflow_instances.get(workflow_id)
        if not instance:
            logger.error(f"Workflow {workflow_id} not found for task completion")
            return

        # Update intermediate data with task result
        instance.intermediate_data.update(task_result)

        # Continue with next task
        instance.current_task_index += 1
        await self._execute_next_task(instance)

    async def task_failed(self, workflow_id: str, error_message: str):
        """Handle task failure callback"""
        instance = self.workflow_instances.get(workflow_id)
        if not instance:
            logger.error(f"Workflow {workflow_id} not found for task failure")
            return

        await self._fail_workflow(instance, error_message)

    async def _complete_workflow(self, instance: WorkflowInstance):
        """Complete the workflow successfully"""
        instance.status = WorkflowStatus.COMPLETED
        instance.completed_at = datetime.now()

        # Prepare final output based on workflow type
        if instance.workflow_type == WorkflowType.QUERY_PROCESSING:
            instance.output_data = {
                "query": instance.input_data.get("query"),
                "answer": instance.intermediate_data.get("answer"),
                "documents": instance.intermediate_data.get("documents", []),
                "execution_time": self._calculate_execution_time(instance),
                "workflow_id": instance.workflow_id
            }
        else:
            instance.output_data = instance.intermediate_data.copy()

        logger.info(f"Completed workflow {instance.workflow_id}")

    async def _fail_workflow(self, instance: WorkflowInstance, error_message: str):
        """Fail the workflow with error message"""
        instance.status = WorkflowStatus.FAILED
        instance.completed_at = datetime.now()
        instance.error_message = error_message

        logger.error(f"Failed workflow {instance.workflow_id}: {error_message}")

    def _calculate_execution_time(self, instance: WorkflowInstance) -> float:
        """Calculate workflow execution time in seconds"""
        if instance.started_at and instance.completed_at:
            return (instance.completed_at - instance.started_at).total_seconds()
        return 0.0

    async def get_workflow_status(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """Get workflow status and progress"""
        instance = self.workflow_instances.get(workflow_id)
        if not instance:
            return None

        current_task = instance.get_current_task()

        return {
            "workflow_id": workflow_id,
            "workflow_type": instance.workflow_type.value,
            "status": instance.status.value,
            "progress_percentage": instance.get_progress_percentage(),
            "current_task": current_task.task_type if current_task else None,
            "total_tasks": len(instance.task_instances),
            "completed_tasks": len([t for t in instance.task_instances if t.status == "completed"]),
            "awaiting_user_input": instance.awaiting_user_input,
            "user_prompt": instance.user_prompt,
            "created_at": instance.created_at.isoformat(),
            "execution_time": self._calculate_execution_time(instance),
            "error_message": instance.error_message
        }

    async def get_workflow_result(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """Get final workflow result"""
        instance = self.workflow_instances.get(workflow_id)
        if not instance or not instance.is_complete():
            return None

        if instance.status == WorkflowStatus.COMPLETED:
            return instance.output_data
        else:
            return {"error": instance.error_message}

    async def cancel_workflow(self, workflow_id: str):
        """Cancel a running workflow"""
        instance = self.workflow_instances.get(workflow_id)
        if not instance:
            raise ValueError(f"Workflow {workflow_id} not found")

        instance.status = WorkflowStatus.CANCELLED
        instance.completed_at = datetime.now()

        # Cancel any running tasks
        current_task = instance.get_current_task()
        if current_task and current_task.status == "running":
            await self.task_model.cancel_task(current_task.task_id)

        logger.info(f"Cancelled workflow {workflow_id}")

    def get_available_workflow_types(self) -> List[Dict[str, Any]]:
        """Get list of available workflow types"""
        return [
            {
                "workflow_type": wf_type.value,
                "name": definition.name,
                "description": definition.description,
                "estimated_duration": definition.estimated_duration,
                "requires_user_input": definition.requires_user_input
            }
            for wf_type, definition in self.workflow_definitions.items()
        ]

    async def cleanup_completed_workflows(self, retention_hours: int = 24):
        """Clean up old completed workflows"""
        cutoff_time = datetime.now() - timedelta(hours=retention_hours)

        workflows_to_remove = []
        for workflow_id, instance in self.workflow_instances.items():
            if (instance.is_complete() and
                    instance.completed_at and
                    instance.completed_at < cutoff_time):
                workflows_to_remove.append(workflow_id)

        for workflow_id in workflows_to_remove:
            del self.workflow_instances[workflow_id]

        logger.info(f"Cleaned up {len(workflows_to_remove)} old workflows")
        return len(workflows_to_remove)