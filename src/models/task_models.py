"""
Task Models - Individual task creation, management, and status tracking
Called by WorkflowModel when tasks need to be created
Delegates complex task logic to TaskService
"""

import uuid
import time
import logging
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class TaskType(str, Enum):
    """Types of tasks that can be executed"""
    # Video processing tasks
    VIDEO_DOWNLOAD = "video_download"
    VIDEO_TRANSCRIPTION = "video_transcription"

    # Document processing tasks
    DOCUMENT_PARSING = "document_parsing"
    CONTENT_EXTRACTION = "content_extraction"
    DOCUMENT_INDEXING = "document_indexing"

    # Query processing tasks
    DOCUMENT_RETRIEVAL = "document_retrieval"
    LLM_INFERENCE = "llm_inference"
    RESPONSE_FORMATTING = "response_formatting"

    # Causation analysis tasks (future)
    DATA_PREPARATION = "data_preparation"
    CAUSATION_DETECTION = "causation_detection"
    RELATIONSHIP_MAPPING = "relationship_mapping"

    # General tasks
    DATA_VALIDATION = "data_validation"
    NOTIFICATION = "notification"


class TaskStatus(str, Enum):
    """Task execution status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RETRYING = "retrying"


class TaskPriority(str, Enum):
    """Task execution priority"""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


class TaskSpec(BaseModel):
    """Specification for a task type"""
    task_type: TaskType
    queue: str  # Queue name for task execution
    timeout_seconds: int = 300
    max_retries: int = 3
    retry_delay_seconds: int = 30
    required_input_fields: List[str] = Field(default_factory=list)
    output_fields: List[str] = Field(default_factory=list)
    dependencies: List[str] = Field(default_factory=list)  # Task types this depends on

    class Config:
        use_enum_values = True


class TaskInstance(BaseModel):
    """Instance of a running task"""
    task_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    task_type: TaskType
    workflow_id: str
    status: TaskStatus = TaskStatus.PENDING
    priority: TaskPriority = TaskPriority.NORMAL

    # Execution details
    queue: str
    assigned_worker: Optional[str] = None

    # Timing
    created_at: datetime = Field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    # Data
    input_data: Dict[str, Any] = Field(default_factory=dict)
    output_data: Dict[str, Any] = Field(default_factory=dict)

    # Error handling
    error_message: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3

    # Metadata
    metadata: Dict[str, Any] = Field(default_factory=dict)
    progress_percentage: float = 0.0
    progress_message: str = ""

    class Config:
        use_enum_values = True

    def get_execution_time(self) -> float:
        """Get task execution time in seconds"""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        elif self.started_at:
            return (datetime.now() - self.started_at).total_seconds()
        return 0.0

    def can_retry(self) -> bool:
        """Check if task can be retried"""
        return self.retry_count < self.max_retries and self.status == TaskStatus.FAILED


class TaskModel:
    """
    Model for task management - called by WorkflowModel
    Handles task creation, persistence, and delegates execution to TaskService
    """

    def __init__(self, task_service):
        self.task_service = task_service

        # In-memory storage for demo (would be database in production)
        self.task_instances = {}
        self.task_specs = {}

        # Initialize built-in task specifications
        self._initialize_task_specs()

    def _initialize_task_specs(self):
        """Initialize built-in task specifications"""

        # Video processing task specs
        self.task_specs[TaskType.VIDEO_DOWNLOAD] = TaskSpec(
            task_type=TaskType.VIDEO_DOWNLOAD,
            queue="cpu_tasks",
            timeout_seconds=600,
            required_input_fields=["url"],
            output_fields=["file_path", "metadata"]
        )

        self.task_specs[TaskType.VIDEO_TRANSCRIPTION] = TaskSpec(
            task_type=TaskType.VIDEO_TRANSCRIPTION,
            queue="transcription_tasks",
            timeout_seconds=1800,
            required_input_fields=["file_path"],
            output_fields=["transcript", "language", "confidence"],
            dependencies=["video_download"]
        )

        # Document processing task specs
        self.task_specs[TaskType.DOCUMENT_PARSING] = TaskSpec(
            task_type=TaskType.DOCUMENT_PARSING,
            queue="cpu_tasks",
            timeout_seconds=300,
            required_input_fields=["file_path"],
            output_fields=["text_content", "metadata", "structure"]
        )

        self.task_specs[TaskType.CONTENT_EXTRACTION] = TaskSpec(
            task_type=TaskType.CONTENT_EXTRACTION,
            queue="cpu_tasks",
            timeout_seconds=180,
            required_input_fields=["text_content"],
            output_fields=["extracted_content", "entities", "keywords"],
            dependencies=["document_parsing"]
        )

        self.task_specs[TaskType.DOCUMENT_INDEXING] = TaskSpec(
            task_type=TaskType.DOCUMENT_INDEXING,
            queue="embedding_tasks",
            timeout_seconds=300,
            required_input_fields=["extracted_content"],
            output_fields=["document_id", "embeddings", "index_status"]
        )

        # Query processing task specs
        self.task_specs[TaskType.DOCUMENT_RETRIEVAL] = TaskSpec(
            task_type=TaskType.DOCUMENT_RETRIEVAL,
            queue="embedding_tasks",
            timeout_seconds=120,
            required_input_fields=["query"],
            output_fields=["documents", "scores", "retrieval_metadata"]
        )

        self.task_specs[TaskType.LLM_INFERENCE] = TaskSpec(
            task_type=TaskType.LLM_INFERENCE,
            queue="llm_tasks",
            timeout_seconds=180,
            required_input_fields=["query", "documents"],
            output_fields=["answer", "confidence", "reasoning"],
            dependencies=["document_retrieval"]
        )

        self.task_specs[TaskType.RESPONSE_FORMATTING] = TaskSpec(
            task_type=TaskType.RESPONSE_FORMATTING,
            queue="cpu_tasks",
            timeout_seconds=60,
            required_input_fields=["answer", "documents"],
            output_fields=["formatted_response", "metadata"],
            dependencies=["llm_inference"]
        )

    async def create_task_instance(self, task_spec: TaskSpec, workflow_id: str,
                                   input_data: Optional[Dict[str, Any]] = None,
                                   priority: TaskPriority = TaskPriority.NORMAL) -> TaskInstance:
        """Create a new task instance from specification"""

        instance = TaskInstance(
            task_type=task_spec.task_type,
            workflow_id=workflow_id,
            queue=task_spec.queue,
            priority=priority,
            max_retries=task_spec.max_retries,
            input_data=input_data or {}
        )

        # Store instance
        self.task_instances[instance.task_id] = instance

        logger.info(f"Created task {instance.task_id} of type {task_spec.task_type}")
        return instance

    async def execute_task(self, task_instance: TaskInstance,
                           workflow_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a task instance - delegates to TaskService"""

        # Validate input data
        task_spec = self.task_specs.get(task_instance.task_type)
        if not task_spec:
            raise ValueError(f"Unknown task type: {task_instance.task_type}")

        # Check required input fields
        missing_fields = []
        for field in task_spec.required_input_fields:
            if field not in task_instance.input_data and field not in workflow_data:
                missing_fields.append(field)

        if missing_fields:
            raise ValueError(f"Missing required input fields: {missing_fields}")

        # Merge workflow data with task input data
        execution_data = {**workflow_data, **task_instance.input_data}

        # Update task status
        task_instance.status = TaskStatus.RUNNING
        task_instance.started_at = datetime.now()

        try:
            # Delegate execution to TaskService
            result = await self.task_service.execute_task(
                task_type=task_instance.task_type,
                task_id=task_instance.task_id,
                input_data=execution_data,
                metadata=task_instance.metadata
            )

            # Update task with result
            task_instance.status = TaskStatus.COMPLETED
            task_instance.completed_at = datetime.now()
            task_instance.output_data = result
            task_instance.progress_percentage = 100.0
            task_instance.progress_message = "Task completed successfully"

            logger.info(f"Completed task {task_instance.task_id}")
            return result

        except Exception as e:
            # Handle task failure
            task_instance.status = TaskStatus.FAILED
            task_instance.completed_at = datetime.now()
            task_instance.error_message = str(e)
            task_instance.retry_count += 1

            logger.error(f"Failed task {task_instance.task_id}: {str(e)}")

            # Check if task can be retried
            if task_instance.can_retry():
                logger.info(f"Task {task_instance.task_id} will be retried")
                task_instance.status = TaskStatus.RETRYING
                # In a real implementation, this would schedule a retry

            raise

    async def cancel_task(self, task_id: str):
        """Cancel a running task"""
        task_instance = self.task_instances.get(task_id)
        if not task_instance:
            raise ValueError(f"Task {task_id} not found")

        if task_instance.status == TaskStatus.RUNNING:
            # Delegate cancellation to TaskService
            await self.task_service.cancel_task(task_id)

            task_instance.status = TaskStatus.CANCELLED
            task_instance.completed_at = datetime.now()
            task_instance.progress_message = "Task cancelled"

            logger.info(f"Cancelled task {task_id}")

    async def retry_task(self, task_id: str) -> bool:
        """Retry a failed task"""
        task_instance = self.task_instances.get(task_id)
        if not task_instance:
            raise ValueError(f"Task {task_id} not found")

        if not task_instance.can_retry():
            return False

        # Reset task for retry
        task_instance.status = TaskStatus.PENDING
        task_instance.started_at = None
        task_instance.completed_at = None
        task_instance.error_message = None
        task_instance.progress_percentage = 0.0
        task_instance.progress_message = "Retrying task"

        logger.info(f"Retrying task {task_id} (attempt {task_instance.retry_count + 1})")
        return True

    async def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get task status and progress"""
        task_instance = self.task_instances.get(task_id)
        if not task_instance:
            return None

        return {
            "task_id": task_id,
            "task_type": task_instance.task_type.value,
            "workflow_id": task_instance.workflow_id,
            "status": task_instance.status.value,
            "priority": task_instance.priority.value,
            "progress_percentage": task_instance.progress_percentage,
            "progress_message": task_instance.progress_message,
            "created_at": task_instance.created_at.isoformat(),
            "started_at": task_instance.started_at.isoformat() if task_instance.started_at else None,
            "completed_at": task_instance.completed_at.isoformat() if task_instance.completed_at else None,
            "execution_time": task_instance.get_execution_time(),
            "retry_count": task_instance.retry_count,
            "max_retries": task_instance.max_retries,
            "can_retry": task_instance.can_retry(),
            "error_message": task_instance.error_message,
            "queue": task_instance.queue,
            "assigned_worker": task_instance.assigned_worker
        }

    async def get_tasks_by_workflow(self, workflow_id: str) -> List[Dict[str, Any]]:
        """Get all tasks for a workflow"""
        workflow_tasks = []

        for task_instance in self.task_instances.values():
            if task_instance.workflow_id == workflow_id:
                task_status = await self.get_task_status(task_instance.task_id)
                if task_status:
                    workflow_tasks.append(task_status)

        # Sort by creation time
        workflow_tasks.sort(key=lambda x: x["created_at"])
        return workflow_tasks

    async def get_tasks_by_status(self, status: TaskStatus) -> List[Dict[str, Any]]:
        """Get all tasks with a specific status"""
        status_tasks = []

        for task_instance in self.task_instances.values():
            if task_instance.status == status:
                task_status = await self.get_task_status(task_instance.task_id)
                if task_status:
                    status_tasks.append(task_status)

        return status_tasks

    async def update_task_progress(self, task_id: str, progress_percentage: float,
                                   progress_message: str = ""):
        """Update task progress"""
        task_instance = self.task_instances.get(task_id)
        if not task_instance:
            return

        task_instance.progress_percentage = max(0.0, min(100.0, progress_percentage))
        task_instance.progress_message = progress_message

        logger.debug(f"Updated task {task_id} progress: {progress_percentage}% - {progress_message}")

    def get_task_spec(self, task_type: TaskType) -> Optional[TaskSpec]:
        """Get task specification for a task type"""
        return self.task_specs.get(task_type)

    def get_available_task_types(self) -> List[Dict[str, Any]]:
        """Get list of available task types"""
        return [
            {
                "task_type": task_type.value,
                "queue": spec.queue,
                "timeout_seconds": spec.timeout_seconds,
                "max_retries": spec.max_retries,
                "required_input_fields": spec.required_input_fields,
                "output_fields": spec.output_fields,
                "dependencies": spec.dependencies
            }
            for task_type, spec in self.task_specs.items()
        ]

    async def cleanup_completed_tasks(self, retention_hours: int = 24):
        """Clean up old completed tasks"""
        from datetime import timedelta

        cutoff_time = datetime.now() - timedelta(hours=retention_hours)

        tasks_to_remove = []
        for task_id, task_instance in self.task_instances.items():
            if (task_instance.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED] and
                    task_instance.completed_at and
                    task_instance.completed_at < cutoff_time):
                tasks_to_remove.append(task_id)

        for task_id in tasks_to_remove:
            del self.task_instances[task_id]

        logger.info(f"Cleaned up {len(tasks_to_remove)} old tasks")
        return len(tasks_to_remove)

    async def get_task_statistics(self) -> Dict[str, Any]:
        """Get task execution statistics"""
        total_tasks = len(self.task_instances)
        status_counts = {}
        type_counts = {}
        queue_counts = {}

        total_execution_time = 0.0
        completed_tasks = 0

        for task_instance in self.task_instances.values():
            # Count by status
            status = task_instance.status.value
            status_counts[status] = status_counts.get(status, 0) + 1

            # Count by type
            task_type = task_instance.task_type.value
            type_counts[task_type] = type_counts.get(task_type, 0) + 1

            # Count by queue
            queue = task_instance.queue
            queue_counts[queue] = queue_counts.get(queue, 0) + 1

            # Calculate execution time for completed tasks
            if task_instance.status == TaskStatus.COMPLETED:
                total_execution_time += task_instance.get_execution_time()
                completed_tasks += 1

        average_execution_time = total_execution_time / completed_tasks if completed_tasks > 0 else 0.0
        success_rate = (status_counts.get("completed", 0) / total_tasks * 100) if total_tasks > 0 else 0.0

        return {
            "total_tasks": total_tasks,
            "status_distribution": status_counts,
            "type_distribution": type_counts,
            "queue_distribution": queue_counts,
            "average_execution_time": average_execution_time,
            "success_rate": success_rate,
            "completed_tasks": completed_tasks,
            "failed_tasks": status_counts.get("failed", 0),
            "retry_rate": sum(
                task.retry_count for task in self.task_instances.values()) / total_tasks if total_tasks > 0 else 0.0
        }