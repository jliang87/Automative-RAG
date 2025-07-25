from datetime import datetime
from typing import Dict, List, Optional, Union, Any
from pydantic import BaseModel, Field

from .enums import JobType, JobStatus


# ============================================================================
# Core Job Models
# ============================================================================

class JobDetails(BaseModel):
    """Detailed information about a background job."""
    job_id: str
    job_type: str
    status: JobStatus
    created_at: float
    updated_at: float
    result: Optional[Any] = None
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

    # Validation data (if applicable)
    validation_workflow: Optional[Any] = None  # ValidationWorkflow from validation_models


class JobMetadata(BaseModel):
    """Metadata for job processing."""
    query: Optional[str] = None
    query_mode: Optional[str] = None
    complexity_level: str = "simple"
    estimated_time: int = 30
    validation_enabled: bool = False
    validation_id: Optional[str] = None
    validation_type: Optional[str] = None
    priority: int = 0
    retry_count: int = 0
    max_retries: int = 3


class JobProgress(BaseModel):
    """Progress information for a job."""
    job_id: str
    current_step: str
    total_steps: int
    completed_steps: int
    progress_percentage: float = Field(ge=0.0, le=100.0)
    estimated_remaining_time: Optional[int] = None
    last_update: datetime


class JobResult(BaseModel):
    """Result data from completed job."""
    job_id: str
    status: JobStatus
    result_data: Dict[str, Any]
    execution_time: float
    completed_at: datetime
    error_details: Optional[str] = None


# ============================================================================
# Job Chain and Orchestration Models
# ============================================================================

class JobChainStep(BaseModel):
    """Individual step in a job chain."""
    step_id: str
    step_name: str
    step_type: str
    depends_on: List[str] = []
    status: str = "pending"
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error: Optional[str] = None
    result: Optional[Dict[str, Any]] = None


class JobChainStatus(BaseModel):
    """Status of an entire job chain."""
    job_id: str
    chain_type: str
    overall_status: str
    current_step: Optional[str] = None
    steps: List[JobChainStep] = []
    started_at: datetime
    updated_at: datetime
    estimated_completion: Optional[datetime] = None


class TaskResult(BaseModel):
    """Result from a specific task execution."""
    task_name: str
    task_id: str
    status: str
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    execution_time: float
    timestamp: datetime


# ============================================================================
# Queue Management Models
# ============================================================================

class QueueStatus(BaseModel):
    """Status of job queues."""
    queue_name: str
    active_jobs: int = 0
    pending_jobs: int = 0
    completed_jobs_today: int = 0
    failed_jobs_today: int = 0
    average_processing_time: float = 0.0
    queue_health: str = "healthy"
    last_updated: datetime


class QueueStatistics(BaseModel):
    """Statistics across all queues."""
    total_active: int = 0
    total_pending: int = 0
    total_completed: int = 0
    total_failed: int = 0
    throughput_per_hour: int = 0
    average_wait_time: float = 0.0
    peak_queue_size: int = 0
    system_load: float = 0.0


class WorkerStatus(BaseModel):
    """Status of job workers."""
    worker_id: str
    worker_type: str
    status: str  # "active", "idle", "error", "offline"
    current_job: Optional[str] = None
    jobs_processed: int = 0
    last_heartbeat: datetime
    performance_metrics: Dict[str, float] = {}


# ============================================================================
# Processing Pipeline Models
# ============================================================================

class PipelineStep(BaseModel):
    """Step in a processing pipeline."""
    step_name: str
    processor_type: str
    config: Dict[str, Any] = {}
    depends_on: List[str] = []
    timeout_seconds: int = 300
    retry_policy: Dict[str, Any] = {}


class PipelineConfig(BaseModel):
    """Configuration for a processing pipeline."""
    pipeline_id: str
    pipeline_name: str
    description: str
    steps: List[PipelineStep]
    default_timeout: int = 1800
    max_retries: int = 3
    failure_policy: str = "fail_fast"


class PipelineExecution(BaseModel):
    """Execution instance of a pipeline."""
    execution_id: str
    pipeline_id: str
    job_id: str
    status: str
    current_step: Optional[str] = None
    started_at: datetime
    completed_at: Optional[datetime] = None
    step_results: List[TaskResult] = []
    final_result: Optional[Dict[str, Any]] = None


# ============================================================================
# Background Processing Models
# ============================================================================

class VideoProcessingJob(BaseModel):
    """Specific job data for video processing."""
    url: str
    platform: str
    video_id: str
    transcription_language: Optional[str] = None
    use_enhanced_processing: bool = True
    output_format: str = "enhanced_chunks"


class PDFProcessingJob(BaseModel):
    """Specific job data for PDF processing."""
    file_path: str
    use_ocr: bool = True
    extract_tables: bool = True
    language_hint: Optional[str] = None
    processing_options: Dict[str, Any] = {}


class TextProcessingJob(BaseModel):
    """Specific job data for text processing."""
    content: str
    source_metadata: Dict[str, Any]
    processing_mode: str = "standard"
    enhancement_options: Dict[str, Any] = {}


class BatchProcessingJob(BaseModel):
    """Job data for batch processing operations."""
    batch_id: str
    item_count: int
    individual_jobs: List[str] = []
    batch_metadata: Dict[str, Any] = {}
    completion_callback: Optional[str] = None


# ============================================================================
# Error and Retry Models
# ============================================================================

class JobError(BaseModel):
    """Error information for failed jobs."""
    error_type: str
    error_code: Optional[str] = None
    message: str
    details: Optional[str] = None
    stack_trace: Optional[str] = None
    timestamp: datetime
    retry_count: int = 0
    recoverable: bool = True


class RetryPolicy(BaseModel):
    """Retry policy for job execution."""
    max_retries: int = 3
    initial_delay: int = 5  # seconds
    backoff_factor: float = 2.0
    max_delay: int = 300  # seconds
    retry_on_errors: List[str] = []
    abort_on_errors: List[str] = []


class JobTimeout(BaseModel):
    """Timeout configuration for jobs."""
    execution_timeout: int = 1800  # seconds
    step_timeout: int = 300  # seconds
    queue_timeout: int = 3600  # seconds
    cleanup_timeout: int = 60  # seconds


# ============================================================================
# Monitoring and Metrics Models
# ============================================================================

class JobMetrics(BaseModel):
    """Performance metrics for jobs."""
    job_id: str
    execution_time: float
    queue_time: float
    memory_usage: Optional[float] = None
    cpu_usage: Optional[float] = None
    network_io: Optional[float] = None
    disk_io: Optional[float] = None
    custom_metrics: Dict[str, float] = {}


class SystemMetrics(BaseModel):
    """Overall system performance metrics."""
    timestamp: datetime
    active_jobs: int
    queue_sizes: Dict[str, int]
    worker_utilization: float
    memory_usage: float
    cpu_usage: float
    disk_usage: float
    network_throughput: float
    error_rate: float


class PerformanceReport(BaseModel):
    """Performance report for a time period."""
    start_time: datetime
    end_time: datetime
    total_jobs: int
    successful_jobs: int
    failed_jobs: int
    average_execution_time: float
    p95_execution_time: float
    throughput: float  # jobs per hour
    error_rate: float
    bottlenecks: List[str] = []
    recommendations: List[str] = []


# ============================================================================
# Job Lifecycle Events
# ============================================================================

class JobEvent(BaseModel):
    """Event in job lifecycle."""
    event_id: str
    job_id: str
    event_type: str  # "created", "started", "completed", "failed", "retried"
    timestamp: datetime
    details: Dict[str, Any] = {}
    metadata: Dict[str, Any] = {}


class JobAuditLog(BaseModel):
    """Audit log for job operations."""
    log_id: str
    job_id: str
    operation: str
    operator: Optional[str] = None  # user/system that triggered
    timestamp: datetime
    before_state: Optional[Dict[str, Any]] = None
    after_state: Optional[Dict[str, Any]] = None
    result: str  # "success", "failure", "partial"


# ============================================================================
# Utility Models
# ============================================================================

class JobSummary(BaseModel):
    """Summary information for job lists."""
    job_id: str
    job_type: str
    status: JobStatus
    created_at: datetime
    progress: Optional[float] = None
    estimated_completion: Optional[datetime] = None
    error_summary: Optional[str] = None


class JobFilter(BaseModel):
    """Filters for job queries."""
    job_types: Optional[List[str]] = None
    statuses: Optional[List[JobStatus]] = None
    created_after: Optional[datetime] = None
    created_before: Optional[datetime] = None
    user_id: Optional[str] = None
    tag: Optional[str] = None


class JobStatistics(BaseModel):
    """Statistical summary of jobs."""
    total_jobs: int
    by_status: Dict[JobStatus, int]
    by_type: Dict[str, int]
    average_execution_time: float
    success_rate: float
    recent_activity: List[JobSummary] = []