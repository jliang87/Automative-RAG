"""
Models package with domain-separated modules.

This module re-exports all models for backwards compatibility while organizing
them into logical domain-specific modules.
"""

# ============================================================================
# Import all enums
# ============================================================================
from .enums import (
    # Core system enums
    DocumentSource,
    JobType,
    JobStatus,
    QueryMode,

    # Validation enums
    ValidationStatus,
    ValidationStep,
    ValidationType,
    ValidationStepType,
    ConfidenceLevel,
    PipelineType,
    SourceType,
    ContributionType,
)

# ============================================================================
# Import query models
# ============================================================================
from .query_models import (
    # Configuration models
    ValidationConfig,
    MetadataFilter,
    QueryModeConfig,

    # Request/Response models
    EnhancedQueryRequest,
    EnhancedQueryResponse,
    EnhancedBackgroundJobResponse,
    DocumentResponse,
    AnalysisStructure,
    ProcessingStatistics,
    ModeMetadata,
    QueryValidationResult,
)


# ============================================================================
# Import document models
# ============================================================================
from .document_models import (
    # Core document models
    DocumentMetadata,
    # DocumentResponse already imported from query_models

    # Ingestion models
    YouTubeIngestRequest,
    BilibiliIngestRequest,
    PDFIngestRequest,
    ManualIngestRequest,
    VideoIngestRequest,
    BatchVideoIngestRequest,

    # Response models
    IngestResponse,
    # BackgroundJobResponse already imported from query_models

    # Processing models
    # ProcessingStatistics already imported from query_models
    DocumentQuality,
    DocumentAnalysis,

    # Enhancement models
    ContentEnhancement,
    VehicleDetection,
    MetadataInjection,

    # Retrieval models
    SearchFilters,
    RetrievalMetadata,
    DocumentChunk,

    # Vector store models
    VectorStoreStats,
    SimilarityResult,
    VectorSearchRequest,
    VectorSearchResponse,
)

# ============================================================================
# Import job models
# ============================================================================
from .job_models import (
    # Core job models
    JobDetails,
    JobMetadata,
    JobProgress,
    JobResult,

    # Chain and orchestration
    JobChainStep,
    JobChainStatus,
    TaskResult,

    # Queue management
    QueueStatus,
    QueueStatistics,
    WorkerStatus,

    # Pipeline models
    PipelineStep,
    PipelineConfig,
    PipelineExecution,

    # Specific job types
    VideoProcessingJob,
    PDFProcessingJob,
    TextProcessingJob,
    BatchProcessingJob,

    # Error and retry models
    JobError,
    RetryPolicy,
    JobTimeout,

    # Monitoring models
    JobMetrics,
    SystemMetrics,
    PerformanceReport,

    # Lifecycle events
    JobEvent,
    JobAuditLog,

    # Utility models
    JobSummary,
    JobFilter,
    JobStatistics,
)


# ============================================================================
# Import system models
# ============================================================================
from .system_models import (
    # Core system models
    SystemCapabilities,
    SystemHealth,
    ComponentStatus,
    SystemMetrics,

    # Version and feature models
    VersionInfo,
    FeatureFlag,
    SystemConfiguration,

    # Error and debug models
    ErrorResponse,
    DebugInfo,
    SystemAlert,

    # Resource models
    ResourceUsage,
    CapacityLimits,
    ResourceAllocation,

    # Service discovery
    ServiceInfo,
    ServiceRegistry,

    # Monitoring models
    LogEntry,
    MetricPoint,
    TraceSpan,

    # Backup and recovery
    BackupInfo,
    RecoveryPlan,

    # Cache and storage
    CacheStats,
    StorageStats,

    # API models
    APIEndpoint,
    APIUsageStats,
    IntegrationStatus,

    # Maintenance models
    MaintenanceWindow,
    DeploymentInfo,
    SystemEvent,

    # Configuration models
    ConfigurationSetting,
    EnvironmentConfig,

    # Utility models
    ValidationMetrics,
    SystemHealthStatus,
    Notification,
)



# ============================================================================
# Export lists for different use cases
# ============================================================================

# Main models for the unified system
__unified_models__ = [
    # Core query models (enhanced)
    "EnhancedQueryRequest",
    "EnhancedQueryResponse",
    "EnhancedBackgroundJobResponse",

    # Validation models (consolidated)
    "ValidationWorkflow",
    "ValidationConfig",
    "ValidationStepResult",
    "ValidationContext",
    "UserChoice",
    "ValidationProgressResponse",
    "UserChoiceRequest",
    "UserChoiceResponse",

    # Supporting models
    "QueryModeConfig",
    "SystemCapabilities",
    "QueryValidationResult",
    "DocumentResponse"
]

# Legacy models (for backward compatibility)
__legacy_models__ = [
    "QueryRequest",
    "QueryResponse",
    "BackgroundJobResponse"
]

# All available models
__all__ = [
    # Core enums
    "DocumentSource",
    "JobType",
    "JobStatus",
    "QueryMode",

    # Validation enums (consolidated)
    "ValidationStatus",
    "ValidationStep",
    "ValidationType",
    "ValidationStepType",
    "ConfidenceLevel",
    "PipelineType",
    "SourceType",
    "ContributionType",

    # Enhanced models (primary)
    "EnhancedQueryRequest",
    "EnhancedQueryResponse",
    "EnhancedBackgroundJobResponse",

    # Supporting models
    "DocumentResponse",
    "QueryModeConfig",
    "SystemCapabilities",
    "QueryValidationResult",
    "ValidationMetrics",
    "SystemHealthStatus",

    # Ingestion models
    "YouTubeIngestRequest",
    "BilibiliIngestRequest",
    "PDFIngestRequest",
    "ManualIngestRequest",
    "IngestResponse",
    "DocumentMetadata",

    # Job models
    "JobDetails",
]