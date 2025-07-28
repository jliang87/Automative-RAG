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
# Import validation models
# ============================================================================
from .validation_models import (
    # Warning and error models
    ValidationWarning,
    PreconditionFailure,
    ValidationGuidance,
    ContributionPrompt,

    # Step and chain results
    ValidationStepResult,
    ConfidenceBreakdown,
    ValidationChainResult,

    # User interaction models
    UserChoice,
    UserContribution,
    LearningCredit,
    ValidationUpdate,

    # Configuration models
    ValidationStepConfig,
    ValidationPipelineConfig,

    # Main workflow models
    ValidationWorkflow,
    ValidationContext,

    # Progress and UI models
    ValidationProgressResponse,
    UserChoiceRequest,
    UserChoiceResponse,
    ValidationRestartRequest,

    # Knowledge base models
    AutomotiveReference,
    SourceAuthority,

    # Utility functions
    create_validation_step_result,
    create_pipeline_config,
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
# Import auth models
# ============================================================================
from .auth_models import (
    # Core auth models
    TokenResponse,
    TokenRequest,
    UserInfo,
    UserSession,
    Permission,
    Role,

    # API key models
    APIKey,
    APIKeyRequest,
    APIKeyResponse,

    # OAuth models
    OAuthProvider,
    OAuthToken,
    ExternalUserInfo,

    # Security models
    LoginAttempt,
    SecurityEvent,
    AuditLog,

    # Password models
    PasswordPolicy,
    PasswordReset,
    PasswordChangeRequest,
    PasswordResetRequest,

    # MFA models
    MFADevice,
    MFAChallenge,
    MFAVerification,

    # Rate limiting models
    RateLimit,
    RateLimitStatus,
    SecuritySettings,

    # Error models
    AuthenticationError,
    AuthorizationError,
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

from .knowledge_models import (
    # Reference data models
    AutomotiveSpecification,
    FuelEconomyReference,
    EngineSpecification,
    SafetyRating,
    VehicleDimensions,

    # Authority models
    SourceAuthority,
    SourceAuthorityDatabase,

    # Constraint models
    PhysicsConstraint,
    ValidationConstraints,

    # Database models
    ValidationReferenceDatabase,
    ReferenceDataQuery,
    ReferenceDataMatch,

    # Consensus models
    FactualClaim,
    ConsensusAnalysisResult,

    # Configuration models
    ConfidenceWeights,
    ConfidenceCalculationConfig,

    # Auto-fetch models
    AutoFetchTarget,
    AutoFetchResult,

    # Migration models
    HardcodedDataMigration,

    # Helper functions
    create_epa_fuel_economy_reference,
    create_source_authority,
    create_physics_constraint
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

    # Consolidated validation models
    "ValidationWorkflow",
    "ValidationConfig",
    "ValidationStepResult",
    "ValidationContext",
    "UserChoice",
    "LearningCredit",
    "ValidationUpdate",
    "UserContribution",
    "ValidationProgressResponse",
    "UserChoiceRequest",
    "UserChoiceResponse",
    "ValidationRestartRequest",

    # Validation support models
    "ValidationWarning",
    "PreconditionFailure",
    "ValidationGuidance",
    "ContributionPrompt",
    "ConfidenceBreakdown",
    "ValidationChainResult",
    "ValidationStepConfig",
    "ValidationPipelineConfig",
    "AutomotiveReference",
    "SourceAuthority",

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

    # Auth models
    "TokenResponse",
    "TokenRequest",

    # Job models
    "JobDetails",
]