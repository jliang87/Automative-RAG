from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Union, Any, Literal
from pydantic import BaseModel, Field, HttpUrl


# ============================================================================
# Core Enums
# ============================================================================

class DocumentSource(str, Enum):
    YOUTUBE = "youtube"
    BILIBILI = "bilibili"
    PDF = "pdf"
    MANUAL = "manual"


class JobType(str, Enum):
    VIDEO_PROCESSING = "video_processing"
    PDF_PROCESSING = "pdf_processing"
    BATCH_VIDEO_PROCESSING = "batch_video_processing"
    MANUAL_TEXT = "manual_text"


class JobStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"


class QueryMode(str, Enum):
    """Query analysis modes for different user intents."""
    FACTS = "facts"
    FEATURES = "features"
    TRADEOFFS = "tradeoffs"
    SCENARIOS = "scenarios"
    DEBATE = "debate"
    QUOTES = "quotes"


# ============================================================================
# Validation Enums and Models (Integrated)
# ============================================================================

class ValidationStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    AWAITING_USER_INPUT = "awaiting_user_input"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ValidationStep(str, Enum):
    DOCUMENT_RETRIEVAL = "document_retrieval"
    RELEVANCE_SCORING = "relevance_scoring"
    CONFIDENCE_ANALYSIS = "confidence_analysis"
    USER_VERIFICATION = "user_verification"
    ANSWER_GENERATION = "answer_generation"
    FINAL_REVIEW = "final_review"


class ValidationType(str, Enum):
    BASIC = "basic"
    COMPREHENSIVE = "comprehensive"
    USER_GUIDED = "user_guided"
    AUTOMATED = "automated"


class ValidationStepResult(BaseModel):
    """Result of a single validation step."""
    step: ValidationStep
    status: ValidationStatus
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    duration_seconds: Optional[float] = None
    confidence_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    issues_found: List[str] = []
    recommendations: List[str] = []
    metadata: Dict[str, Any] = {}


class UserChoice(BaseModel):
    """User input for validation decisions."""
    choice_id: str
    step: ValidationStep
    user_decision: str  # "approve", "reject", "modify", "restart"
    user_feedback: Optional[str] = None
    modified_data: Optional[Dict[str, Any]] = None
    timestamp: datetime = Field(default_factory=datetime.now)


class ValidationWorkflow(BaseModel):
    """Complete validation workflow data model."""
    validation_id: str
    job_id: str
    validation_type: ValidationType
    overall_status: ValidationStatus
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None

    steps: List[ValidationStepResult] = []
    current_step: Optional[ValidationStep] = None

    user_choices: List[UserChoice] = []
    awaiting_user_input: bool = False
    user_input_prompt: Optional[str] = None

    overall_confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    total_duration_seconds: Optional[float] = None

    validation_passed: Optional[bool] = None
    issues_identified: List[str] = []
    final_recommendations: List[str] = []

    metadata: Dict[str, Any] = {}


class ValidationConfig(BaseModel):
    """Configuration for validation in a query."""
    enabled: bool = False
    validation_type: ValidationType = ValidationType.BASIC
    require_user_approval: bool = False
    confidence_threshold: float = Field(0.7, ge=0.0, le=1.0)
    auto_approve_high_confidence: bool = True
    high_confidence_threshold: float = Field(0.9, ge=0.0, le=1.0)


# ============================================================================
# Enhanced Query Models
# ============================================================================

class EnhancedQueryRequest(BaseModel):
    """Enhanced query request with validation support."""
    query: str = Field(..., description="The user's query about automotive specifications")
    metadata_filter: Optional[Dict[str, Union[str, List[str], int, List[int]]]] = Field(
        None, description="Optional metadata filters to narrow the search"
    )
    top_k: Optional[int] = Field(5, description="Number of documents to retrieve")
    query_mode: QueryMode = Field(QueryMode.FACTS, description="Query analysis mode")
    prompt_template: Optional[str] = Field(None, description="Custom prompt template override")

    validation_config: Optional[ValidationConfig] = Field(
        None, description="Validation configuration for this query"
    )

    class Config:
        schema_extra = {
            "example": {
                "query": "2023年宝马X5的后备箱容积是多少？",
                "metadata_filter": {"manufacturer": "宝马", "year": 2023},
                "top_k": 5,
                "query_mode": "facts",
                "prompt_template": None,
                "validation_config": {
                    "enabled": True,
                    "validation_type": "comprehensive",
                    "require_user_approval": False,
                    "confidence_threshold": 0.8
                }
            }
        }


class DocumentResponse(BaseModel):
    id: str
    content: str
    metadata: Dict[str, Any]
    relevance_score: float


class EnhancedQueryResponse(BaseModel):
    """Enhanced query response with validation data."""
    query: str
    answer: str
    documents: List[DocumentResponse]
    query_mode: QueryMode
    analysis_structure: Optional[Dict[str, str]] = Field(
        None, description="Structured analysis sections for two-layer modes"
    )
    metadata_filters_used: Optional[Dict[str, Union[str, List[str], int, List[int]]]] = None
    execution_time: float
    status: str = "completed"
    job_id: Optional[str] = None
    mode_metadata: Optional[Dict[str, Any]] = Field(
        None, description="Mode-specific metadata and processing info"
    )

    # Validation data integration
    validation: Optional[ValidationWorkflow] = Field(
        None, description="Complete validation workflow data if validation was enabled"
    )
    validation_summary: Optional[Dict[str, Any]] = Field(
        None, description="Summary of validation results for quick reference"
    )

    class Config:
        schema_extra = {
            "example": {
                "query": "2023年宝马X5的后备箱容积是多少？",
                "answer": "根据提供的文档，2023年宝马X5的后备箱容积为650升。",
                "documents": [],
                "query_mode": "facts",
                "analysis_structure": None,
                "metadata_filters_used": {"manufacturer": "宝马"},
                "execution_time": 8.5,
                "status": "completed",
                "job_id": "12345-67890",
                "mode_metadata": {
                    "processing_mode": "facts",
                    "complexity_level": "simple",
                    "unified_system": True,
                    "validation_enabled": True
                },
                "validation_summary": {
                    "validation_id": "val_12345",
                    "status": "completed",
                    "confidence": 0.85,
                    "passed": True
                }
            }
        }


class EnhancedBackgroundJobResponse(BaseModel):
    """Enhanced background job response with validation support."""
    message: str
    job_id: str
    job_type: str
    query_mode: QueryMode = Field(QueryMode.FACTS, description="Query analysis mode")
    expected_processing_time: Optional[int] = Field(None, description="Expected time in seconds")
    status: str = "pending"
    complexity_level: Optional[str] = None

    # Validation info
    validation_enabled: bool = False
    validation_id: Optional[str] = None
    validation_type: Optional[ValidationType] = None

    class Config:
        schema_extra = {
            "example": {
                "message": "Query processing in '车辆规格查询' mode with comprehensive validation",
                "job_id": "12345-67890",
                "job_type": "comprehensive_validation",
                "query_mode": "facts",
                "expected_processing_time": 25,
                "status": "processing",
                "complexity_level": "simple",
                "validation_enabled": True,
                "validation_id": "val_12345",
                "validation_type": "comprehensive"
            }
        }


# ============================================================================
# Supporting Models
# ============================================================================

class QueryModeConfig(BaseModel):
    """Configuration for query modes in the UI."""
    mode: QueryMode
    icon: str
    name: str
    description: str
    use_case: str
    two_layer: bool
    examples: List[str]
    is_default: Optional[bool] = Field(False, description="Whether this is the default mode")

    class Config:
        schema_extra = {
            "example": {
                "mode": "facts",
                "icon": "📌",
                "name": "车辆规格查询",
                "description": "验证具体的车辆规格参数",
                "use_case": "查询确切的技术规格、配置信息",
                "two_layer": False,
                "is_default": True,
                "examples": [
                    "2023年宝马X5的后备箱容积是多少？",
                    "特斯拉Model 3的充电速度参数"
                ]
            }
        }


class SystemCapabilities(BaseModel):
    """System capabilities and supported query modes."""
    supported_modes: List[QueryModeConfig]
    current_load: Dict[str, int]
    estimated_response_times: Dict[QueryMode, int]
    feature_flags: Dict[str, bool]
    system_status: Literal["healthy", "degraded", "maintenance"]

    # Validation capabilities
    validation_supported: bool = True
    validation_types: List[ValidationType] = [
        ValidationType.BASIC,
        ValidationType.COMPREHENSIVE,
        ValidationType.USER_GUIDED
    ]

    class Config:
        schema_extra = {
            "example": {
                "supported_modes": [],
                "current_load": {"inference_tasks": 2},
                "estimated_response_times": {"facts": 10, "features": 30},
                "feature_flags": {
                    "unified_query_system": True,
                    "validation_workflows": True,
                    "user_guided_validation": True
                },
                "system_status": "healthy",
                "validation_supported": True,
                "validation_types": ["basic", "comprehensive", "user_guided"]
            }
        }


class QueryValidationResult(BaseModel):
    """Result of query validation for specific modes."""
    is_valid: bool
    mode_compatibility: Dict[QueryMode, bool]
    recommendations: List[str]
    warnings: List[str] = []
    suggested_mode: Optional[QueryMode] = Field(QueryMode.FACTS, description="Suggested mode")
    confidence_score: float = Field(ge=0.0, le=1.0)

    # Validation recommendations
    suggested_validation_type: Optional[ValidationType] = None
    validation_recommended: bool = False

    class Config:
        schema_extra = {
            "example": {
                "is_valid": True,
                "mode_compatibility": {"facts": True, "features": False},
                "recommendations": ["建议使用车辆规格查询模式", "建议启用comprehensive验证"],
                "warnings": [],
                "suggested_mode": "facts",
                "confidence_score": 0.85,
                "suggested_validation_type": "comprehensive",
                "validation_recommended": True
            }
        }


# ============================================================================
# Ingestion Models (unchanged)
# ============================================================================

class YouTubeIngestRequest(BaseModel):
    url: HttpUrl
    metadata: Optional[Dict[str, str]] = None


class BilibiliIngestRequest(BaseModel):
    url: HttpUrl
    metadata: Optional[Dict[str, str]] = None


class PDFIngestRequest(BaseModel):
    file_path: str
    metadata: Optional[Dict[str, str]] = None


class DocumentMetadata(BaseModel):
    source: DocumentSource
    source_id: str
    url: Optional[HttpUrl] = None
    title: Optional[str] = None
    author: Optional[str] = None
    published_date: Optional[datetime] = None
    manufacturer: Optional[str] = None
    model: Optional[str] = None
    year: Optional[int] = None
    category: Optional[str] = None
    engine_type: Optional[str] = None
    transmission: Optional[str] = None
    custom_metadata: Optional[Dict[str, str]] = None


class ManualIngestRequest(BaseModel):
    content: str
    metadata: DocumentMetadata


class IngestResponse(BaseModel):
    message: str
    document_count: int
    document_ids: List[str]


# ============================================================================
# Auth Models (unchanged)
# ============================================================================

class TokenResponse(BaseModel):
    access_token: str
    token_type: str


class TokenRequest(BaseModel):
    username: str
    password: str


# ============================================================================
# Job Models (enhanced with validation)
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

    # Validation data
    validation_workflow: Optional[ValidationWorkflow] = None


# ============================================================================
# Backward Compatibility (Legacy Models)
# ============================================================================

# These models are kept for backward compatibility but are deprecated
# New code should use the Enhanced versions above

class QueryRequest(BaseModel):
    """DEPRECATED: Use EnhancedQueryRequest instead."""
    query: str = Field(..., description="The user's query about automotive specifications")
    metadata_filter: Optional[Dict[str, Union[str, List[str], int, List[int]]]] = Field(
        None, description="Optional metadata filters to narrow the search"
    )
    top_k: Optional[int] = Field(5, description="Number of documents to retrieve")

    class Config:
        schema_extra = {
            "deprecated": True,
            "alternative": "EnhancedQueryRequest"
        }


class QueryResponse(BaseModel):
    """DEPRECATED: Use EnhancedQueryResponse instead."""
    query: str
    answer: str
    documents: List[DocumentResponse]
    metadata_filters_used: Optional[Dict[str, Union[str, List[str], int, List[int]]]] = None
    execution_time: float
    status: str = "completed"
    job_id: Optional[str] = None

    class Config:
        schema_extra = {
            "deprecated": True,
            "alternative": "EnhancedQueryResponse"
        }


class BackgroundJobResponse(BaseModel):
    """DEPRECATED: Use EnhancedBackgroundJobResponse instead."""
    message: str
    job_id: str
    job_type: str
    status: str = "pending"

    class Config:
        schema_extra = {
            "deprecated": True,
            "alternative": "EnhancedBackgroundJobResponse"
        }


# ============================================================================
# Validation-Related Supporting Models
# ============================================================================

class ValidationProgressResponse(BaseModel):
    """Response for validation progress queries."""
    validation_id: str
    job_id: str
    validation_type: ValidationType
    overall_status: ValidationStatus
    current_step: Optional[ValidationStep] = None
    progress_percentage: float = Field(ge=0.0, le=100.0)
    steps_completed: int
    total_steps: int
    overall_confidence: Optional[float] = None
    awaiting_user_input: bool = False
    user_input_prompt: Optional[str] = None
    estimated_completion_time: Optional[int] = None
    steps: List[Dict[str, Any]] = []
    issues_identified: List[str] = []
    final_recommendations: List[str] = []


class UserChoiceRequest(BaseModel):
    """Request for submitting user choices in validation."""
    decision: str = Field(..., description="User decision: approve, reject, modify, restart")
    feedback: Optional[str] = Field(None, description="Optional user feedback")
    modified_data: Optional[Dict[str, Any]] = Field(None, description="Modified data if decision is 'modify'")

    class Config:
        schema_extra = {
            "example": {
                "decision": "approve",
                "feedback": "The validation looks correct",
                "modified_data": None
            }
        }


class UserChoiceResponse(BaseModel):
    """Response after submitting user choice."""
    message: str
    validation_id: str
    choice_accepted: bool
    next_step: Optional[str] = None
    requires_user_input: bool = False
    validation_status: str


class ValidationRestartRequest(BaseModel):
    """Request for restarting validation workflow."""
    step: Optional[ValidationStep] = Field(None, description="Step to restart from (optional)")

    class Config:
        schema_extra = {
            "example": {
                "step": "confidence_analysis"
            }
        }


# ============================================================================
# Advanced Validation Models (for future use)
# ============================================================================

class ContributionType(str, Enum):
    """Types of user contributions to validation."""
    URL_LINK = "url_link"
    FILE_UPLOAD = "file_upload"
    DATABASE_LINK = "database_link"
    TEXT_INPUT = "text_input"


class UserContribution(BaseModel):
    """User-contributed validation resource."""
    contribution_id: str
    user_id: str
    validation_id: str
    failed_step: ValidationStep
    contribution_type: ContributionType
    source_url: Optional[str] = None
    file_path: Optional[str] = None
    text_content: Optional[str] = None
    additional_context: Optional[str] = None
    vehicle_scope: str
    status: str = "pending"
    rejection_reason: Optional[str] = None
    submitted_at: datetime
    processed_at: Optional[datetime] = None


class LearningCredit(BaseModel):
    """Credit earned for contributing to system knowledge."""
    credit_id: str
    user_id: str
    contribution_id: str
    credit_points: float
    credit_category: str
    impact_summary: str
    benefits: List[str]
    validation_improvement: float
    future_benefit_estimate: str
    earned_at: datetime


class ValidationUpdate(BaseModel):
    """Result of processing user contribution and re-validating."""
    update_id: str
    original_validation_id: str
    contribution_id: str
    status: str
    new_step_result: Optional[ValidationStepResult] = None
    confidence_improvement: float = 0.0
    contribution_accepted: bool = False
    learning_credit: Optional[LearningCredit] = None
    knowledge_base_updates: List[str] = []
    future_validation_impact: str = ""
    processed_at: datetime


# ============================================================================
# Context Models for Validation
# ============================================================================

class ValidationContext(BaseModel):
    """Context information for validation processes."""
    query_id: str
    query_text: str
    query_mode: str
    manufacturer: Optional[str] = None
    model: Optional[str] = None
    year: Optional[int] = None
    trim: Optional[str] = None
    market: str = "US"
    documents: List[Dict[str, Any]] = []
    user_id: Optional[str] = None
    user_preferences: Dict[str, Any] = {}
    retrieval_metadata: Dict[str, Any] = {}
    processing_metadata: Dict[str, Any] = {}


# ============================================================================
# Utility Classes
# ============================================================================

class ValidationMetrics(BaseModel):
    """Metrics for validation performance tracking."""
    total_validations: int = 0
    successful_validations: int = 0
    failed_validations: int = 0
    user_interventions: int = 0
    average_confidence: float = 0.0
    average_processing_time: float = 0.0
    user_satisfaction_rate: float = 0.0


class SystemHealthStatus(BaseModel):
    """Overall system health including validation capabilities."""
    status: Literal["healthy", "degraded", "maintenance"]
    validation_system_status: Literal["operational", "limited", "offline"]
    active_validations: int = 0
    queue_status: Dict[str, Any] = {}
    last_updated: datetime = Field(default_factory=datetime.now)


# ============================================================================
# Export Configuration
# ============================================================================

# Main models for the unified system
__unified_models__ = [
    # Core query models (enhanced)
    "EnhancedQueryRequest",
    "EnhancedQueryResponse",
    "EnhancedBackgroundJobResponse",

    # Validation models
    "ValidationWorkflow",
    "ValidationConfig",
    "ValidationStepResult",
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

    # Validation enums
    "ValidationStatus",
    "ValidationStep",
    "ValidationType",
    "ContributionType",

    # Enhanced models (primary)
    "EnhancedQueryRequest",
    "EnhancedQueryResponse",
    "EnhancedBackgroundJobResponse",

    # Validation models
    "ValidationWorkflow",
    "ValidationConfig",
    "ValidationStepResult",
    "UserChoice",
    "LearningCredit",
    "ValidationUpdate",
    "UserContribution",
    "ValidationContext",
    "ValidationProgressResponse",
    "UserChoiceRequest",
    "UserChoiceResponse",
    "ValidationRestartRequest",

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

    # Legacy models (deprecated)
    "QueryRequest",
    "QueryResponse",
    "BackgroundJobResponse"
]

# ============================================================================
# Model Aliases for Backward Compatibility
# ============================================================================

# These aliases allow existing code to work while migrating to the new models
# The Enhanced versions are the canonical models going forward

# Primary aliases (recommended for new code)
QueryRequest = EnhancedQueryRequest
QueryResponse = EnhancedQueryResponse
BackgroundJobResponse = EnhancedBackgroundJobResponse

# Note: The enhanced models are now the primary models.
# The "Enhanced" prefix will be dropped in future versions,
# and these will become the standard QueryRequest, QueryResponse, etc.