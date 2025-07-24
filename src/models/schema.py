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
# Consolidated Validation Enums and Models
# ============================================================================

class ValidationStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    AWAITING_USER_INPUT = "awaiting_user_input"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PASSED = "passed"
    WARNING = "warning"
    UNVERIFIABLE = "unverifiable"


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


class ValidationStepType(str, Enum):
    """Types of validation steps in the validation pipeline."""
    RETRIEVAL = "retrieval"
    SOURCE_CREDIBILITY = "source_credibility"
    TECHNICAL_CONSISTENCY = "technical_consistency"
    COMPLETENESS = "completeness"
    CONSENSUS = "consensus"
    LLM_INFERENCE = "llm_inference"


class ConfidenceLevel(str, Enum):
    """Confidence levels for validation results."""
    EXCELLENT = "excellent"  # 90-100%
    HIGH = "high"  # 80-89%
    MEDIUM = "medium"  # 70-79%
    LOW = "low"  # 60-69%
    POOR = "poor"  # <60%


class PipelineType(str, Enum):
    """Types of validation pipelines."""
    SPECIFICATION_VERIFICATION = "specification_verification"
    FEATURE_COMPARISON = "feature_comparison"
    TRADEOFF_ANALYSIS = "tradeoff_analysis"
    USE_CASE_SCENARIOS = "use_case_scenarios"
    EXPERT_DEBATE = "expert_debate"
    USER_EXPERIENCE = "user_experience"


class SourceType(str, Enum):
    """Types of information sources."""
    OFFICIAL = "official"  # Manufacturer, EPA, NHTSA
    PROFESSIONAL = "professional"  # Automotive journalism, industry publications
    USER_GENERATED = "user_generated"  # Forums, social media, reviews
    ACADEMIC = "academic"  # Research papers, studies
    REGULATORY = "regulatory"  # Government databases, standards


class ContributionType(str, Enum):
    """Types of user contributions to validation."""
    URL_LINK = "url_link"
    FILE_UPLOAD = "file_upload"
    DATABASE_LINK = "database_link"
    TEXT_INPUT = "text_input"


# ============================================================================
# Validation Warning and Error Models
# ============================================================================

class ValidationWarning(BaseModel):
    """Individual validation warning."""
    category: str
    severity: str  # "critical", "caution", "info"
    message: str
    explanation: str
    suggestion: Optional[str] = None


class PreconditionFailure(BaseModel):
    """Details about failed preconditions."""
    resource_type: str
    resource_name: str
    failure_reason: str
    impact_description: str
    suggested_action: str
    confidence_impact: float = 0.0


class ValidationGuidance(BaseModel):
    """Guidance for resolving validation failures."""
    guidance_type: str
    primary_message: str
    suggestions: List[Dict[str, Any]]
    user_prompt: str
    learning_opportunity: str
    estimated_confidence_boost: float = 0.0


class ContributionPrompt(BaseModel):
    """Prompt for user to contribute missing information."""
    needed_resource_type: str
    specific_need_description: str
    contribution_types: List[ContributionType]
    examples: List[str]
    confidence_impact: float
    future_benefit_description: str


# ============================================================================
# Core Validation Result Models
# ============================================================================

class ValidationStepResult(BaseModel):
    """Result from a single validation step - consolidated from both files."""
    # Core identification
    step_id: str
    step_type: ValidationStepType
    step_name: str
    status: ValidationStatus
    confidence_impact: float = 0.0

    # Legacy compatibility fields (from original schema.py)
    step: Optional[ValidationStep] = None  # For backward compatibility

    # Core result data
    summary: str
    details: Dict[str, Any] = Field(default_factory=dict)

    # Timing information
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    duration_seconds: Optional[float] = None
    duration_ms: Optional[int] = None

    # Validation-specific data
    warnings: List[ValidationWarning] = Field(default_factory=list)
    sources_used: List[str] = Field(default_factory=list)
    issues_found: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    # Meta-validation data
    precondition_failures: List[PreconditionFailure] = Field(default_factory=list)
    guidance: Optional[ValidationGuidance] = None
    contribution_prompt: Optional[ContributionPrompt] = None

    # Retry capability
    auto_retry_enabled: bool = False
    retry_count: int = 0

    # Confidence scoring
    confidence_score: Optional[float] = Field(None, ge=0.0, le=1.0)

    class Config:
        use_enum_values = True


class ConfidenceBreakdown(BaseModel):
    """Detailed confidence score breakdown."""
    total_score: float
    level: ConfidenceLevel

    # Component scores
    source_credibility: float = 0.0
    technical_consistency: float = 0.0
    completeness: float = 0.0
    consensus: float = 0.0
    llm_quality: float = 0.0

    # Meta-validation adjustments
    verification_coverage: float = 100.0  # Percentage of steps that could be verified
    unverifiable_penalty: float = 0.0

    # Calculation details
    calculation_method: str = "weighted_average"
    weights_used: Dict[str, float] = Field(default_factory=dict)

    class Config:
        use_enum_values = True


class ValidationChainResult(BaseModel):
    """Complete validation chain result."""
    chain_id: str
    query_id: str
    pipeline_type: PipelineType

    # Overall results
    overall_status: ValidationStatus
    confidence: ConfidenceBreakdown

    # Step results
    validation_steps: List[ValidationStepResult]

    # Chain metadata
    started_at: datetime
    completed_at: Optional[datetime] = None
    total_duration_ms: Optional[int] = None

    # Trust trail data
    step_progression: List[str]  # Visual representation of step flow
    interactive_elements: List[str] = Field(default_factory=list)

    # Learning opportunities
    contribution_opportunities: List[ContributionPrompt] = Field(default_factory=list)
    learning_credits_available: float = 0.0

    class Config:
        use_enum_values = True


# ============================================================================
# User Interaction Models
# ============================================================================

class UserChoice(BaseModel):
    """User input for validation decisions."""
    choice_id: str
    step: Union[ValidationStep, ValidationStepType]  # Support both for compatibility
    user_decision: str  # "approve", "reject", "modify", "restart"
    user_feedback: Optional[str] = None
    modified_data: Optional[Dict[str, Any]] = None
    timestamp: datetime = Field(default_factory=datetime.now)


class UserContribution(BaseModel):
    """User-contributed validation resource."""
    contribution_id: str
    user_id: str
    validation_id: str
    failed_step: Union[ValidationStep, ValidationStepType]

    # Contribution data
    contribution_type: ContributionType
    source_url: Optional[str] = None
    file_path: Optional[str] = None
    text_content: Optional[str] = None
    additional_context: Optional[str] = None

    # Validation scope
    vehicle_scope: str  # Description of what vehicles this helps validate

    # Processing status
    status: str = "pending"  # "pending", "accepted", "rejected"
    rejection_reason: Optional[str] = None

    # Metadata
    submitted_at: datetime
    processed_at: Optional[datetime] = None

    class Config:
        use_enum_values = True


class LearningCredit(BaseModel):
    """Credit earned for contributing to system knowledge."""
    credit_id: str
    user_id: str
    contribution_id: str

    # Credit details
    credit_points: float
    credit_category: str
    impact_summary: str
    benefits: List[str]

    # Context
    validation_improvement: float  # Confidence improvement achieved
    future_benefit_estimate: str

    # Metadata
    earned_at: datetime

    class Config:
        use_enum_values = True


class ValidationUpdate(BaseModel):
    """Result of processing user contribution and re-validating."""
    update_id: str
    original_validation_id: str
    contribution_id: str

    # Update results
    status: str  # "validation_updated", "contribution_rejected", "no_change"
    new_step_result: Optional[ValidationStepResult] = None
    confidence_improvement: float = 0.0

    # Learning data
    contribution_accepted: bool = False
    learning_credit: Optional[LearningCredit] = None

    # System impact
    knowledge_base_updates: List[str] = Field(default_factory=list)
    future_validation_impact: str = ""

    # Metadata
    processed_at: datetime

    class Config:
        use_enum_values = True


# ============================================================================
# Validation Configuration Models
# ============================================================================

class ValidationStepConfig(BaseModel):
    """Configuration for a single validation step."""
    step_type: ValidationStepType
    step_name: str
    enabled: bool = True
    weight: float = 1.0

    # Step-specific configuration
    parameters: Dict[str, Any] = Field(default_factory=dict)
    required_resources: List[str] = Field(default_factory=list)

    # Precondition requirements
    preconditions: Dict[str, Any] = Field(default_factory=dict)
    fallback_behavior: str = "warn"  # "fail", "warn", "skip"

    class Config:
        use_enum_values = True


class ValidationPipelineConfig(BaseModel):
    """Configuration for a complete validation pipeline."""
    pipeline_type: PipelineType
    pipeline_name: str
    description: str

    # Pipeline steps in order
    steps: List[ValidationStepConfig]

    # Confidence calculation
    confidence_weights: Dict[ValidationStepType, float] = Field(default_factory=dict)
    minimum_confidence_threshold: float = 60.0

    # Meta-validation settings
    enable_meta_validation: bool = True
    enable_guided_trust_loop: bool = True
    auto_retry_on_contribution: bool = True

    class Config:
        use_enum_values = True


class ValidationConfig(BaseModel):
    """Configuration for validation in a query."""
    enabled: bool = False
    validation_type: ValidationType = ValidationType.BASIC
    require_user_approval: bool = False
    confidence_threshold: float = Field(0.7, ge=0.0, le=1.0)
    auto_approve_high_confidence: bool = True
    high_confidence_threshold: float = Field(0.9, ge=0.0, le=1.0)


# ============================================================================
# Main Validation Workflow Model
# ============================================================================

class ValidationWorkflow(BaseModel):
    """Complete validation workflow data model - consolidated from both files."""
    validation_id: str
    job_id: str
    validation_type: ValidationType
    overall_status: ValidationStatus
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None

    steps: List[ValidationStepResult] = []
    current_step: Optional[Union[ValidationStep, ValidationStepType]] = None

    user_choices: List[UserChoice] = []
    awaiting_user_input: bool = False
    user_input_prompt: Optional[str] = None

    overall_confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    total_duration_seconds: Optional[float] = None

    validation_passed: Optional[bool] = None
    issues_identified: List[str] = []
    final_recommendations: List[str] = []

    metadata: Dict[str, Any] = {}


# ============================================================================
# Context and Knowledge Base Models
# ============================================================================

class ValidationContext(BaseModel):
    """Context information for validation processes."""
    query_id: str
    query_text: str
    query_mode: str

    # Vehicle context (if applicable)
    manufacturer: Optional[str] = None
    model: Optional[str] = None
    year: Optional[int] = None
    trim: Optional[str] = None
    market: str = "US"

    # Retrieved documents
    documents: List[Dict[str, Any]] = Field(default_factory=list)

    # User context
    user_id: Optional[str] = None
    user_preferences: Dict[str, Any] = Field(default_factory=dict)

    # System context
    retrieval_metadata: Dict[str, Any] = Field(default_factory=dict)
    processing_metadata: Dict[str, Any] = Field(default_factory=dict)


class AutomotiveReference(BaseModel):
    """Reference data for automotive validation."""
    reference_id: str
    reference_type: str  # "epa_mpg", "manufacturer_spec", "safety_rating"

    # Vehicle scope
    manufacturer: Optional[str] = None
    model: Optional[str] = None
    year: Optional[int] = None
    trim: Optional[str] = None
    market: str = "US"

    # Reference data
    data: Dict[str, Any]

    # Metadata
    source_url: Optional[str] = None
    source_authority: SourceType
    created_at: datetime
    updated_at: datetime
    expires_at: Optional[datetime] = None

    # Quality metrics
    reliability_score: float = 1.0
    verification_count: int = 0

    class Config:
        use_enum_values = True


class SourceAuthority(BaseModel):
    """Source authority and credibility information."""
    domain: str
    source_type: SourceType
    authority_score: float  # 0.0 to 1.0

    # Credibility factors
    expertise_level: str  # "expert", "professional", "enthusiast", "general"
    bias_score: float = 0.0  # 0.0 = unbiased, 1.0 = highly biased
    accuracy_history: float = 1.0  # Historical accuracy rate

    # Metadata
    description: Optional[str] = None
    verification_date: Optional[datetime] = None

    class Config:
        use_enum_values = True


# ============================================================================
# Enhanced Query Models (unchanged from original)
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
# Supporting Models (unchanged from original)
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
# Validation Progress and User Interface Models
# ============================================================================

class ValidationProgressResponse(BaseModel):
    """Response for validation progress queries."""
    validation_id: str
    job_id: str
    validation_type: ValidationType
    overall_status: ValidationStatus
    current_step: Optional[Union[ValidationStep, ValidationStepType]] = None
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
    step: Optional[Union[ValidationStep, ValidationStepType]] = Field(None,
                                                                      description="Step to restart from (optional)")

    class Config:
        schema_extra = {
            "example": {
                "step": "confidence_analysis"
            }
        }


# ============================================================================
# Ingestion Models (unchanged from original)
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
# Auth Models (unchanged from original)
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
# Utility and Metrics Models
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

class ValidationType(str, Enum):
    BASIC = "basic"
    COMPREHENSIVE = "comprehensive"
    USER_GUIDED = "user_guided"


class ValidationWorkflow(BaseModel):
    validation_id: str
    job_id: str
    validation_type: ValidationType
    steps: List[Any] = []
    overall_status: str
    overall_confidence: float
    validation_passed: bool
    awaiting_user_input: bool
    current_step: Optional[str] = None
    issues_identified: List[str] = []
    final_recommendations: List[str] = []


class QueryModeConfig(BaseModel):
    mode: QueryMode
    name: str
    description: str
    icon: str
    is_default: bool = False


class SystemCapabilities(BaseModel):
    available_modes: List[QueryMode]
    validation_enabled: bool
    max_query_length: int
    supported_languages: List[str]
    version: str


class QueryValidationResult(BaseModel):
    is_valid: bool
    mode_compatibility: Dict[QueryMode, bool]
    recommendations: List[str]
    suggested_mode: QueryMode
    confidence_score: float
    suggested_validation_type: Optional[ValidationType] = None
    validation_recommended: bool = False


# ============================================================================
# Utility Functions for Model Creation
# ============================================================================

def create_validation_step_result(
        step_type: ValidationStepType,
        status: ValidationStatus,
        summary: str,
        confidence_impact: float = 0.0,
        **kwargs
) -> ValidationStepResult:
    """Helper function to create validation step results."""
    return ValidationStepResult(
        step_id=f"{step_type.value}_{datetime.now().isoformat()}",
        step_type=step_type,
        step_name=step_type.value.replace("_", " ").title(),
        status=status,
        confidence_impact=confidence_impact,
        summary=summary,
        started_at=datetime.now(),
        **kwargs
    )


def create_pipeline_config(
        pipeline_type: PipelineType,
        steps: List[ValidationStepType],
        confidence_weights: Optional[Dict[ValidationStepType, float]] = None
) -> ValidationPipelineConfig:
    """Helper function to create pipeline configurations."""
    if confidence_weights is None:
        # Default weights
        confidence_weights = {
            ValidationStepType.SOURCE_CREDIBILITY: 0.4,
            ValidationStepType.TECHNICAL_CONSISTENCY: 0.3,
            ValidationStepType.COMPLETENESS: 0.2,
            ValidationStepType.CONSENSUS: 0.1
        }

    step_configs = []
    for step_type in steps:
        step_configs.append(ValidationStepConfig(
            step_type=step_type,
            step_name=step_type.value.replace("_", " ").title(),
            weight=confidence_weights.get(step_type, 1.0)
        ))

    return ValidationPipelineConfig(
        pipeline_type=pipeline_type,
        pipeline_name=pipeline_type.value.replace("_", " ").title(),
        description=f"Validation pipeline for {pipeline_type.value}",
        steps=step_configs,
        confidence_weights=confidence_weights
    )


# ============================================================================
# Export Configuration
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


# ==============================================================================
# Enums (add these if missing)
# ==============================================================================

class QueryMode(str, Enum):
    """Available query processing modes."""
    FACTS = "facts"
    FEATURES = "features"
    TRADEOFFS = "tradeoffs"
    SCENARIOS = "scenarios"
    DEBATE = "debate"
    QUOTES = "quotes"


class ValidationType(str, Enum):
    """Types of validation workflows."""
    BASIC = "basic"
    COMPREHENSIVE = "comprehensive"
    USER_GUIDED = "user_guided"


class ValidationStatus(str, Enum):
    """Validation workflow status."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ValidationStepType(str, Enum):
    """Types of validation steps."""
    CONTENT_CHECK = "content_check"
    FACTUAL_VERIFICATION = "factual_verification"
    USER_CONFIRMATION = "user_confirmation"
    QUALITY_ASSESSMENT = "quality_assessment"


# ==============================================================================
# Base Models (add these if missing)
# ==============================================================================

class ValidationConfig(BaseModel):
    """Configuration for query validation."""
    enabled: bool = False
    validation_type: ValidationType = ValidationType.BASIC
    confidence_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    require_user_confirmation: bool = False
    custom_validation_steps: Optional[List[str]] = None


class MetadataFilter(BaseModel):
    """Metadata filtering for document retrieval."""
    manufacturer: Optional[str] = None
    model: Optional[str] = None
    year: Optional[int] = None
    category: Optional[str] = None
    custom_filters: Optional[Dict[str, Any]] = None


class DocumentResponse(BaseModel):
    """Document response from vector store."""
    id: str
    content: str
    metadata: Dict[str, Any] = {}
    score: float = 0.0
    relevance_score: Optional[float] = None


# ==============================================================================
# Request Models (add/update these)
# ==============================================================================

class EnhancedQueryRequest(BaseModel):
    """Enhanced query request with validation support."""
    query: str = Field(..., min_length=1, max_length=1000)
    query_mode: QueryMode = QueryMode.FACTS
    metadata_filter: Optional[MetadataFilter] = None
    top_k: int = Field(default=10, ge=1, le=20)
    prompt_template: Optional[str] = None
    validation_config: Optional[ValidationConfig] = None
    include_sources: bool = True
    response_format: str = "markdown"


# ==============================================================================
# Validation Models (add these if missing)
# ==============================================================================

class ValidationStep(BaseModel):
    """Individual validation step."""
    step_id: str
    step_type: ValidationStepType
    name: str
    description: str
    status: ValidationStatus = ValidationStatus.PENDING
    confidence: float = 0.0
    result: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    user_input_required: bool = False
    completed_at: Optional[datetime] = None


class ValidationWorkflow(BaseModel):
    """Complete validation workflow."""
    validation_id: str
    job_id: str
    validation_type: ValidationType
    steps: List[ValidationStep] = []
    overall_status: ValidationStatus = ValidationStatus.PENDING
    overall_confidence: float = 0.0
    validation_passed: bool = False
    awaiting_user_input: bool = False
    current_step: Optional[ValidationStepType] = None
    issues_identified: List[str] = []
    final_recommendations: List[str] = []
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class QueryValidationResult(BaseModel):
    """Result of query validation for mode compatibility."""
    is_valid: bool
    mode_compatibility: Dict[QueryMode, bool]
    recommendations: List[str]
    suggested_mode: QueryMode
    confidence_score: float = Field(ge=0.0, le=1.0)
    suggested_validation_type: Optional[ValidationType] = None
    validation_recommended: bool = False
    reasoning: Optional[str] = None


# ==============================================================================
# Response Models (add/update these)
# ==============================================================================

class AnalysisStructure(BaseModel):
    """Structured analysis for complex query modes."""
    summary: str
    main_points: List[str] = []
    detailed_sections: List[Dict[str, str]] = []
    conclusion: Optional[str] = None
    recommendations: List[str] = []


class ProcessingStatistics(BaseModel):
    """Statistics about document processing."""
    total_documents: int = 0
    documents_used: int = 0
    average_relevance_score: float = 0.0
    processing_time_ms: float = 0.0
    sources_by_type: Dict[str, int] = {}


class ModeMetadata(BaseModel):
    """Metadata specific to the query mode used."""
    processing_mode: str
    complexity_level: str = "simple"
    mode_name: str = ""
    processing_stats: Optional[ProcessingStatistics] = None
    validation_enabled: bool = False
    unified_system: bool = True
    estimated_time: Optional[int] = None


class ValidationSummary(BaseModel):
    """Summary of validation results."""
    validation_id: str
    status: ValidationStatus
    type: ValidationType
    confidence: float = 0.0
    passed: bool = False
    steps_completed: int = 0
    total_steps: int = 0
    awaiting_user_input: bool = False
    current_step: Optional[ValidationStepType] = None
    issues_identified: int = 0
    recommendations: int = 0


class EnhancedQueryResponse(BaseModel):
    """Enhanced response with validation integration."""
    query: str
    answer: str
    documents: List[DocumentResponse] = []
    query_mode: QueryMode
    analysis_structure: Optional[AnalysisStructure] = None
    metadata_filters_used: Optional[MetadataFilter] = None
    execution_time: float = 0.0
    status: str = "completed"
    job_id: str
    mode_metadata: Optional[ModeMetadata] = None
    validation: Optional[ValidationWorkflow] = None
    validation_summary: Optional[ValidationSummary] = None
    created_at: Optional[datetime] = None


class EnhancedBackgroundJobResponse(BaseModel):
    """Enhanced background job response."""
    message: str
    job_id: str
    job_type: str = "llm_inference"
    query_mode: QueryMode
    expected_processing_time: int = 30  # seconds
    status: str = "processing"
    complexity_level: str = "simple"
    validation_enabled: bool = False
    validation_id: Optional[str] = None
    validation_type: Optional[ValidationType] = None
    created_at: Optional[datetime] = None


# ==============================================================================
# System Models (add these if missing)
# ==============================================================================

class QueryModeConfig(BaseModel):
    """Configuration for a query mode."""
    mode: QueryMode
    name: str
    description: str
    icon: str
    use_case: Optional[str] = None
    is_default: bool = False
    two_layer: bool = False
    examples: List[str] = []


class SystemCapabilities(BaseModel):
    """System capabilities and status."""
    available_modes: List[QueryMode]
    validation_enabled: bool = True
    max_query_length: int = 1000
    supported_languages: List[str] = ["zh", "en"]
    version: str = "1.0.0"
    vector_store_status: str = "connected"
    redis_status: str = "connected"
    job_queue_status: str = "active"


class SystemHealth(BaseModel):
    """System health check response."""
    status: str = "healthy"
    timestamp: datetime
    components: Dict[str, str] = {}
    metrics: Dict[str, Any] = {}


# ==============================================================================
# Error Models (add these if missing)
# ==============================================================================

class ErrorResponse(BaseModel):
    """Standard error response."""
    error: str
    detail: Optional[str] = None
    error_code: Optional[str] = None
    job_id: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ValidationError(BaseModel):
    """Validation-specific error."""
    validation_id: str
    step_id: Optional[str] = None
    error_type: str
    message: str
    recoverable: bool = True
    suggested_action: Optional[str] = None


# ==============================================================================
# Additional Utility Models (add these if missing)
# ==============================================================================

class JobStatus(BaseModel):
    """Job status information."""
    job_id: str
    status: str
    progress: float = Field(ge=0.0, le=1.0)
    current_step: Optional[str] = None
    estimated_completion: Optional[datetime] = None
    error: Optional[str] = None


class QueueStatus(BaseModel):
    """Queue status information."""
    active_jobs: int = 0
    pending_jobs: int = 0
    completed_jobs_today: int = 0
    failed_jobs_today: int = 0
    average_processing_time: float = 0.0
    queue_health: str = "healthy"


class DebugInfo(BaseModel):
    """Debug information for development."""
    query: str
    documents_retrieved: int
    search_time_ms: float
    processing_steps: List[str] = []
    internal_metrics: Dict[str, Any] = {}


# ==============================================================================
# Model Configuration (add this at the end)
# ==============================================================================

# Configure all models to use enum values in JSON
for model_class in [
    ValidationConfig, EnhancedQueryRequest, ValidationStep, ValidationWorkflow,
    QueryValidationResult, EnhancedQueryResponse, EnhancedBackgroundJobResponse,
    QueryModeConfig, SystemCapabilities
]:
    if hasattr(model_class, 'model_config'):
        model_class.model_config = {"use_enum_values": True}


# ==============================================================================
# Instructions for Integration
# ==============================================================================

"""
To integrate these models into your existing src/models/schema.py:

1. Compare with your existing file to see which models already exist
2. Add only the missing models and enums
3. Update existing models if they're missing fields
4. Make sure all imports are present at the top of your file
5. If any model exists but has different fields, merge them carefully

Key models that MUST exist for the system to work:
- QueryMode enum with all 6 values
- EnhancedQueryRequest 
- EnhancedQueryResponse
- EnhancedBackgroundJobResponse
- QueryModeConfig
- SystemCapabilities
- QueryValidationResult
- ValidationConfig
- MetadataFilter
- DocumentResponse

If you're unsure about any conflicts, you can copy this entire file as your new schema.py
"""