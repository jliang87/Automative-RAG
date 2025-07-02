"""
Validation Framework Data Models
Defines all data structures for the validation system
"""

from enum import Enum
from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field
from datetime import datetime


# ============================================================================
# Core Enums
# ============================================================================

class ValidationStatus(str, Enum):
    PASSED = "passed"
    WARNING = "warning"
    FAILED = "failed"
    UNVERIFIABLE = "unverifiable"
    PENDING = "pending"


class ConfidenceLevel(str, Enum):
    EXCELLENT = "excellent"  # 90-100%
    HIGH = "high"  # 80-89%
    MEDIUM = "medium"  # 70-79%
    LOW = "low"  # 60-69%
    POOR = "poor"  # <60%


class ValidationStepType(str, Enum):
    RETRIEVAL = "retrieval"
    SOURCE_CREDIBILITY = "source_credibility"
    TECHNICAL_CONSISTENCY = "technical_consistency"
    COMPLETENESS = "completeness"
    CONSENSUS = "consensus"
    LLM_INFERENCE = "llm_inference"


class PipelineType(str, Enum):
    SPECIFICATION_VERIFICATION = "specification_verification"
    FEATURE_COMPARISON = "feature_comparison"
    TRADEOFF_ANALYSIS = "tradeoff_analysis"
    USE_CASE_SCENARIOS = "use_case_scenarios"
    EXPERT_DEBATE = "expert_debate"
    USER_EXPERIENCE = "user_experience"


class SourceType(str, Enum):
    OFFICIAL = "official"  # Manufacturer, EPA, NHTSA
    PROFESSIONAL = "professional"  # Automotive journalism, industry publications
    USER_GENERATED = "user_generated"  # Forums, social media, reviews
    ACADEMIC = "academic"  # Research papers, studies
    REGULATORY = "regulatory"  # Government databases, standards


class ContributionType(str, Enum):
    URL_LINK = "url_link"
    FILE_UPLOAD = "file_upload"
    DATABASE_LINK = "database_link"
    TEXT_INPUT = "text_input"


# ============================================================================
# Validation Result Models
# ============================================================================

class ValidationWarning(BaseModel):
    """Individual validation warning"""
    category: str
    severity: str  # "critical", "caution", "info"
    message: str
    explanation: str
    suggestion: Optional[str] = None


class PreconditionFailure(BaseModel):
    """Details about failed preconditions"""
    resource_type: str
    resource_name: str
    failure_reason: str
    impact_description: str
    suggested_action: str
    confidence_impact: float = 0.0


class ValidationGuidance(BaseModel):
    """Guidance for resolving validation failures"""
    guidance_type: str
    primary_message: str
    suggestions: List[Dict[str, Any]]
    user_prompt: str
    learning_opportunity: str
    estimated_confidence_boost: float = 0.0


class ContributionPrompt(BaseModel):
    """Prompt for user to contribute missing information"""
    needed_resource_type: str
    specific_need_description: str
    contribution_types: List[ContributionType]
    examples: List[str]
    confidence_impact: float
    future_benefit_description: str


class ValidationStepResult(BaseModel):
    """Result from a single validation step"""
    step_id: str
    step_type: ValidationStepType
    step_name: str
    status: ValidationStatus
    confidence_impact: float = 0.0

    # Core result data
    summary: str
    details: Dict[str, Any] = Field(default_factory=dict)

    # Timing information
    started_at: datetime
    completed_at: Optional[datetime] = None
    duration_ms: Optional[int] = None

    # Validation-specific data
    warnings: List[ValidationWarning] = Field(default_factory=list)
    sources_used: List[str] = Field(default_factory=list)

    # Meta-validation data
    precondition_failures: List[PreconditionFailure] = Field(default_factory=list)
    guidance: Optional[ValidationGuidance] = None
    contribution_prompt: Optional[ContributionPrompt] = None

    # Retry capability
    auto_retry_enabled: bool = False
    retry_count: int = 0

    class Config:
        use_enum_values = True


class ConfidenceBreakdown(BaseModel):
    """Detailed confidence score breakdown"""
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
    """Complete validation chain result"""
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
# Pipeline Configuration Models
# ============================================================================

class ValidationStepConfig(BaseModel):
    """Configuration for a single validation step"""
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
    """Configuration for a complete validation pipeline"""
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


# ============================================================================
# Knowledge Base Models
# ============================================================================

class AutomotiveReference(BaseModel):
    """Reference data for automotive validation"""
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
    """Source authority and credibility information"""
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
# Guided Trust Loop Models
# ============================================================================

class UserContribution(BaseModel):
    """User-contributed validation resource"""
    contribution_id: str
    user_id: str
    validation_id: str
    failed_step: ValidationStepType

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
    """Credit earned for contributing to system knowledge"""
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
    """Result of processing user contribution and re-validating"""
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
# Context Models
# ============================================================================

class ValidationContext(BaseModel):
    """Context information for validation"""
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


# ============================================================================
# Utility Functions
# ============================================================================

def create_validation_step_result(
        step_type: ValidationStepType,
        status: ValidationStatus,
        summary: str,
        confidence_impact: float = 0.0,
        **kwargs
) -> ValidationStepResult:
    """Helper function to create validation step results"""
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
    """Helper function to create pipeline configurations"""
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