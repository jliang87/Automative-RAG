from datetime import datetime
from typing import Dict, List, Optional, Union, Any
from pydantic import BaseModel, Field

from .enums import (
    ValidationStatus, ValidationStep, ValidationType, ValidationStepType,
    ConfidenceLevel, PipelineType, SourceType, ContributionType
)


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
# Validation Step Results
# ============================================================================

class ValidationStepResult(BaseModel):
    """Result from a single validation step."""
    # Core identification
    step_id: str
    step_type: ValidationStepType
    step_name: str
    status: ValidationStatus
    confidence_impact: float = 0.0

    # Legacy compatibility fields
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


# ============================================================================
# Confidence and Chain Results
# ============================================================================

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


# ============================================================================
# Main Validation Workflow Models
# ============================================================================

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
# Knowledge Base and Reference Models
# ============================================================================

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
# Utility Functions
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