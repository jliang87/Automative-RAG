from enum import Enum


# ============================================================================
# Core System Enums
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
# Validation Enums
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