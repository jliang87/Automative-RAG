from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Union, Any, Literal

from pydantic import BaseModel, Field, HttpUrl


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


class BackgroundJobResponse(BaseModel):
    """Response for a background job submission."""
    message: str
    job_id: str
    job_type: str
    status: str = "pending"


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


class Document(BaseModel):
    id: Optional[str] = None
    content: str
    metadata: DocumentMetadata


class YouTubeIngestRequest(BaseModel):
    url: HttpUrl
    metadata: Optional[Dict[str, str]] = None


class BilibiliIngestRequest(BaseModel):
    url: HttpUrl
    metadata: Optional[Dict[str, str]] = None


class PDFIngestRequest(BaseModel):
    file_path: str
    metadata: Optional[Dict[str, str]] = None


class ManualIngestRequest(BaseModel):
    content: str
    metadata: DocumentMetadata


class QueryRequest(BaseModel):
    query: str = Field(..., description="The user's query about automotive specifications")
    metadata_filter: Optional[Dict[str, Union[str, List[str], int, List[int]]]] = Field(
        None, description="Optional metadata filters to narrow the search"
    )
    top_k: Optional[int] = Field(5, description="Number of documents to retrieve")


class DocumentResponse(BaseModel):
    id: str
    content: str
    metadata: DocumentMetadata
    relevance_score: float


class IngestResponse(BaseModel):
    message: str
    document_count: int
    document_ids: List[str]


class TokenResponse(BaseModel):
    access_token: str
    token_type: str


class TokenRequest(BaseModel):
    username: str
    password: str


class QueryResponse(BaseModel):
    query: str
    answer: str
    documents: List[DocumentResponse]
    metadata_filters_used: Optional[Dict[str, Union[str, List[str], int, List[int]]]] = None
    execution_time: float
    status: str = "completed"
    job_id: Optional[str] = None


# Add new query mode enum
class QueryMode(str, Enum):
    """Query analysis modes for different user intents."""
    FACTS = "facts"  # Verify concrete vehicle specs
    FEATURES = "features"  # Suggest new features evaluation
    TRADEOFFS = "tradeoffs"  # Evaluate design choice pros/cons
    SCENARIOS = "scenarios"  # Think in user scenarios
    DEBATE = "debate"  # Multi-perspective debate
    QUOTES = "quotes"  # Extract raw user quotes


# Enhanced query request with mode support
class EnhancedQueryRequest(BaseModel):
    """Enhanced query request with analysis mode support."""
    query: str = Field(..., description="The user's query about automotive specifications")
    metadata_filter: Optional[Dict[str, Union[str, List[str], int, List[int]]]] = Field(
        None, description="Optional metadata filters to narrow the search"
    )
    top_k: Optional[int] = Field(5, description="Number of documents to retrieve")
    query_mode: QueryMode = Field(QueryMode.FACTS, description="Query analysis mode")
    prompt_template: Optional[str] = Field(None, description="Custom prompt template override")


# Enhanced query response with mode information
class EnhancedQueryResponse(BaseModel):
    """Enhanced query response with mode-specific formatting."""
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


# Query mode configuration for UI
class QueryModeConfig(BaseModel):
    """Configuration for query modes in the UI."""
    mode: QueryMode
    icon: str
    name: str
    description: str
    use_case: str
    two_layer: bool
    examples: List[str]

    class Config:
        schema_extra = {
            "example": {
                "mode": "facts",
                "icon": "📌",
                "name": "车辆规格查询",
                "description": "验证具体的车辆规格参数",
                "use_case": "查询确切的技术规格、配置信息",
                "two_layer": True,
                "examples": [
                    "2023年宝马X5的后备箱容积是多少？",
                    "特斯拉Model 3的充电速度参数"
                ]
            }
        }


# Structured analysis section for two-layer responses
class AnalysisSection(BaseModel):
    """Individual section in a structured analysis."""
    title: str
    content: str
    section_type: Literal["grounded", "inferred"] = Field(
        description="Whether content is grounded in documents or inferred"
    )


# Multi-perspective analysis for debate mode
class PerspectiveAnalysis(BaseModel):
    """Analysis from a specific role perspective."""
    role: str
    icon: str
    viewpoint: str
    key_points: List[str]


class DebateAnalysis(BaseModel):
    """Structured debate analysis with multiple perspectives."""
    topic: str
    perspectives: List[PerspectiveAnalysis]
    consensus_points: List[str]
    disagreement_points: List[str]
    recommendation: Optional[str] = None


# Tradeoff analysis structure
class TradeoffPoint(BaseModel):
    """Individual pro or con point in tradeoff analysis."""
    point: str
    evidence_type: Literal["document_based", "inferred"]
    source_reference: Optional[str] = None


class TradeoffAnalysis(BaseModel):
    """Structured tradeoff analysis."""
    decision_topic: str
    pros: List[TradeoffPoint]
    cons: List[TradeoffPoint]
    recommendation: Optional[str] = None


# Scenario analysis structure
class ScenarioAnalysis(BaseModel):
    """User scenario analysis structure."""
    scenario_name: str
    target_users: List[str]
    use_cases: List[str]
    optimal_conditions: List[str]
    potential_issues: List[str]
    improvement_suggestions: List[str]


# User quote extraction
class UserQuote(BaseModel):
    """Extracted user quote with source information."""
    quote_text: str
    source_id: str
    source_title: Optional[str] = None
    context: Optional[str] = None
    relevance_score: Optional[float] = None


class QuoteAnalysis(BaseModel):
    """Collection of user quotes on a topic."""
    topic: str
    quotes: List[UserQuote]
    total_found: int
    sentiment_summary: Optional[str] = None


# Feature evaluation structure
class FeatureEvaluation(BaseModel):
    """Feature evaluation analysis."""
    feature_name: str
    user_benefits: List[str]
    technical_feasibility: str
    market_advantage: str
    cost_benefit_ratio: str
    recommendation: str
    confidence_level: Literal["high", "medium", "low"]


# Job metadata for enhanced queries
class EnhancedJobMetadata(BaseModel):
    """Enhanced job metadata with query mode information."""
    query: str
    query_mode: QueryMode
    metadata_filter: Optional[Dict[str, Any]] = None
    top_k: int = 5
    has_custom_template: bool = False
    expected_structure: str = Field(description="Expected response structure")
    processing_complexity: Literal["simple", "moderate", "complex"] = "simple"


# Response formatter configuration
class ResponseFormatterConfig(BaseModel):
    """Configuration for formatting mode-specific responses."""
    mode: QueryMode
    section_headers: List[str]
    requires_parsing: bool = True
    output_format: Literal["structured", "freeform", "debate", "quotes"]
    ui_components: List[str] = Field(description="UI components to render")


# Enhanced background job response with mode info
class EnhancedBackgroundJobResponse(BaseModel):
    """Enhanced background job response with query mode information."""
    message: str
    job_id: str
    job_type: str
    query_mode: Optional[QueryMode] = None
    expected_processing_time: Optional[int] = Field(None, description="Expected time in seconds")
    status: str = "pending"
    complexity_level: Optional[str] = None


# Query mode statistics for analytics
class QueryModeStats(BaseModel):
    """Statistics for query mode usage."""
    mode: QueryMode
    total_queries: int
    avg_processing_time: float
    success_rate: float
    user_satisfaction: Optional[float] = None
    most_common_topics: List[str]


# System capability response
class SystemCapabilities(BaseModel):
    """System capabilities and supported query modes."""
    supported_modes: List[QueryModeConfig]
    current_load: Dict[str, int]
    estimated_response_times: Dict[QueryMode, int]
    feature_flags: Dict[str, bool]
    system_status: Literal["healthy", "degraded", "maintenance"]


# Enhanced document metadata for mode-aware processing
class EnhancedDocumentMetadata(BaseModel):
    """Enhanced document metadata with analysis capabilities."""
    # Existing fields
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

    # New fields for enhanced analysis
    contains_user_reviews: bool = False
    contains_technical_specs: bool = False
    contains_comparisons: bool = False
    sentiment_indicators: Optional[List[str]] = None
    analysis_tags: Optional[List[str]] = None
    complexity_level: Optional[Literal["basic", "intermediate", "advanced"]] = None


# Query processing pipeline status
class QueryPipelineStatus(BaseModel):
    """Status of the query processing pipeline."""
    job_id: str
    current_stage: str
    completed_stages: List[str]
    remaining_stages: List[str]
    stage_timings: Dict[str, float]
    mode_specific_progress: Optional[Dict[str, Any]] = None
    estimated_completion: Optional[datetime] = None


# Mode-specific prompt template
class PromptTemplate(BaseModel):
    """Prompt template for specific query modes."""
    mode: QueryMode
    template_id: str
    name: str
    description: str
    template_text: str
    required_variables: List[str]
    optional_variables: List[str] = []
    output_format: str
    example_input: Optional[str] = None
    example_output: Optional[str] = None


# API response wrapper for enhanced queries
class EnhancedAPIResponse(BaseModel):
    """Wrapper for enhanced API responses."""
    success: bool
    data: Optional[Union[EnhancedQueryResponse, EnhancedBackgroundJobResponse]] = None
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    processing_info: Optional[Dict[str, Any]] = None
    api_version: str = "2.0"


# Query validation result
class QueryValidationResult(BaseModel):
    """Result of query validation for specific modes."""
    is_valid: bool
    mode_compatibility: Dict[QueryMode, bool]
    recommendations: List[str]
    warnings: List[str] = []
    suggested_mode: Optional[QueryMode] = None
    confidence_score: float = Field(ge=0.0, le=1.0)