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


# UNIFIED: Enhanced query mode enum (only mode that matters now)
class QueryMode(str, Enum):
    """Query analysis modes for different user intents."""
    FACTS = "facts"  # DEFAULT: Direct verification (replaces normal queries)
    FEATURES = "features"  # Suggest new features evaluation
    TRADEOFFS = "tradeoffs"  # Evaluate design choice pros/cons
    SCENARIOS = "scenarios"  # Think in user scenarios
    DEBATE = "debate"  # Multi-perspective debate
    QUOTES = "quotes"  # Extract raw user quotes


# UNIFIED: Only enhanced models are used now
class UnifiedQueryRequest(BaseModel):
    """
    UNIFIED: Single query request model (replaces both QueryRequest and EnhancedQueryRequest).
    Facts mode is the default and replaces normal queries.
    """
    query: str = Field(..., description="The user's query about automotive specifications")
    metadata_filter: Optional[Dict[str, Union[str, List[str], int, List[int]]]] = Field(
        None, description="Optional metadata filters to narrow the search"
    )
    top_k: Optional[int] = Field(5, description="Number of documents to retrieve")
    query_mode: QueryMode = Field(QueryMode.FACTS, description="Query analysis mode (defaults to facts)")
    prompt_template: Optional[str] = Field(None, description="Custom prompt template override")

    class Config:
        schema_extra = {
            "example": {
                "query": "2023年宝马X5的后备箱容积是多少？",
                "metadata_filter": {"manufacturer": "宝马", "year": 2023},
                "top_k": 5,
                "query_mode": "facts",
                "prompt_template": None
            }
        }


class UnifiedBackgroundJobResponse(BaseModel):
    """
    UNIFIED: Single background job response (replaces both BackgroundJobResponse and EnhancedBackgroundJobResponse).
    """
    message: str
    job_id: str
    job_type: str
    query_mode: QueryMode = Field(QueryMode.FACTS, description="Query mode being processed")
    expected_processing_time: Optional[int] = Field(None, description="Expected time in seconds")
    status: str = "pending"
    complexity_level: Optional[str] = None

    class Config:
        schema_extra = {
            "example": {
                "message": "Query processing in '车辆规格查询' mode",
                "job_id": "12345-67890",
                "job_type": "llm_inference",
                "query_mode": "facts",
                "expected_processing_time": 10,
                "status": "processing",
                "complexity_level": "simple"
            }
        }


class DocumentResponse(BaseModel):
    id: str
    content: str
    metadata: Dict[str, Any]  # Simplified metadata
    relevance_score: float


class UnifiedQueryResponse(BaseModel):
    """
    UNIFIED: Single query response (replaces both QueryResponse and EnhancedQueryResponse).
    All queries now return this enhanced format.
    """
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
                    "unified_system": True
                }
            }
        }


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


# System capability response
class SystemCapabilities(BaseModel):
    """System capabilities and supported query modes."""
    supported_modes: List[QueryModeConfig]
    current_load: Dict[str, int]
    estimated_response_times: Dict[QueryMode, int]
    feature_flags: Dict[str, bool]
    system_status: Literal["healthy", "degraded", "maintenance"]

    class Config:
        schema_extra = {
            "example": {
                "supported_modes": [],
                "current_load": {"inference_tasks": 2},
                "estimated_response_times": {"facts": 10, "features": 30},
                "feature_flags": {
                    "unified_query_system": True,
                    "facts_as_default": True
                },
                "system_status": "healthy"
            }
        }


# Query validation result
class QueryValidationResult(BaseModel):
    """Result of query validation for specific modes."""
    is_valid: bool
    mode_compatibility: Dict[QueryMode, bool]
    recommendations: List[str]
    warnings: List[str] = []
    suggested_mode: Optional[QueryMode] = Field(QueryMode.FACTS, description="Suggested mode (defaults to facts)")
    confidence_score: float = Field(ge=0.0, le=1.0)

    class Config:
        schema_extra = {
            "example": {
                "is_valid": True,
                "mode_compatibility": {"facts": True, "features": False},
                "recommendations": ["建议使用车辆规格查询模式"],
                "warnings": [],
                "suggested_mode": "facts",
                "confidence_score": 0.85
            }
        }


# LEGACY MODELS: Keep for backward compatibility but mark as deprecated

class DocumentMetadata(BaseModel):
    """Legacy document metadata model - use with caution."""
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
    """Legacy document model - use DocumentResponse instead."""
    id: Optional[str] = None
    content: str
    metadata: DocumentMetadata


# Ingestion models (unchanged)
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


class IngestResponse(BaseModel):
    message: str
    document_count: int
    document_ids: List[str]


# Auth models (unchanged)
class TokenResponse(BaseModel):
    access_token: str
    token_type: str


class TokenRequest(BaseModel):
    username: str
    password: str


# Job models (unchanged)
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


# DEPRECATED MODELS - Kept for backward compatibility only
# These will be removed in future versions

class QueryRequest(BaseModel):
    """DEPRECATED: Use UnifiedQueryRequest instead."""
    query: str = Field(..., description="The user's query about automotive specifications")
    metadata_filter: Optional[Dict[str, Union[str, List[str], int, List[int]]]] = Field(
        None, description="Optional metadata filters to narrow the search"
    )
    top_k: Optional[int] = Field(5, description="Number of documents to retrieve")

    class Config:
        schema_extra = {
            "deprecated": True,
            "alternative": "UnifiedQueryRequest"
        }


class QueryResponse(BaseModel):
    """DEPRECATED: Use UnifiedQueryResponse instead."""
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
            "alternative": "UnifiedQueryResponse"
        }


class BackgroundJobResponse(BaseModel):
    """DEPRECATED: Use UnifiedBackgroundJobResponse instead."""
    message: str
    job_id: str
    job_type: str
    status: str = "pending"

    class Config:
        schema_extra = {
            "deprecated": True,
            "alternative": "UnifiedBackgroundJobResponse"
        }


class EnhancedQueryRequest(BaseModel):
    """DEPRECATED: Use UnifiedQueryRequest instead."""
    query: str = Field(..., description="The user's query about automotive specifications")
    metadata_filter: Optional[Dict[str, Union[str, List[str], int, List[int]]]] = Field(
        None, description="Optional metadata filters to narrow the search"
    )
    top_k: Optional[int] = Field(5, description="Number of documents to retrieve")
    query_mode: QueryMode = Field(QueryMode.FACTS, description="Query analysis mode")
    prompt_template: Optional[str] = Field(None, description="Custom prompt template override")

    class Config:
        schema_extra = {
            "deprecated": True,
            "alternative": "UnifiedQueryRequest"
        }


class EnhancedQueryResponse(BaseModel):
    """DEPRECATED: Use UnifiedQueryResponse instead."""
    query: str
    answer: str
    documents: List[DocumentResponse]
    query_mode: QueryMode
    analysis_structure: Optional[Dict[str, str]] = None
    metadata_filters_used: Optional[Dict[str, Union[str, List[str], int, List[int]]]] = None
    execution_time: float
    status: str = "completed"
    job_id: Optional[str] = None
    mode_metadata: Optional[Dict[str, Any]] = None

    class Config:
        schema_extra = {
            "deprecated": True,
            "alternative": "UnifiedQueryResponse"
        }


class EnhancedBackgroundJobResponse(BaseModel):
    """DEPRECATED: Use UnifiedBackgroundJobResponse instead."""
    message: str
    job_id: str
    job_type: str
    query_mode: Optional[QueryMode] = None
    expected_processing_time: Optional[int] = Field(None, description="Expected time in seconds")
    status: str = "pending"
    complexity_level: Optional[str] = None

    class Config:
        schema_extra = {
            "deprecated": True,
            "alternative": "UnifiedBackgroundJobResponse"
        }


# Alias mappings for smooth transition
# These allow existing code to continue working while we migrate

# Main aliases (recommended for new code)
QueryRequest = UnifiedQueryRequest
QueryResponse = UnifiedQueryResponse
BackgroundJobResponse = UnifiedBackgroundJobResponse

# Enhanced aliases (for backward compatibility)
EnhancedQueryRequest = UnifiedQueryRequest
EnhancedQueryResponse = UnifiedQueryResponse
EnhancedBackgroundJobResponse = UnifiedBackgroundJobResponse

# Export list for the unified system
__all__ = [
    # Core enums
    "DocumentSource",
    "JobType",
    "JobStatus",
    "QueryMode",

    # Unified models (primary)
    "UnifiedQueryRequest",
    "UnifiedQueryResponse",
    "UnifiedBackgroundJobResponse",

    # Aliases (for compatibility)
    "QueryRequest",
    "QueryResponse",
    "BackgroundJobResponse",
    "EnhancedQueryRequest",
    "EnhancedQueryResponse",
    "EnhancedBackgroundJobResponse",

    # Supporting models
    "DocumentResponse",
    "QueryModeConfig",
    "SystemCapabilities",
    "QueryValidationResult",

    # Ingestion models
    "YouTubeIngestRequest",
    "PDFIngestRequest",
    "ManualIngestRequest",
    "IngestResponse",
    "BilibiliIngestRequest",

    # Legacy models (deprecated)
    "DocumentMetadata",
    "Document",

    # Auth models
    "TokenResponse",
    "TokenRequest",

    # Job models
    "JobDetails"
]
