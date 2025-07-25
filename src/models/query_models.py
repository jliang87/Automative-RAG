from datetime import datetime
from typing import Dict, List, Optional, Union, Any, Literal
from pydantic import BaseModel, Field, HttpUrl

from .enums import QueryMode, ValidationType


# ============================================================================
# Query Configuration and Mode Models
# ============================================================================

class ValidationConfig(BaseModel):
    """Configuration for validation in a query."""
    enabled: bool = False
    validation_type: ValidationType = ValidationType.BASIC
    require_user_approval: bool = False
    confidence_threshold: float = Field(0.7, ge=0.0, le=1.0)
    auto_approve_high_confidence: bool = True
    high_confidence_threshold: float = Field(0.9, ge=0.0, le=1.0)


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


# ============================================================================
# Query Request Models
# ============================================================================

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


# ============================================================================
# Query Response Models
# ============================================================================

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
    validation: Optional[Any] = None  # Will be ValidationWorkflow from validation_models
    validation_summary: Optional[Dict[str, Any]] = None
    created_at: Optional[datetime] = None

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
# Query Mode Configuration Models
# ============================================================================

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
# Model Configuration
# ============================================================================

# Configure all models to use enum values in JSON
for model_class in [
    ValidationConfig, EnhancedQueryRequest, EnhancedQueryResponse,
    EnhancedBackgroundJobResponse, QueryModeConfig, QueryValidationResult
]:
    if hasattr(model_class, 'model_config'):
        model_class.model_config = {"use_enum_values": True}