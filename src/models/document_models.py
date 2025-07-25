from datetime import datetime
from typing import Dict, List, Optional, Union, Any
from pydantic import BaseModel, Field, HttpUrl

from .enums import DocumentSource


# ============================================================================
# Document Metadata and Processing Models
# ============================================================================

class DocumentMetadata(BaseModel):
    """Metadata for processed documents."""
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


class DocumentResponse(BaseModel):
    """Document response from retrieval."""
    id: str
    content: str
    metadata: Dict[str, Any]
    relevance_score: float


# ============================================================================
# Ingestion Request Models
# ============================================================================

class YouTubeIngestRequest(BaseModel):
    """Request for YouTube video ingestion."""
    url: HttpUrl
    metadata: Optional[Dict[str, str]] = None


class BilibiliIngestRequest(BaseModel):
    """Request for Bilibili video ingestion."""
    url: HttpUrl
    metadata: Optional[Dict[str, str]] = None


class PDFIngestRequest(BaseModel):
    """Request for PDF document ingestion."""
    file_path: str
    metadata: Optional[Dict[str, str]] = None


class ManualIngestRequest(BaseModel):
    """Request for manual text ingestion."""
    content: str
    metadata: DocumentMetadata


class VideoIngestRequest(BaseModel):
    """Request model for video ingestion from any platform."""
    url: HttpUrl
    metadata: Optional[Dict[str, str]] = None


class BatchVideoIngestRequest(BaseModel):
    """Request model for batch video ingestion."""
    urls: List[HttpUrl]
    metadata: Optional[List[Dict[str, str]]] = None


# ============================================================================
# Ingestion Response Models
# ============================================================================

class IngestResponse(BaseModel):
    """Response from document ingestion."""
    message: str
    document_count: int
    document_ids: List[str]


class BackgroundJobResponse(BaseModel):
    """Response for background job submission."""
    message: str
    job_id: str
    job_type: str
    status: str = "processing"


# ============================================================================
# Document Processing and Analysis Models
# ============================================================================

class ProcessingStatistics(BaseModel):
    """Statistics about document processing."""
    total_documents: int = 0
    documents_used: int = 0
    average_relevance_score: float = 0.0
    processing_time_ms: float = 0.0
    sources_by_type: Dict[str, int] = {}


class DocumentQuality(BaseModel):
    """Quality assessment for a document."""
    content_length: int
    metadata_completeness: float
    source_authority: float
    relevance_score: float
    processing_errors: List[str] = []


class DocumentAnalysis(BaseModel):
    """Analysis results for a document."""
    document_id: str
    content_type: str  # "specifications", "reviews", "features", etc.
    automotive_relevance: float
    extracted_entities: Dict[str, List[str]] = {}
    quality_assessment: DocumentQuality
    processing_metadata: Dict[str, Any] = {}


# ============================================================================
# Content Enhancement Models
# ============================================================================

class ContentEnhancement(BaseModel):
    """Enhanced content with metadata injection."""
    original_content: str
    enhanced_content: str
    injected_metadata: Dict[str, Any]
    enhancement_type: str
    confidence_score: float


class VehicleDetection(BaseModel):
    """Vehicle detection results from content."""
    detected: bool
    manufacturer: Optional[str] = None
    model: Optional[str] = None
    year: Optional[int] = None
    confidence: float = 0.0
    extraction_method: str


class MetadataInjection(BaseModel):
    """Metadata injection results."""
    successful: bool
    injected_fields: List[str]
    injection_patterns: List[str]
    original_length: int
    enhanced_length: int


# ============================================================================
# Retrieval and Search Models
# ============================================================================

class SearchFilters(BaseModel):
    """Filters for document search."""
    manufacturer: Optional[str] = None
    model: Optional[str] = None
    year_range: Optional[tuple[int, int]] = None
    source_types: Optional[List[str]] = None
    content_types: Optional[List[str]] = None
    min_relevance: float = 0.0


class RetrievalMetadata(BaseModel):
    """Metadata about document retrieval."""
    query: str
    filters_applied: Optional[SearchFilters] = None
    total_candidates: int
    filtered_count: int
    final_count: int
    search_time_ms: float
    ranking_method: str


class DocumentChunk(BaseModel):
    """A chunk of a larger document."""
    chunk_id: str
    parent_document_id: str
    content: str
    metadata: Dict[str, Any]
    chunk_index: int
    total_chunks: int
    overlap_with_previous: int = 0


# ============================================================================
# Vector Store Models
# ============================================================================

class VectorStoreStats(BaseModel):
    """Statistics from the vector store."""
    total_documents: int
    total_chunks: int
    collections: List[str]
    index_size_mb: float
    last_updated: datetime


class SimilarityResult(BaseModel):
    """Result from similarity search."""
    document: DocumentChunk
    score: float
    rank: int


class VectorSearchRequest(BaseModel):
    """Request for vector similarity search."""
    query: str
    top_k: int = 10
    filters: Optional[SearchFilters] = None
    include_metadata: bool = True
    score_threshold: Optional[float] = None


class VectorSearchResponse(BaseModel):
    """Response from vector similarity search."""
    query: str
    results: List[SimilarityResult]
    total_results: int
    search_time_ms: float
    filters_applied: Optional[SearchFilters] = None