from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Union, Any

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


class QueryResponse(BaseModel):
    query: str
    answer: str
    documents: List[DocumentResponse]
    metadata_filters_used: Optional[Dict[str, Union[str, List[str], int, List[int]]]] = None
    execution_time: float


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