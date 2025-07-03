"""
Core RAG System
Restructured modular architecture with clean separation of concerns.

Modules:
- background: Infrastructure (Redis, models, workers)
- ingestion: Document processing and embedding  
- orchestration: Job management and workflow
- query: Document retrieval and LLM inference
"""

# High-level convenience imports
from .background import JobStatus
from .orchestration import job_chain, JobType

# Most specific imports should come from submodules
__all__ = [
    "JobStatus",
    "job_chain", 
    "JobType"
]
