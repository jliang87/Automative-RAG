"""
Unified Ingestion System
Internal module for consistent document processing across all ingestion types.
All processors use EnhancedTranscriptProcessor for superior metadata extraction.

This module is for internal use only - external APIs remain unchanged.
"""

# Internal exports only - no public API changes
from .factory import ProcessorFactory
from .base.processor import BaseIngestionProcessor
from .processors.text_processor import TextProcessor
from .processors.pdf_processor import PDFProcessor
from .processors.video_processor import VideoProcessor

__all__ = [
    "ProcessorFactory",
    "BaseIngestionProcessor",
    "TextProcessor",
    "PDFProcessor",
    "VideoProcessor"
]