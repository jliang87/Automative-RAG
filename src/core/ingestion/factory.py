"""
Processor Factory
Internal factory for creating appropriate ingestion processors.
"""

import logging
from typing import Dict, Any

from .base.processor import BaseIngestionProcessor
from .processors.text_processor import TextProcessor
from .processors.pdf_processor import PDFProcessor
from .processors.video_processor import VideoProcessor

logger = logging.getLogger(__name__)


class ProcessorFactory:
    """
    Internal factory for creating appropriate ingestion processors.

    This factory is for internal use only to maintain the unified
    ingestion system architecture.
    """

    @staticmethod
    def create_processor(ingestion_type: str) -> BaseIngestionProcessor:
        """
        Create processor based on ingestion type.

        Args:
            ingestion_type: Type of processor to create ("text", "pdf", "video")

        Returns:
            Appropriate processor instance

        Raises:
            ValueError: If ingestion type is not supported
        """
        ingestion_type = ingestion_type.lower().strip()

        if ingestion_type == "text":
            logger.info("Creating TextProcessor with EnhancedTranscriptProcessor")
            return TextProcessor()

        elif ingestion_type == "pdf":
            logger.info("Creating PDFProcessor with EnhancedTranscriptProcessor")
            return PDFProcessor()

        elif ingestion_type == "video":
            logger.info("Creating VideoProcessor with existing VideoTranscriber")
            return VideoProcessor()

        else:
            supported_types = ["text", "pdf", "video"]
            raise ValueError(
                f"Unknown ingestion type: '{ingestion_type}'. "
                f"Supported types: {supported_types}"
            )

    @staticmethod
    def get_supported_types() -> Dict[str, Dict[str, Any]]:
        """
        Get information about supported ingestion types.

        Returns:
            Dictionary with information about each supported type
        """
        return {
            "text": {
                "processor_class": "TextProcessor",
                "description": "Process raw text with automotive metadata extraction",
                "input_format": "string",
                "uses_enhanced_transcript_processor": True,
                "supports_automotive_detection": True,
                "example_input": "Raw text content about vehicles..."
            },
            "pdf": {
                "processor_class": "PDFProcessor",
                "description": "Extract and process PDF documents with OCR support",
                "input_format": "file_path",
                "uses_enhanced_transcript_processor": True,
                "supports_automotive_detection": True,
                "example_input": "/path/to/automotive_document.pdf"
            },
            "video": {
                "processor_class": "VideoProcessor",
                "description": "Transcribe and process video content from URLs",
                "input_format": "url",
                "uses_enhanced_transcript_processor": True,
                "supports_automotive_detection": True,
                "example_input": "https://youtube.com/watch?v=abc123"
            }
        }

    @staticmethod
    def validate_processor_type(ingestion_type: str) -> bool:
        """
        Validate if processor type is supported.

        Args:
            ingestion_type: Type to validate

        Returns:
            True if supported, False otherwise
        """
        supported_types = ProcessorFactory.get_supported_types()
        return ingestion_type.lower().strip() in supported_types

    @staticmethod
    def get_processor_info(ingestion_type: str) -> Dict[str, Any]:
        """
        Get detailed information about a specific processor type.

        Args:
            ingestion_type: Type of processor

        Returns:
            Processor information dictionary

        Raises:
            ValueError: If processor type is not supported
        """
        if not ProcessorFactory.validate_processor_type(ingestion_type):
            raise ValueError(f"Unsupported processor type: {ingestion_type}")

        supported_types = ProcessorFactory.get_supported_types()
        return supported_types[ingestion_type.lower().strip()]

    @staticmethod
    def create_processor_with_validation(ingestion_type: str, source: Any) -> BaseIngestionProcessor:
        """
        Create processor and validate source in one step.

        Args:
            ingestion_type: Type of processor to create
            source: Source data to validate

        Returns:
            Validated processor instance

        Raises:
            ValueError: If processor type or source is invalid
        """
        # Create processor
        processor = ProcessorFactory.create_processor(ingestion_type)

        # Validate source
        validation_result = processor.validate_source(source)

        if not validation_result.get('valid', False):
            error_msg = validation_result.get('error', 'Source validation failed')
            raise ValueError(f"Source validation failed: {error_msg}")

        logger.info(f"Created and validated {ingestion_type} processor")
        return processor