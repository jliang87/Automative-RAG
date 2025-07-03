"""
Base Ingestion Processor
Unified processing pipeline using EnhancedTranscriptProcessor for all ingestion types.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from datetime import datetime
from langchain_core.documents import Document

from src.config.settings import settings

logger = logging.getLogger(__name__)


class BaseIngestionProcessor(ABC):
    """
    Base class for all ingestion processors.

    All processors use EnhancedTranscriptProcessor for consistent, high-quality
    metadata extraction and content enhancement across video, PDF, and text.
    """

    def __init__(self):
        """Initialize with EnhancedTranscriptProcessor for unified processing."""
        from src.core.ingestion.loaders.enhanced_transcript_processor import EnhancedTranscriptProcessor
        self.transcript_processor = EnhancedTranscriptProcessor()
        logger.info(f"Initialized {self.__class__.__name__} with EnhancedTranscriptProcessor")

    @abstractmethod
    async def extract_raw_content(self, source: Any) -> str:
        """
        Extract raw text content from the source.

        Args:
            source: Source data (URL, file path, text string, etc.)

        Returns:
            Raw text content

        Raises:
            ValueError: If content extraction fails
        """
        pass

    @abstractmethod
    def build_video_metadata_format(self, source: Any) -> Dict[str, Any]:
        """
        Convert source information to video metadata format.

        This allows EnhancedTranscriptProcessor to work with any content type
        by converting source info to the expected video metadata structure.

        Args:
            source: Source data

        Returns:
            Metadata in video format for EnhancedTranscriptProcessor
        """
        pass

    def process(self, source: Any, custom_metadata: Optional[Dict] = None) -> List[Document]:
        """
        Main processing pipeline using EnhancedTranscriptProcessor.

        This is the unified processing flow for all ingestion types:
        1. Extract raw content from source
        2. Convert source info to video metadata format
        3. Use EnhancedTranscriptProcessor for metadata extraction and chunking
        4. Add custom metadata

        Args:
            source: Source to process (URL, file path, text, etc.)
            custom_metadata: Optional additional metadata

        Returns:
            List of enhanced Document objects with metadata injection

        Raises:
            ValueError: If processing fails
        """
        try:
            logger.info(f"Processing {self.__class__.__name__} source with enhanced processor")

            # Step 1: Extract raw content
            raw_content = self.extract_raw_content(source)
            if not raw_content or not raw_content.strip():
                raise ValueError("No content extracted from source")

            # Step 2: Build metadata in video format for compatibility
            video_metadata = self.build_video_metadata_format(source)

            # Validate essential metadata fields
            required_fields = ['title', 'id', 'url', 'uploader']
            for field in required_fields:
                if not video_metadata.get(field):
                    logger.warning(f"Missing required metadata field: {field}")
                    video_metadata[field] = f"unknown_{field}"

            # Step 3: Use EnhancedTranscriptProcessor for ALL processing
            logger.info("Applying EnhancedTranscriptProcessor for metadata extraction and chunking")
            documents = self.transcript_processor.process_transcript_chunks(
                transcript=raw_content,
                video_metadata=video_metadata,
                chunk_size=getattr(settings, 'chunk_size', 1000),
                chunk_overlap=getattr(settings, 'chunk_overlap', 200)
            )

            # Step 4: Add source-specific and custom metadata
            for doc in documents:
                # Add processor-specific metadata
                doc.metadata.update({
                    'processor_type': self.__class__.__name__,
                    'processing_method': 'enhanced_transcript_processor',
                    'enhanced_processing_used': True,
                    'metadata_injection_applied': True,
                    'unified_ingestion_system': True
                })

                # Add custom metadata if provided
                if custom_metadata:
                    doc.metadata.update(custom_metadata)

            logger.info(f"✅ Enhanced processing complete: {len(documents)} documents created")

            # Log enhancement statistics
            if documents:
                sample_doc = documents[0]
                vehicle_detected = sample_doc.metadata.get('vehicleDetected', False)
                metadata_injected = sample_doc.metadata.get('metadataInjected', False)

                logger.info(f"🚗 Vehicle detected: {vehicle_detected}")
                logger.info(f"🏷️ Metadata injected: {metadata_injected}")

                # Count embedded patterns
                import re
                embedded_patterns = re.findall(r'【[^】]+】', sample_doc.page_content)
                logger.info(f"📝 Embedded patterns: {len(embedded_patterns)}")

            return documents

        except Exception as e:
            logger.error(f"Error in {self.__class__.__name__} processing: {str(e)}")
            raise ValueError(f"Processing failed: {str(e)}")

    def get_processing_stats(self) -> Dict[str, Any]:
        """
        Get processing statistics and capabilities.

        Returns:
            Dictionary with processor information
        """
        return {
            'processor_name': self.__class__.__name__,
            'enhanced_processing_enabled': True,
            'uses_enhanced_transcript_processor': True,
            'metadata_extraction_enabled': True,
            'content_injection_enabled': True,
            'automotive_optimization': True,
            'unified_ingestion_system': True,
            'vehicle_detection_capable': True,
            'chinese_text_support': True,
            'version': '2.0_unified'
        }

    def validate_source(self, source: Any) -> Dict[str, Any]:
        """
        Validate source before processing.

        Args:
            source: Source to validate

        Returns:
            Validation result dictionary
        """
        try:
            # Basic validation - subclasses can override for specific validation
            if source is None:
                return {'valid': False, 'error': 'Source is None'}

            if isinstance(source, str) and not source.strip():
                return {'valid': False, 'error': 'Source is empty string'}

            return {'valid': True, 'source_type': type(source).__name__}

        except Exception as e:
            return {'valid': False, 'error': f'Validation error: {str(e)}'}

    @staticmethod
    def _create_fallback_metadata(source: Any) -> Dict[str, Any]:
        """
        Create fallback metadata when source-specific extraction fails.

        Args:
            source: Source data

        Returns:
            Basic metadata dictionary
        """
        import hashlib

        # Create basic fallback metadata
        source_str = str(source)
        source_hash = hashlib.md5(source_str.encode()).hexdigest()[:8]

        return {
            'title': f"Document {source_hash}",
            'id': f"doc_{source_hash}",
            'url': f'processed://document_{source_hash}',
            'uploader': 'System Generated',
            'upload_date': datetime.now().strftime('%Y%m%d'),
            'duration': len(source_str) // 10,  # Rough estimate
            'view_count': 0,
            'description': f'Processed document: {source_str[:100]}...' if len(source_str) > 100 else source_str,
        }