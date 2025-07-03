"""
Text Processor
Enhanced text processing using EnhancedTranscriptProcessor for superior metadata extraction.
Replaces the basic text processing from job_chain.py with automotive-aware processing.
"""

import logging
from typing import Dict, Any
from datetime import datetime

from ..base.processor import BaseIngestionProcessor

logger = logging.getLogger(__name__)


class TextProcessor(BaseIngestionProcessor):
    """
    Text processor using EnhancedTranscriptProcessor for superior metadata extraction.

    This replaces the basic text processing from job_chain.py and provides:
    - Automotive metadata detection and extraction
    - Content injection with embedded metadata
    - Enhanced chunking strategies
    - Consistent processing pipeline with video and PDF
    """

    def __init__(self):
        """Initialize text processor with enhanced capabilities."""
        super().__init__()
        logger.info("TextProcessor initialized with EnhancedTranscriptProcessor")

    async def extract_raw_content(self, text: str) -> str:
        """
        Extract and validate raw text content.

        Args:
            text: Input text string

        Returns:
            Cleaned and validated text content

        Raises:
            ValueError: If text is empty or invalid
        """
        if not isinstance(text, str):
            raise ValueError(f"Expected string input, got {type(text)}")

        if not text or not text.strip():
            raise ValueError("Text input is empty or contains only whitespace")

        # Basic text cleaning
        cleaned_text = text.strip()

        # Log text statistics
        logger.info(f"Processing text: {len(cleaned_text)} characters")
        if len(cleaned_text) > 1000:
            logger.info(f"Preview: {cleaned_text[:200]}...")

        return cleaned_text

    def build_video_metadata_format(self, text: str) -> Dict[str, Any]:
        """
        Convert text to video metadata format for EnhancedTranscriptProcessor.

        This creates a metadata structure that allows EnhancedTranscriptProcessor
        to extract automotive information from any text content.

        Args:
            text: Input text content

        Returns:
            Metadata in video format for EnhancedTranscriptProcessor compatibility
        """
        import hashlib

        # Create unique ID from text content
        text_hash = hashlib.md5(text.encode('utf-8')).hexdigest()[:8]

        # Extract preview for title and description
        preview = text[:100].replace('\n', ' ').strip()
        if len(text) > 100:
            preview += "..."

        # Try to detect if this looks like automotive content
        automotive_keywords = [
            '车', 'car', 'vehicle', 'auto', '汽车', 'mpg', 'hp', 'horsepower',
            'engine', '发动机', 'transmission', '变速', 'fuel', '油耗',
            'toyota', 'honda', 'ford', 'bmw', 'mercedes', 'audi',
            '丰田', '本田', '宝马', '奔驰', '奥迪', '大众'
        ]

        text_lower = text.lower()
        automotive_score = sum(1 for keyword in automotive_keywords if keyword in text_lower)

        # Build metadata structure compatible with EnhancedTranscriptProcessor
        metadata = {
            'title': f"Text Document: {preview}",
            'id': f"text_{text_hash}",
            'url': f'text://document_{text_hash}',
            'uploader': 'Manual Text Input',
            'upload_date': datetime.now().strftime('%Y%m%d'),
            'duration': max(60, len(text) // 10),  # Rough reading time estimate
            'view_count': 0,
            'description': preview,

            # Additional metadata for processing
            'source_type': 'text_input',
            'content_length': len(text),
            'automotive_keywords_found': automotive_score,
            'processing_timestamp': datetime.now().isoformat(),
        }

        logger.info(f"Built metadata for text document: {text_hash}")
        logger.info(f"Automotive keywords detected: {automotive_score}")

        return metadata

    def validate_source(self, text: str) -> Dict[str, Any]:
        """
        Validate text input before processing.

        Args:
            text: Text to validate

        Returns:
            Validation result with detailed information
        """
        try:
            if not isinstance(text, str):
                return {
                    'valid': False,
                    'error': f'Expected string input, got {type(text).__name__}',
                    'source_type': 'invalid'
                }

            if not text.strip():
                return {
                    'valid': False,
                    'error': 'Text is empty or contains only whitespace',
                    'source_type': 'empty'
                }

            # Check length constraints
            if len(text) < 10:
                return {
                    'valid': False,
                    'error': 'Text too short (minimum 10 characters)',
                    'source_type': 'too_short',
                    'length': len(text)
                }

            if len(text) > 100000:  # 100KB limit
                return {
                    'valid': False,
                    'error': 'Text too long (maximum 100,000 characters)',
                    'source_type': 'too_long',
                    'length': len(text)
                }

            # Successful validation
            return {
                'valid': True,
                'source_type': 'text',
                'length': len(text),
                'estimated_chunks': len(text) // 1000 + 1,
                'preview': text[:100] + ('...' if len(text) > 100 else '')
            }

        except Exception as e:
            return {
                'valid': False,
                'error': f'Validation error: {str(e)}',
                'source_type': 'error'
            }

    def get_processing_stats(self) -> Dict[str, Any]:
        """
        Get text processor statistics and capabilities.

        Returns:
            Dictionary with processor information
        """
        base_stats = super().get_processing_stats()

        text_specific_stats = {
            'input_type': 'text_string',
            'supports_automotive_detection': True,
            'supports_multilingual': True,
            'supports_chinese_text': True,
            'min_text_length': 10,
            'max_text_length': 100000,
            'optimal_chunk_size': 1000,
            'processing_features': [
                'automotive_metadata_extraction',
                'content_injection',
                'enhanced_chunking',
                'vehicle_detection',
                'manufacturer_identification',
                'specification_extraction'
            ]
        }

        return {**base_stats, **text_specific_stats}

    @staticmethod
    def estimate_processing_time(text: str) -> float:
        """
        Estimate processing time for text content.

        Args:
            text: Text to process

        Returns:
            Estimated processing time in seconds
        """
        if not text:
            return 0.0

        # Rough estimates based on text length and complexity
        base_time = 0.5  # Base processing overhead
        char_time = len(text) * 0.00001  # Time per character
        chunk_time = (len(text) // 1000) * 0.1  # Time per chunk

        return base_time + char_time + chunk_time