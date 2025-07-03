"""
PDF Processor
Enhanced PDF processing using EnhancedTranscriptProcessor for consistent metadata extraction.
Wraps existing PDFLoader and adds automotive-aware processing.
"""

import os
import logging
from typing import Dict, Any
from datetime import datetime

from ..base.processor import BaseIngestionProcessor

logger = logging.getLogger(__name__)


class PDFProcessor(BaseIngestionProcessor):
    """
    PDF processor using EnhancedTranscriptProcessor for consistent metadata extraction.

    This enhances the existing PDFLoader with:
    - Automotive metadata detection from PDF content
    - Content injection with embedded metadata
    - Enhanced chunking strategies
    - Consistent processing pipeline with video and text
    """

    def __init__(self):
        """Initialize PDF processor with enhanced capabilities."""
        super().__init__()

        # Import PDFLoader for PDF-specific operations
        from src.core.pdf_loader import PDFLoader
        self.pdf_loader = PDFLoader()

        logger.info("PDFProcessor initialized with PDFLoader and EnhancedTranscriptProcessor")

    async def extract_raw_content(self, file_path: str) -> str:
        """
        Extract raw text content from PDF file.

        Args:
            file_path: Path to PDF file

        Returns:
            Combined text content from all PDF pages

        Raises:
            ValueError: If PDF extraction fails or file not found
        """
        if not isinstance(file_path, str):
            raise ValueError(f"Expected string file path, got {type(file_path)}")

        if not os.path.exists(file_path):
            raise ValueError(f"PDF file not found: {file_path}")

        if not file_path.lower().endswith('.pdf'):
            raise ValueError(f"File is not a PDF: {file_path}")

        try:
            logger.info(f"Extracting content from PDF: {file_path}")

            # Use existing PDFLoader to extract text
            documents = self.pdf_loader.load_pdf(file_path)

            if not documents:
                raise ValueError(f"No content extracted from PDF: {file_path}")

            # Combine all page content
            combined_content = []
            for i, doc in enumerate(documents):
                page_content = doc.page_content.strip()
                if page_content:
                    # Add page markers for better context
                    combined_content.append(f"[Page {i + 1}] {page_content}")

            if not combined_content:
                raise ValueError(f"PDF contains no readable text: {file_path}")

            full_content = "\n\n".join(combined_content)

            logger.info(f"Extracted {len(full_content)} characters from {len(documents)} pages")
            return full_content

        except Exception as e:
            logger.error(f"Error extracting content from PDF {file_path}: {str(e)}")
            raise ValueError(f"PDF content extraction failed: {str(e)}")

    def build_video_metadata_format(self, file_path: str) -> Dict[str, Any]:
        """
        Convert PDF file info to video metadata format for EnhancedTranscriptProcessor.

        Args:
            file_path: Path to PDF file

        Returns:
            Metadata in video format for EnhancedTranscriptProcessor compatibility
        """
        import hashlib

        # Extract file information
        filename = os.path.basename(file_path)
        file_size = os.path.getsize(file_path) if os.path.exists(file_path) else 0
        mod_time = os.path.getmtime(file_path) if os.path.exists(file_path) else 0

        # Create unique ID from file path
        file_hash = hashlib.md5(file_path.encode('utf-8')).hexdigest()[:8]

        # Try to extract additional PDF metadata using existing PDFLoader
        pdf_metadata = {}
        try:
            pdf_info = self.pdf_loader.get_pdf_info(file_path)
            pdf_metadata = pdf_info.get('metadata', {})
        except Exception as e:
            logger.warning(f"Could not extract PDF metadata: {str(e)}")

        # Extract title from filename or PDF metadata
        title = pdf_metadata.get('title') or filename
        if title.lower().endswith('.pdf'):
            title = title[:-4]  # Remove .pdf extension

        # Extract author from PDF metadata
        author = pdf_metadata.get('author') or pdf_metadata.get('creator') or 'PDF Document'

        # Build metadata structure compatible with EnhancedTranscriptProcessor
        metadata = {
            'title': f"PDF: {title}",
            'id': f"pdf_{file_hash}",
            'url': f'file://{file_path}',
            'uploader': author,
            'upload_date': datetime.fromtimestamp(mod_time).strftime('%Y%m%d') if mod_time else datetime.now().strftime(
                '%Y%m%d'),
            'duration': max(60, file_size // 1000),  # Rough reading time based on file size
            'view_count': 0,
            'description': f'PDF document: {filename}',

            # Additional PDF-specific metadata
            'source_type': 'pdf_document',
            'file_path': file_path,
            'filename': filename,
            'file_size': file_size,
            'pdf_metadata': pdf_metadata,
            'processing_timestamp': datetime.now().isoformat(),
        }

        logger.info(f"Built metadata for PDF: {filename}")
        if pdf_metadata:
            logger.info(f"PDF metadata available: {list(pdf_metadata.keys())}")

        return metadata

    def validate_source(self, file_path: str) -> Dict[str, Any]:
        """
        Validate PDF file before processing.

        Args:
            file_path: Path to PDF file

        Returns:
            Validation result with detailed information
        """
        try:
            if not isinstance(file_path, str):
                return {
                    'valid': False,
                    'error': f'Expected string file path, got {type(file_path).__name__}',
                    'source_type': 'invalid'
                }

            if not file_path.strip():
                return {
                    'valid': False,
                    'error': 'File path is empty',
                    'source_type': 'empty'
                }

            if not os.path.exists(file_path):
                return {
                    'valid': False,
                    'error': f'File does not exist: {file_path}',
                    'source_type': 'not_found'
                }

            if not os.path.isfile(file_path):
                return {
                    'valid': False,
                    'error': f'Path is not a file: {file_path}',
                    'source_type': 'not_file'
                }

            if not file_path.lower().endswith('.pdf'):
                return {
                    'valid': False,
                    'error': f'File is not a PDF: {file_path}',
                    'source_type': 'wrong_format'
                }

            # Check file size
            file_size = os.path.getsize(file_path)
            if file_size == 0:
                return {
                    'valid': False,
                    'error': 'PDF file is empty',
                    'source_type': 'empty_file',
                    'file_size': file_size
                }

            if file_size > 50 * 1024 * 1024:  # 50MB limit
                return {
                    'valid': False,
                    'error': 'PDF file too large (maximum 50MB)',
                    'source_type': 'too_large',
                    'file_size': file_size
                }

            # Try to get basic PDF info
            try:
                pdf_info = self.pdf_loader.get_pdf_info(file_path)
                page_count = pdf_info.get('page_count', 0)

                if page_count == 0:
                    return {
                        'valid': False,
                        'error': 'PDF has no pages',
                        'source_type': 'no_pages'
                    }

            except Exception as e:
                return {
                    'valid': False,
                    'error': f'PDF validation failed: {str(e)}',
                    'source_type': 'corrupt'
                }

            # Successful validation
            return {
                'valid': True,
                'source_type': 'pdf',
                'file_size': file_size,
                'page_count': page_count,
                'filename': os.path.basename(file_path),
                'estimated_processing_time': self.estimate_processing_time(file_path)
            }

        except Exception as e:
            return {
                'valid': False,
                'error': f'Validation error: {str(e)}',
                'source_type': 'error'
            }

    def get_processing_stats(self) -> Dict[str, Any]:
        """
        Get PDF processor statistics and capabilities.

        Returns:
            Dictionary with processor information
        """
        base_stats = super().get_processing_stats()

        pdf_specific_stats = {
            'input_type': 'pdf_file',
            'supports_ocr': getattr(self.pdf_loader, 'use_ocr', False),
            'supports_tables': True,
            'supports_automotive_detection': True,
            'max_file_size_mb': 50,
            'supported_extensions': ['.pdf'],
            'processing_features': [
                'text_extraction',
                'ocr_fallback',
                'automotive_metadata_extraction',
                'content_injection',
                'enhanced_chunking',
                'page_aware_processing'
            ]
        }

        return {**base_stats, **pdf_specific_stats}

    @staticmethod
    def estimate_processing_time(file_path: str) -> float:
        """
        Estimate processing time for PDF file.

        Args:
            file_path: Path to PDF file

        Returns:
            Estimated processing time in seconds
        """
        try:
            if not os.path.exists(file_path):
                return 0.0

            file_size = os.path.getsize(file_path)

            # Rough estimates based on file size
            base_time = 2.0  # Base processing overhead
            size_time = file_size / (1024 * 1024) * 1.5  # 1.5 seconds per MB

            return base_time + size_time

        except Exception:
            return 5.0  # Default estimate