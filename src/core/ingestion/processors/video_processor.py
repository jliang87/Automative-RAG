"""
Video Processor
Wrapper around existing VideoTranscriber for consistency with unified ingestion system.
VideoTranscriber already uses EnhancedTranscriptProcessor optimally.
"""

import logging
from typing import Dict, Any, List
from langchain_core.documents import Document

from ..base.processor import BaseIngestionProcessor

logger = logging.getLogger(__name__)


class VideoProcessor(BaseIngestionProcessor):
    """
    Video processor that wraps existing VideoTranscriber logic.

    VideoTranscriber already uses EnhancedTranscriptProcessor optimally,
    so this mainly provides consistency with the unified ingestion system.
    """

    def __init__(self):
        """Initialize video processor with existing VideoTranscriber."""
        # Note: We don't call super().__init__() because VideoTranscriber
        # already has its own EnhancedTranscriptProcessor instance
        from src.core.ingestion.loaders.video_transcriber import VideoTranscriber
        self.video_transcriber = VideoTranscriber()

        logger.info("VideoProcessor initialized with existing VideoTranscriber")

    async def extract_raw_content(self, url: str) -> str:
        """
        Extract raw transcript content from video URL.

        This is mainly for interface compliance - VideoTranscriber
        handles the full pipeline internally.

        Args:
            url: Video URL

        Returns:
            Raw transcript text
        """
        # VideoTranscriber handles this internally in process_video()
        # This method is mainly for interface compliance
        raise NotImplementedError(
            "VideoProcessor uses VideoTranscriber.process_video() directly. "
            "Use process() method instead."
        )

    def build_video_metadata_format(self, url: str) -> Dict[str, Any]:
        """
        Video URLs are already in the correct format.

        Args:
            url: Video URL

        Returns:
            Video metadata (extracted by VideoTranscriber)
        """
        # VideoTranscriber handles this internally
        # This method is mainly for interface compliance
        return self.video_transcriber.get_video_metadata(url)

    def process(self, url: str, custom_metadata: Dict = None) -> List[Document]:
        """
        Process video URL using existing VideoTranscriber.

        VideoTranscriber already implements the optimal video processing
        pipeline with EnhancedTranscriptProcessor, so we delegate to it.

        Args:
            url: Video URL to process
            custom_metadata: Optional additional metadata

        Returns:
            List of enhanced Document objects
        """
        try:
            logger.info(f"Processing video URL with VideoTranscriber: {url}")

            # Use existing VideoTranscriber which already uses EnhancedTranscriptProcessor
            documents = self.video_transcriber.process_video(url, custom_metadata)

            # Add unified ingestion system markers
            for doc in documents:
                doc.metadata.update({
                    'processor_type': 'VideoProcessor',
                    'unified_ingestion_system': True,
                    'video_processor_wrapper': True
                })

            logger.info(f"✅ Video processing complete: {len(documents)} documents")
            return documents

        except Exception as e:
            logger.error(f"Video processing failed for {url}: {str(e)}")
            raise ValueError(f"Video processing failed: {str(e)}")

    def validate_source(self, url: str) -> Dict[str, Any]:
        """
        Validate video URL using existing VideoTranscriber validation.

        Args:
            url: Video URL to validate

        Returns:
            Validation result
        """
        try:
            # Use existing VideoTranscriber validation
            validation_result = self.video_transcriber.validate_url(url)

            # Enhance with processor-specific info
            validation_result['processor_type'] = 'VideoProcessor'
            validation_result['uses_video_transcriber'] = True

            return validation_result

        except Exception as e:
            return {
                'valid': False,
                'error': f'Video validation error: {str(e)}',
                'processor_type': 'VideoProcessor'
            }

    def get_processing_stats(self) -> Dict[str, Any]:
        """
        Get video processor statistics and capabilities.

        Returns:
            Dictionary with processor information
        """
        # Get base stats from VideoTranscriber
        video_stats = self.video_transcriber.get_processing_stats()

        # Add processor-specific info
        processor_stats = {
            'processor_name': 'VideoProcessor',
            'wraps_video_transcriber': True,
            'unified_ingestion_system': True,
            'enhanced_processing_enabled': True,
            'uses_enhanced_transcript_processor': True,
            'input_type': 'video_url',
            'processor_version': '2.0_unified'
        }

        return {**video_stats, **processor_stats}

    def extract_video_id(self, url: str) -> str:
        """
        Extract video ID from URL.

        Args:
            url: Video URL

        Returns:
            Video ID
        """
        return self.video_transcriber.extract_video_id(url)

    def detect_platform(self, url: str) -> str:
        """
        Detect video platform from URL.

        Args:
            url: Video URL

        Returns:
            Platform name
        """
        return self.video_transcriber.detect_platform(url)

    def get_video_metadata(self, url: str) -> Dict[str, Any]:
        """
        Get video metadata from URL.

        Args:
            url: Video URL

        Returns:
            Video metadata dictionary
        """
        return self.video_transcriber.get_video_metadata(url)

    @staticmethod
    def estimate_processing_time(url: str) -> float:
        """
        Estimate processing time for video URL.

        Args:
            url: Video URL

        Returns:
            Estimated processing time in seconds
        """
        # Video processing time depends on video length and transcription
        # This is a rough estimate - actual time varies significantly
        return 30.0  # Base estimate for unknown video length