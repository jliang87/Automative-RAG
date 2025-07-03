import os
import re
import subprocess
import json
from typing import Dict, List, Optional, Tuple, Union, Literal

import torch
import logging
from langchain_core.documents import Document
from faster_whisper import WhisperModel

from src.models.schema import DocumentSource
from src.config.settings import settings
# ✅ FIXED IMPORT - Only import what exists
from src.core.ingestion.loaders.enhanced_transcript_processor import EnhancedTranscriptProcessor

logger = logging.getLogger(__name__)


class VideoTranscriber:
    """
    Enhanced unified video transcriber that handles multiple platforms (YouTube, Bilibili, etc.)
    using faster-whisper for CPU-optimized transcription with advanced metadata injection.

    UPDATED: Now includes enhanced transcript processing with vehicle detection and metadata injection.
    """

    def __init__(
            self,
            output_dir: str = "data/videos",
            whisper_model_size: str = "medium",
            device: Optional[str] = "cpu",
            num_workers: int = 3
    ):
        """
        Initialize the enhanced video transcriber.

        Args:
            output_dir: Directory to save downloaded videos and audio
            whisper_model_size: Size of the Whisper model (tiny, base, small, medium, large)
            device: Device to run Whisper on (should be "cpu" for faster-whisper)
            num_workers: Number of workers for parallel processing
        """
        self.output_dir = output_dir
        self.audio_dir = os.path.join(output_dir, "audio")
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(self.audio_dir, exist_ok=True)

        # Determine device (use CUDA if available)
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.whisper_model_size = whisper_model_size
        self.whisper_model = None  # Lazy-load the model when needed
        self.num_workers = num_workers

        # ✅ FIXED: Only initialize what exists
        self.transcript_processor = EnhancedTranscriptProcessor()

        # Add Chinese converter if available
        try:
            import opencc
            self.chinese_converter = opencc.OpenCC('t2s')  # Traditional to Simplified
            print("Chinese character conversion enabled")
        except ImportError:
            print("Warning: opencc-python-reimplemented not installed. Chinese character conversion disabled.")
            self.chinese_converter = None

    def _load_whisper_model(self):
        """Load the faster-whisper model if not already loaded, optimized for CPU."""
        if self.whisper_model is None:
            print(f"Loading faster-whisper {self.whisper_model_size} model on CPU with {self.num_workers} workers...")

            try:
                # Check if using custom model path
                if hasattr(settings, 'whisper_model_full_path') and settings.whisper_model_full_path and os.path.exists(
                        settings.whisper_model_full_path):
                    model_path = settings.whisper_model_full_path
                    print(f"Loading faster-whisper model from local path: {model_path}")
                else:
                    model_path = self.whisper_model_size

                # Initialize the faster-whisper model with CPU optimizations
                self.whisper_model = WhisperModel(
                    model_path,
                    device="cpu",
                    compute_type="int8",  # Use int8 quantization for CPU efficiency
                    cpu_threads=self.num_workers,
                    num_workers=self.num_workers
                )

                print(f"faster-whisper model loaded successfully with {self.num_workers} parallel workers")
            except Exception as e:
                print(f"Error loading faster-whisper model: {str(e)}")
                raise

    def detect_platform(self, url: str) -> Literal["youtube", "bilibili", "unknown"]:
        """
        Detect the platform based on the URL.

        Args:
            url: Video URL

        Returns:
            Platform name ("youtube", "bilibili", or "unknown")
        """
        if "youtube.com" in url or "youtu.be" in url:
            return "youtube"
        elif "bilibili.com" in url:
            return "bilibili"
        else:
            return "unknown"

    def extract_video_id(self, url: str) -> str:
        """
        Extract the video ID from a URL based on the detected platform.

        Args:
            url: Video URL

        Returns:
            Video ID

        Raises:
            ValueError: If video ID cannot be extracted
        """
        platform = self.detect_platform(url)

        if platform == "youtube":
            # YouTube URL patterns
            patterns = [
                r'(?:youtube\.com/watch\?v=|youtu\.be/)([a-zA-Z0-9_-]{11})',
                r'youtube\.com/embed/([a-zA-Z0-9_-]{11})',
                r'youtube\.com/v/([a-zA-Z0-9_-]{11})',
            ]

            for pattern in patterns:
                match = re.search(pattern, url)
                if match:
                    return match.group(1)

            raise ValueError(f"Could not extract YouTube video ID from URL: {url}")

        elif platform == "bilibili":
            # Extract Bilibili video ID (BV or AV id)
            match = re.search(r'/(BV\w+|av\d+)', url)
            if not match:
                raise ValueError(f"Could not extract Bilibili video ID from URL: {url}")
            return match.group(1)

        else:
            raise ValueError(f"Unsupported platform for URL: {url}")

    def extract_audio(self, url: str) -> str:
        """
        Extract audio directly from a video URL using yt-dlp.

        Args:
            url: Video URL

        Returns:
            Path to the extracted audio file or video file if direct audio extraction fails
        """
        video_id = self.extract_video_id(url)
        audio_path = os.path.join(self.audio_dir, f"{video_id}.mp3")

        # Check if we already have this audio file
        if os.path.exists(audio_path):
            print(f"Audio already exists for video ID: {video_id}")
            return audio_path

        # Download audio using yt-dlp
        print(f"Downloading audio for video: {video_id}")

        try:
            # Use yt-dlp to download audio directly in mp3 format
            subprocess.run([
                "yt-dlp",
                "-x",  # Extract audio
                "--audio-format", "mp3",  # Convert to mp3
                "--audio-quality", "0",  # Best quality
                "-o", audio_path,  # Output file
                url
            ], check=True)

            if os.path.exists(audio_path):
                return audio_path

            # If direct audio download failed, fall back to downloading video
            print(f"Direct audio download failed, falling back to video download for: {video_id}")
            return self.download_video(url)
        except subprocess.CalledProcessError as e:
            print(f"Error downloading audio with yt-dlp: {str(e)}")
            print(f"Falling back to video download for: {video_id}")
            return self.download_video(url)
        except FileNotFoundError:
            raise ValueError("yt-dlp not found. Please install it with: pip install yt-dlp")

    def download_video(self, url: str) -> str:
        """
        Download a video file.

        Args:
            url: Video URL

        Returns:
            Path to the downloaded video file
        """
        video_id = self.extract_video_id(url)
        video_path = os.path.join(self.output_dir, f"{video_id}.mp4")

        # Check if we already have this video file
        if os.path.exists(video_path):
            print(f"Video already exists for ID: {video_id}")
            return video_path

        # Download using yt-dlp
        print(f"Downloading video: {video_id}")

        try:
            subprocess.run([
                "yt-dlp",
                "-f", "best",
                "-o", video_path,
                url
            ], check=True)

            return video_path
        except subprocess.CalledProcessError as e:
            raise ValueError(f"Error downloading video: {str(e)}")
        except FileNotFoundError:
            raise ValueError("yt-dlp not found. Please install it with: pip install yt-dlp")

    def get_video_metadata(self, url: str) -> Dict[str, Union[str, int]]:
        """
        Get metadata from a video using yt-dlp with enhanced validation.

        Returns:
            Dictionary with video metadata
        """
        try:
            video_id = self.extract_video_id(url)

            # Use UTF-8 encoding for subprocess
            result = subprocess.run([
                "yt-dlp",
                "--dump-json",
                "--skip-download",
                url
            ], capture_output=True, text=True, check=True, encoding='utf-8')

            if not result.stdout:
                raise ValueError(f"yt-dlp returned empty output for {url}")

            # Parse JSON - Global patch will clean any escapes automatically
            data = json.loads(result.stdout)

            # Validate essential metadata exists
            if not data.get("title") or not data.get("uploader"):
                raise ValueError(f"Missing essential metadata (title/uploader) for {url}")

            # Extract metadata - no manual Unicode cleaning needed
            metadata = {
                "title": data.get("title", ""),
                "uploader": data.get("uploader", ""),  # ✅ Use 'uploader' consistently
                "published_date": data.get("upload_date"),
                "video_id": data.get("id", video_id),
                "url": url,
                "duration": int(data.get("duration", 0)),
                "view_count": data.get("view_count", 0),
                "description": data.get("description", ""),
            }

            # Basic validation - no need to check for Unicode escapes
            if not metadata["title"] or metadata["title"] in ["", "Unknown Video"]:
                raise ValueError(f"Invalid title for {url}")

            if not metadata["uploader"] or metadata["uploader"] in ["", "Unknown", "Unknown Author"]:
                raise ValueError(f"Invalid uploader for {url}")

            logger.info(f"Successfully extracted metadata for {video_id}")
            logger.info(f"Title: {metadata['title']}")
            logger.info(f"Uploader: {metadata['uploader']}")

            return metadata

        except subprocess.CalledProcessError as e:
            error_msg = f"yt-dlp command failed for {url}: {e.stderr}"
            logger.error(error_msg)
            raise ValueError(error_msg)

        except json.JSONDecodeError as e:
            error_msg = f"Failed to parse yt-dlp JSON output for {url}: {str(e)}"
            logger.error(error_msg)
            raise ValueError(error_msg)

        except Exception as e:
            error_msg = f"Metadata extraction failed for {url}: {str(e)}"
            logger.error(error_msg)
            raise ValueError(error_msg)

    def transcribe_with_whisper(self, media_path: str) -> Tuple[str, any]:
        """
        Transcribe audio or video file using faster-whisper with CPU parallelization.

        Args:
            media_path: Path to the audio or video file

        Returns:
            Tuple of (transcribed text, transcription info)
        """
        # Load model if not already loaded
        self._load_whisper_model()

        print(f"Transcribing with faster-whisper ({self.whisper_model_size}) using {self.num_workers} CPU workers...")

        # Use faster-whisper to transcribe
        segments, info = self.whisper_model.transcribe(
            media_path,
            beam_size=5,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500)
        )

        all_text = [segment.text for segment in segments]
        transcript = " ".join(all_text)

        # Convert to Simplified Chinese if detected language is Chinese
        if info.language.startswith("zh") and self.chinese_converter:
            print("Detected Chinese. Converting to Simplified Chinese...")
            transcript = self.chinese_converter.convert(transcript)

        print(f"Transcription complete. Detected language: {info.language}, Duration: {info.duration:.2f}s")
        return transcript, info

    def process_video(
            self, url: str, custom_metadata: Optional[Dict[str, str]] = None
    ) -> List[Document]:
        """
        ENHANCED: Process a video and return enhanced Langchain documents with metadata injection.

        Args:
            url: Video URL
            custom_metadata: Optional custom metadata

        Returns:
            List of enhanced Langchain Document objects with injected metadata
        """
        # Detect platform from URL
        platform = self.detect_platform(url)

        # Extract metadata - no manual Unicode cleaning needed
        video_metadata = self.get_video_metadata(url)

        # ✅ REMOVED: Early vehicle detection (now handled internally by enhanced processor)
        logger.info(f"🔧 Processing video with enhanced transcript processor")

        # Determine the source based on platform
        source = DocumentSource.YOUTUBE if platform == "youtube" else DocumentSource.BILIBILI

        # Transcribe using Whisper
        try:
            # Extract audio
            media_path = self.extract_audio(url)

            # Transcribe with Whisper
            transcript_text, info = self.transcribe_with_whisper(media_path)

            # Add language to custom metadata
            if custom_metadata is None:
                custom_metadata = {}
            custom_metadata["language"] = info.language
            custom_metadata["transcription_method"] = "whisper"
            custom_metadata["whisper_model"] = self.whisper_model_size
            custom_metadata["platform"] = platform

            logger.info(f"Using Whisper transcription for video ID: {video_metadata['video_id']}")
        except Exception as e:
            raise ValueError(f"Error transcribing video with Whisper: {str(e)}")

        if not transcript_text:
            raise ValueError("Transcription failed: no text was generated")

        # ✅ ENHANCED: Use enhanced transcript processing with metadata injection
        logger.info("🔧 Applying enhanced transcript processing with metadata injection...")

        enhanced_documents = self.transcript_processor.process_transcript_chunks(
            transcript=transcript_text,
            video_metadata=video_metadata,
            chunk_size=getattr(settings, 'chunk_size', 1000),
            chunk_overlap=getattr(settings, 'chunk_overlap', 200)
        )

        # ✅ NEW: Add custom metadata to all enhanced documents
        for doc in enhanced_documents:
            doc.metadata.update(custom_metadata)

        logger.info(f"✅ Enhanced processing complete: {len(enhanced_documents)} documents with metadata injection")

        # Log enhanced processing results
        if enhanced_documents:
            sample_doc = enhanced_documents[0]

            # Check for embedded metadata in content
            import re
            embedded_patterns = re.findall(r'【[^】]+】', sample_doc.page_content)

            vehicle_info = {
                'manufacturer': sample_doc.metadata.get('manufacturer'),
                'vehicleModel': sample_doc.metadata.get('vehicleModel'),
                'vehicleDetected': sample_doc.metadata.get('vehicleDetected', False),
                'metadataInjected': sample_doc.metadata.get('metadataInjected', False),
                'embedded_patterns_count': len(embedded_patterns)
            }
            logger.info(f"🚗 Final vehicle detection: {vehicle_info}")

            # Log sample enhanced content
            original_length = sample_doc.metadata.get('originalChunkLength', 0)
            enhanced_length = sample_doc.metadata.get('enhancedChunkLength', len(sample_doc.page_content))
            logger.info(
                f"📝 Content enhancement: {original_length} → {enhanced_length} chars (+{enhanced_length - original_length})")

            # Show embedded patterns for debugging
            if embedded_patterns:
                logger.info(f"🏷️ Embedded patterns: {embedded_patterns[:3]}...")

        return enhanced_documents

    def batch_process_videos(
            self, urls: List[str], custom_metadata: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, Union[List[str], Dict[str, str]]]:
        """
        ENHANCED: Process multiple videos in batch with enhanced metadata processing.

        Args:
            urls: List of Video URLs
            custom_metadata: Optional list of custom metadata dictionaries (same length as urls)

        Returns:
            Dictionary mapping URLs to lists of document IDs or error messages
        """
        # Ensure custom_metadata is the right length or None
        if custom_metadata and len(custom_metadata) != len(urls):
            raise ValueError("custom_metadata list must be the same length as urls")

        results = {}
        enhanced_stats = {
            'total_videos': len(urls),
            'successful_processing': 0,
            'vehicle_detection_count': 0,
            'metadata_injection_count': 0,
            'total_enhanced_documents': 0
        }

        # Load Whisper model once for all videos
        self._load_whisper_model()

        # Process each video
        for i, url in enumerate(urls):
            metadata = custom_metadata[i] if custom_metadata else None
            try:
                enhanced_documents = self.process_video(url, metadata)

                # Extract document IDs
                doc_ids = [doc.metadata.get("chunkId", f"doc_{i}") for doc in enhanced_documents]
                results[url] = doc_ids

                # Update stats
                enhanced_stats['successful_processing'] += 1
                enhanced_stats['total_enhanced_documents'] += len(enhanced_documents)

                if enhanced_documents:
                    sample_doc = enhanced_documents[0]
                    if sample_doc.metadata.get('vehicleDetected', False):
                        enhanced_stats['vehicle_detection_count'] += 1
                    if sample_doc.metadata.get('metadataInjected', False):
                        enhanced_stats['metadata_injection_count'] += 1

                logger.info(f"✅ Successfully processed video {i + 1}/{len(urls)}: {url}")

            except Exception as e:
                error_msg = str(e)
                logger.error(f"❌ Error processing video {i + 1}/{len(urls)}: {url}")
                logger.error(f"Error: {error_msg}")
                results[url] = {"error": error_msg}

        # Log batch processing summary
        logger.info(f"📊 Batch processing complete:")
        logger.info(f"  Successful: {enhanced_stats['successful_processing']}/{enhanced_stats['total_videos']}")
        logger.info(f"  Vehicle detected: {enhanced_stats['vehicle_detection_count']} videos")
        logger.info(f"  Metadata injected: {enhanced_stats['metadata_injection_count']} videos")
        logger.info(f"  Total enhanced documents: {enhanced_stats['total_enhanced_documents']}")

        # Add summary to results
        results['_batch_summary'] = enhanced_stats

        return results

    def get_processing_stats(self) -> Dict[str, any]:
        """
        Get statistics about the enhanced processing capabilities.

        Returns:
            Dictionary with processing statistics and capabilities
        """
        return {
            'whisper_model_size': self.whisper_model_size,
            'device': self.device,
            'num_workers': self.num_workers,
            'enhanced_processing_enabled': True,
            'vehicle_extraction_enabled': True,
            'metadata_injection_enabled': True,
            'supported_platforms': ['youtube', 'bilibili'],
            'chinese_conversion_available': self.chinese_converter is not None,
            'transcript_processor_version': 'enhanced_v2_fixed',
            'capabilities': {
                'vehicle_detection': True,
                'metadata_injection': True,
                'fallback_metadata': True,
                'score_normalization_ready': True,
                'chinese_text_processing': True,
                'automotive_domain_optimization': True
            }
        }

    def validate_url(self, url: str) -> Dict[str, any]:
        """
        Validate a video URL before processing.

        Args:
            url: Video URL to validate

        Returns:
            Dictionary with validation results
        """
        validation_result = {
            'valid': False,
            'platform': 'unknown',
            'video_id': None,
            'errors': []
        }

        try:
            # Detect platform
            platform = self.detect_platform(url)
            validation_result['platform'] = platform

            if platform == 'unknown':
                validation_result['errors'].append('Unsupported platform')
                return validation_result

            # Extract video ID
            video_id = self.extract_video_id(url)
            validation_result['video_id'] = video_id

            # Basic URL format validation
            if not url.startswith(('http://', 'https://')):
                validation_result['errors'].append('Invalid URL format')
                return validation_result

            validation_result['valid'] = True

        except ValueError as e:
            validation_result['errors'].append(str(e))
        except Exception as e:
            validation_result['errors'].append(f'Validation error: {str(e)}')

        return validation_result

    def cleanup_temp_files(self, keep_audio: bool = True, keep_video: bool = False):
        """
        Clean up temporary files to save disk space.

        Args:
            keep_audio: Whether to keep downloaded audio files
            keep_video: Whether to keep downloaded video files
        """
        cleaned_files = 0

        if not keep_video:
            for file in os.listdir(self.output_dir):
                if file.endswith('.mp4'):
                    file_path = os.path.join(self.output_dir, file)
                    try:
                        os.remove(file_path)
                        cleaned_files += 1
                    except Exception as e:
                        logger.warning(f"Could not delete {file_path}: {e}")

        if not keep_audio:
            for file in os.listdir(self.audio_dir):
                if file.endswith('.mp3'):
                    file_path = os.path.join(self.audio_dir, file)
                    try:
                        os.remove(file_path)
                        cleaned_files += 1
                    except Exception as e:
                        logger.warning(f"Could not delete {file_path}: {e}")

        logger.info(f"🧹 Cleaned up {cleaned_files} temporary files")
        return cleaned_files


# ✅ FIXED: Convenience function for easy integration
def create_enhanced_video_transcriber(**kwargs) -> VideoTranscriber:
    """
    Create an enhanced video transcriber with optimal settings.

    Args:
        **kwargs: Optional arguments to override defaults

    Returns:
        Configured VideoTranscriber instance
    """
    default_settings = {
        'whisper_model_size': 'medium',
        'device': 'cpu',
        'num_workers': 3,
        'output_dir': 'data/videos'
    }

    # Override defaults with provided kwargs
    settings_to_use = {**default_settings, **kwargs}

    transcriber = VideoTranscriber(**settings_to_use)

    logger.info("🚀 Created enhanced video transcriber with:")
    logger.info(f"  Whisper model: {settings_to_use['whisper_model_size']}")
    logger.info(f"  Device: {settings_to_use['device']}")
    logger.info(f"  Workers: {settings_to_use['num_workers']}")
    logger.info(f"  Enhanced processing: ENABLED")
    logger.info(f"  Vehicle detection: ENABLED")
    logger.info(f"  Metadata injection: ENABLED")

    return transcriber