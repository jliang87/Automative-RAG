"""
Module for transcribing videos from various platforms with GPU-accelerated Whisper.

This module provides a unified transcriber for YouTube, Bilibili, and other video platforms,
with GPU acceleration using Whisper for high-quality transcription.
"""

import os
import re
import tempfile
import subprocess
import json
from typing import Dict, List, Optional, Tuple, Union, Literal

import torch
import logging
from langchain_core.documents import Document
# import whisper
# Import the faster-whisper package for transcription on CPU
from faster_whisper import WhisperModel

from src.models.schema import DocumentMetadata, DocumentSource
from src.config.settings import settings
from src.utils.helpers import (
    extract_metadata_from_text,
    generate_unique_id
)

logger = logging.getLogger(__name__)

class VideoTranscriber:
    """
    Unified video transcriber that handles multiple platforms (YouTube, Bilibili, etc.)
    using faster-whisper for CPU-optimized transcription.
    """

    def __init__(
            self,
            output_dir: str = "data/videos",
            whisper_model_size: str = "medium",
            device: Optional[str] = "cpu",
            num_workers: int = 3  # Number of threads for parallel processing
    ):
        """
        Initialize the video transcriber with CPU-optimized faster-whisper.

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
                    # Load from local path
                    model_path = settings.whisper_model_full_path
                    print(f"Loading faster-whisper model from local path: {model_path}")
                else:
                    # Use model name for faster-whisper to download
                    model_path = self.whisper_model_size

                # Initialize the faster-whisper model with CPU optimizations
                self.whisper_model = WhisperModel(
                    model_path,
                    device="cpu",
                    compute_type="int8",  # Use int8 quantization for CPU efficiency
                    cpu_threads=self.num_workers,  # Use multiple CPU threads
                    num_workers=self.num_workers  # Number of workers for parallel processing
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
        Get metadata from a video using yt-dlp.

        Args:
            url: Video URL

        Returns:
            Dictionary containing video metadata
        """
        try:
            video_id = self.extract_video_id(url)

            # Use yt-dlp to get metadata
            result = subprocess.run([
                "yt-dlp",
                "--dump-json",
                "--skip-download",
                url
            ], capture_output=True, text=True, check=True)

            if result.stdout:
                data = json.loads(result.stdout)

                # Extract relevant metadata
                metadata = {
                    "title": data.get("title", f"Video {video_id}"),
                    "author": data.get("uploader", "Unknown"),
                    "published_date": data.get("upload_date"),
                    "video_id": video_id,
                    "url": url,  # ✅ Always preserve the original URL
                    "length": data.get("duration", 0),
                    "views": data.get("view_count", 0),
                    "description": data.get("description", ""),
                }

                # Debug log to verify URL is set
                logger.info(f"Video metadata extracted for {video_id}: URL={url}")
                return metadata
            else:
                raise ValueError("No metadata returned from yt-dlp")

        except Exception as e:
            # Fallback metadata if yt-dlp fails
            print(f"Warning: Error fetching video metadata: {str(e)}")

            # Try to extract video_id if not already done
            if 'video_id' not in locals():
                try:
                    video_id = self.extract_video_id(url)
                except Exception:
                    video_id = "unknown"

            # CRITICAL FIX: Always preserve URL in fallback metadata
            fallback_metadata = {
                "title": f"Video {video_id}",
                "author": "Unknown Author",
                "published_date": None,
                "video_id": video_id,
                "url": url,  # ✅ Always preserve URL even in fallback
                "length": 0,
                "views": 0,
                "description": "",
            }

            logger.warning(f"Using fallback metadata for {video_id}: URL={url}")
            return fallback_metadata

    def transcribe_with_whisper(self, media_path: str) -> str:
        """
        Transcribe audio or video file using faster-whisper with CPU parallelization.

        Args:
            media_path: Path to the audio or video file

        Returns:
            Transcribed text
        """
        # # Try to free up GPU memory
        # if self.device.startswith("cuda"):
        #     torch.cuda.empty_cache()

        # Load model if not already loaded
        self._load_whisper_model()

        print(f"Transcribing with faster-whisper ({self.whisper_model_size}) using {self.num_workers} CPU workers...")

        # Use faster-whisper to transcribe
        segments, info = self.whisper_model.transcribe(
            media_path,
            beam_size=5,  # Larger beam size for better accuracy
            vad_filter=True,  # Voice activity detection for better segmentation
            vad_parameters=dict(  # Fine-tuned VAD parameters
                min_silence_duration_ms=500
            )
        )

        all_text = [segment.text for segment in segments]
        transcript = " ".join(all_text)

        # ✅ Convert only if detected language is Chinese
        if info.language.startswith("zh") and self.chinese_converter:
            print("Detected Chinese. Converting to Simplified Chinese...")
            transcript = self.chinese_converter.convert(transcript)

        print(f"Transcription complete. Detected language: {info.language}, Duration: {info.duration:.2f}s")
        return transcript, info

    def process_video(
            self, url: str, custom_metadata: Optional[Dict[str, str]] = None
    ) -> List[Document]:
        """
        Process a video and return Langchain documents using Whisper for transcription.

        Args:
            url: Video URL
            custom_metadata: Optional custom metadata

        Returns:
            List of Langchain Document objects
        """
        # Detect platform from URL
        platform = self.detect_platform(url)

        # Extract metadata
        video_metadata = self.get_video_metadata(url)

        # Extract automotive metadata using helper function
        auto_metadata = extract_metadata_from_text(video_metadata.get("title", "") + " " +
                                                  video_metadata.get("description", ""))

        # Determine the source based on platform
        source = DocumentSource.YOUTUBE if platform == "youtube" else DocumentSource.BILIBILI

        # Transcribe using Whisper
        try:
            # Extract audio
            media_path = self.extract_audio(url)

            # Transcribe with Whisper
            transcript_text, info = self.transcribe_with_whisper(media_path)
            custom_metadata = custom_metadata or {}
            custom_metadata["language"] = info.language
            print(f"Using Whisper transcription for video ID: {video_metadata['video_id']}")
        except Exception as e:
            raise ValueError(f"Error transcribing video with Whisper: {str(e)}")

        if not transcript_text:
            raise ValueError("Transcription failed: no text was generated")

        # Create metadata object
        metadata = DocumentMetadata(
            source=source,
            source_id=video_metadata["video_id"],
            url=url,
            title=video_metadata["title"],
            author=video_metadata["author"],
            published_date=video_metadata["published_date"],
            manufacturer=auto_metadata.get("manufacturer"),
            model=custom_metadata.get("model") if custom_metadata else None,
            year=auto_metadata.get("year"),
            category=auto_metadata.get("category"),
            engine_type=auto_metadata.get("engine_type"),
            transmission=auto_metadata.get("transmission"),
            custom_metadata=custom_metadata,
        )

        # Add transcription info to metadata
        metadata.custom_metadata["transcription_method"] = "whisper"
        metadata.custom_metadata["whisper_model"] = self.whisper_model_size

        # Add platform information
        metadata.custom_metadata["platform"] = platform

        # Create document
        document = Document(
            page_content=transcript_text,
            metadata=metadata.dict(),
        )

        return [document]

    def batch_process_videos(
            self, urls: List[str], custom_metadata: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, Union[List[str], Dict[str, str]]]:
        """
        Process multiple videos in batch using Whisper for transcription.

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

        # Load Whisper model once for all videos
        self._load_whisper_model()

        # Process each video
        for i, url in enumerate(urls):
            metadata = custom_metadata[i] if custom_metadata else None
            try:
                documents = self.process_video(url, metadata)
                # Extract document IDs after they're added to the vector store
                # This will be handled by the DocumentProcessor that calls this method
                results[url] = [doc.metadata.get("id", "") for doc in documents]
                print(f"Successfully processed video {i + 1}/{len(urls)}: {url}")
            except Exception as e:
                print(f"Error processing video {i + 1}/{len(urls)}: {url}")
                print(f"Error: {str(e)}")
                results[url] = {"error": str(e)}

        return results