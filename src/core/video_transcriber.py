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
from langchain_core.documents import Document
import whisper

from src.models.schema import DocumentMetadata, DocumentSource
from src.config.settings import settings
from src.utils.helpers import (
    extract_metadata_from_text,
    generate_unique_id
)


class VideoTranscriber:
    """
    Unified video transcriber that handles multiple platforms (YouTube, Bilibili, etc.)
    using Whisper for all transcription tasks.
    """

    def __init__(
            self,
            output_dir: str = "data/videos",
            whisper_model_size: str = "medium",
            device: Optional[str] = None
    ):
        """
        Initialize the video transcriber.

        Args:
            output_dir: Directory to save downloaded videos and audio
            whisper_model_size: Size of the Whisper model (tiny, base, small, medium, large)
            device: Device to run Whisper on (cuda or cpu), defaults to cuda if available
        """
        self.output_dir = output_dir
        self.audio_dir = os.path.join(output_dir, "audio")
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(self.audio_dir, exist_ok=True)

        # Determine device (use CUDA if available)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.whisper_model_size = whisper_model_size
        self.whisper_model = None  # Lazy-load the model when needed

        # Add Chinese converter if available
        try:
            import opencc
            self.chinese_converter = opencc.OpenCC('t2s')  # Traditional to Simplified
            print("Chinese character conversion enabled")
        except ImportError:
            print("Warning: opencc-python-reimplemented not installed. Chinese character conversion disabled.")
            self.chinese_converter = None

    def _load_whisper_model(self):
        """Load the Whisper model if not already loaded."""
        if self.whisper_model is None:
            print(f"Loading Whisper {self.whisper_model_size} model on {self.device}...")

            # Check if using custom model path
            if hasattr(settings, 'whisper_model_full_path') and settings.whisper_model_full_path and os.path.exists(
                    settings.whisper_model_full_path):
                # Load from local path
                print(f"Loading Whisper model from local path: {settings.whisper_model_full_path}")

                self.whisper_model = whisper.load_model(
                    name=self.whisper_model_size,
                    device=self.device,
                    download_root=settings.whisper_model_full_path
                )
            else:
                # Load from default location
                self.whisper_model = whisper.load_model(
                    self.whisper_model_size,
                    device=self.device
                )

            print("Whisper model loaded successfully")

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
                    "url": url,
                    "length": data.get("duration", 0),
                    "views": data.get("view_count", 0),
                    "description": data.get("description", ""),
                }
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

            # Create minimal metadata to allow processing to continue
            return {
                "title": f"Video {video_id}",
                "author": "Unknown Author",
                "published_date": None,
                "video_id": video_id,
                "url": url,
                "length": 0,
                "views": 0,
                "description": "",
            }

    def transcribe_with_whisper(self, media_path: str) -> str:
        """
        Transcribe audio or video file using Whisper with GPU acceleration.

        Args:
            media_path: Path to the audio or video file

        Returns:
            Transcribed text
        """
        # Load model if not already loaded
        self._load_whisper_model()

        print(f"Transcribing with Whisper ({self.whisper_model_size}) on {self.device}...")

        # Whisper can directly handle both audio and video files
        result = self.whisper_model.transcribe(media_path)
        transcript = result["text"]

        # Convert traditional to simplified Chinese if needed
        if hasattr(self, 'chinese_converter') and self.chinese_converter:
            transcript = self.chinese_converter.convert(transcript)

        return transcript

    def format_transcript(self, transcript: str, is_srt: bool = False) -> str:
        """
        Format the transcript to plain text.

        Args:
            transcript: Transcript text
            is_srt: Whether the transcript is in SRT format

        Returns:
            Plain text transcript
        """
        if not is_srt:
            return transcript  # Already plain text (from Whisper)

        # Format SRT/VTT transcript
        lines = transcript.split('\n')
        formatted_lines = []
        current_text = ""

        # For VTT format
        if "WEBVTT" in lines[0]:
            in_cue = False
            for line in lines:
                # Skip header and timing lines
                if "-->" in line or line.strip() == "" or line.startswith("WEBVTT"):
                    if in_cue and current_text:
                        formatted_lines.append(current_text.strip())
                        current_text = ""
                    in_cue = False
                    continue

                # We're in a text cue
                in_cue = True
                current_text += " " + line.strip()

        # For SRT format
        else:
            i = 0
            while i < len(lines):
                # Skip index lines (numbers)
                if lines[i].strip().isdigit():
                    i += 1
                    continue

                # Skip timestamp lines
                if '-->' in lines[i]:
                    i += 1
                    continue

                # Add text content
                if lines[i].strip():
                    formatted_lines.append(lines[i].strip())

                i += 1

        # Convert to plain text
        formatted_text = ' '.join(formatted_lines)

        # Convert traditional to simplified Chinese if needed
        if hasattr(self, 'chinese_converter') and self.chinese_converter:
            formatted_text = self.chinese_converter.convert(formatted_text)

        return formatted_text



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
            transcript_text = self.transcribe_with_whisper(media_path)
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
            custom_metadata=custom_metadata or {},
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