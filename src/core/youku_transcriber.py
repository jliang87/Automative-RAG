"""
Module for transcribing Youku videos with GPU-accelerated Whisper.
"""

import os
import re
import tempfile
from typing import Dict, List, Optional, Tuple, Union

import torch
from langchain_core.documents import Document

from src.models.schema import DocumentMetadata, DocumentSource
from src.core.youtube_transcriber import YouTubeTranscriber


class YoukuTranscriber(YouTubeTranscriber):
    """
    Class for downloading and transcribing Youku videos with GPU acceleration.

    Extends YouTubeTranscriber to handle Youku videos.
    """

    def __init__(
            self,
            output_dir: str = "data/youku",
            whisper_model_size: str = "medium",
            device: Optional[str] = None,
            force_whisper: bool = True
    ):
        """
        Initialize the Youku transcriber.

        Args:
            output_dir: Directory to save downloaded videos and audio
            whisper_model_size: Size of the Whisper model (tiny, base, small, medium, large)
            device: Device to run Whisper on (cuda or cpu), defaults to cuda if available
            force_whisper: Always use Whisper for transcription
        """
        super().__init__(
            output_dir=output_dir,
            whisper_model_size=whisper_model_size,
            device=device,
            use_youtube_captions=False,  # Always disable captions for Youku
            use_whisper_as_fallback=True,  # Always use Whisper for Youku
            force_whisper=force_whisper  # Pass through force_whisper parameter
        )

    def extract_video_id(self, url: str) -> str:
        """
        Extract the video ID from a Youku URL.

        Args:
            url: Youku URL

        Returns:
            Video ID or None if invalid
        """
        # Extract Youku video ID
        # Youku URLs often look like: https://v.youku.com/v_show/id_XNTk1NDAxMzgwNA==.html
        match = re.search(r'id_([^.=]+)', url)
        if not match:
            raise ValueError(f"Could not extract Youku video ID from URL: {url}")
        return match.group(1)

    def download_youku_video(self, url: str) -> str:
        """
        Download a Youku video.

        Args:
            url: Youku URL

        Returns:
            Path to the downloaded video file
        """
        import subprocess

        video_id = self.extract_video_id(url)
        video_path = os.path.join(self.output_dir, f"{video_id}.mp4")

        # Check if we already have this video file
        if os.path.exists(video_path):
            print(f"Video already exists for ID: {video_id}")
            return video_path

        # Download using yt-dlp (which supports Youku)
        # Make sure you have yt-dlp installed: pip install yt-dlp
        print(f"Downloading Youku video: {video_id}")

        try:
            subprocess.run([
                "yt-dlp",
                "-f", "best",
                "-o", video_path,
                url
            ], check=True)

            return video_path
        except subprocess.CalledProcessError as e:
            raise ValueError(f"Error downloading Youku video: {str(e)}")
        except FileNotFoundError:
            raise ValueError("yt-dlp not found. Please install it with: pip install yt-dlp")

    def extract_audio_from_video(self, video_path: str) -> str:
        """
        Extract audio from a video file.

        Args:
            video_path: Path to the video file

        Returns:
            Path to the extracted audio file
        """
        import subprocess

        # Generate output audio path
        video_filename = os.path.basename(video_path)
        video_id = os.path.splitext(video_filename)[0]
        audio_path = os.path.join(self.audio_dir, f"{video_id}.mp3")

        # Check if we already have this audio file
        if os.path.exists(audio_path):
            print(f"Audio already exists for video ID: {video_id}")
            return audio_path

        # Extract audio using ffmpeg
        print(f"Extracting audio from video: {video_id}")

        try:
            subprocess.run([
                "ffmpeg",
                "-i", video_path,
                "-q:a", "0",
                "-map", "a",
                "-f", "mp3",
                audio_path
            ], check=True)

            return audio_path
        except subprocess.CalledProcessError as e:
            raise ValueError(f"Error extracting audio: {str(e)}")
        except FileNotFoundError:
            raise ValueError("ffmpeg not found. Please install it first.")

    def get_video_metadata(self, url: str) -> Dict[str, Union[str, int]]:
        """
        Get metadata from a Youku video.

        Args:
            url: Youku URL

        Returns:
            Dictionary containing video metadata
        """
        import subprocess
        import json

        video_id = self.extract_video_id(url)

        try:
            # Use yt-dlp to get metadata
            result = subprocess.run([
                "yt-dlp",
                "--dump-json",
                url
            ], capture_output=True, text=True, check=True)

            metadata = json.loads(result.stdout)

            return {
                "title": metadata.get("title", ""),
                "author": metadata.get("uploader", ""),
                "published_date": metadata.get("upload_date", ""),
                "video_id": video_id,
                "url": url,
                "length": metadata.get("duration", 0),
                "views": metadata.get("view_count", 0),
                "description": metadata.get("description", ""),
            }
        except Exception as e:
            # Fallback with minimal metadata if yt-dlp fails
            print(f"Warning: Could not get full Youku metadata: {str(e)}")
            return {
                "title": "Youku Video",
                "video_id": video_id,
                "url": url,
            }

    def process_video(
            self, url: str, custom_metadata: Optional[Dict[str, str]] = None, force_whisper: bool = True
    ) -> List[Document]:
        """
        Process a Youku video and return Langchain documents.

        Args:
            url: Youku URL
            custom_metadata: Optional custom metadata
            force_whisper: Always use Whisper for transcription (default: True)

        Returns:
            List of Langchain Document objects
        """
        # Extract metadata
        video_metadata = self.get_video_metadata(url)

        # Extract automotive metadata
        auto_metadata = self.extract_automotive_metadata(video_metadata)

        try:
            # Download video
            video_path = self.download_youku_video(url)

            # Extract audio
            audio_path = self.extract_audio_from_video(video_path)

            # Always use Whisper for transcription
            print(f"Using Whisper for Youku video transcription: {video_metadata['video_id']}")
            transcript_text = self.transcribe_with_whisper(audio_path)

            # Create metadata object
            metadata = DocumentMetadata(
                source=DocumentSource.YOUTUBE,  # Reuse YouTube enum for simplicity
                source_id=video_metadata["video_id"],
                url=url,
                title=video_metadata["title"],
                author=video_metadata.get("author"),
                published_date=video_metadata.get("published_date"),
                manufacturer=auto_metadata.get("manufacturer"),
                model=custom_metadata.get("model") if custom_metadata else None,
                year=auto_metadata.get("year"),
                category=auto_metadata.get("category"),
                engine_type=auto_metadata.get("engine_type"),
                transmission=auto_metadata.get("transmission"),
                custom_metadata=custom_metadata or {},
            )

            # Add platform and transcription info to metadata
            metadata.custom_metadata["platform"] = "youku"
            metadata.custom_metadata["transcription_method"] = "whisper"
            metadata.custom_metadata["whisper_model"] = self.whisper_model_size

            # Create document
            document = Document(
                page_content=transcript_text,
                metadata=metadata.dict(),
            )

            return [document]
        except Exception as e:
            raise ValueError(f"Error processing Youku video: {str(e)}")