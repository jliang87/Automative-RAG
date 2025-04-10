import os
import re
import tempfile
import subprocess
import json
from typing import Dict, List, Optional, Tuple, Union

import torch
from langchain_core.documents import Document
import whisper

from src.models.schema import DocumentMetadata, DocumentSource
from src.config.settings import settings


class YouTubeTranscriber:
    """
    Enhanced class for downloading and transcribing YouTube videos with GPU acceleration.

    Uses yt-dlp for video downloading and Whisper for high-quality transcription.
    Supports loading Whisper from local paths.
    """

    def __init__(
            self,
            output_dir: str = "data/youtube",
            whisper_model_size: str = "medium",
            device: Optional[str] = None,
            use_youtube_captions: bool = False,
            use_whisper_as_fallback: bool = False,
            force_whisper: bool = True
    ):
        """
        Initialize the YouTube transcriber.

        Args:
            output_dir: Directory to save downloaded videos and audio
            whisper_model_size: Size of the Whisper model (tiny, base, small, medium, large)
            device: Device to run Whisper on (cuda or cpu), defaults to cuda if available
            use_youtube_captions: Whether to use YouTube's captions if available
            use_whisper_as_fallback: Whether to use Whisper as fallback when YouTube captions aren't available
        """
        self.output_dir = output_dir
        self.audio_dir = os.path.join(output_dir, "audio")
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(self.audio_dir, exist_ok=True)

        # Determine device (use CUDA if available)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.whisper_model_size = whisper_model_size
        self.whisper_model = None  # Lazy-load the model when needed

        # Configuration options
        self.use_youtube_captions = use_youtube_captions
        self.use_whisper_as_fallback = use_whisper_as_fallback
        self.force_whisper = force_whisper

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

            # Check if we should use models directory for downloading/loading
            models_dir = None
            if hasattr(settings, 'models_dir') and settings.models_dir:
                models_dir = os.path.join(settings.models_dir, settings.whisper_model_path)
                print(f"Using models directory: {models_dir}")

                # Ensure directory exists
                os.makedirs(models_dir, exist_ok=True)

            # Load model from cache or download if needed
            self.whisper_model = whisper.load_model(
                name=self.whisper_model_size,
                device=self.device,
                download_root=models_dir
            )

            print("Whisper model loaded successfully")

    def extract_video_id(self, url: str) -> str:
        """
        Extract the video ID from a YouTube URL.

        Args:
            url: YouTube URL

        Returns:
            Video ID
        """
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

    def extract_audio(self, url: str) -> str:
        """
        Extract audio from a YouTube video using yt-dlp.

        Args:
            url: YouTube URL

        Returns:
            Path to the extracted audio file
        """
        video_id = self.extract_video_id(url)

        # Define output file path
        audio_path = os.path.join(self.audio_dir, f"{video_id}.mp3")

        # Check if we already have this audio file
        if os.path.exists(audio_path):
            print(f"Audio already exists for video ID: {video_id}")
            return audio_path

        # Download audio using yt-dlp
        print(f"Downloading audio for YouTube video: {video_id}")

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
            else:
                raise ValueError(f"Audio file not found after download: {audio_path}")

        except subprocess.CalledProcessError as e:
            raise ValueError(f"Error downloading audio with yt-dlp: {str(e)}")
        except FileNotFoundError:
            raise ValueError("yt-dlp not found. Please install it with: pip install yt-dlp")

    def get_video_metadata(self, url: str) -> Dict[str, Union[str, int]]:
        """
        Get metadata from a YouTube video using yt-dlp.

        Args:
            url: YouTube URL

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
                    "title": data.get("title", f"YouTube Video {video_id}"),
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
            print(f"Warning: Error fetching YouTube metadata: {str(e)}")
            print(f"Using fallback metadata for video ID: {video_id if 'video_id' in locals() else 'unknown'}")

            # Try to extract video_id if not already done
            if 'video_id' not in locals():
                try:
                    video_id = self.extract_video_id(url)
                except Exception:
                    video_id = "unknown"

            # Create minimal metadata to allow processing to continue
            return {
                "title": f"YouTube Video {video_id}",
                "author": "Unknown Author",
                "published_date": None,
                "video_id": video_id,
                "url": url,
                "length": 0,
                "views": 0,
                "description": "",
            }

    def download_youtube_captions(self, url: str) -> Optional[str]:
        """
        Download the transcript from a YouTube video's captions using yt-dlp.

        Args:
            url: YouTube URL

        Returns:
            Video transcript text or None if no captions are available
        """
        try:
            video_id = self.extract_video_id(url)

            # Create a temporary directory for the subtitles
            with tempfile.TemporaryDirectory() as temp_dir:
                srt_path = os.path.join(temp_dir, f"{video_id}.en.vtt")

                # Try to download subtitles
                try:
                    subprocess.run([
                        "yt-dlp",
                        "--skip-download",  # Don't download the video
                        "--write-sub",  # Write subtitles
                        "--write-auto-sub",  # Write auto-generated subtitles if available
                        "--sub-lang", "en",  # English subtitles
                        "--sub-format", "vtt",  # VTT format
                        "-o", os.path.join(temp_dir, video_id),  # Output filename
                        url
                    ], check=True, capture_output=True)
                except subprocess.CalledProcessError:
                    return None

                # Check if subtitles were downloaded
                srt_files = [f for f in os.listdir(temp_dir) if f.endswith('.vtt')]
                if not srt_files:
                    return None

                # Read the subtitle file
                srt_path = os.path.join(temp_dir, srt_files[0])
                with open(srt_path, 'r', encoding='utf-8') as f:
                    return f.read()

        except Exception as e:
            print(f"Warning: Could not download YouTube captions: {str(e)}")
            return None

    def transcribe_with_whisper(self, audio_path: str) -> str:
        """
        Transcribe audio file using Whisper with GPU acceleration.

        Args:
            audio_path: Path to the audio file

        Returns:
            Transcribed text
        """
        # Load model if not already loaded
        self._load_whisper_model()

        print(f"Transcribing with Whisper ({self.whisper_model_size}) on {self.device}...")

        # Transcribe the audio
        result = self.whisper_model.transcribe(audio_path)

        # Get the transcript text
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

    def extract_automotive_metadata(self, metadata: Dict[str, any]) -> Dict[str, any]:
        """
        Extract automotive-specific metadata from video title and description.

        Args:
            metadata: Video metadata

        Returns:
            Dictionary with automotive metadata
        """
        title = metadata.get("title", "")
        description = metadata.get("description", "")

        # Combine for search
        text = title + " " + description

        auto_metadata = {}

        # Common manufacturers
        manufacturers = [
            "Toyota", "Honda", "Ford", "Chevrolet", "BMW", "Mercedes", "Audi",
            "Volkswagen", "Nissan", "Hyundai", "Kia", "Subaru", "Mazda",
            "Porsche", "Ferrari", "Lamborghini", "Tesla", "Volvo", "Jaguar",
            "Land Rover", "Lexus", "Acura", "Infiniti", "Cadillac", "Jeep",
            # Chinese manufacturers
            "BYD", "NIO", "Xpeng", "Li Auto", "Geely", "Great Wall", "Chery", "SAIC"
        ]

        # Look for manufacturer
        for manufacturer in manufacturers:
            if manufacturer.lower() in text.lower():
                auto_metadata["manufacturer"] = manufacturer
                break

        # Extract year (4-digit number between 1900 and 2100)
        year_match = re.search(r'(19\d{2}|20\d{2})', text)
        if year_match:
            auto_metadata["year"] = int(year_match.group(0))

        # Common categories
        categories = {
            "sedan": ["sedan", "saloon"],
            "suv": ["suv", "crossover"],
            "truck": ["truck", "pickup"],
            "sports": ["sports car", "supercar", "hypercar"],
            "minivan": ["minivan", "van"],
            "coupe": ["coupe", "coupé"],
            "convertible": ["convertible", "cabriolet"],
            "hatchback": ["hatchback", "hot hatch"],
            "wagon": ["wagon", "estate"],
        }

        for category, keywords in categories.items():
            for keyword in keywords:
                if keyword.lower() in text.lower():
                    auto_metadata["category"] = category
                    break
            if "category" in auto_metadata:
                break

        # Engine types
        engine_types = {
            "gasoline": ["gasoline", "petrol", "gas"],
            "diesel": ["diesel"],
            "electric": ["electric", "ev", "battery"],
            "hybrid": ["hybrid", "phev", "plug-in"],
            "hydrogen": ["hydrogen", "fuel cell"],
        }

        for engine_type, keywords in engine_types.items():
            for keyword in keywords:
                if keyword.lower() in text.lower():
                    auto_metadata["engine_type"] = engine_type
                    break
            if "engine_type" in auto_metadata:
                break

        # Transmission types
        transmission_types = {
            "automatic": ["automatic", "auto"],
            "manual": ["manual", "stick", "6-speed"],
            "cvt": ["cvt", "continuously variable"],
            "dct": ["dct", "dual-clutch"],
        }

        for transmission, keywords in transmission_types.items():
            for keyword in keywords:
                if keyword.lower() in text.lower():
                    auto_metadata["transmission"] = transmission
                    break
            if "transmission" in auto_metadata:
                break

        return auto_metadata

    def process_video(
            self, url: str, custom_metadata: Optional[Dict[str, str]] = None, force_whisper: bool = False
    ) -> List[Document]:
        """
        Process a YouTube video and return Langchain documents.

        Args:
            url: YouTube URL
            custom_metadata: Optional custom metadata
            force_whisper: Whether to force using Whisper even if YouTube captions are available

        Returns:
            List of Langchain Document objects
        """
        # Extract metadata
        video_metadata = self.get_video_metadata(url)

        # Extract automotive metadata
        auto_metadata = self.extract_automotive_metadata(video_metadata)

        # Get transcript
        transcript_text = None
        used_whisper = False

        # Use Whisper directly if force_whisper is enabled
        if self.force_whisper or force_whisper:
            print(f"Using Whisper for transcription as configured for video ID: {video_metadata['video_id']}")
        # Otherwise, try YouTube captions first if enabled and not forced to use Whisper
        elif self.use_youtube_captions and not force_whisper:
            youtube_captions = self.download_youtube_captions(url)
            if youtube_captions:
                transcript_text = self.format_transcript(youtube_captions, is_srt=True)
                print(f"Using YouTube captions for video ID: {video_metadata['video_id']}")

        # Use Whisper if no transcript yet or force_whisper is True
        if transcript_text is None and (self.use_whisper_as_fallback or self.force_whisper or force_whisper):
            try:
                # Extract audio
                audio_path = self.extract_audio(url)

                # Transcribe with Whisper
                transcript_text = self.transcribe_with_whisper(audio_path)
                used_whisper = True
                print(f"Using Whisper transcription for video ID: {video_metadata['video_id']}")
            except Exception as e:
                raise ValueError(f"Error transcribing video with Whisper: {str(e)}")

        if transcript_text is None:
            raise ValueError("No transcript available for this video and transcription failed")

        # Create metadata object
        metadata = DocumentMetadata(
            source=DocumentSource.YOUTUBE,
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

        # Add transcription method to metadata
        metadata.custom_metadata["transcription_method"] = "whisper" if used_whisper else "youtube_captions"
        if used_whisper:
            metadata.custom_metadata["whisper_model"] = self.whisper_model_size

        # Create document
        document = Document(
            page_content=transcript_text,
            metadata=metadata.dict(),
        )

        return [document]

    def batch_process_videos(
            self, urls: List[str], custom_metadata: Optional[List[Dict[str, str]]] = None
    ) -> List[Document]:
        """
        Process multiple YouTube videos in batch.

        Args:
            urls: List of YouTube URLs
            custom_metadata: Optional list of custom metadata dictionaries (same length as urls)

        Returns:
            List of Langchain Document objects
        """
        # Ensure custom_metadata is the right length or None
        if custom_metadata and len(custom_metadata) != len(urls):
            raise ValueError("custom_metadata list must be the same length as urls")

        documents = []

        # Load model once for all videos
        if self.use_whisper_as_fallback:
            self._load_whisper_model()

        # Process each video
        for i, url in enumerate(urls):
            metadata = custom_metadata[i] if custom_metadata else None
            try:
                video_docs = self.process_video(url, metadata)
                documents.extend(video_docs)
                print(f"Successfully processed video {i + 1}/{len(urls)}: {url}")
            except Exception as e:
                print(f"Error processing video {i + 1}/{len(urls)}: {url}")
                print(f"Error: {str(e)}")

        return documents


class BilibiliTranscriber(YouTubeTranscriber):
    """
    Class for downloading and transcribing Bilibili videos with GPU acceleration.

    Extends YouTubeTranscriber to handle Bilibili videos.
    """

    def __init__(
            self,
            output_dir: str = "data/bilibili",
            whisper_model_size: str = "medium",
            device: Optional[str] = None,
            force_whisper: bool = True
    ):
        """
        Initialize the Bilibili transcriber.

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
            use_youtube_captions=False,  # Always disable captions for Bilibili
            use_whisper_as_fallback=True,  # Always use Whisper for Bilibili
            force_whisper=force_whisper   # Pass through force_whisper parameter
        )

    def extract_video_id(self, url: str) -> str:
        """
        Extract the video ID from a Bilibili URL.

        Args:
            url: Bilibili URL

        Returns:
            Video ID
        """
        # Extract Bilibili video ID (BV or AV id)
        match = re.search(r'/(BV\w+|av\d+)', url)
        if not match:
            raise ValueError(f"Could not extract Bilibili video ID from URL: {url}")
        return match.group(1)

    def download_bilibili_video(self, url: str) -> str:
        """
        Download a Bilibili video.

        Args:
            url: Bilibili URL

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

        # Download using youtube-dl or yt-dlp (which supports Bilibili)
        # Make sure you have yt-dlp installed: pip install yt-dlp
        print(f"Downloading Bilibili video: {video_id}")

        try:
            subprocess.run([
                "yt-dlp",
                "-f", "30280",
                "-o", video_path,
                url
            ], check=True)

            return video_path
        except subprocess.CalledProcessError as e:
            raise ValueError(f"Error downloading Bilibili video: {str(e)}")
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
        Get metadata from a Bilibili video.

        Args:
            url: Bilibili URL

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
            print(f"Warning: Could not get full Bilibili metadata: {str(e)}")
            return {
                "title": "Bilibili Video",
                "video_id": video_id,
                "url": url,
            }

    def process_video(
            self, url: str, custom_metadata: Optional[Dict[str, str]] = None, force_whisper: bool = True
    ) -> List[Document]:
        """
        Process a Bilibili video and return Langchain documents.

        Args:
            url: Bilibili URL
            custom_metadata: Optional custom metadata

        Returns:
            List of Langchain Document objects
        """
        # Extract metadata
        video_metadata = self.get_video_metadata(url)

        # Extract automotive metadata
        auto_metadata = self.extract_automotive_metadata(video_metadata)

        try:
            # Download video
            video_path = self.download_bilibili_video(url)

            # Extract audio
            audio_path = self.extract_audio_from_video(video_path)

            # Always use Whisper for transcription
            print(f"Using Whisper for Bilibili video transcription: {video_metadata['video_id']}")
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
            metadata.custom_metadata["platform"] = "bilibili"
            metadata.custom_metadata["transcription_method"] = "whisper"
            metadata.custom_metadata["whisper_model"] = self.whisper_model_size

            # Create document
            document = Document(
                page_content=transcript_text,
                metadata=metadata.dict(),
            )

            return [document]
        except Exception as e:
            raise ValueError(f"Error processing Bilibili video: {str(e)}")