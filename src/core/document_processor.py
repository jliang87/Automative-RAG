import os
import uuid
from typing import Dict, List, Optional, Union

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.core.pdf_loader import PDFLoader
from src.core.vectorstore import QdrantStore
from src.core.base_video_transcriber import YouTubeTranscriber, BilibiliTranscriber, create_transcriber_for_url

from src.models.schema import DocumentMetadata, DocumentSource, ManualIngestRequest
from src.config.settings import settings


class DocumentProcessor:
    """
    Class for processing documents from various sources with GPU acceleration.
    
    Handles YouTube/Bilibili videos, PDFs, and manual text entry.
    Chunks documents and adds them to the vector store.
    """

    def __init__(
        self,
        vector_store: QdrantStore,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        upload_dir: str = "data/uploads",
        device: Optional[str] = None,
        youtube_transcriber: Optional[YouTubeTranscriber] = None,
        bilibili_transcriber: Optional[BilibiliTranscriber] = None,
        pdf_loader: Optional[PDFLoader] = None,
    ):
        """
        Initialize the document processor.

        Args:
            vector_store: Vector store to add documents to
            chunk_size: Size of document chunks
            chunk_overlap: Overlap between chunks
            upload_dir: Directory for uploaded files
            device: Device to use for GPU-accelerated tasks (cuda or cpu)
        """
        self.vector_store = vector_store
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.upload_dir = upload_dir
        self.device = device or settings.device

        self.youtube_transcriber = youtube_transcriber
        self.bilibili_transcriber = bilibili_transcriber
        self.pdf_loader = pdf_loader

        # Always create the text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        os.makedirs(upload_dir, exist_ok=True)

    def process_youtube_video(
        self, url: str, custom_metadata: Optional[Dict[str, str]] = None, force_whisper: bool = False
    ) -> List[str]:
        """
        Process a YouTube video with GPU-accelerated transcription.

        Args:
            url: YouTube URL
            custom_metadata: Optional custom metadata
            force_whisper: Whether to force using Whisper even if YouTube captions are available

        Returns:
            List of document IDs
        """
        # Get documents from transcriber
        documents = self.youtube_transcriber.process_video(
            url=url,
            custom_metadata=custom_metadata,
            force_whisper=force_whisper
        )
        
        # Split into chunks
        chunked_documents = self.text_splitter.split_documents(documents)
        
        # Assign IDs to chunks
        video_id = self.youtube_transcriber.extract_video_id(url)
        for i, doc in enumerate(chunked_documents):
            doc.metadata["id"] = f"youtube-{video_id}-{i}"
            doc.metadata["chunk_id"] = i
        
        # Add to vector store
        return self.vector_store.add_documents(chunked_documents)

    def process_bilibili_video(
            self, url: str, custom_metadata: Optional[Dict[str, str]] = None, force_whisper: bool = True
    ) -> List[str]:
        """
        Process a Bilibili video with GPU-accelerated transcription.

        Args:
            url: Bilibili URL
            custom_metadata: Optional custom metadata
            force_whisper: Whether to force using Whisper for transcription (default: True)

        Returns:
            List of document IDs
        """
        # Get documents from transcriber
        documents = self.bilibili_transcriber.process_video(
            url=url,
            custom_metadata=custom_metadata
        )

        # Split into chunks
        chunked_documents = self.text_splitter.split_documents(documents)

        # Assign IDs to chunks
        video_id = self.bilibili_transcriber.extract_video_id(url)
        for i, doc in enumerate(chunked_documents):
            doc.metadata["id"] = f"bilibili-{video_id}-{i}"
            doc.metadata["chunk_id"] = i

        # Add to vector store
        return self.vector_store.add_documents(chunked_documents)


    def process_pdf(
        self, file_path: str, custom_metadata: Optional[Dict[str, str]] = None
    ) -> List[str]:
        """
        Process a PDF file with GPU-accelerated OCR if needed.

        Args:
            file_path: Path to the PDF file
            custom_metadata: Optional custom metadata

        Returns:
            List of document IDs
        """
        # Get documents from PDF loader
        documents = self.pdf_loader.process_pdf(
            file_path=file_path,
            custom_metadata=custom_metadata,
        )
        
        # Assign IDs to chunks
        base_name = os.path.basename(file_path)
        for i, doc in enumerate(documents):
            doc.metadata["id"] = f"pdf-{base_name}-{i}"
            doc.metadata["chunk_id"] = i
        
        # Add to vector store
        return self.vector_store.add_documents(documents)

    def process_text(self, request: ManualIngestRequest) -> List[str]:
        """
        Process manually entered text.

        Args:
            request: Manual ingest request

        Returns:
            List of document IDs
        """
        # Create document
        document = Document(
            page_content=request.content,
            metadata=request.metadata.dict(),
        )
        
        # Split into chunks
        chunked_documents = self.text_splitter.split_documents([document])
        
        # Assign IDs to chunks
        doc_id = str(uuid.uuid4())
        for i, doc in enumerate(chunked_documents):
            doc.metadata["id"] = f"manual-{doc_id}-{i}"
            doc.metadata["chunk_id"] = i
        
        # Add to vector store
        return self.vector_store.add_documents(chunked_documents)

    def batch_process_videos(
            self, urls: List[str], platform: str = "youtube", custom_metadata: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, List[str]]:
        """
        Process multiple videos in batch with GPU acceleration.

        Args:
            urls: List of video URLs
            platform: Platform of the videos ("youtube", "bilibili")
            custom_metadata: Optional list of custom metadata (same length as urls)

        Returns:
            Dictionary mapping URLs to lists of document IDs
        """
        result = {}

        # Validate inputs
        if custom_metadata and len(custom_metadata) != len(urls):
            raise ValueError("If provided, custom_metadata must have the same length as urls")

        for i, url in enumerate(urls):
            metadata = custom_metadata[i] if custom_metadata else None
            try:
                if platform.lower() == "youtube":
                    doc_ids = self.process_youtube_video(url, metadata)
                elif platform.lower() == "bilibili":
                    doc_ids = self.process_bilibili_video(url, metadata)
                else:
                    raise ValueError(f"Unsupported platform: {platform}")

                result[url] = doc_ids
            except Exception as e:
                result[url] = {"error": str(e)}

        return result

    def delete_documents(self, ids: List[str]) -> None:
        """
        Delete documents from the vector store.

        Args:
            ids: List of document IDs to delete
        """
        self.vector_store.delete_by_ids(ids)

    def get_source_documents(self, source_id: str) -> List[Document]:
        """
        Get all documents with the given source ID.

        Args:
            source_id: Source ID to filter by

        Returns:
            List of documents
        """
        # This would require implementing a method to search by metadata in QdrantStore
        # For simplicity, we'll just return a stub
        return []

    def process_video_generic(self, url, custom_metadata=None, force_whisper=True):
        """Process any supported video URL automatically."""
        transcriber = create_transcriber_for_url(
            url,
            device=self.device,
            force_whisper=force_whisper
        )

        documents = transcriber.process_video(
            url=url,
            custom_metadata=custom_metadata,
            force_whisper=force_whisper
        )

        # Split into chunks
        chunked_documents = self.text_splitter.split_documents(documents)

        # Assign IDs to chunks
        video_id = transcriber.extract_video_id(url)
        platform = transcriber.__class__.__name__.replace('Transcriber', '').lower()

        for i, doc in enumerate(chunked_documents):
            doc.metadata["id"] = f"{platform}-{video_id}-{i}"
            doc.metadata["chunk_id"] = i
            doc.metadata["platform"] = platform

        # Add to vector store
        return self.vector_store.add_documents(chunked_documents)