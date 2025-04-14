import os
import uuid
from typing import Dict, List, Optional, Union

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.core.pdf_loader import PDFLoader
from src.core.vectorstore import QdrantStore
from src.core.video_transcriber import VideoTranscriber

from src.models.schema import DocumentMetadata, DocumentSource, ManualIngestRequest
from src.config.settings import settings


class DocumentProcessor:
    """
    Class for processing documents from various sources with GPU acceleration.

    Handles videos (YouTube, Bilibili), PDFs, and manual text entry.
    Chunks documents and adds them to the vector store.
    """

    def __init__(
            self,
            vector_store: QdrantStore,
            chunk_size: int = 1000,
            chunk_overlap: int = 200,
            upload_dir: str = "data/uploads",
            device: Optional[str] = None,
            video_transcriber: Optional[VideoTranscriber] = None,
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

        # Initialize video transcriber if not provided
        self.video_transcriber = video_transcriber or VideoTranscriber(
            whisper_model_size=settings.whisper_model_size,
            device=self.device
        )

        # Initialize PDF loader if not provided
        self.pdf_loader = pdf_loader or PDFLoader(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            device=self.device,
            use_ocr=settings.use_pdf_ocr,
            ocr_languages=settings.ocr_languages
        )

        # Create the text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        os.makedirs(upload_dir, exist_ok=True)

    def process_video(
            self, url: str, custom_metadata: Optional[Dict[str, str]] = None
    ) -> List[str]:
        """
        Process a video from any platform with GPU-accelerated Whisper transcription.

        Args:
            url: Video URL
            custom_metadata: Optional custom metadata

        Returns:
            List of document IDs
        """
        # Get platform from the video URL
        platform = self.video_transcriber.detect_platform(url)

        # Get documents from transcriber
        documents = self.video_transcriber.process_video(
            url=url,
            custom_metadata=custom_metadata
        )

        # Split into chunks
        chunked_documents = self.text_splitter.split_documents(documents)

        # Assign IDs to chunks
        video_id = self.video_transcriber.extract_video_id(url)
        for i, doc in enumerate(chunked_documents):
            doc.metadata["id"] = f"{platform}-{video_id}-{i}"
            doc.metadata["chunk_id"] = i
            doc.metadata["platform"] = platform

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
            self, urls: List[str], custom_metadata: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, List[str]]:
        """
        Process multiple videos in batch with GPU acceleration.

        Args:
            urls: List of video URLs
            custom_metadata: Optional list of custom metadata (same length as urls)

        Returns:
            Dictionary mapping URLs to lists of document IDs
        """
        result = {}

        # Validate inputs
        if custom_metadata and len(custom_metadata) != len(urls):
            raise ValueError("If provided, custom_metadata must have the same length as urls")

        # Process each URL
        for i, url in enumerate(urls):
            metadata = custom_metadata[i] if custom_metadata else None
            try:
                doc_ids = self.process_video(url, metadata)
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