import os
import uuid
import time
import logging
from typing import Dict, List, Optional, Union

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.core.pdf_loader import PDFLoader
from src.core.vectorstore import QdrantStore
from src.core.video_transcriber import VideoTranscriber

from src.models.schema import DocumentMetadata, DocumentSource, ManualIngestRequest
from src.config.settings import settings

# Configure logging
logger = logging.getLogger(__name__)


class DocumentProcessor:
    """
    Enhanced class for processing documents from various sources with GPU acceleration.

    Handles videos (YouTube, Bilibili), PDFs, and manual text entry.
    Chunks documents and adds them to the vector store with proper metadata.
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

        logger.info(f"Document processor initialized with chunk size: {chunk_size}, overlap: {chunk_overlap}")
        logger.info(f"Using device: {self.device}")

    def process_video(
            self, url: str, custom_metadata: Optional[Dict[str, str]] = None, force_refresh: bool = False
    ) -> List[str]:
        """
        Process a video from any platform with enhanced metadata tracking.

        Args:
            url: Video URL
            custom_metadata: Optional custom metadata
            force_refresh: Force reprocessing even if video was previously ingested

        Returns:
            List of document IDs
        """
        # Get platform from the video URL
        platform = self.video_transcriber.detect_platform(url)
        video_id = self.video_transcriber.extract_video_id(url)

        logger.info(f"Processing video: {url}")
        logger.info(f"Detected platform: {platform}, video ID: {video_id}")

        # Check if we already have this video if not forcing refresh
        if not force_refresh:
            existing_docs = self.vector_store.search_by_metadata({
                "source": platform,
                "source_id": video_id,
            })

            if existing_docs:
                logger.info(f"Video already ingested with {len(existing_docs)} chunks, returning existing document IDs")
                return [doc.metadata.get("id") for doc in existing_docs]

        # Get documents from transcriber
        documents = self.video_transcriber.process_video(
            url=url,
            custom_metadata=custom_metadata
        )

        logger.info(f"Transcription completed, got {len(documents)} documents")

        # Split into chunks
        chunked_documents = self.text_splitter.split_documents(documents)
        logger.info(f"Split into {len(chunked_documents)} chunks")

        # Assign IDs and additional metadata to chunks
        current_time = time.time()
        for i, doc in enumerate(chunked_documents):
            # Ensure an ID exists (might be overwritten by vector store)
            doc_id = f"{platform}-{video_id}-{i}"
            doc.metadata["id"] = doc_id
            doc.metadata["chunk_id"] = i
            doc.metadata["platform"] = platform
            doc.metadata["ingestion_time"] = current_time

            # Add total chunks info to help with retrieval context
            doc.metadata["total_chunks"] = len(chunked_documents)

            # Ensure source and source_id are set for consistent retrieval
            doc.metadata["source"] = platform
            doc.metadata["source_id"] = video_id

        logger.info(f"Adding {len(chunked_documents)} chunks to vector store")

        # Add to vector store
        doc_ids = self.vector_store.add_documents(chunked_documents)
        logger.info(f"Successfully added to vector store, got {len(doc_ids)} document IDs")

        return doc_ids

    def process_pdf(
            self, file_path: str, custom_metadata: Optional[Dict[str, str]] = None, force_refresh: bool = False
    ) -> List[str]:
        """
        Process a PDF file with enhanced metadata tracking.

        Args:
            file_path: Path to the PDF file
            custom_metadata: Optional custom metadata
            force_refresh: Force reprocessing even if PDF was previously ingested

        Returns:
            List of document IDs
        """
        base_name = os.path.basename(file_path)
        logger.info(f"Processing PDF: {base_name}")

        # Check if we already have this PDF if not forcing refresh
        if not force_refresh:
            existing_docs = self.vector_store.search_by_metadata({
                "source": "pdf",
                "source_id": base_name,
            })

            if existing_docs:
                logger.info(f"PDF already ingested with {len(existing_docs)} chunks, returning existing document IDs")
                return [doc.metadata.get("id") for doc in existing_docs]

        # Get documents from PDF loader
        documents = self.pdf_loader.process_pdf(
            file_path=file_path,
            custom_metadata=custom_metadata,
        )

        logger.info(f"PDF processing completed, got {len(documents)} documents")

        # Add additional metadata
        current_time = time.time()
        for i, doc in enumerate(documents):
            # Add unique ID if not already present
            if "id" not in doc.metadata:
                doc.metadata["id"] = f"pdf-{base_name}-{i}"

            doc.metadata["chunk_id"] = i
            doc.metadata["ingestion_time"] = current_time
            doc.metadata["total_chunks"] = len(documents)

            # Ensure source and source_id are set for consistent retrieval
            doc.metadata["source"] = "pdf"
            doc.metadata["source_id"] = base_name

        logger.info(f"Adding {len(documents)} documents to vector store")

        # Add to vector store
        doc_ids = self.vector_store.add_documents(documents)
        logger.info(f"Successfully added to vector store, got {len(doc_ids)} document IDs")

        return doc_ids

    def process_text(
            self, request: ManualIngestRequest, force_refresh: bool = False
    ) -> List[str]:
        """
        Process manually entered text with enhanced metadata tracking.

        Args:
            request: Manual ingest request
            force_refresh: Force reprocessing even if identical content exists

        Returns:
            List of document IDs
        """
        logger.info(f"Processing manual text input: {request.metadata.title}")

        # Generate a unique identifier for this text entry
        doc_id = str(uuid.uuid4())

        # Create document
        document = Document(
            page_content=request.content,
            metadata=request.metadata.dict(),
        )

        # Split into chunks
        chunked_documents = self.text_splitter.split_documents([document])
        logger.info(f"Split into {len(chunked_documents)} chunks")

        # Assign IDs and metadata to chunks
        current_time = time.time()
        for i, doc in enumerate(chunked_documents):
            doc.metadata["id"] = f"manual-{doc_id}-{i}"
            doc.metadata["chunk_id"] = i
            doc.metadata["ingestion_time"] = current_time
            doc.metadata["total_chunks"] = len(chunked_documents)

            # Ensure source and source_id are set for consistent retrieval
            doc.metadata["source"] = "manual"
            doc.metadata["source_id"] = doc_id

        logger.info(f"Adding {len(chunked_documents)} chunks to vector store")

        # Add to vector store
        doc_ids = self.vector_store.add_documents(chunked_documents)
        logger.info(f"Successfully added to vector store, got {len(doc_ids)} document IDs")

        return doc_ids

    def batch_process_videos(
            self, urls: List[str], custom_metadata: Optional[List[Dict[str, str]]] = None,
            force_refresh: bool = False
    ) -> Dict[str, Union[List[str], Dict[str, str]]]:
        """
        Process multiple videos in batch with enhanced logging and error handling.

        Args:
            urls: List of video URLs
            custom_metadata: Optional list of custom metadata (same length as urls)
            force_refresh: Force reprocessing even if videos were previously ingested

        Returns:
            Dictionary mapping URLs to lists of document IDs or error messages
        """
        # Validate inputs
        if custom_metadata and len(custom_metadata) != len(urls):
            logger.warning("custom_metadata length doesn't match urls length")
            raise ValueError("If provided, custom_metadata must have the same length as urls")

        result = {}
        logger.info(f"Batch processing {len(urls)} videos")

        # Process each URL
        for i, url in enumerate(urls):
            metadata = custom_metadata[i] if custom_metadata else None
            try:
                logger.info(f"Processing video {i + 1}/{len(urls)}: {url}")
                doc_ids = self.process_video(url, metadata, force_refresh)
                result[url] = doc_ids
                logger.info(f"Successfully processed video {i + 1}: {url}")
            except Exception as e:
                logger.error(f"Error processing video {i + 1}: {url} - {str(e)}")
                result[url] = {"error": str(e)}

        logger.info(
            f"Batch processing completed, successfully processed: {sum(1 for v in result.values() if isinstance(v, list))}/{len(urls)}")
        return result

    def delete_documents(self, ids: List[str]) -> None:
        """
        Delete documents from the vector store with enhanced logging.

        Args:
            ids: List of document IDs to delete
        """
        logger.info(f"Deleting {len(ids)} documents from vector store")
        self.vector_store.delete_by_ids(ids)
        logger.info("Documents deleted successfully")

    def verify_ingestion(self, url: str) -> Dict[str, any]:
        """
        Verify that a video was properly ingested and retrievable.

        Args:
            url: Video URL to verify

        Returns:
            Dictionary with verification results
        """
        platform = self.video_transcriber.detect_platform(url)
        video_id = self.video_transcriber.extract_video_id(url)

        logger.info(f"Verifying ingestion for {platform} video: {video_id}")

        # Search for documents from this video
        documents = self.vector_store.search_by_metadata({
            "source": platform,
            "source_id": video_id,
        })

        # Check if we found any documents
        found = len(documents) > 0

        # Get embedding statistics if available
        embedding_stats = {}
        if documents:
            embedding_sample = self.vector_store.get_embedding(documents[0].metadata.get("id"))
            if embedding_sample:
                embedding_stats = {
                    "dimensions": len(embedding_sample),
                    "min_value": min(embedding_sample),
                    "max_value": max(embedding_sample),
                    "avg_value": sum(embedding_sample) / len(embedding_sample),
                }

        result = {
            "url": url,
            "platform": platform,
            "video_id": video_id,
            "documents_found": len(documents),
            "verification_success": found,
            "embedding_stats": embedding_stats
        }

        logger.info(f"Verification result: {'Success' if found else 'Failed'}, found {len(documents)} documents")
        return result

    def repair_vector_store(self) -> Dict[str, any]:
        """
        Attempt to repair vector store indices and consistency.

        Returns:
            Dictionary with repair results
        """
        logger.info("Starting vector store repair process")

        # Get statistics before repair
        before_stats = self.vector_store.get_stats()

        # Attempt to repair collections (implementation depends on vector store)
        repair_result = self.vector_store.repair_indices()

        # Get statistics after repair
        after_stats = self.vector_store.get_stats()

        result = {
            "before": before_stats,
            "after": after_stats,
            "repair_result": repair_result
        }

        logger.info(f"Vector store repair completed: {repair_result}")
        return result