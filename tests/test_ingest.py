"""
Tests for the document ingestion components.
"""

import os
import pytest
from unittest.mock import patch, MagicMock, mock_open
import torch
from langchain_core.documents import Document

from src.core.document_processor import DocumentProcessor
from src.core.base_video_transcriber import YouTubeTranscriber, BilibiliTranscriber, create_transcriber_for_url
from src.core.pdf_loader import PDFLoader


def test_document_processor_initialization(vector_store):
    """Test document processor initialization."""
    processor = DocumentProcessor(vector_store=vector_store)
    
    assert processor.vector_store == vector_store
    assert processor.chunk_size == 1000
    assert processor.chunk_overlap == 200
    assert isinstance(processor.youtube_transcriber, YouTubeTranscriber)
    assert isinstance(processor.bilibili_transcriber, BilibiliTranscriber)
    assert isinstance(processor.pdf_loader, PDFLoader)


def test_process_youtube_video(vector_store):
    """Test processing a YouTube video."""
    processor = DocumentProcessor(vector_store=vector_store)
    
    # Mock the YouTube transcriber
    with patch.object(processor.youtube_transcriber, 'process_video') as mock_process_video, \
         patch.object(processor.youtube_transcriber, 'extract_video_id', return_value="dQw4w9WgXcQ"), \
         patch.object(processor.text_splitter, 'split_documents', return_value=[
             Document(page_content="Part 1 of transcript", metadata={}),
             Document(page_content="Part 2 of transcript", metadata={})
         ]), \
         patch.object(processor.vector_store, 'add_documents', return_value=["doc1", "doc2"]):
        
        # Mock the transcriber to return a document
        mock_process_video.return_value = [
            Document(page_content="Test transcript", metadata={"title": "Test Video"})
        ]
        
        # Process the video
        result = processor.process_youtube_video(
            url="https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            custom_metadata={"manufacturer": "Toyota"}
        )
        
        # Verify the result
        assert result == ["doc1", "doc2"]
        
        # Verify that the transcriber was called with the correct arguments
        mock_process_video.assert_called_once_with(
            url="https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            custom_metadata={"manufacturer": "Toyota"},
            force_whisper=False
        )


def test_process_bilibili_video(vector_store):
    """Test processing a Bilibili video."""
    processor = DocumentProcessor(vector_store=vector_store)
    
    # Mock the Bilibili transcriber
    with patch.object(processor.bilibili_transcriber, 'process_video') as mock_process_video, \
         patch.object(processor.bilibili_transcriber, 'extract_video_id', return_value="BV1xx411c7mD"), \
         patch.object(processor.text_splitter, 'split_documents', return_value=[
             Document(page_content="Part 1 of transcript", metadata={}),
             Document(page_content="Part 2 of transcript", metadata={})
         ]), \
         patch.object(processor.vector_store, 'add_documents', return_value=["doc1", "doc2"]):
        
        # Mock the transcriber to return a document
        mock_process_video.return_value = [
            Document(page_content="Test transcript", metadata={"title": "Test Video"})
        ]
        
        # Process the video
        result = processor.process_bilibili_video(
            url="https://www.bilibili.com/video/BV1xx411c7mD",
            custom_metadata={"manufacturer": "Toyota"}
        )
        
        # Verify the result
        assert result == ["doc1", "doc2"]
        
        # Verify that the transcriber was called with the correct arguments
        mock_process_video.assert_called_once_with(
            url="https://www.bilibili.com/video/BV1xx411c7mD",
            custom_metadata={"manufacturer": "Toyota"}
        )


def test_process_pdf(vector_store):
    """Test processing a PDF file."""
    processor = DocumentProcessor(vector_store=vector_store)
    
    # Mock the PDF loader
    with patch.object(processor.pdf_loader, 'process_pdf') as mock_process_pdf, \
         patch.object(processor.vector_store, 'add_documents', return_value=["doc1", "doc2", "doc3"]):
        
        # Mock the PDF loader to return documents
        mock_process_pdf.return_value = [
            Document(page_content="Page 1 content", metadata={}),
            Document(page_content="Page 2 content", metadata={}),
            Document(page_content="Page 3 content", metadata={})
        ]
        
        # Process the PDF
        result = processor.process_pdf(
            file_path="test.pdf",
            custom_metadata={"manufacturer": "Honda", "model": "Civic"}
        )
        
        # Verify the result
        assert result == ["doc1", "doc2", "doc3"]
        
        # Verify that the PDF loader was called with the correct arguments
        mock_process_pdf.assert_called_once_with(
            file_path="test.pdf",
            custom_metadata={"manufacturer": "Honda", "model": "Civic"}
        )


def test_process_text(vector_store):
    """Test processing manual text entry."""
    processor = DocumentProcessor(vector_store=vector_store)
    
    # Create a manual ingest request
    from src.models.schema import DocumentMetadata, DocumentSource, ManualIngestRequest
    
    request = ManualIngestRequest(
        content="The 2023 Toyota Camry has 208 horsepower.",
        metadata=DocumentMetadata(
            source=DocumentSource.MANUAL,
            source_id="manual-test",
            title="Toyota Camry Specs",
            manufacturer="Toyota",
            model="Camry",
            year=2023
        )
    )
    
    # Mock the text splitter and vector store
    with patch.object(processor.text_splitter, 'split_documents', return_value=[
             Document(page_content="The 2023 Toyota Camry has 208 horsepower.", metadata={})
         ]), \
         patch.object(processor.vector_store, 'add_documents', return_value=["doc1"]):
        
        # Process the text
        result = processor.process_text(request)
        
        # Verify the result
        assert result == ["doc1"]


def test_batch_process_videos(vector_store):
    """Test batch processing of videos."""
    processor = DocumentProcessor(vector_store=vector_store)
    
    # Mock the processing methods
    with patch.object(processor, 'process_youtube_video') as mock_process_youtube, \
         patch.object(processor, 'process_bilibili_video') as mock_process_bilibili:
        
        # Mock return values
        mock_process_youtube.side_effect = [["doc1", "doc2"], ["doc3", "doc4"]]
        mock_process_bilibili.return_value = ["doc5", "doc6"]
        
        # Batch process YouTube videos
        youtube_result = processor.batch_process_videos(
            urls=["https://www.youtube.com/watch?v=video1", "https://www.youtube.com/watch?v=video2"],
            platform="youtube",
            custom_metadata=[{"manufacturer": "Toyota"}, {"manufacturer": "Honda"}]
        )
        
        # Verify YouTube results
        assert "https://www.youtube.com/watch?v=video1" in youtube_result
        assert youtube_result["https://www.youtube.com/watch?v=video1"] == ["doc1", "doc2"]
        assert "https://www.youtube.com/watch?v=video2" in youtube_result
        assert youtube_result["https://www.youtube.com/watch?v=video2"] == ["doc3", "doc4"]
        
        # Batch process Bilibili videos
        bilibili_result = processor.batch_process_videos(
            urls=["https://www.bilibili.com/video/BV123"],
            platform="bilibili",
            custom_metadata=[{"manufacturer": "Tesla"}]
        )
        
        # Verify Bilibili results
        assert "https://www.bilibili.com/video/BV123" in bilibili_result
        assert bilibili_result["https://www.bilibili.com/video/BV123"] == ["doc5", "doc6"]
        
        # Verify that the processing methods were called with the correct arguments
        mock_process_youtube.assert_any_call(
            url="https://www.youtube.com/watch?v=video1", 
            metadata={"manufacturer": "Toyota"}
        )
        mock_process_youtube.assert_any_call(
            url="https://www.youtube.com/watch?v=video2", 
            metadata={"manufacturer": "Honda"}
        )
        mock_process_bilibili.assert_called_once_with(
            url="https://www.bilibili.com/video/BV123", 
            metadata={"manufacturer": "Tesla"}
        )


def test_delete_documents(vector_store):
    """Test deleting documents."""
    processor = DocumentProcessor(vector_store=vector_store)
    
    # Mock the vector store delete method
    with patch.object(processor.vector_store, 'delete_by_ids') as mock_delete:
        # Delete documents
        processor.delete_documents(["doc1", "doc2"])
        
        # Verify that the delete method was called with the correct arguments
        mock_delete.assert_called_once_with(["doc1", "doc2"])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No GPU available")
def test_youtube_transcriber_with_whisper():
    """Test YouTube transcriber with Whisper."""
    # Create a transcriber with whisper enabled
    transcriber = YouTubeTranscriber(
        whisper_model_size="tiny",  # Use tiny model for faster tests
        device="cuda:0",
        use_youtube_captions=False,
        use_whisper_as_fallback=True
    )
    
    # Mock the whisper model and other methods
    with patch.object(transcriber, '_load_whisper_model') as mock_load_model, \
         patch.object(transcriber, 'extract_audio', return_value="test_audio.mp3"), \
         patch.object(transcriber, 'get_video_metadata', return_value={
             "title": "Test Toyota Camry Review",
             "author": "Test Channel",
             "published_date": None,
             "video_id": "test123",
             "url": "https://www.youtube.com/watch?v=test123",
             "description": "Review of the Toyota Camry"
         }), \
         patch.object(transcriber, 'transcribe_with_whisper', return_value="This is a test transcript for the Toyota Camry."):
        
        # Process the video
        documents = transcriber.process_video(
            url="https://www.youtube.com/watch?v=test123",
            force_whisper=True
        )
        
        # Verify the result
        assert len(documents) == 1
        assert documents[0].page_content == "This is a test transcript for the Toyota Camry."
        assert documents[0].metadata["source"] == "youtube"
        assert documents[0].metadata["title"] == "Test Toyota Camry Review"
        assert documents[0].metadata["manufacturer"] == "Toyota"
        assert documents[0].metadata["custom_metadata"]["transcription_method"] == "whisper"


def test_pdf_loader_with_ocr():
    """Test PDF loader with OCR."""
    # Create a PDF loader with OCR enabled
    pdf_loader = PDFLoader(
        use_ocr=True,
        device="cpu"  # Use CPU for tests
    )
    
    # Mock initialization
    with patch.object(pdf_loader, '_initialize_ocr') as mock_init_ocr:
        # Ensure initialization was called
        assert mock_init_ocr.called
    
    # Mock loading and OCR methods
    with patch('langchain_community.document_loaders.PyPDFLoader') as mock_pypdf, \
         patch.object(pdf_loader, '_apply_ocr') as mock_apply_ocr:
        
        # Setup mocks
        mock_loader = MagicMock()
        mock_loader.load.return_value = [
            Document(page_content="", metadata={"page": 1})  # Empty content to trigger OCR
        ]
        mock_pypdf.return_value = mock_loader
        
        mock_apply_ocr.return_value = [
            Document(page_content="OCR extracted text about Toyota vehicles.", metadata={"page": 1, "ocr_applied": True})
        ]
        
        # Load the PDF
        documents = pdf_loader.load_pdf("test_scan.pdf", use_ocr=True)
        
        # Verify that OCR was applied
        assert mock_apply_ocr.called
        assert len(documents) == 1
        assert documents[0].metadata["ocr_applied"] is True
        assert "Toyota" in documents[0].page_content