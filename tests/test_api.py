"""
Tests for the FastAPI implementation of the Automotive Specs RAG system.
"""

import json
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient


def test_health_endpoint(test_client: TestClient):
    """Test the health check endpoint."""
    response = test_client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_api_key_authentication(test_client: TestClient):
    """Test that API key authentication works correctly."""
    # Test with invalid API key
    response = test_client.get("/ingest/status", headers={"x-token": "invalid-key"})
    assert response.status_code == 401
    
    # Test with valid API key
    from src.config.settings import settings
    response = test_client.get("/ingest/status", headers={"x-token": settings.api_key})
    assert response.status_code == 200


def test_query_endpoint(test_client: TestClient, sample_documents):
    """Test the query endpoint."""
    # Mock the retriever and LLM dependencies
    with patch("src.api.dependencies.get_retriever") as mock_get_retriever, \
         patch("src.api.dependencies.get_local_llm") as mock_get_llm:
        
        # Configure mocks
        mock_retriever = MagicMock()
        mock_llm = MagicMock()
        
        # Setup retriever to return sample documents
        from langchain_core.documents import Document
        docs = [
            (Document(page_content=doc["content"], metadata=doc["metadata"]), 0.95 - i*0.05) 
            for i, doc in enumerate(sample_documents)
        ]
        mock_retriever.retrieve.return_value = (docs, 0.05)
        
        # Setup LLM to return a mock answer
        mock_llm.answer_query_with_sources.return_value = (
            "The Toyota Camry has 208 horsepower.", 
            [{"id": doc["id"], "title": doc["metadata"]["title"]} for doc in sample_documents]
        )
        
        # Assign mocks to the dependency functions
        mock_get_retriever.return_value = mock_retriever
        mock_get_llm.return_value = mock_llm
        
        # Make request to query endpoint
        response = test_client.post(
            "/query/",
            headers={"x-token": settings.api_key},
            json={
                "query": "What is the horsepower of the Toyota Camry?",
                "metadata_filter": {"manufacturer": "Toyota"},
                "top_k": 3
            }
        )
        
        # Verify response
        assert response.status_code == 200
        data = response.json()
        assert data["query"] == "What is the horsepower of the Toyota Camry?"
        assert data["answer"] == "The Toyota Camry has 208 horsepower."
        assert len(data["documents"]) == 3
        assert "execution_time" in data
        
        # Verify that the retriever was called with the correct arguments
        mock_retriever.retrieve.assert_called_once_with(
            query="What is the horsepower of the Toyota Camry?",
            metadata_filter={"manufacturer": "Toyota"},
            rerank=True
        )


def test_ingest_youtube_endpoint(test_client: TestClient):
    """Test the YouTube ingestion endpoint."""
    # Mock the document processor
    with patch("src.api.routers.ingest.DocumentProcessor") as mock_processor_class:
        # Configure mock
        mock_processor = MagicMock()
        mock_processor.process_youtube_video.return_value = ["doc1", "doc2"]
        mock_processor_class.return_value = mock_processor
        
        # Make request to ingest endpoint
        response = test_client.post(
            "/ingest/youtube",
            headers={"x-token": settings.api_key},
            json={
                "url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
                "metadata": {
                    "manufacturer": "Tesla",
                    "model": "Model S"
                }
            }
        )
        
        # Verify response
        assert response.status_code == 200
        data = response.json()
        assert data["document_count"] == 2
        assert data["document_ids"] == ["doc1", "doc2"]
        
        # Verify that the processor was called with the correct arguments
        mock_processor.process_youtube_video.assert_called_once_with(
            url="https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            custom_metadata={"manufacturer": "Tesla", "model": "Model S"},
            force_whisper=False
        )


def test_ingest_pdf_endpoint(test_client: TestClient):
    """Test the PDF ingestion endpoint."""
    # Mock the document processor
    with patch("src.api.routers.ingest.DocumentProcessor") as mock_processor_class:
        # Configure mock
        mock_processor = MagicMock()
        mock_processor.process_pdf.return_value = ["doc1", "doc2", "doc3"]
        mock_processor_class.return_value = mock_processor
        
        # Create a simple test PDF file
        pdf_content = b"%PDF-1.5\nTest PDF"
        
        # Prepare metadata
        metadata = {
            "manufacturer": "BMW",
            "model": "X5",
            "year": 2023
        }
        
        # Make request to ingest endpoint
        response = test_client.post(
            "/ingest/pdf",
            headers={"x-token": settings.api_key},
            files={
                "file": ("test.pdf", pdf_content, "application/pdf")
            },
            data={
                "metadata": json.dumps(metadata),
                "use_ocr": "true",
                "extract_tables": "true"
            }
        )
        
        # Verify response
        assert response.status_code == 200
        data = response.json()
        assert data["document_count"] == 3
        assert data["document_ids"] == ["doc1", "doc2", "doc3"]
        
        # Verify that the processor was called
        mock_processor.process_pdf.assert_called_once()
        # Note: We can't easily verify the exact arguments because the file is saved to disk


def test_manufacturer_endpoint(test_client: TestClient):
    """Test the manufacturers endpoint."""
    response = test_client.get(
        "/query/manufacturers",
        headers={"x-token": settings.api_key}
    )
    
    assert response.status_code == 200
    manufacturers = response.json()
    assert isinstance(manufacturers, list)
    assert "Toyota" in manufacturers
    assert "Honda" in manufacturers


def test_models_endpoint(test_client: TestClient):
    """Test the models endpoint."""
    response = test_client.get(
        "/query/models?manufacturer=Toyota",
        headers={"x-token": settings.api_key}
    )
    
    assert response.status_code == 200
    models = response.json()
    assert isinstance(models, list)
    assert "Camry" in models
    assert "Corolla" in models


def test_llm_info_endpoint(test_client: TestClient):
    """Test the LLM info endpoint."""
    # Mock the LLM dependency
    with patch("src.api.dependencies.get_local_llm") as mock_get_llm:
        # Configure mock
        mock_llm = MagicMock()
        mock_llm.model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
        mock_llm.device = "cuda:0"
        mock_llm.temperature = 0.1
        mock_llm.max_tokens = 512
        mock_llm.use_4bit = True
        mock_llm.use_8bit = False
        mock_llm.torch_dtype = "torch.float16"
        
        # Assign mock to the dependency function
        mock_get_llm.return_value = mock_llm
        
        # Make request to LLM info endpoint
        response = test_client.get(
            "/query/llm-info",
            headers={"x-token": settings.api_key}
        )
        
        # Verify response
        assert response.status_code == 200
        data = response.json()
        assert data["model_name"] == "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
        assert data["device"] == "cuda:0"
        assert data["temperature"] == 0.1
        assert data["max_tokens"] == 512
        assert data["quantization"] == "4-bit"
        assert data["torch_dtype"] == "torch.float16"