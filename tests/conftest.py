"""
Pytest configuration for the Automotive Specs RAG system.

This module provides fixtures and configurations used by the test suite.
"""

import os
import sys
import pytest
import tempfile
from typing import Dict, List, Generator, Any

import torch
from fastapi.testclient import TestClient
from qdrant_client import QdrantClient
from langchain_community.embeddings import HuggingFaceEmbeddings

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.api.main import app
from src.core.vectorstore import QdrantStore
from src.core.colbert_reranker import ColBERTReranker
from src.core.llm import LocalLLM


@pytest.fixture
def test_client() -> Generator[TestClient, None, None]:
    """
    Create a FastAPI test client.
    
    Returns:
        TestClient: FastAPI test client
    """
    client = TestClient(app)
    yield client


@pytest.fixture
def temp_data_dir() -> Generator[str, None, None]:
    """
    Create a temporary directory for test data.
    
    Returns:
        str: Path to temporary directory
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


@pytest.fixture
def embedding_function() -> HuggingFaceEmbeddings:
    """
    Create an embedding function for testing.
    
    Returns:
        HuggingFaceEmbeddings: Embedding function
    """
    # Use a small, fast model for testing
    return HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={"device": "cpu"},  # Use CPU for tests
        encode_kwargs={"normalize_embeddings": True},
    )


@pytest.fixture
def mock_qdrant_client(temp_data_dir: str) -> QdrantClient:
    """
    Create a Qdrant client with an in-memory database for testing.
    
    Args:
        temp_data_dir: Temporary directory for storage
        
    Returns:
        QdrantClient: In-memory Qdrant client
    """
    return QdrantClient(location=":memory:")


@pytest.fixture
def vector_store(mock_qdrant_client: QdrantClient, embedding_function: HuggingFaceEmbeddings) -> QdrantStore:
    """
    Create a vector store for testing.
    
    Args:
        mock_qdrant_client: Qdrant client
        embedding_function: Embedding function
        
    Returns:
        QdrantStore: Vector store
    """
    return QdrantStore(
        client=mock_qdrant_client,
        collection_name="test_collection",
        embedding_function=embedding_function,
    )


@pytest.fixture
def colbert_reranker() -> ColBERTReranker:
    """
    Create a ColBERT reranker for testing.
    
    Returns:
        ColBERTReranker: Reranker instance
    """
    # Mock the ColBERT reranker if needed
    if os.environ.get("CI") or not torch.cuda.is_available():
        # Create a lightweight version for CI or when GPU is not available
        class MockColBERTReranker:
            def rerank(self, query, documents, top_k=None):
                # Just return documents with mock scores
                return [(doc, 0.9 - i * 0.1) for i, doc in enumerate(documents)]
                
        return MockColBERTReranker()
    else:
        # Use actual reranker with small model
        return ColBERTReranker(
            model_name="colbert-ir/colbertv2.0",
            device="cpu",  # Use CPU for tests
            max_query_length=32,
            max_doc_length=128,  # Smaller for tests
        )


@pytest.fixture
def sample_documents() -> List[Dict[str, Any]]:
    """
    Provide sample documents for testing.
    
    Returns:
        List[Dict]: Sample documents
    """
    return [
        {
            "id": "doc1",
            "content": "The 2023 Toyota Camry Hybrid combines a 2.5-liter four-cylinder engine with an electric motor for a total system output of 208 horsepower.",
            "metadata": {
                "source": "manual",
                "title": "Toyota Camry Specifications",
                "manufacturer": "Toyota",
                "model": "Camry",
                "year": 2023,
                "category": "sedan",
                "engine_type": "hybrid",
            }
        },
        {
            "id": "doc2",
            "content": "The 2022 Honda Civic has a 2.0-liter naturally aspirated four-cylinder engine that produces 158 horsepower and 138 lb-ft of torque.",
            "metadata": {
                "source": "manual",
                "title": "Honda Civic Specifications",
                "manufacturer": "Honda",
                "model": "Civic",
                "year": 2022,
                "category": "sedan",
                "engine_type": "gasoline",
            }
        },
        {
            "id": "doc3",
            "content": "The Tesla Model 3 Long Range features dual electric motors with a combined output of approximately 346 horsepower.",
            "metadata": {
                "source": "manual",
                "title": "Tesla Model 3 Specifications",
                "manufacturer": "Tesla",
                "model": "Model 3",
                "year": 2022,
                "category": "sedan",
                "engine_type": "electric",
            }
        }
    ]


@pytest.fixture
def mock_llm() -> LocalLLM:
    """
    Create a mock LLM for testing.
    
    Returns:
        MockLLM: Mock LLM instance
    """
    class MockLLM:
        def answer_query(self, query, documents, metadata_filter=None):
            return f"Mock answer for: {query}"
            
        def answer_query_with_sources(self, query, documents, metadata_filter=None):
            answer = f"Mock answer for: {query}"
            sources = [{"id": doc.metadata.get("id", ""), "title": doc.metadata.get("title", "")} for doc, _ in documents]
            return answer, sources
            
    return MockLLM()


@pytest.fixture(scope="session")
def skip_if_no_gpu():
    """
    Skip a test if no GPU is available.
    """
    if not torch.cuda.is_available():
        pytest.skip("Test requires GPU")