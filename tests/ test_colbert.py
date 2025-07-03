"""
Tests for the ColBERT reranker component.
"""

import pytest
import torch
from unittest.mock import patch
from langchain_core.documents import Document

from src.core.query.llm.rerankers import ColBERTReranker


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda:0", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="No GPU available"))])
def test_colbert_reranker_initialization(device):
    """Test ColBERT reranker initialization with different devices."""
    # Use smaller model parameters for tests
    reranker = ColBERTReranker(
        model_name="colbert-ir/colbertv2.0",
        device=device,
        max_query_length=32,
        max_doc_length=128,
        batch_size=2,
        use_fp16=False,  # Disable FP16 for tests
    )
    
    assert reranker.device == device
    assert reranker.model_name == "colbert-ir/colbertv2.0"
    assert reranker.max_query_length == 32
    assert reranker.max_doc_length == 128
    assert reranker.batch_size == 2
    assert reranker.use_fp16 is False


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No GPU available")
def test_encode_query():
    """Test query encoding with ColBERT."""
    reranker = ColBERTReranker(
        model_name="colbert-ir/colbertv2.0",
        device="cuda:0",
        max_query_length=32,
    )
    
    # Mock the ColBERT query method to avoid actual computation
    with patch.object(reranker.colbert, 'query', return_value=torch.rand(1, 32, 128)) as mock_query:
        # Test query encoding
        query = "What is the horsepower of the Toyota Camry?"
        query_embeddings = reranker._encode_query(query)
        
        # Verify the result shape
        assert query_embeddings.shape == (1, 32, 128)
        
        # Verify that the ColBERT query method was called
        mock_query.assert_called_once()


def test_encode_documents_batched():
    """Test document encoding with batching."""
    # Use CPU for consistent testing
    reranker = ColBERTReranker(
        model_name="colbert-ir/colbertv2.0",
        device="cpu",
        max_doc_length=128,
        batch_size=2,
    )
    
    # Create mock documents
    documents = [
        Document(page_content="The Toyota Camry has 208 horsepower.", metadata={"id": "doc1"}),
        Document(page_content="The Honda Civic has 158 horsepower.", metadata={"id": "doc2"}),
        Document(page_content="The Tesla Model 3 has 346 horsepower.", metadata={"id": "doc3"}),
    ]
    
    # Mock the ColBERT doc method to avoid actual computation
    with patch.object(reranker.colbert, 'doc', return_value=torch.rand(2, 128, 128)) as mock_doc:
        # Test document encoding
        doc_embeddings = reranker._encode_documents_batched(documents)
        
        # Verify that we got embeddings for each document
        assert len(doc_embeddings) == 3
        
        # Verify that the ColBERT doc method was called twice (for 3 docs with batch size 2)
        assert mock_doc.call_count == 2


def test_score_documents_batched():
    """Test document scoring with batching."""
    # Use CPU for consistent testing
    reranker = ColBERTReranker(
        model_name="colbert-ir/colbertv2.0",
        device="cpu",
        batch_size=2,
    )
    
    # Create mock query and document embeddings
    query_embeddings = torch.rand(1, 32, 128)
    doc_embeddings_list = [
        torch.rand(1, 128, 128),
        torch.rand(1, 128, 128),
        torch.rand(1, 128, 128),
    ]
    
    # Test document scoring
    with patch('torch.bmm', return_value=torch.rand(2, 32, 128)) as mock_bmm:
        # Override max to return known values for deterministic testing
        with patch('torch.max', return_value=(torch.ones(2, 32), torch.zeros(2, 32))):
            scores = reranker._score_documents_batched(query_embeddings, doc_embeddings_list)
            
            # Verify that we got scores for each document
            assert len(scores) == 3
            
            # Each score should be 32.0 (sum of 32 ones)
            assert all(score == 32.0 for score in scores)
            
            # Verify that torch.bmm was called twice (for 3 docs with batch size 2)
            assert mock_bmm.call_count == 2


def test_rerank_empty_documents():
    """Test reranking with empty document list."""
    reranker = ColBERTReranker(
        model_name="colbert-ir/colbertv2.0",
        device="cpu",
    )
    
    # Test reranking with empty document list
    query = "What is the horsepower of the Toyota Camry?"
    result = reranker.rerank(query, [])
    
    # Verify that the result is an empty list
    assert result == []


def test_rerank_documents():
    """Test document reranking."""
    # Use CPU for consistent testing
    reranker = ColBERTReranker(
        model_name="colbert-ir/colbertv2.0",
        device="cpu",
    )
    
    # Create mock documents
    documents = [
        Document(page_content="The Toyota Camry has 208 horsepower.", metadata={"id": "doc1"}),
        Document(page_content="The Honda Civic has 158 horsepower.", metadata={"id": "doc2"}),
        Document(page_content="The Tesla Model 3 has 346 horsepower.", metadata={"id": "doc3"}),
    ]
    
    # Mock the internal methods
    with patch.object(reranker, '_encode_query', return_value=torch.rand(1, 32, 128)), \
         patch.object(reranker, '_encode_documents_batched', return_value=[torch.rand(1, 128, 128) for _ in range(3)]), \
         patch.object(reranker, '_score_documents_batched', return_value=[0.9, 0.7, 0.8]):
        
        # Test reranking
        result = reranker.rerank(
            query="What is the horsepower of the Toyota Camry?",
            documents=documents,
            top_k=2
        )
        
        # Verify the result format
        assert len(result) == 2
        assert isinstance(result[0], tuple)
        assert isinstance(result[0][0], Document)
        assert isinstance(result[0][1], float)
        
        # Verify sorting by score (highest first)
        assert result[0][1] == 0.9  # doc1 with highest score
        assert result[1][1] == 0.8  # doc3 with second highest score


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No GPU available")
def test_rerank_with_explanations():
    """Test reranking with explanations."""
    reranker = ColBERTReranker(
        model_name="colbert-ir/colbertv2.0",
        device="cuda:0",
    )
    
    # Create mock documents
    documents = [
        Document(page_content="The Toyota Camry has 208 horsepower.", metadata={"id": "doc1"}),
        Document(page_content="The Honda Civic has 158 horsepower.", metadata={"id": "doc2"}),
    ]
    
    # Mock the tokenizer and similarity matrix for deterministic testing
    with patch.object(reranker.tokenizer, 'convert_ids_to_tokens', return_value=["car", "has", "horsepower"]), \
         patch('torch.matmul', return_value=torch.ones(1, 32, 128)), \
         patch('torch.max', return_value=(torch.ones(32), torch.zeros(32))):
        
        # Test reranking with explanations
        result = reranker.rerank_with_explanations(
            query="What is the horsepower of Toyota cars?",
            documents=documents,
            top_k=1
        )
        
        # Verify the result format
        assert len(result) == 1
        assert "document" in result[0]
        assert "score" in result[0]
        assert "explanations" in result[0]
        
        # Verify explanations format
        explanations = result[0]["explanations"]
        assert len(explanations) <= 5  # Top 5 explanations
        if explanations:
            assert "query_token" in explanations[0]
            assert "doc_token" in explanations[0]
            assert "similarity" in explanations[0]