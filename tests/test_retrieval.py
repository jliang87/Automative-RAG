"""
Tests for the retrieval components.
"""

import pytest
from unittest.mock import patch, MagicMock
import torch
from langchain_core.documents import Document

from src.core.vectorstore import QdrantStore
from src.core.retriever import HybridRetriever


def test_vector_store_initialization(mock_qdrant_client, embedding_function):
    """Test vector store initialization."""
    vector_store = QdrantStore(
        client=mock_qdrant_client,
        collection_name="test_collection",
        embedding_function=embedding_function,
    )
    
    assert vector_store.client == mock_qdrant_client
    assert vector_store.collection_name == "test_collection"
    assert vector_store.embedding_function == embedding_function


def test_vector_store_collection_creation(mock_qdrant_client, embedding_function):
    """Test vector store collection creation."""
    # Mock the collection listing
    mock_collections = MagicMock()
    mock_collections.collections = []
    mock_qdrant_client.get_collections.return_value = mock_collections
    
    # Mock the embedding dimension
    embedding_function.embed_query = MagicMock(return_value=[0.1] * 384)
    
    with patch.object(mock_qdrant_client, 'create_collection') as mock_create_collection, \
         patch.object(mock_qdrant_client, 'create_payload_index') as mock_create_index:
        
        # Initialize the vector store
        vector_store = QdrantStore(
            client=mock_qdrant_client,
            collection_name="test_collection",
            embedding_function=embedding_function,
        )
        
        # Verify that the collection was created
        mock_create_collection.assert_called_once()
        
        # Verify that payload indexes were created for metadata fields
        assert mock_create_index.call_count > 0


def test_add_documents(vector_store):
    """Test adding documents to the vector store."""
    # Create test documents
    documents = [
        Document(page_content="The Toyota Camry has 208 horsepower.", metadata={"manufacturer": "Toyota"}),
        Document(page_content="The Honda Civic has 158 horsepower.", metadata={"manufacturer": "Honda"}),
    ]
    
    # Mock the Langchain Qdrant wrapper
    with patch.object(vector_store.langchain_qdrant, 'add_documents', return_value=["doc1", "doc2"]) as mock_add:
        # Add documents
        result = vector_store.add_documents(documents)
        
        # Verify the result
        assert result == ["doc1", "doc2"]
        
        # Verify that the add_documents method was called with the correct arguments
        mock_add.assert_called_once_with(documents)


def test_similarity_search_with_filter(vector_store):
    """Test similarity search with metadata filtering."""
    # Mock the filter building and Langchain Qdrant wrapper
    with patch.object(vector_store, '_build_filter') as mock_build_filter, \
         patch.object(vector_store.langchain_qdrant, 'similarity_search_with_score') as mock_search:
        
        # Configure mocks
        filter_obj = {"must": [{"key": "metadata.manufacturer", "match": {"value": "Toyota"}}]}
        mock_build_filter.return_value = filter_obj
        
        mock_search.return_value = [
            (Document(page_content="Test document", metadata={}), 0.95)
        ]
        
        # Perform search with filter
        result = vector_store.similarity_search_with_score(
            query="What is the horsepower?",
            k=5,
            metadata_filter={"manufacturer": "Toyota"}
        )
        
        # Verify the result
        assert len(result) == 1
        assert isinstance(result[0][0], Document)
        assert result[0][1] == 0.95
        
        # Verify that the filter was built correctly
        mock_build_filter.assert_called_once_with({"manufacturer": "Toyota"})
        
        # Verify that the search method was called with the correct arguments
        mock_search.assert_called_once_with(
            query="What is the horsepower?",
            k=5,
            filter=filter_obj
        )
        
        # Test without filter
        mock_search.reset_mock()
        vector_store.similarity_search_with_score(
            query="What is the horsepower?",
            k=5
        )
        
        # Verify that the search method was called without a filter
        mock_search.assert_called_once_with(
            query="What is the horsepower?",
            k=5
        )


def test_build_filter(vector_store):
    """Test building Qdrant filters from metadata."""
    # Test single value filter
    single_filter = vector_store._build_filter({"manufacturer": "Toyota"})
    assert len(single_filter.must) == 1
    assert single_filter.must[0].key == "metadata.manufacturer"
    assert single_filter.must[0].match.value == "Toyota"
    
    # Test list value filter
    list_filter = vector_store._build_filter({"manufacturer": ["Toyota", "Honda"]})
    assert len(list_filter.must) == 1
    assert len(list_filter.must[0].should) == 2
    assert list_filter.must[0].should[0].key == "metadata.manufacturer"
    assert list_filter.must[0].should[0].match.value == "Toyota"
    assert list_filter.must[0].should[1].key == "metadata.manufacturer"
    assert list_filter.must[0].should[1].match.value == "Honda"
    
    # Test year range filter
    year_filter = vector_store._build_filter({"year": 2023})
    assert len(year_filter.must) == 1
    assert year_filter.must[0].key == "metadata.year"
    assert year_filter.must[0].range.gte == 2023
    assert year_filter.must[0].range.lte == 2023
    
    # Test multiple field filter
    multi_filter = vector_store._build_filter({
        "manufacturer": "Toyota",
        "category": "sedan",
        "year": 2023
    })
    assert len(multi_filter.must) == 3


def test_delete_by_ids(vector_store):
    """Test deleting documents by IDs."""
    with patch.object(vector_store.client, 'delete') as mock_delete:
        # Delete documents
        vector_store.delete_by_ids(["doc1", "doc2"])
        
        # Verify that the delete method was called with the correct arguments
        mock_delete.assert_called_once()
        assert mock_delete.call_args[1]["collection_name"] == "test_collection"
        assert mock_delete.call_args[1]["points_selector"].points == ["doc1", "doc2"]


def test_get_stats(vector_store):
    """Test getting collection statistics."""
    with patch.object(vector_store.client, 'get_collection') as mock_get_collection:
        # Configure mock
        mock_collection = MagicMock()
        mock_collection.dict.return_value = {
            "name": "test_collection",
            "vectors_count": 100,
            "disk_data_size": 1024
        }
        mock_get_collection.return_value = mock_collection
        
        # Get stats
        stats = vector_store.get_stats()
        
        # Verify the result
        assert stats["name"] == "test_collection"
        assert stats["vectors_count"] == 100
        assert stats["disk_data_size"] == 1024
        
        # Verify that the get_collection method was called with the correct arguments
        mock_get_collection.assert_called_once_with("test_collection")


def test_hybrid_retriever_initialization(vector_store, colbert_reranker):
    """Test hybrid retriever initialization."""
    retriever = HybridRetriever(
        vector_store=vector_store,
        reranker=colbert_reranker,
        top_k=20,
        rerank_top_k=5
    )
    
    assert retriever.vector_store == vector_store
    assert retriever.reranker == colbert_reranker
    assert retriever.top_k == 20
    assert retriever.rerank_top_k == 5


def test_retrieve_with_reranking(vector_store, colbert_reranker):
    """Test retrieval with reranking."""
    retriever = HybridRetriever(
        vector_store=vector_store,
        reranker=colbert_reranker,
        top_k=10,
        rerank_top_k=3
    )
    
    # Create test documents
    documents = [
        (Document(page_content="Doc 1", metadata={"id": "1"}), 0.9),
        (Document(page_content="Doc 2", metadata={"id": "2"}), 0.8),
        (Document(page_content="Doc 3", metadata={"id": "3"}), 0.7),
    ]
    
    # Mock the vector store and reranker
    with patch.object(vector_store, 'similarity_search_with_score', return_value=documents) as mock_search, \
         patch.object(colbert_reranker, 'rerank') as mock_rerank:
        
        # Configure mock reranker
        reranked_docs = [
            (Document(page_content="Doc 3", metadata={"id": "3"}), 0.95),  # Reordered
            (Document(page_content="Doc 1", metadata={"id": "1"}), 0.90),
            (Document(page_content="Doc 2", metadata={"id": "2"}), 0.85),
        ]
        mock_rerank.return_value = reranked_docs
        
        # Perform retrieval with reranking
        results, execution_time = retriever.retrieve(
            query="What is the horsepower?",
            metadata_filter={"manufacturer": "Toyota"},
            rerank=True
        )
        
        # Verify the results
        assert results == reranked_docs
        assert execution_time > 0
        
        # Verify that the search and rerank methods were called with the correct arguments
        mock_search.assert_called_once_with(
            query="What is the horsepower?",
            k=10,
            metadata_filter={"manufacturer": "Toyota"}
        )
        
        # Extract just the documents from the search results (without scores)
        search_docs = [doc for doc, _ in documents]
        mock_rerank.assert_called_once_with(
            query="What is the horsepower?",
            documents=search_docs,
            top_k=3
        )
        
        # Test retrieval without reranking
        mock_search.reset_mock()
        mock_rerank.reset_mock()
        
        results, execution_time = retriever.retrieve(
            query="What is the horsepower?",
            metadata_filter={"manufacturer": "Toyota"},
            rerank=False
        )
        
        # Verify that the results are from the initial search (limited to rerank_top_k)
        assert results == documents[:3]
        assert execution_time > 0
        
        # Verify that the search method was called and rerank was not
        mock_search.assert_called_once()
        mock_rerank.assert_not_called()


def test_retrieve_and_format(vector_store, colbert_reranker):
    """Test retrieval with formatting for API response."""
    retriever = HybridRetriever(
        vector_store=vector_store,
        reranker=colbert_reranker
    )
    
    # Mock the retrieve method
    with patch.object(retriever, 'retrieve') as mock_retrieve:
        # Configure mock
        mock_retrieve.return_value = (
            [
                (Document(
                    page_content="The Toyota Camry has 208 horsepower.",
                    metadata={"id": "doc1", "manufacturer": "Toyota", "model": "Camry"}
                ), 0.95),
                (Document(
                    page_content="The Honda Civic has 158 horsepower.",
                    metadata={"id": "doc2", "manufacturer": "Honda", "model": "Civic"}
                ), 0.85),
            ],
            0.05  # execution time
        )
        
        # Perform retrieval and formatting
        formatted_results, execution_time = retriever.retrieve_and_format(
            query="What is the horsepower of the Toyota Camry?",
            metadata_filter={"manufacturer": "Toyota"}
        )
        
        # Verify the results
        assert len(formatted_results) == 2
        assert formatted_results[0]["id"] == "doc1"
        assert formatted_results[0]["content"] == "The Toyota Camry has 208 horsepower."
        assert formatted_results[0]["metadata"]["manufacturer"] == "Toyota"
        assert formatted_results[0]["relevance_score"] == 0.95
        
        assert formatted_results[1]["id"] == "doc2"
        assert formatted_results[1]["metadata"]["manufacturer"] == "Honda"
        assert formatted_results[1]["relevance_score"] == 0.85
        
        assert execution_time == 0.05
        
        # Verify that the retrieve method was called with the correct arguments
        mock_retrieve.assert_called_once_with(
            query="What is the horsepower of the Toyota Camry?",
            metadata_filter={"manufacturer": "Toyota"},
            rerank=True
        )