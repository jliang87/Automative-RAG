import time
from typing import Dict, List, Optional, Tuple, Union

from langchain_core.documents import Document

from src.core.colbert_reranker import ColBERTReranker
from src.core.vectorstore import QdrantStore


class HybridRetriever:
    """
    Hybrid retriever combining vector search with metadata filtering and ColBERT reranking.
    
    This retriever:
    1. Performs an initial retrieval using Qdrant with optional metadata filters
    2. Reranks the results using ColBERT's late interaction model for precise semantic matching
    """

    def __init__(
        self,
        vector_store: QdrantStore,
        reranker: ColBERTReranker,
        top_k: int = 20,
        rerank_top_k: int = 5,
    ):
        """
        Initialize the hybrid retriever.

        Args:
            vector_store: Vector store for initial retrieval
            reranker: ColBERT reranker for late interaction reranking
            top_k: Number of documents to retrieve in the initial pass
            rerank_top_k: Number of documents to return after reranking
        """
        self.vector_store = vector_store
        self.reranker = reranker
        self.top_k = top_k
        self.rerank_top_k = rerank_top_k

    def retrieve(
        self,
        query: str,
        metadata_filter: Optional[Dict[str, Union[str, List[str], int, List[int]]]] = None,
        rerank: bool = True,
    ) -> Tuple[List[Tuple[Document, float]], float]:
        """
        Retrieve documents using hybrid search and late interaction reranking.

        Args:
            query: Query string
            metadata_filter: Optional metadata filters
            rerank: Whether to perform reranking

        Returns:
            Tuple of (results, execution_time)
            - results: List of (document, score) tuples
            - execution_time: Time taken for retrieval in seconds
        """
        start_time = time.time()
        
        # Step 1: Initial retrieval with vector search + metadata filtering
        initial_results = self.vector_store.similarity_search_with_score(
            query=query,
            k=self.top_k,
            metadata_filter=metadata_filter,
        )
        
        # Extract documents from results
        documents = [doc for doc, _ in initial_results]
        
        # Step 2: Rerank using ColBERT if enabled
        if rerank and documents:
            reranked_results = self.reranker.rerank(
                query=query,
                documents=documents,
                top_k=self.rerank_top_k,
            )
            results = reranked_results
        else:
            # Use initial results but limit to rerank_top_k
            results = initial_results[:self.rerank_top_k]
        
        execution_time = time.time() - start_time
        
        return results, execution_time

    def retrieve_and_format(
        self,
        query: str,
        metadata_filter: Optional[Dict[str, Union[str, List[str], int, List[int]]]] = None,
        rerank: bool = True,
    ) -> Tuple[List[Dict], float]:
        """
        Retrieve documents and format them for API response.

        Args:
            query: Query string
            metadata_filter: Optional metadata filters
            rerank: Whether to perform reranking

        Returns:
            Tuple of (formatted_results, execution_time)
            - formatted_results: List of formatted document dictionaries
            - execution_time: Time taken for retrieval in seconds
        """
        results, execution_time = self.retrieve(
            query=query,
            metadata_filter=metadata_filter,
            rerank=rerank,
        )
        
        formatted_results = []
        
        for doc, score in results:
            formatted_doc = {
                "id": doc.metadata.get("id", ""),
                "content": doc.page_content,
                "metadata": doc.metadata,
                "relevance_score": score,
            }
            formatted_results.append(formatted_doc)
        
        return formatted_results, execution_time
