import time
from typing import Dict, List, Optional, Tuple, Union
import logging

from langchain_core.documents import Document

from src.core.colbert_reranker import ColBERTReranker
from src.core.vectorstore import QdrantStore

# Configure logging
logger = logging.getLogger(__name__)


class HybridRetriever:
    """
    Enhanced hybrid retriever combining vector search with metadata filtering and ColBERT reranking.

    This retriever:
    1. Performs an initial retrieval using Qdrant with optional metadata filters
    2. Reranks the results using ColBERT's late interaction model for precise semantic matching
    3. Ensures proper diversity in the results to avoid over-representing certain documents
    """

    def __init__(
            self,
            vector_store: QdrantStore,
            reranker: ColBERTReranker,
            top_k: int = 30,  # Increased from 20 to ensure more candidates
            rerank_top_k: int = 10,  # Increased from 5 to ensure more diversity
            source_boost: float = 1.0,  # Boost factor for source diversity
            max_recency_boost: float = 0.2,  # Maximum boost for recent documents
    ):
        """
        Initialize the enhanced hybrid retriever.

        Args:
            vector_store: Vector store for initial retrieval
            reranker: ColBERT reranker for late interaction reranking
            top_k: Number of documents to retrieve in the initial pass
            rerank_top_k: Number of documents to return after reranking
            source_boost: Boost factor for source diversity
            max_recency_boost: Maximum boost for recent documents
        """
        self.vector_store = vector_store
        self.reranker = reranker
        self.top_k = top_k
        self.rerank_top_k = rerank_top_k
        self.source_boost = source_boost
        self.max_recency_boost = max_recency_boost

    def retrieve(
            self,
            query: str,
            metadata_filter: Optional[Dict[str, Union[str, List[str], int, List[int]]]] = None,
            rerank: bool = True,
            ensure_source_diversity: bool = True,  # New parameter to ensure diverse sources
    ) -> Tuple[List[Tuple[Document, float]], float]:
        """
        Retrieve documents using enhanced hybrid search and late interaction reranking.

        Args:
            query: Query string
            metadata_filter: Optional metadata filters
            rerank: Whether to perform reranking
            ensure_source_diversity: Whether to ensure diverse sources in results

        Returns:
            Tuple of (results, execution_time)
            - results: List of (document, score) tuples
            - execution_time: Time taken for retrieval in seconds
        """
        start_time = time.time()

        logger.info(f"Retrieving documents for query: {query}")
        logger.info(f"Metadata filter: {metadata_filter}")
        logger.info(f"Using reranking: {rerank}")
        logger.info(f"Ensuring source diversity: {ensure_source_diversity}")

        # Step 1: Initial retrieval with vector search + metadata filtering
        # Use a larger top_k to ensure diversity
        initial_top_k = min(self.top_k * 2, 50) if ensure_source_diversity else self.top_k

        initial_results = self.vector_store.similarity_search_with_score(
            query=query,
            k=initial_top_k,
            metadata_filter=metadata_filter,
        )

        # Log statistics about initial results
        logger.info(f"Initial retrieval returned {len(initial_results)} documents")
        if initial_results:
            source_counts = {}
            for doc, score in initial_results:
                source = doc.metadata.get("source", "unknown")
                source_id = doc.metadata.get("source_id", "unknown")
                key = f"{source}:{source_id}"
                source_counts[key] = source_counts.get(key, 0) + 1

            logger.info(f"Source distribution in initial results: {source_counts}")

        # Extract documents from results
        documents = [doc for doc, _ in initial_results]

        # Step 2: Apply source diversity if enabled
        if ensure_source_diversity and documents:
            # Get unique source/source_id combinations
            seen_sources = set()
            diverse_docs = []

            # First pass: Get one document from each unique source/source_id
            for doc in documents:
                source = doc.metadata.get("source", "unknown")
                source_id = doc.metadata.get("source_id", "unknown")
                key = f"{source}:{source_id}"

                if key not in seen_sources:
                    diverse_docs.append(doc)
                    seen_sources.add(key)

            # Second pass: Fill remaining slots with other documents
            remaining_slots = self.top_k - len(diverse_docs)
            if remaining_slots > 0:
                for doc in documents:
                    if doc not in diverse_docs and len(diverse_docs) < self.top_k:
                        diverse_docs.append(doc)

            # Update documents list with diverse set
            if len(diverse_docs) > 0:
                documents = diverse_docs[:self.top_k]
                logger.info(f"Applied source diversity, reduced to {len(documents)} documents")

        # Step 3: Rerank using ColBERT if enabled
        if rerank and documents:
            reranked_results = self.reranker.rerank(
                query=query,
                documents=documents,
                top_k=self.rerank_top_k,
            )
            results = reranked_results

            # Log statistics about reranked results
            logger.info(f"Reranking returned {len(results)} documents")
            if results:
                source_counts = {}
                for doc, score in results:
                    source = doc.metadata.get("source", "unknown")
                    source_id = doc.metadata.get("source_id", "unknown")
                    key = f"{source}:{source_id}"
                    source_counts[key] = source_counts.get(key, 0) + 1

                logger.info(f"Source distribution in reranked results: {source_counts}")
        else:
            # Use initial results but limit to rerank_top_k
            # Recreate the score tuples since we might have modified the document list
            results = []
            seen_docs = set()

            for doc, score in initial_results:
                if doc in documents and doc.page_content not in seen_docs:
                    results.append((doc, score))
                    seen_docs.add(doc.page_content)

            results = results[:self.rerank_top_k]

            logger.info(f"Using non-reranked results, returning {len(results)} documents")

        # Apply recency boost if available
        if self.max_recency_boost > 0:
            results = self._apply_recency_boost(results)

        execution_time = time.time() - start_time
        logger.info(f"Retrieval completed in {execution_time:.2f}s")

        return results, execution_time

    def _apply_recency_boost(self, results: List[Tuple[Document, float]]) -> List[Tuple[Document, float]]:
        """Apply a recency boost to favor more recent documents."""
        # Find most recent document time
        most_recent_time = 0
        current_time = time.time()

        for doc, _ in results:
            doc_time = doc.metadata.get("ingestion_time")
            if not doc_time:
                continue

            try:
                doc_time = float(doc_time)
                most_recent_time = max(most_recent_time, doc_time)
            except (ValueError, TypeError):
                continue

        if most_recent_time == 0:
            return results

        # Apply boost based on recency
        boosted_results = []
        for doc, score in results:
            doc_time = doc.metadata.get("ingestion_time")
            if not doc_time:
                boosted_results.append((doc, score))
                continue

            try:
                doc_time = float(doc_time)
                # Calculate age factor (0 to 1, where 1 is most recent)
                age_factor = (doc_time - most_recent_time) / (current_time - most_recent_time)

                # Apply boost (maximum of max_recency_boost)
                recency_boost = self.max_recency_boost * age_factor
                boosted_score = score * (1 + recency_boost)

                boosted_results.append((doc, boosted_score))
            except (ValueError, TypeError, ZeroDivisionError):
                boosted_results.append((doc, score))

        # Sort by boosted score
        boosted_results.sort(key=lambda x: x[1], reverse=True)
        return boosted_results

    def retrieve_and_format(
            self,
            query: str,
            metadata_filter: Optional[Dict[str, Union[str, List[str], int, List[int]]]] = None,
            rerank: bool = True,
            ensure_source_diversity: bool = True,
    ) -> Tuple[List[Dict], float]:
        """
        Retrieve documents and format them for API response with enhanced diversity.

        Args:
            query: Query string
            metadata_filter: Optional metadata filters
            rerank: Whether to perform reranking
            ensure_source_diversity: Whether to ensure diverse sources

        Returns:
            Tuple of (formatted_results, execution_time)
            - formatted_results: List of formatted document dictionaries
            - execution_time: Time taken for retrieval in seconds
        """
        results, execution_time = self.retrieve(
            query=query,
            metadata_filter=metadata_filter,
            rerank=rerank,
            ensure_source_diversity=ensure_source_diversity,
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