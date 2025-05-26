# src/core/background/actors/inference.py (SIMPLIFIED - Remove All Priority Queue Logic)

"""
Simplified inference actors that contain only the core work logic.
All coordination is handled by the job chain system.
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
import dramatiq

logger = logging.getLogger(__name__)

# Keep only the core work functions - remove all priority queue coordination


def retrieve_documents_for_query(query: str, metadata_filter: Optional[Dict] = None, top_k: int = 30) -> List[Dict]:
    """Retrieve documents for a query. Pure work function."""
    from src.core.background.models import get_vector_store

    vector_store = get_vector_store()

    # Perform retrieval
    results = vector_store.similarity_search_with_score(
        query=query,
        k=top_k,
        metadata_filter=metadata_filter
    )

    # Format results
    serialized_docs = []
    for doc, score in results:
        serialized_docs.append({
            "content": doc.page_content,
            "metadata": doc.metadata,
            "relevance_score": score
        })

    return serialized_docs


def rerank_documents_with_colbert(query: str, documents: List[Dict], top_k: int = 10) -> List[Tuple]:
    """Rerank documents using ColBERT. Pure work function."""
    from src.core.background.models import get_colbert_reranker
    from langchain_core.documents import Document

    # Convert to Document objects
    doc_objects = []
    for doc_dict in documents:
        doc = Document(
            page_content=doc_dict["content"],
            metadata=doc_dict.get("metadata", {})
        )
        doc_objects.append(doc)

    # Perform reranking
    reranker = get_colbert_reranker()
    if reranker is not None:
        reranked_docs = reranker.rerank(query, doc_objects, top_k)
    else:
        # Fallback: use original order with scores
        reranked_docs = [(doc, doc_dict.get("relevance_score", 0))
                        for doc, doc_dict in zip(doc_objects, documents[:top_k])]

    return reranked_docs


def generate_llm_answer(query: str, documents: List[Tuple], metadata_filter: Optional[Dict] = None) -> str:
    """Generate LLM answer from documents. Pure work function."""
    from src.core.background.models import get_llm_model

    # Get LLM model and perform inference
    llm = get_llm_model()
    answer = llm.answer_query(
        query=query,
        documents=documents,
        metadata_filter=metadata_filter
    )

    return answer


def perform_complete_llm_inference(query: str, documents: List[Dict], metadata_filter: Optional[Dict] = None) -> Dict:
    """Complete LLM inference pipeline. Pure work function."""
    from src.config.settings import settings

    # Step 1: Rerank documents
    reranked_docs = rerank_documents_with_colbert(query, documents, settings.reranker_top_k)

    # Step 2: Generate answer
    answer = generate_llm_answer(query, reranked_docs, metadata_filter)

    # Step 3: Format response
    formatted_documents = []
    for doc, score in reranked_docs:
        formatted_doc = {
            "id": doc.metadata.get("id", ""),
            "content": doc.page_content,
            "metadata": doc.metadata,
            "relevance_score": score,
        }
        formatted_documents.append(formatted_doc)

    return {
        "query": query,
        "answer": answer,
        "documents": formatted_documents
    }


# Legacy actors for backward compatibility (minimal versions)

@dramatiq.actor(queue_name="inference_tasks", store_results=True, max_retries=2)
def simple_llm_inference(query: str, documents: List[Dict], metadata_filter: Optional[Dict] = None):
    """Simple LLM inference actor. Use job chain instead when possible."""
    try:
        return perform_complete_llm_inference(query, documents, metadata_filter)
    except Exception as e:
        logger.error(f"LLM inference failed: {str(e)}")
        raise


@dramatiq.actor(queue_name="embedding_tasks", store_results=True, max_retries=2)
def simple_document_retrieval(query: str, metadata_filter: Optional[Dict] = None):
    """Simple document retrieval actor. Use job chain instead when possible."""
    try:
        return retrieve_documents_for_query(query, metadata_filter)
    except Exception as e:
        logger.error(f"Document retrieval failed: {str(e)}")
        raise