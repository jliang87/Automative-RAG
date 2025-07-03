#!/usr/bin/env python3
"""
Complete ingestion verification script for the RAG system.
"""

import os
import sys
import json
import argparse
import logging
from typing import List, Dict, Any

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import project modules
from src.config.settings import settings
from src.core.query.retrieval.vectorstore import QdrantStore
from qdrant_client import QdrantClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [%(name)s] - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/verification.log"),
    ]
)
logger = logging.getLogger("verify_ingestion")


def initialize_vector_store():
    """Initialize vector store for verification."""
    logger.info("Initializing vector store...")

    qdrant_client = QdrantClient(
        host=settings.qdrant_host,
        port=settings.qdrant_port,
    )

    # Initialize in metadata-only mode for verification
    vector_store = QdrantStore(
        client=qdrant_client,
        collection_name=settings.qdrant_collection,
        embedding_function=None,  # Metadata-only mode
    )

    return vector_store


def search_documents_by_job_id(vector_store: QdrantStore, job_id: str) -> List[Dict[str, Any]]:
    """
    Search for documents by job ID.

    Args:
        vector_store: Vector store instance
        job_id: Job ID to search for

    Returns:
        List of documents found
    """
    logger.info(f"Searching for documents with job_id: {job_id}")

    try:
        # Search by metadata
        documents = vector_store.search_by_metadata(
            metadata_filter={"job_id": job_id},
            limit=100
        )

        logger.info(f"Found {len(documents)} documents for job {job_id}")
        return [{"content": doc.page_content, "metadata": doc.metadata} for doc in documents]

    except Exception as e:
        logger.error(f"Error searching documents: {e}")
        return []


def search_documents_by_content(vector_store: QdrantStore, search_term: str) -> List[Dict[str, Any]]:
    """
    Search for documents containing specific content.

    Args:
        vector_store: Vector store instance
        search_term: Term to search for in document content

    Returns:
        List of matching documents
    """
    logger.info(f"Searching for documents containing: '{search_term}'")

    try:
        # Get collection info first
        collection_info = vector_store.client.get_collection(vector_store.collection_name)
        total_vectors = getattr(collection_info, 'vectors_count', 0)
        logger.info(f"Collection has {total_vectors} total documents")

        if total_vectors == 0:
            logger.warning("Collection appears to be empty!")
            return []

        # Scroll through all documents in batches
        all_matching_docs = []
        offset = None
        batch_size = 100
        total_processed = 0

        while True:
            # Scroll through documents
            scroll_result = vector_store.client.scroll(
                collection_name=vector_store.collection_name,
                limit=batch_size,
                offset=offset,
                with_payload=True,
                with_vectors=False
            )

            points, next_offset = scroll_result

            if not points:
                break

            logger.info(f"Processing batch of {len(points)} documents (total processed: {total_processed})")

            for point in points:
                content = point.payload.get("page_content", "")
                metadata = point.payload.get("metadata", {})

                # Check if search term is in content (case insensitive)
                if search_term.lower() in content.lower():
                    all_matching_docs.append({
                        "id": str(point.id),
                        "content": content,
                        "metadata": metadata,
                        "relevance": "content_match"
                    })

                # Also check metadata fields for partial matches
                elif any(search_term.lower() in str(v).lower() for v in metadata.values() if v):
                    all_matching_docs.append({
                        "id": str(point.id),
                        "content": content,
                        "metadata": metadata,
                        "relevance": "metadata_match"
                    })

            total_processed += len(points)

            # Continue with next batch
            if next_offset is None:
                break
            offset = next_offset

        logger.info(
            f"Processed {total_processed} total documents, found {len(all_matching_docs)} matches for '{search_term}'")
        return all_matching_docs

    except Exception as e:
        logger.error(f"Error searching by content: {e}")
        return []


def get_collection_stats(vector_store: QdrantStore) -> Dict[str, Any]:
    """Get detailed collection statistics."""
    logger.info("Getting collection statistics...")

    try:
        # Get basic collection info first
        collection_info = vector_store.client.get_collection(vector_store.collection_name)

        # Try multiple methods to get document count
        vector_count = None
        try:
            # Method 1: Try vectors_count attribute
            vector_count = getattr(collection_info, 'vectors_count', None)
        except:
            pass

        if vector_count is None:
            try:
                # Method 2: Try points_count attribute
                vector_count = getattr(collection_info, 'points_count', None)
            except:
                pass

        if vector_count is None:
            try:
                # Method 3: Use count API directly
                count_result = vector_store.client.count(vector_store.collection_name)
                vector_count = count_result.count
            except:
                pass

        # Sample documents to get actual count
        all_docs = []
        offset = None
        actual_count = 0

        while True:
            try:
                scroll_result = vector_store.client.scroll(
                    collection_name=vector_store.collection_name,
                    limit=100,
                    offset=offset,
                    with_payload=True,
                    with_vectors=False
                )

                points, next_offset = scroll_result
                if not points:
                    break

                actual_count += len(points)

                # Keep first 5 for samples
                if len(all_docs) < 5:
                    all_docs.extend(points[:5 - len(all_docs)])

                if next_offset is None:
                    break
                offset = next_offset

            except Exception as e:
                logger.error(f"Error during scroll: {e}")
                break

        sample_metadata = []
        for point in all_docs[:5]:
            metadata = point.payload.get("metadata", {})
            sample_metadata.append({
                "point_id": str(point.id),
                "source": metadata.get("source"),
                "job_id": metadata.get("job_id"),
                "title": metadata.get("title", "")[:50] + "..." if metadata.get("title") else None,
                "chunk_id": metadata.get("chunk_id"),
                "ingestion_time": metadata.get("ingestion_time"),
                "content_length": len(point.payload.get("page_content", ""))
            })

        enhanced_stats = {
            "collection_name": vector_store.collection_name,
            "vector_count_api": vector_count,  # What API reports
            "actual_document_count": actual_count,  # What we actually found
            "collection_status": str(getattr(collection_info, 'status', 'unknown')),
            "sample_documents": sample_metadata,
            "total_sample_docs": len(sample_metadata),
            "count_method_used": "scroll_count" if vector_count is None else "api_count"
        }

        return enhanced_stats

    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        return {"error": str(e)}


def analyze_job_documents(vector_store: QdrantStore, job_id: str) -> Dict[str, Any]:
    """
    Comprehensive analysis of documents for a specific job.

    Args:
        vector_store: Vector store instance
        job_id: Job ID to analyze

    Returns:
        Analysis results
    """
    logger.info(f"Analyzing documents for job: {job_id}")

    documents = search_documents_by_job_id(vector_store, job_id)

    if not documents:
        return {
            "job_id": job_id,
            "found": False,
            "message": "No documents found for this job ID"
        }

    # Analyze the documents
    analysis = {
        "job_id": job_id,
        "found": True,
        "document_count": len(documents),
        "documents": [],
        "content_analysis": {
            "total_characters": 0,
            "avg_chunk_length": 0,
            "sources": set(),
            "keywords": {}
        }
    }

    total_chars = 0
    keyword_counts = {}

    for i, doc in enumerate(documents):
        content = doc["content"]
        metadata = doc["metadata"]

        total_chars += len(content)

        # Extract key info
        doc_info = {
            "document_number": i + 1,
            "content_length": len(content),
            "content_preview": content[:200] + "..." if len(content) > 200 else content,
            "metadata": {
                "source": metadata.get("source"),
                "chunk_id": metadata.get("chunk_id"),
                "title": metadata.get("title"),
                "ingestion_time": metadata.get("ingestion_time")
            }
        }

        analysis["documents"].append(doc_info)

        # Track sources
        if metadata.get("source"):
            analysis["content_analysis"]["sources"].add(metadata["source"])

        # Simple keyword extraction (look for key terms)
        key_terms = ["吉利", "星越", "卡车", "碰撞", "事故", "安全"]
        for term in key_terms:
            if term in content:
                keyword_counts[term] = keyword_counts.get(term, 0) + 1

    # Finalize analysis
    analysis["content_analysis"]["total_characters"] = total_chars
    analysis["content_analysis"]["avg_chunk_length"] = total_chars / len(documents) if documents else 0
    analysis["content_analysis"]["sources"] = list(analysis["content_analysis"]["sources"])
    analysis["content_analysis"]["keywords"] = keyword_counts

    return analysis


def debug_query_pipeline(vector_store: QdrantStore, query: str, job_id: str = None) -> Dict[str, Any]:
    """
    Debug why a query might not be finding expected documents.

    Args:
        vector_store: Vector store instance
        query: Query to debug
        job_id: Optional job ID to focus on

    Returns:
        Debug information
    """
    logger.info(f"Debugging query: '{query}'")

    debug_info = {
        "query": query,
        "job_id": job_id,
        "steps": {}
    }

    # Step 1: Check if documents exist for the job
    if job_id:
        job_docs = search_documents_by_job_id(vector_store, job_id)
        debug_info["steps"]["job_documents"] = {
            "found": len(job_docs),
            "sample_content": [doc["content"][:100] + "..." for doc in job_docs[:3]]
        }

    # Step 2: Search for query terms in content
    query_terms = query.split()
    for term in query_terms:
        matching_docs = search_documents_by_content(vector_store, term)
        debug_info["steps"][f"term_search_{term}"] = {
            "term": term,
            "matches": len(matching_docs),
            "sample_matches": [
                {
                    "job_id": doc["metadata"].get("job_id"),
                    "source": doc["metadata"].get("source"),
                    "content_preview": doc["content"][:100] + "..."
                }
                for doc in matching_docs[:3]
            ]
        }

    # Step 3: Check collection status
    stats = get_collection_stats(vector_store)
    debug_info["steps"]["collection_status"] = stats

    return debug_info


def main():
    parser = argparse.ArgumentParser(description="Debug document ingestion and retrieval")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Search by job ID
    job_parser = subparsers.add_parser("search-job", help="Search documents by job ID")
    job_parser.add_argument("job_id", help="Job ID to search for")

    # Search by content
    content_parser = subparsers.add_parser("search-content", help="Search documents by content")
    content_parser.add_argument("search_term", help="Term to search for in document content")

    # Get collection stats
    stats_parser = subparsers.add_parser("stats", help="Get collection statistics")

    # Analyze job
    analyze_parser = subparsers.add_parser("analyze-job", help="Analyze documents for a specific job")
    analyze_parser.add_argument("job_id", help="Job ID to analyze")

    # Debug query
    debug_parser = subparsers.add_parser("debug-query", help="Debug why a query might not work")
    debug_parser.add_argument("query", help="Query to debug")
    debug_parser.add_argument("--job-id", help="Optional job ID to focus debugging on")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    # Create logs directory
    os.makedirs("logs", exist_ok=True)

    # Initialize vector store
    vector_store = initialize_vector_store()

    if args.command == "search-job":
        documents = search_documents_by_job_id(vector_store, args.job_id)
        result = {
            "job_id": args.job_id,
            "documents_found": len(documents),
            "documents": documents
        }
        print(json.dumps(result, indent=2, ensure_ascii=False))

    elif args.command == "search-content":
        documents = search_documents_by_content(vector_store, args.search_term)
        result = {
            "search_term": args.search_term,
            "documents_found": len(documents),
            "documents": documents
        }
        print(json.dumps(result, indent=2, ensure_ascii=False))

    elif args.command == "stats":
        stats = get_collection_stats(vector_store)
        print(json.dumps(stats, indent=2, ensure_ascii=False))

    elif args.command == "analyze-job":
        analysis = analyze_job_documents(vector_store, args.job_id)
        print(json.dumps(analysis, indent=2, ensure_ascii=False))

    elif args.command == "debug-query":
        debug_info = debug_query_pipeline(vector_store, args.query, args.job_id)
        print(json.dumps(debug_info, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()