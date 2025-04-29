#!/usr/bin/env python3
"""
Ingestion verification script for the RAG system.

This script helps verify that documents are properly ingested into the vector store
and can be retrieved correctly. It can also help diagnose issues with document
processing, embedding, and retrieval.
"""

import os
import sys
import json
import argparse
import logging
from typing import List, Dict, Any, Optional

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import project modules
from src.config.settings import settings
from src.core.document_processor import DocumentProcessor
from src.core.vectorstore import QdrantStore
from src.core.retriever import HybridRetriever
from src.core.colbert_reranker import ColBERTReranker

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


def initialize_components():
    """Initialize the components needed for verification."""
    logger.info("Initializing components...")

    # Initialize vector store
    from qdrant_client import QdrantClient
    qdrant_client = QdrantClient(
        host=settings.qdrant_host,
        port=settings.qdrant_port,
    )
    vector_store = QdrantStore(
        client=qdrant_client,
        collection_name=settings.qdrant_collection,
        embedding_function=settings.embedding_function,
    )

    # Initialize ColBERT reranker
    reranker = ColBERTReranker(
        model_name=settings.default_colbert_model,
        device=settings.device,
        batch_size=settings.colbert_batch_size,
        use_fp16=settings.use_fp16,
        use_bge_reranker=settings.use_bge_reranker,
        colbert_weight=settings.colbert_weight,
        bge_weight=settings.bge_weight,
        bge_model_name=settings.default_bge_reranker_model
    )

    # Initialize retriever
    retriever = HybridRetriever(
        vector_store=vector_store,
        reranker=reranker,
        top_k=settings.retriever_top_k,
        rerank_top_k=settings.reranker_top_k,
    )

    # Initialize document processor
    processor = DocumentProcessor(
        vector_store=vector_store,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        device=settings.device,
    )

    return vector_store, retriever, processor


def verify_video(processor: DocumentProcessor, url: str, force_refresh: bool = False) -> Dict[str, Any]:
    """
    Verify that a video was properly ingested and is retrievable.

    Args:
        processor: Document processor
        url: Video URL to verify
        force_refresh: Whether to force reprocessing of the video

    Returns:
        Dictionary with verification results
    """
    logger.info(f"Verifying video ingestion for: {url}")

    if force_refresh:
        # Process the video (reprocess if it already exists)
        doc_ids = processor.process_video(url, force_refresh=True)
        logger.info(f"Reprocessed video, got {len(doc_ids)} document IDs")

    # Verify ingestion
    result = processor.verify_ingestion(url)

    # Print details
    if result["verification_success"]:
        logger.info(f"Verification successful! Found {result['documents_found']} documents")
    else:
        logger.error(f"Verification failed! No documents found for {url}")

    return result


def verify_all_videos(processor: DocumentProcessor, urls: List[str], force_refresh: bool = False) -> Dict[str, Any]:
    """
    Verify multiple videos.

    Args:
        processor: Document processor
        urls: List of video URLs to verify
        force_refresh: Whether to force reprocessing

    Returns:
        Dictionary with verification results
    """
    results = {
        "total": len(urls),
        "successful": 0,
        "failed": 0,
        "details": []
    }

    for url in urls:
        result = verify_video(processor, url, force_refresh)
        results["details"].append(result)

        if result["verification_success"]:
            results["successful"] += 1
        else:
            results["failed"] += 1

    logger.info(f"Verification complete: {results['successful']}/{results['total']} successful")
    return results


def test_query(retriever: HybridRetriever, query: str, metadata_filter: Optional[Dict[str, Any]] = None) -> Dict[
    str, Any]:
    """
    Test a query to see what documents are retrieved.

    Args:
        retriever: HybridRetriever instance
        query: Query string to test
        metadata_filter: Optional metadata filter

    Returns:
        Dictionary with query results
    """
    logger.info(f"Testing query: '{query}'")
    if metadata_filter:
        logger.info(f"With metadata filter: {metadata_filter}")

    # Retrieve documents
    docs, execution_time = retriever.retrieve(
        query=query,
        metadata_filter=metadata_filter,
        rerank=True,
        ensure_source_diversity=True
    )

    # Format results
    sources = {}
    results = {
        "query": query,
        "metadata_filter": metadata_filter,
        "execution_time": execution_time,
        "documents_retrieved": len(docs),
        "sources": {},
        "results": []
    }

    for doc, score in docs:
        source = doc.metadata.get("source", "unknown")
        source_id = doc.metadata.get("source_id", "unknown")

        # Track source statistics
        source_key = f"{source}:{source_id}"
        if source_key not in sources:
            sources[source_key] = 0
        sources[source_key] += 1

        # Add to results
        results["results"].append({
            "score": score,
            "content": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
            "metadata": {
                "source": source,
                "source_id": source_id,
                "title": doc.metadata.get("title", "Unknown"),
                "chunk_id": doc.metadata.get("chunk_id", -1),
                "total_chunks": doc.metadata.get("total_chunks", 0)
            }
        })

    # Add source counts
    for source, count in sources.items():
        results["sources"][source] = count

    logger.info(f"Query returned {len(docs)} documents from {len(sources)} sources")
    return results


def repair_vector_store(vector_store: QdrantStore) -> Dict[str, Any]:
    """
    Attempt to repair the vector store.

    Args:
        vector_store: Vector store to repair

    Returns:
        Dictionary with repair results
    """
    logger.info("Attempting to repair vector store...")

    # Get stats before repair
    before_stats = vector_store.get_stats()
    logger.info(f"Before repair: {before_stats.get('vectors_count', 0)} vectors in collection")

    # Perform repair
    repair_result = vector_store.repair_indices()

    # Get stats after repair
    after_stats = vector_store.get_stats()
    logger.info(f"After repair: {after_stats.get('vectors_count', 0)} vectors in collection")

    return {
        "before": before_stats,
        "after": after_stats,
        "repair_result": repair_result
    }


def main():
    parser = argparse.ArgumentParser(description="Verify document ingestion and retrieval")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Verify video command
    verify_parser = subparsers.add_parser("verify-video", help="Verify a video was properly ingested")
    verify_parser.add_argument("url", help="URL of the video to verify")
    verify_parser.add_argument("--force-refresh", action="store_true", help="Force reprocessing")

    # Verify all videos command
    verify_all_parser = subparsers.add_parser("verify-all", help="Verify all videos in a list")
    verify_all_parser.add_argument("urls_file", help="JSON file containing a list of video URLs")
    verify_all_parser.add_argument("--force-refresh", action="store_true", help="Force reprocessing")

    # Test query command
    query_parser = subparsers.add_parser("test-query", help="Test a query")
    query_parser.add_argument("query", help="Query string to test")
    query_parser.add_argument("--metadata", help="JSON metadata filter")

    # Repair vector store command
    repair_parser = subparsers.add_parser("repair", help="Repair vector store")

    args = parser.parse_args()

    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)

    # Initialize components
    vector_store, retriever, processor = initialize_components()

    if args.command == "verify-video":
        result = verify_video(processor, args.url, args.force_refresh)
        print(json.dumps(result, indent=2))

    elif args.command == "verify-all":
        with open(args.urls_file, "r") as f:
            urls = json.load(f)

        if not isinstance(urls, list):
            logger.error("URLs file must contain a JSON array of URLs")
            sys.exit(1)

        results = verify_all_videos(processor, urls, args.force_refresh)
        print(json.dumps(results, indent=2))

        # Save results to file
        output_file = f"verification_results_{int(time.time())}.json"
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {output_file}")

    elif args.command == "test-query":
        metadata_filter = None
        if args.metadata:
            try:
                metadata_filter = json.loads(args.metadata)
            except json.JSONDecodeError:
                logger.error("Invalid metadata JSON")
                sys.exit(1)

        result = test_query(retriever, args.query, metadata_filter)
        print(json.dumps(result, indent=2))

    elif args.command == "repair":
        result = repair_vector_store(vector_store)
        print(json.