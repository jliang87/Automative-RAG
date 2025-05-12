"""
Dramatiq actors for inference tasks.

This module defines actors for inference tasks like query answering
with LLM and document reranking. These tasks are typically executed
by GPU-inference workers.
"""

import time
import json
import logging
from typing import Dict, List, Optional, Any

import dramatiq
import torch

from ..common import JobStatus
from ..job_tracker import job_tracker
from ..priority_queue import priority_queue
from ..models import get_colbert_reranker, get_llm_model, get_vector_store

logger = logging.getLogger(__name__)

# Specialized actor for LLM inference with priority handling
@dramatiq.actor(
    queue_name="inference_tasks",
    max_retries=3,
    time_limit=300000,  # 5 minutes
    min_backoff=10000,  # 10 seconds
    max_backoff=300000,  # 5 minutes
    store_results=True
)
def perform_llm_inference(job_id: str, query: str, documents: List[Dict], metadata_filter: Optional[Dict] = None):
    """Perform LLM inference using reranking and local LLM with priority handling."""
    try:
        # Register task with the priority system
        task_id = f"inference_{job_id}"
        priority_queue.register_task("inference_tasks", task_id, {"job_id": job_id, "query": query})

        # Update job status to show we're queued
        job_tracker.update_job_status(
            job_id,
            JobStatus.PROCESSING,
            result={"message": "In priority queue for inference resources"},
            stage="priority_queue"
        )

        # Wait for priority system to allow this task to run
        # For inference tasks, can_run_task will always return True if there's no active inference task
        wait_start = time.time()
        while not priority_queue.can_run_task("inference_tasks", task_id):
            # Log every 10 seconds of waiting
            if int(time.time() - wait_start) % 10 == 0:
                logger.info(f"Inference task {task_id} waiting in priority queue")
            time.sleep(0.5)

        # Mark this task as now active on GPU
        priority_queue.mark_task_active({
            "task_id": task_id,
            "queue_name": "inference_tasks",
            "priority": 1,
            "job_id": job_id,
            "registered_at": time.time()
        })

        logger.info(f"Starting inference for job {job_id} with priority handling")

        try:
            # Update job status to processing
            job_tracker.update_job_status(
                job_id,
                JobStatus.PROCESSING,
                result={"message": "Processing query with LLM"},
                stage="reranking"
            )

            # Convert document dictionaries back to Document objects
            from langchain_core.documents import Document
            doc_objects = []
            for doc_dict in documents:
                doc = Document(
                    page_content=doc_dict["content"],
                    metadata=doc_dict.get("metadata", {})
                )
                score = doc_dict.get("relevance_score", 0)
                doc_objects.append((doc, score))

            # Perform reranking if available
            reranker = get_colbert_reranker()
            if reranker is not None:
                logger.info(f"Reranking {len(doc_objects)} documents")
                reranked_docs = reranker.rerank(query, [doc for doc, _ in doc_objects], 5)
            else:
                logger.warning("Reranker not available, using original document order")
                reranked_docs = doc_objects[:5]

            # Update job status to LLM inference stage
            job_tracker.update_job_status(
                job_id,
                JobStatus.PROCESSING,
                result={"message": "Generating answer with LLM"},
                stage="llm_inference"
            )

            # Get LLM model
            llm = get_llm_model()

            # Perform inference
            start_time = time.time()
            answer = llm.answer_query(
                query=query,
                documents=reranked_docs,
                metadata_filter=metadata_filter
            )
            inference_time = time.time() - start_time
            logger.info(f"Inference completed in {inference_time:.2f}s")

            # Prepare formatted documents for response
            formatted_documents = []
            for doc, score in reranked_docs:
                formatted_doc = {
                    "id": doc.metadata.get("id", ""),
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "relevance_score": score,
                }
                formatted_documents.append(formatted_doc)

            # Update job with success result
            job_tracker.update_job_status(
                job_id,
                JobStatus.COMPLETED,
                result={
                    "query": query,
                    "answer": answer,
                    "documents": formatted_documents,
                    "metadata_filters_used": metadata_filter,
                    "execution_time": inference_time
                }
            )

            return {
                "answer": answer,
                "documents": formatted_documents,
                "execution_time": inference_time
            }
        finally:
            # Always mark task as completed, even if it failed
            priority_queue.mark_task_completed(task_id)

            # Clear cache if needed
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    except Exception as e:
        import traceback
        error_detail = f"Error performing LLM inference: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_detail)

        # Update job with error
        job_tracker.update_job_status(
            job_id,
            JobStatus.FAILED,
            error=error_detail
        )

        # Clean up GPU memory on error
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Make sure to mark task as completed even in error case
        priority_queue.mark_task_completed(f"inference_{job_id}")

        # Re-raise for dramatiq retry mechanism
        raise


@dramatiq.actor(
    queue_name="reranking_tasks",
    max_retries=2,
    store_results=True
)
def rerank_documents(job_id: str, query: str, documents: List[Dict], top_k: int = 5):
    """Perform document reranking using ColBERT."""
    try:
        # Register task with the priority system
        task_id = f"rerank_{job_id}"
        priority_queue.register_task("reranking_tasks", task_id, {"job_id": job_id, "query": query})

        # Update job status
        job_tracker.update_job_status(
            job_id,
            JobStatus.PROCESSING,
            result={"message": "Reranking documents"},
            stage="reranking"
        )

        # Wait for priority system to allow this task to run
        wait_start = time.time()
        while not priority_queue.can_run_task("reranking_tasks", task_id):
            # Log every 10 seconds of waiting
            if int(time.time() - wait_start) % 10 == 0:
                logger.info(f"Reranking task {task_id} waiting in priority queue")
            time.sleep(0.5)

        # Mark this task as now active on GPU
        priority_queue.mark_task_active({
            "task_id": task_id,
            "queue_name": "reranking_tasks",
            "priority": 2,
            "job_id": job_id,
            "registered_at": time.time()
        })

        try:
            # Convert document dictionaries back to Document objects
            from langchain_core.documents import Document
            doc_objects = []
            for doc_dict in documents:
                doc = Document(
                    page_content=doc_dict["content"],
                    metadata=doc_dict.get("metadata", {})
                )
                doc_objects.append(doc)

            # Perform reranking
            reranker = get_colbert_reranker()
            start_time = time.time()
            reranked_docs = reranker.rerank(query, doc_objects, top_k)
            reranking_time = time.time() - start_time

            # Format results
            results = []
            for doc, score in reranked_docs:
                results.append({
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "relevance_score": score
                })

            # Update job with success result
            job_tracker.update_job_status(
                job_id,
                JobStatus.COMPLETED,
                result={
                    "reranked_documents": results,
                    "execution_time": reranking_time
                }
            )

            return {
                "reranked_documents": results,
                "execution_time": reranking_time
            }
        finally:
            # Always mark task as completed
            priority_queue.mark_task_completed(task_id)

            # Clear cache if needed
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    except Exception as e:
        import traceback
        error_detail = f"Error reranking documents: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_detail)

        # Update job with error
        job_tracker.update_job_status(
            job_id,
            JobStatus.FAILED,
            error=error_detail
        )

        # Clean up GPU memory on error
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Make sure to mark task as completed
        priority_queue.mark_task_completed(f"rerank_{job_id}")

        # Re-raise for dramatiq retry mechanism
        raise


@dramatiq.actor(
    queue_name="embedding_tasks",  # This runs on the worker-gpu-embedding worker
    max_retries=3,
    store_results=True
)
def retrieve_documents(job_id: str, query: str, metadata_filter: Optional[Dict] = None):
    """
    Retrieve documents from the vector store for a query.
    This actor runs on the worker-gpu-embedding to ensure efficient embedding model usage.
    """
    try:
        # Register with priority system
        task_id = f"retrieval_{job_id}"
        priority_queue.register_task("embedding_tasks", task_id, {"job_id": job_id, "query": query})

        # Update job status
        job_tracker.update_job_status(
            job_id,
            JobStatus.PROCESSING,
            result={"message": "Retrieving documents"},
            stage="retrieval"
        )

        # Wait for priority system to allow task to run
        wait_start = time.time()
        while not priority_queue.can_run_task("embedding_tasks", task_id):
            # Log every 10 seconds of waiting
            if int(time.time() - wait_start) % 10 == 0:
                logger.info(f"Retrieval task {task_id} waiting in priority queue")
            time.sleep(1)

        # Mark task as active
        priority_queue.mark_task_active({
            "task_id": task_id,
            "queue_name": "embedding_tasks",
            "priority": 3,  # Same priority as other embedding tasks
            "job_id": job_id,
            "registered_at": time.time()
        })

        try:
            # Get vector store (which uses the preloaded embedding model)
            vector_store = get_vector_store()

            # Get job metadata to find top_k value
            job_data = job_tracker.get_job(job_id)
            parent_job_id = job_data.get("metadata", {}).get("parent_job_id")
            parent_job = job_tracker.get_job(parent_job_id) if parent_job_id else None

            # Get top_k from parent job or use default
            top_k = parent_job.get("metadata", {}).get("top_k", 20) if parent_job else 20

            # Perform retrieval
            start_time = time.time()

            # Use QdrantStore's similarity_search_with_score method
            results = vector_store.similarity_search_with_score(
                query=query,
                k=top_k,
                metadata_filter=metadata_filter
            )

            # Format results for transfer between workers
            serialized_docs = []
            for doc, score in results:
                serialized_docs.append({
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "relevance_score": score
                })

            retrieval_time = time.time() - start_time

            # Update job status
            job_tracker.update_job_status(
                job_id,
                JobStatus.COMPLETED,
                result={
                    "documents": serialized_docs,
                    "document_count": len(serialized_docs),
                    "retrieval_time": retrieval_time
                }
            )

            # Return documents
            return {
                "documents": serialized_docs,
                "document_count": len(serialized_docs),
                "retrieval_time": retrieval_time
            }

        finally:
            # Always mark task as completed
            priority_queue.mark_task_completed(task_id)

    except Exception as e:
        import traceback
        error_detail = f"Error retrieving documents: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_detail)

        # Update job with error
        job_tracker.update_job_status(
            job_id,
            JobStatus.FAILED,
            error=error_detail
        )

        # Re-raise for dramatiq retry mechanism
        raise


@dramatiq.actor(
    queue_name="cpu_tasks",  # Start on CPU for orchestration
    max_retries=3,
    store_results=True
)
def process_query_request(job_id: str, query: str, metadata_filter: Optional[Dict] = None):
    """
    Process a query request by coordinating multiple stages across different workers.

    This actor orchestrates the following sequence:
    1. Document retrieval (performed by worker-gpu-embedding)
    2. Document reranking (performed by worker-gpu-inference)
    3. LLM inference (performed by worker-gpu-inference)
    """
    try:
        # Update job status to show we're starting
        job_tracker.update_job_status(
            job_id,
            JobStatus.PROCESSING,
            result={"message": "Query processing started"},
            stage="query_processing"
        )

        # Step 1: Submit retrieval job to the GPU embedding worker
        retrieval_job_id = f"{job_id}_retrieval"

        # Create a sub-job for retrieval
        job_tracker.create_job(
            job_id=retrieval_job_id,
            job_type="document_retrieval",
            metadata={
                "parent_job_id": job_id,
                "query": query,
                "metadata_filter": metadata_filter
            }
        )

        # Send to retrieve_documents actor (on embedding_tasks queue)
        # Use the retrieve_documents actor defined above in the same file
        retrieve_result = retrieve_documents.send(retrieval_job_id, query, metadata_filter)

        # Wait for retrieval to complete
        retrieval_result = retrieve_result.get_result(block=True, timeout=120)  # 2-minute timeout

        if not retrieval_result or "documents" not in retrieval_result:
            raise ValueError("Document retrieval failed")

        documents = retrieval_result["documents"]

        # Step 2: Submit to LLM inference (which includes reranking)
        # Now send to LLM inference worker
        perform_llm_inference.send(job_id, query, documents, metadata_filter)

        # The job is now being processed by perform_llm_inference, so we're done here
        logger.info(f"Query {job_id} passed to LLM inference worker after retrieval")

        return {
            "message": "Query processing handed off to LLM inference worker",
            "document_count": len(documents)
        }

    except Exception as e:
        import traceback
        error_detail = f"Error processing query: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_detail)

        # Update job with error
        job_tracker.update_job_status(
            job_id,
            JobStatus.FAILED,
            error=error_detail
        )

        # Re-raise for dramatiq retry mechanism
        raise
