"""
Embedding generation tasks - Extracted from JobChain
Handles document embedding and vector storage
"""

import time
import logging
from typing import Dict, List
import dramatiq

from core.orchestration.job_tracker import job_tracker
from core.orchestration.job_chain import job_chain

logger = logging.getLogger(__name__)


@dramatiq.actor(queue_name="embedding_tasks", store_results=True, max_retries=2)
def generate_embeddings_task(job_id: str, documents: List[Dict]):
    """Generate embeddings - Unicode cleaning happens automatically!"""
    try:
        logger.info(f"Generating embeddings for job {job_id}: {len(documents)} documents")

        from core.background.models import get_vector_store
        from langchain_core.documents import Document

        # Validate documents exist
        if not documents:
            error_msg = f"No documents provided for embedding generation in job {job_id}"
            logger.error(error_msg)
            job_chain.task_failed(job_id, error_msg)
            return

        # Convert back to Document objects
        doc_objects = []
        for doc_dict in documents:
            if not doc_dict.get("content") or not doc_dict.get("metadata"):
                error_msg = f"Invalid document structure in job {job_id} - missing content or metadata"
                logger.error(error_msg)
                job_chain.task_failed(job_id, error_msg)
                return

            doc = Document(
                page_content=doc_dict["content"],
                metadata=doc_dict["metadata"]
            )
            doc_objects.append(doc)

        # Log sample for verification
        if doc_objects:
            sample_doc = doc_objects[0]
            logger.info(f"Sample document for job {job_id}")
            logger.info(f"  Sample title: {sample_doc.metadata.get('title', 'NO_TITLE')}")
            logger.info(f"  Sample author: {sample_doc.metadata.get('author', 'NO_AUTHOR')}")
            logger.info(f"  Total documents: {len(doc_objects)}")

        # Get existing job data for context
        current_job = job_tracker.get_job(job_id, include_progress=False)
        existing_result = current_job.get("result", {}) if current_job else {}

        if isinstance(existing_result, str):
            try:
                import json
                existing_result = json.loads(existing_result)
            except:
                existing_result = {}

        # Add ingestion timestamp and job ID to ALL documents
        current_time = time.time()
        for doc in doc_objects:
            # Ensure job_id is always present
            doc.metadata["job_id"] = job_id

            # Add ingestion timestamp if not present
            if "ingestion_time" not in doc.metadata:
                doc.metadata["ingestion_time"] = current_time

            # Ensure document has an ID for proper indexing
            if "id" not in doc.metadata or not doc.metadata["id"]:
                doc.metadata["id"] = f"doc-{job_id}-{len(doc_objects)}-{int(current_time)}"

        # Add to vector store using preloaded embedding model
        vector_store = get_vector_store()
        doc_ids = vector_store.add_documents(doc_objects)

        if not doc_ids:
            error_msg = f"Vector store failed to generate document IDs for job {job_id}"
            logger.error(error_msg)
            job_chain.task_failed(job_id, error_msg)
            return

        logger.info(f"Successfully added {len(doc_ids)} documents to vector store")

        # Create final result while PRESERVING ALL existing data
        final_result = {}
        final_result.update(existing_result)

        # Add the new embedding data
        final_result.update({
            "document_ids": doc_ids,
            "document_count": len(doc_ids),
            "embedding_completed_at": time.time(),
            "ingestion_completed": True
        })

        logger.info(f"Embedding generation completed for job {job_id}")

        # Complete the job (this is the final step for most processing jobs)
        job_chain.task_completed(job_id, final_result)

    except Exception as e:
        error_msg = f"Embedding generation failed for job {job_id}: {str(e)}"
        logger.error(error_msg)
        job_chain.task_failed(job_id, error_msg)


def start_embedding_generation(job_id: str, data: Dict):
    """
    Start embedding generation workflow

    Args:
        job_id: Job identifier
        data: Job data containing documents to embed
    """
    logger.info(f"Starting embedding generation workflow for job {job_id}")

    # Validate required data
    if "documents" not in data:
        error_msg = "documents required for embedding generation"
        logger.error(error_msg)
        job_chain.task_failed(job_id, error_msg)
        return

    # Start the embedding generation task
    generate_embeddings_task.send(job_id, data["documents"])