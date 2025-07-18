"""
Document retrieval tasks - Extracted from JobChain
Handles document search and retrieval workflows
"""

import time
import logging
from typing import Dict, Optional
import dramatiq
import numpy as np

from src.core.orchestration.job_tracker import job_tracker
from src.core.orchestration.job_chain import job_chain
from src.core.query.llm.mode_config import mode_config, QueryMode
from src.core.orchestration.queue_manager import queue_manager, QueueNames

logger = logging.getLogger(__name__)


def apply_token_budget_management(documents, token_budget: int):
    """Apply token budget limits while preserving highest similarity documents."""
    if not documents:
        return documents

    from src.core.query.llm.mode_config import estimate_token_count

    selected_docs = []
    total_tokens = 0

    for doc, score in documents:
        # Estimate tokens for this document
        content_tokens = estimate_token_count(doc.page_content)

        # Check if adding this doc would exceed budget
        if total_tokens + content_tokens > token_budget and selected_docs:
            break  # Stop if we'd exceed budget (but keep at least one doc)

        # Add the document
        selected_docs.append((doc, score))
        total_tokens += content_tokens

        # Safety limit
        if len(selected_docs) >= 12:  # Reasonable upper limit
            break

    logger.info(f"Token budget: {total_tokens}/{token_budget} tokens used")
    return selected_docs


@queue_manager.create_task_decorator(QueueNames.EMBEDDING_TASKS.value)
def retrieve_documents_task(job_id: str, query: str, metadata_filter: Optional[Dict] = None, query_mode: str = "facts"):
    """
    Enhanced document retrieval with full validation pipeline
    """
    try:
        from src.core.background.models import get_vector_store

        logger.info(f"Enhanced retrieval + validation for job {job_id} in '{query_mode}' mode: {query}")

        vector_store = get_vector_store()

        # Parse query mode
        try:
            mode_enum = QueryMode(query_mode)
        except ValueError:
            logger.warning(f"Invalid query mode '{query_mode}', falling back to facts")
            mode_enum = QueryMode.FACTS

        # Get mode-specific parameters
        retrieval_params = mode_config.get_retrieval_params(mode_enum)
        context_params = mode_config.get_context_params(mode_enum)

        # Initial document retrieval
        initial_k = retrieval_params["retrieval_k"]
        results = vector_store.similarity_search_with_score(
            query=query,
            k=initial_k,
            metadata_filter=metadata_filter
        )

        logger.info(f"Initial retrieval returned {len(results)} results")

        if not results:
            logger.warning(f"No documents found for query: {query}")
            job_chain.task_completed(job_id, {
                "documents": [],
                "document_count": 0,
                "retrieval_completed_at": time.time(),
                "retrieval_method": f"enhanced_validation_{query_mode}",
                "query_used": query,
                "query_mode": query_mode,
                "validation_result": None
            })
            return

        # Apply validation pipeline if available
        validation_result = None
        try:
            from core.validation.validation_engine import validation_engine

            # Convert to validation framework format
            documents_for_validation = []
            for doc, score in results:
                json_safe_score = float(score) if isinstance(score, (np.floating, np.float32, np.float64)) else score

                cleaned_metadata = {}
                for key, value in doc.metadata.items():
                    if isinstance(value, (np.floating, np.float32, np.float64)):
                        cleaned_metadata[key] = float(value)
                    elif isinstance(value, (np.integer, np.int32, np.int64)):
                        cleaned_metadata[key] = int(value)
                    elif isinstance(value, np.ndarray):
                        cleaned_metadata[key] = value.tolist()
                    else:
                        cleaned_metadata[key] = value

                documents_for_validation.append({
                    "content": doc.page_content,
                    "metadata": cleaned_metadata,
                    "relevance_score": json_safe_score
                })

            # Apply full validation pipeline
            validation_result = validation_engine.validate_documents(
                documents=documents_for_validation,
                query=query,
                query_mode=query_mode,
                metadata_filter=metadata_filter,
                job_id=job_id
            )
            logger.info("✅ Validation pipeline applied successfully")

        except ImportError:
            logger.info("Validation framework not available, proceeding without validation")
        except Exception as e:
            logger.warning(f"Validation pipeline failed: {str(e)}, proceeding without validation")

        # Apply token budget management to documents
        token_budget = context_params["max_context_tokens"]
        final_documents = apply_token_budget_management(results, token_budget)

        # Calculate enhanced statistics
        if final_documents:
            relevance_scores = [score for _, score in final_documents]
            avg_relevance_score = sum(relevance_scores) / len(relevance_scores)
        else:
            avg_relevance_score = 0.0

        # Format results for transfer to inference worker
        serialized_docs = []
        for doc, score in final_documents:
            json_safe_score = float(score) if isinstance(score, (np.floating, np.float32, np.float64)) else score

            cleaned_metadata = {}
            for key, value in doc.metadata.items():
                if isinstance(value, (np.floating, np.float32, np.float64)):
                    cleaned_metadata[key] = float(value)
                elif isinstance(value, (np.integer, np.int32, np.int64)):
                    cleaned_metadata[key] = int(value)
                elif isinstance(value, np.ndarray):
                    cleaned_metadata[key] = value.tolist()
                else:
                    cleaned_metadata[key] = value

            # Add validation metadata if available
            if validation_result:
                cleaned_metadata["validation_status"] = "validated"
                cleaned_metadata["validation_confidence"] = validation_result.confidence.total_score

            serialized_docs.append({
                "content": doc.page_content,
                "metadata": cleaned_metadata,
                "relevance_score": json_safe_score
            })

        logger.info(f"Document retrieval completed for job {job_id}: {len(final_documents)} documents")
        if validation_result:
            logger.info(f"Validation confidence: {validation_result.confidence.total_score:.1f}%")

        # Build comprehensive result
        result = {
            "documents": serialized_docs,
            "document_count": len(serialized_docs),
            "retrieval_completed_at": time.time(),
            "retrieval_method": f"enhanced_validation_{query_mode}",
            "query_mode": query_mode,
            "avg_relevance_score": avg_relevance_score,
            "token_budget_used": token_budget,
            "enhanced_with_validation_framework": bool(validation_result)
        }

        # Add validation results if available
        if validation_result:
            result["validation_result"] = {
                "chain_id": validation_result.chain_id,
                "pipeline_type": validation_result.pipeline_type.value,
                "overall_status": validation_result.overall_status.value,
                "confidence": {
                    "total_score": validation_result.confidence.total_score,
                    "level": validation_result.confidence.level.value,
                    "source_credibility": validation_result.confidence.source_credibility,
                    "technical_consistency": validation_result.confidence.technical_consistency,
                    "completeness": validation_result.confidence.completeness,
                    "consensus": validation_result.confidence.consensus,
                    "verification_coverage": validation_result.confidence.verification_coverage
                },
                "validation_steps": [
                    {
                        "step_name": step.step_name,
                        "status": step.status.value,
                        "confidence_impact": step.confidence_impact,
                        "summary": step.summary,
                        "warnings": [
                            {
                                "category": w.category,
                                "severity": w.severity,
                                "message": w.message,
                                "explanation": w.explanation,
                                "suggestion": w.suggestion
                            }
                            for w in step.warnings
                        ]
                    }
                    for step in validation_result.validation_steps
                ],
                "trust_trail": validation_result.step_progression,
                "contribution_opportunities": [
                    {
                        "needed_resource": prompt.needed_resource_type,
                        "description": prompt.specific_need_description,
                        "confidence_impact": prompt.confidence_impact,
                        "future_benefit": prompt.future_benefit_description
                    }
                    for prompt in validation_result.contribution_opportunities
                ]
            }

        # Complete the task
        job_chain.task_completed(job_id, result)

    except Exception as e:
        logger.error(f"Enhanced retrieval failed for job {job_id}: {str(e)}")
        job_chain.task_failed(job_id, f"Enhanced retrieval failed: {str(e)}")


def start_document_retrieval(job_id: str, data: Dict):
    """
    Start document retrieval workflow

    Args:
        job_id: Job identifier
        data: Job data containing query, metadata_filter, and query_mode
    """
    logger.info(f"Starting document retrieval workflow for job {job_id}")

    # Validate required data
    if "query" not in data:
        error_msg = "query required for document retrieval"
        logger.error(error_msg)
        job_chain.task_failed(job_id, error_msg)
        return

    # Start the document retrieval task
    retrieve_documents_task.send(
        job_id,
        data["query"],
        data.get("metadata_filter"),
        data.get("query_mode", "facts")
    )