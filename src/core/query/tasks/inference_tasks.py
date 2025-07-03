"""
LLM inference tasks - Extracted from JobChain
Handles query processing and answer generation
"""

import time
import logging
from typing import Dict, List, Any
import dramatiq

from core.orchestration.job_tracker import job_tracker
from core.orchestration.job_chain import job_chain
from core.query.llm.mode_config import mode_config, QueryMode

logger = logging.getLogger(__name__)


@dramatiq.actor(queue_name="inference_tasks", store_results=True, max_retries=2)
def llm_inference_task(job_id: str, query: str, documents: List[Dict], query_mode: str = "facts"):
    """
    Enhanced LLM inference with answer validation
    """
    try:
        from core.background.models import get_llm_model
        from langchain_core.documents import Document

        logger.info(f"Enhanced LLM inference + answer validation for job {job_id} in '{query_mode}' mode")

        # Get the preloaded LLM model
        llm_model = get_llm_model()

        # Convert serialized documents back to Document objects
        doc_objects = []
        relevance_scores = []

        for doc_dict in documents:
            doc = Document(
                page_content=doc_dict["content"],
                metadata=doc_dict["metadata"]
            )
            doc_objects.append(doc)

            # Extract relevance score
            relevance_score = doc_dict.get("relevance_score", 0.0)
            relevance_scores.append(relevance_score)

        # Create documents with scores for LLM processing
        documents_with_scores = list(zip(doc_objects, relevance_scores))

        # Calculate average relevance for confidence assessment
        avg_relevance_score = sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0.0

        logger.info(f"Processing {len(documents)} documents with avg relevance: {avg_relevance_score:.3f}")

        # Parse query mode for mode-specific parameters
        try:
            mode_enum = QueryMode(query_mode)
        except ValueError:
            mode_enum = QueryMode.FACTS

        # Get mode-specific LLM parameters
        llm_params = mode_config.get_llm_params(mode_enum)

        # Generate answer with mode-specific parameters
        base_answer = llm_model.answer_query_with_mode_specific_params(
            query=query,
            documents=documents_with_scores,
            query_mode=query_mode,
            temperature=llm_params["temperature"],
            max_tokens=llm_params["max_tokens"],
            top_p=llm_params.get("top_p", 0.85),
            repetition_penalty=llm_params.get("repetition_penalty", 1.1)
        )

        # Apply answer validation if available
        answer_validation = {}
        validation_confidence = 0
        final_answer = base_answer

        try:
            from core.validation.validation_engine import validation_engine

            logger.info("Applying enhanced answer validation...")

            # Convert documents to validation format
            doc_list = []
            for doc in doc_objects:
                doc_dict = {
                    "content": doc.page_content,
                    "metadata": doc.metadata
                }
                doc_list.append(doc_dict)

            # Validate answer using the validation framework
            answer_validation = await validation_engine.validate_answer(
                answer=base_answer,
                documents=doc_list,
                query=query,
                query_mode=query_mode,
                job_id=job_id
            )

            # Format validation warnings for user display
            warning_footnotes = validation_engine.format_automotive_warnings_for_user(answer_validation)

            # Create final answer with validation footnotes
            final_answer = base_answer + warning_footnotes
            validation_confidence = answer_validation.get("confidence_score", 0)

            logger.info("✅ Answer validation applied successfully")

        except ImportError:
            logger.info("Validation framework not available, proceeding without answer validation")
        except Exception as e:
            logger.warning(f"Answer validation failed: {str(e)}, proceeding without validation")

        # Enhanced confidence assessment
        simple_confidence = min(100, avg_relevance_score * 50 + validation_confidence * 0.5)

        # Get current job data
        current_job = job_tracker.get_job(job_id, include_progress=False)
        existing_result = current_job.get("result", {}) if current_job else {}

        if isinstance(existing_result, str):
            try:
                import json
                existing_result = json.loads(existing_result)
            except:
                existing_result = {}

        # Get validation result from retrieval step
        retrieval_validation = existing_result.get("validation_result", {})

        # Create comprehensive inference result
        inference_result = {
            "answer": final_answer,
            "base_answer": base_answer,  # Answer without validation footnotes
            "query": query,
            "query_mode": query_mode,
            "inference_completed_at": time.time(),
            "llm_parameters_used": llm_params,

            # Enhanced confidence metrics
            "simple_confidence": simple_confidence,
            "avg_relevance_score": avg_relevance_score,
            "documents_used": len(documents),

            # Legacy compatibility
            "automotive_validation": answer_validation,
            "enhanced_with_validation_framework": bool(answer_validation)
        }

        # Add comprehensive validation integration if available
        if answer_validation:
            inference_result["enhanced_validation"] = {
                "answer_validation": answer_validation,
                "retrieval_validation": retrieval_validation,
                "overall_confidence": validation_confidence,
                "confidence_level": answer_validation.get("automotive_confidence", "unknown"),
                "validation_warnings": answer_validation.get("answer_warnings", []),
                "source_warnings": answer_validation.get("source_warnings", []),

                # Trust trail information
                "trust_trail_available": bool(retrieval_validation.get("trust_trail")),
                "validation_steps_completed": len(retrieval_validation.get("validation_steps", [])),
                "pipeline_type": retrieval_validation.get("pipeline_type", "unknown"),

                # User guidance
                "contribution_opportunities": retrieval_validation.get("contribution_opportunities", []),
                "guided_trust_loop_enabled": True,

                # Validation transparency
                "validation_methodology": {
                    "retrieval_pipeline": retrieval_validation.get("pipeline_type"),
                    "answer_validation_applied": True,
                    "multi_step_validation": True,
                    "meta_validation_enabled": True,
                    "user_contribution_enabled": True
                }
            }

        # Preserve existing data and add inference result
        final_result = {}
        final_result.update(existing_result)
        final_result.update(inference_result)

        logger.info(f"Enhanced LLM inference completed for job {job_id}")
        logger.info(f"Simple confidence: {simple_confidence:.1f}%, Validation confidence: {validation_confidence:.1f}%")

        if answer_validation.get("answer_warnings"):
            logger.info(f"Answer warnings: {len(answer_validation['answer_warnings'])}")
        if answer_validation.get("source_warnings"):
            logger.info(f"Source warnings: {len(answer_validation['source_warnings'])}")

        # Complete the job
        job_chain.task_completed(job_id, final_result)

    except Exception as e:
        error_msg = f"Enhanced LLM inference failed for job {job_id}: {str(e)}"
        logger.error(error_msg)
        job_chain.task_failed(job_id, error_msg)


@dramatiq.actor(queue_name="inference_tasks", store_results=True, max_retries=2)
def process_user_contribution_task(job_id: str, step_type: str, contribution_data: Dict[str, Any]):
    """
    Process user contribution and retry validation (Guided Trust Loop)
    """
    try:
        logger.info(f"Processing user contribution for job {job_id}, step {step_type}")

        # Process contribution using validation engine if available
        try:
            from core.validation.validation_engine import validation_engine

            contribution_result = await validation_engine.process_user_contribution(
                job_id=job_id,
                step_type=step_type,
                contribution_data=contribution_data
            )

            if contribution_result.get("success"):
                # Update job with new validation results
                new_confidence = contribution_result.get("new_confidence", 0)
                learning_credit = contribution_result.get("learning_credit")

                # Get current job data
                current_job = job_tracker.get_job(job_id, include_progress=False)
                existing_result = current_job.get("result", {}) if current_job else {}

                if isinstance(existing_result, str):
                    try:
                        import json
                        existing_result = json.loads(existing_result)
                    except:
                        existing_result = {}

                # Update validation results
                validation_update = {
                    "contribution_processed": True,
                    "contribution_accepted": True,
                    "updated_confidence": new_confidence,
                    "learning_credit_earned": learning_credit,
                    "contribution_timestamp": time.time(),
                    "updated_validation": contribution_result.get("validation_updated", False)
                }

                # Update enhanced validation section
                if "enhanced_validation" in existing_result:
                    existing_result["enhanced_validation"]["user_contributions"] = existing_result[
                        "enhanced_validation"].get("user_contributions", [])
                    existing_result["enhanced_validation"]["user_contributions"].append(validation_update)
                    existing_result["enhanced_validation"]["overall_confidence"] = new_confidence

                # Update job
                job_tracker.update_job_status(
                    job_id,
                    "completed",  # Job remains completed, just updated
                    result=existing_result,
                    stage="contribution_processed",
                    replace_result=True
                )

                logger.info(f"User contribution processed successfully for job {job_id}")

            else:
                error_msg = contribution_result.get("error", "Unknown error processing contribution")
                logger.error(f"Contribution processing failed for job {job_id}: {error_msg}")

        except ImportError:
            logger.warning("Validation framework not available for user contribution processing")
        except Exception as e:
            logger.error(f"Error processing user contribution: {str(e)}")

    except Exception as e:
        error_msg = f"Contribution processing task failed for job {job_id}: {str(e)}"
        logger.error(error_msg)


def start_llm_inference(job_id: str, data: Dict):
    """
    Start LLM inference workflow

    Args:
        job_id: Job identifier
        data: Job data containing query, documents, and query_mode
    """
    logger.info(f"Starting LLM inference workflow for job {job_id}")

    # Validate required data
    if "query" not in data:
        error_msg = "query required for LLM inference"
        logger.error(error_msg)
        job_chain.task_failed(job_id, error_msg)
        return

    if "documents" not in data:
        error_msg = "documents required for LLM inference"
        logger.error(error_msg)
        job_chain.task_failed(job_id, error_msg)
        return

    # Start the LLM inference task
    llm_inference_task.send(
        job_id,
        data["query"],
        data["documents"],
        data.get("query_mode", "facts")
    )