"""
WorkflowService - Complex workflow orchestration logic
Handles complex business logic that would make models fat
Called by WorkflowModel for complex orchestration scenarios
"""

import logging
import time
from typing import Dict, List, Any, Optional
from datetime import datetime

from src.models.workflow_models import WorkflowType, WorkflowInstance
from src.core.orchestration.queue_manager import queue_manager
from src.core.orchestration.job_tracker import job_tracker

logger = logging.getLogger(__name__)


class WorkflowService:
    """
    Service for complex workflow orchestration logic
    Implements anti-fat model pattern by handling complex business logic
    """

    def __init__(self, task_service, document_service, query_service, causation_service=None):
        self.task_service = task_service
        self.document_service = document_service
        self.query_service = query_service
        self.causation_service = causation_service

    async def orchestrate_complex_workflow(self, workflow_instance: WorkflowInstance,
                                           workflow_config: Dict[str, Any]) -> Dict[str, Any]:
        """Orchestrate complex multi-step workflows with conditional logic"""

        logger.info(f"Orchestrating complex workflow {workflow_instance.workflow_id}")

        try:
            # Prepare workflow execution context
            execution_context = await self._prepare_execution_context(
                workflow_instance, workflow_config
            )

            # Execute workflow based on type
            if workflow_instance.workflow_type == WorkflowType.QUERY_PROCESSING:
                result = await self._orchestrate_query_workflow(workflow_instance, execution_context)
            elif workflow_instance.workflow_type == WorkflowType.VIDEO_PROCESSING:
                result = await self._orchestrate_video_workflow(workflow_instance, execution_context)
            elif workflow_instance.workflow_type == WorkflowType.DOCUMENT_PROCESSING:
                result = await self._orchestrate_document_workflow(workflow_instance, execution_context)
            elif workflow_instance.workflow_type == WorkflowType.CAUSATION_ANALYSIS:
                result = await self._orchestrate_causation_workflow(workflow_instance, execution_context)
            else:
                raise ValueError(f"Unsupported workflow type: {workflow_instance.workflow_type}")

            # Post-process results
            final_result = await self._post_process_workflow_result(
                workflow_instance, result, execution_context
            )

            logger.info(f"Completed complex workflow orchestration for {workflow_instance.workflow_id}")
            return final_result

        except Exception as e:
            logger.error(f"Error orchestrating workflow {workflow_instance.workflow_id}: {str(e)}")
            await self._handle_workflow_error(workflow_instance, str(e))
            raise

    async def _prepare_execution_context(self, workflow_instance: WorkflowInstance,
                                         config: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare execution context with all necessary data and configurations"""

        context = {
            "workflow_id": workflow_instance.workflow_id,
            "workflow_type": workflow_instance.workflow_type.value,
            "start_time": time.time(),
            "config": config,
            "input_data": workflow_instance.input_data.copy(),
            "metadata": workflow_instance.metadata.copy(),
            "execution_steps": [],
            "performance_metrics": {},
            "error_recovery": {
                "max_retries": config.get("max_retries", 3),
                "retry_count": 0,
                "timeout_seconds": config.get("timeout_seconds", 1800)
            }
        }

        # Add workflow-specific context
        if workflow_instance.workflow_type == WorkflowType.QUERY_PROCESSING:
            context.update(await self._prepare_query_context(workflow_instance))
        elif workflow_instance.workflow_type == WorkflowType.VIDEO_PROCESSING:
            context.update(await self._prepare_video_context(workflow_instance))
        elif workflow_instance.workflow_type == WorkflowType.DOCUMENT_PROCESSING:
            context.update(await self._prepare_document_context(workflow_instance))

        return context

    async def _prepare_query_context(self, workflow_instance: WorkflowInstance) -> Dict[str, Any]:
        """Prepare context specific to query processing workflows"""

        input_data = workflow_instance.input_data

        return {
            "query_mode": input_data.get("query_mode", "facts"),
            "validation_enabled": bool(input_data.get("validation_config")),
            "retrieval_config": {
                "top_k": input_data.get("top_k", 10),
                "metadata_filter": input_data.get("metadata_filter"),
                "include_sources": input_data.get("include_sources", True)
            },
            "response_config": {
                "format": input_data.get("response_format", "markdown"),
                "template": input_data.get("prompt_template")
            },
            "performance_targets": {
                "max_response_time": 30,  # seconds
                "min_confidence": 0.7
            }
        }

    async def _prepare_video_context(self, workflow_instance: WorkflowInstance) -> Dict[str, Any]:
        """Prepare context specific to video processing workflows"""

        input_data = workflow_instance.input_data

        return {
            "video_url": input_data.get("url"),
            "platform": input_data.get("platform", "youtube"),
            "processing_config": {
                "extract_audio": True,
                "generate_transcript": True,
                "extract_metadata": True,
                "create_embeddings": True
            },
            "quality_settings": {
                "audio_quality": "high",
                "transcript_language": input_data.get("language", "auto")
            }
        }

    async def _prepare_document_context(self, workflow_instance: WorkflowInstance) -> Dict[str, Any]:
        """Prepare context specific to document processing workflows"""

        input_data = workflow_instance.input_data

        return {
            "document_type": input_data.get("document_type", "pdf"),
            "source_path": input_data.get("file_path"),
            "content": input_data.get("content"),
            "processing_config": {
                "extract_text": True,
                "extract_metadata": True,
                "create_chunks": True,
                "generate_embeddings": True
            },
            "quality_settings": {
                "ocr_enabled": True,
                "language_detection": True,
                "content_enhancement": True
            }
        }

    async def _orchestrate_query_workflow(self, workflow_instance: WorkflowInstance,
                                          context: Dict[str, Any]) -> Dict[str, Any]:
        """Orchestrate query processing workflow with complex validation logic"""

        query = context["input_data"]["query"]
        query_mode = context["query_mode"]

        logger.info(f"Orchestrating query workflow for: {query[:50]}...")

        # Step 1: Document retrieval with optimization
        retrieval_start = time.time()
        documents = await self.query_service.retrieve_documents(
            query=query,
            config=context["retrieval_config"],
            optimization_hints=self._get_retrieval_optimization_hints(query_mode)
        )

        context["execution_steps"].append({
            "step": "document_retrieval",
            "duration": time.time() - retrieval_start,
            "documents_found": len(documents),
            "success": True
        })

        # Step 2: Document quality assessment and filtering
        quality_start = time.time()
        filtered_documents = await self.document_service.assess_and_filter_documents(
            documents=documents,
            query=query,
            quality_threshold=context.get("performance_targets", {}).get("min_confidence", 0.7)
        )

        context["execution_steps"].append({
            "step": "document_filtering",
            "duration": time.time() - quality_start,
            "documents_after_filter": len(filtered_documents),
            "success": True
        })

        # Step 3: LLM inference with mode-specific processing
        inference_start = time.time()
        answer_data = await self.query_service.generate_answer(
            query=query,
            documents=filtered_documents,
            query_mode=query_mode,
            config=context["response_config"]
        )

        context["execution_steps"].append({
            "step": "llm_inference",
            "duration": time.time() - inference_start,
            "answer_length": len(answer_data.get("answer", "")),
            "success": True
        })

        # Step 4: Response post-processing and validation
        processing_start = time.time()
        final_response = await self.query_service.post_process_response(
            answer_data=answer_data,
            documents=filtered_documents,
            query=query,
            query_mode=query_mode
        )

        context["execution_steps"].append({
            "step": "response_processing",
            "duration": time.time() - processing_start,
            "success": True
        })

        return {
            "query": query,
            "answer": final_response["answer"],
            "documents": final_response["documents"],
            "query_mode": query_mode,
            "metadata": final_response.get("metadata", {}),
            "execution_context": context
        }

    async def _orchestrate_video_workflow(self, workflow_instance: WorkflowInstance,
                                          context: Dict[str, Any]) -> Dict[str, Any]:
        """Orchestrate video processing workflow with quality control"""

        video_url = context["video_url"]

        logger.info(f"Orchestrating video workflow for: {video_url}")

        # Step 1: Video download and validation
        download_start = time.time()
        video_data = await self.document_service.download_and_validate_video(
            url=video_url,
            platform=context["platform"],
            quality_settings=context["quality_settings"]
        )

        context["execution_steps"].append({
            "step": "video_download",
            "duration": time.time() - download_start,
            "file_size": video_data.get("file_size", 0),
            "success": True
        })

        # Step 2: Audio extraction and transcription
        transcription_start = time.time()
        transcript_data = await self.document_service.extract_and_transcribe_audio(
            video_path=video_data["file_path"],
            config=context["processing_config"]
        )

        context["execution_steps"].append({
            "step": "transcription",
            "duration": time.time() - transcription_start,
            "transcript_length": len(transcript_data.get("transcript", "")),
            "confidence": transcript_data.get("confidence", 0.0),
            "success": True
        })

        # Step 3: Content processing and indexing
        indexing_start = time.time()
        indexed_content = await self.document_service.process_and_index_content(
            transcript=transcript_data["transcript"],
            metadata=video_data["metadata"],
            config=context["processing_config"]
        )

        context["execution_steps"].append({
            "step": "content_indexing",
            "duration": time.time() - indexing_start,
            "chunks_created": indexed_content.get("chunk_count", 0),
            "success": True
        })

        return {
            "video_url": video_url,
            "document_id": indexed_content["document_id"],
            "transcript": transcript_data["transcript"],
            "metadata": video_data["metadata"],
            "processing_stats": indexed_content["stats"],
            "execution_context": context
        }

    async def _orchestrate_document_workflow(self, workflow_instance: WorkflowInstance,
                                             context: Dict[str, Any]) -> Dict[str, Any]:
        """Orchestrate document processing workflow"""

        document_type = context["document_type"]

        logger.info(f"Orchestrating document workflow for {document_type}")

        # Step 1: Document parsing and text extraction
        parsing_start = time.time()
        if context.get("source_path"):
            parsed_content = await self.document_service.parse_document_file(
                file_path=context["source_path"],
                document_type=document_type,
                config=context["processing_config"]
            )
        else:
            parsed_content = await self.document_service.parse_text_content(
                content=context["content"],
                config=context["processing_config"]
            )

        context["execution_steps"].append({
            "step": "document_parsing",
            "duration": time.time() - parsing_start,
            "content_length": len(parsed_content.get("text", "")),
            "success": True
        })

        # Step 2: Content enhancement and metadata extraction
        enhancement_start = time.time()
        enhanced_content = await self.document_service.enhance_content(
            text=parsed_content["text"],
            metadata=parsed_content["metadata"],
            config=context["quality_settings"]
        )

        context["execution_steps"].append({
            "step": "content_enhancement",
            "duration": time.time() - enhancement_start,
            "entities_extracted": len(enhanced_content.get("entities", [])),
            "success": True
        })

        # Step 3: Indexing and embedding generation
        indexing_start = time.time()
        indexed_content = await self.document_service.process_and_index_content(
            text=enhanced_content["text"],
            metadata=enhanced_content["metadata"],
            config=context["processing_config"]
        )

        context["execution_steps"].append({
            "step": "content_indexing",
            "duration": time.time() - indexing_start,
            "chunks_created": indexed_content.get("chunk_count", 0),
            "success": True
        })

        return {
            "document_type": document_type,
            "document_id": indexed_content["document_id"],
            "text_content": enhanced_content["text"],
            "metadata": enhanced_content["metadata"],
            "processing_stats": indexed_content["stats"],
            "execution_context": context
        }

    async def _orchestrate_causation_workflow(self, workflow_instance: WorkflowInstance,
                                              context: Dict[str, Any]) -> Dict[str, Any]:
        """Orchestrate causation analysis workflow (placeholder for future)"""

        if not self.causation_service:
            raise ValueError("Causation analysis service not available")

        logger.info(f"Orchestrating causation workflow (placeholder)")

        # Placeholder implementation
        return {
            "analysis_type": "placeholder",
            "status": "not_implemented",
            "message": "Causation analysis workflow will be implemented in the future",
            "execution_context": context
        }

    def _get_retrieval_optimization_hints(self, query_mode: str) -> Dict[str, Any]:
        """Get retrieval optimization hints based on query mode"""

        optimization_hints = {
            "facts": {
                "prefer_authoritative_sources": True,
                "emphasize_specifications": True,
                "boost_official_documents": 1.5
            },
            "features": {
                "prefer_comparative_content": True,
                "emphasize_reviews": True,
                "boost_feature_lists": 1.3
            },
            "tradeoffs": {
                "prefer_analytical_content": True,
                "emphasize_pros_cons": True,
                "boost_comparison_docs": 1.4
            },
            "scenarios": {
                "prefer_user_experiences": True,
                "emphasize_use_cases": True,
                "boost_scenario_docs": 1.3
            }
        }

        return optimization_hints.get(query_mode, {})

    async def _post_process_workflow_result(self, workflow_instance: WorkflowInstance,
                                            result: Dict[str, Any],
                                            context: Dict[str, Any]) -> Dict[str, Any]:
        """Post-process workflow results with quality enhancements"""

        # Calculate execution metrics
        total_duration = time.time() - context["start_time"]
        step_durations = {step["step"]: step["duration"] for step in context["execution_steps"]}

        # Add performance metrics
        result["performance_metrics"] = {
            "total_execution_time": total_duration,
            "step_durations": step_durations,
            "workflow_efficiency": self._calculate_workflow_efficiency(context),
            "quality_score": await self._calculate_quality_score(result, context)
        }

        # Add execution summary
        result["execution_summary"] = {
            "workflow_id": workflow_instance.workflow_id,
            "workflow_type": workflow_instance.workflow_type.value,
            "steps_executed": len(context["execution_steps"]),
            "total_duration": total_duration,
            "success": True,
            "timestamp": datetime.now().isoformat()
        }

        return result

    def _calculate_workflow_efficiency(self, context: Dict[str, Any]) -> float:
        """Calculate workflow execution efficiency score"""

        total_steps = len(context["execution_steps"])
        successful_steps = sum(1 for step in context["execution_steps"] if step.get("success", False))

        if total_steps == 0:
            return 0.0

        success_rate = successful_steps / total_steps

        # Factor in execution time efficiency
        total_duration = sum(step.get("duration", 0) for step in context["execution_steps"])
        expected_duration = context.get("config", {}).get("expected_duration", 60)

        time_efficiency = min(1.0, expected_duration / max(total_duration, 1))

        return (success_rate * 0.7) + (time_efficiency * 0.3)

    async def _calculate_quality_score(self, result: Dict[str, Any], context: Dict[str, Any]) -> float:
        """Calculate overall quality score for workflow output"""

        quality_factors = []

        # Content quality (if applicable)
        if "answer" in result:
            answer_length = len(result["answer"])
            if 50 <= answer_length <= 1000:  # Optimal length range
                quality_factors.append(1.0)
            elif answer_length < 50:
                quality_factors.append(0.5)
            else:
                quality_factors.append(0.8)

        # Document quality (if applicable)
        if "documents" in result:
            doc_count = len(result["documents"])
            if doc_count >= 3:  # Good document coverage
                quality_factors.append(1.0)
            elif doc_count >= 1:
                quality_factors.append(0.7)
            else:
                quality_factors.append(0.3)

        # Execution efficiency
        efficiency = self._calculate_workflow_efficiency(context)
        quality_factors.append(efficiency)

        # Default quality if no specific factors
        if not quality_factors:
            quality_factors.append(0.8)

        return sum(quality_factors) / len(quality_factors)

    async def _handle_workflow_error(self, workflow_instance: WorkflowInstance, error_message: str):
        """Handle workflow errors with recovery strategies"""

        logger.error(f"Workflow {workflow_instance.workflow_id} encountered error: {error_message}")

        # Update job tracker if available
        try:
            job_tracker.update_job_status(
                workflow_instance.workflow_id,
                "failed",
                error=error_message
            )
        except Exception as e:
            logger.warning(f"Could not update job tracker: {str(e)}")

        # Implement error recovery strategies here if needed
        # For now, just log the error

    async def coordinate_with_core_orchestration(self, workflow_instance: WorkflowInstance,
                                                 prepared_specs: Dict[str, Any]):
        """Coordinate with core orchestration components"""

        try:
            # Call core orchestration with prepared specifications
            from src.core.orchestration.job_chain import job_chain
            from src.core.orchestration.task_router import JobType

            # Map workflow type to job type
            job_type_mapping = {
                WorkflowType.QUERY_PROCESSING: JobType.LLM_INFERENCE,
                WorkflowType.VIDEO_PROCESSING: JobType.VIDEO_PROCESSING,
                WorkflowType.DOCUMENT_PROCESSING: JobType.PDF_PROCESSING,
                WorkflowType.CAUSATION_ANALYSIS: JobType.LLM_INFERENCE  # Future
            }

            job_type = job_type_mapping.get(workflow_instance.workflow_type, JobType.LLM_INFERENCE)

            # Start job chain with prepared specifications
            job_chain.start_job_chain(
                job_id=workflow_instance.workflow_id,
                job_type=job_type,
                initial_data=prepared_specs
            )

            logger.info(f"Coordinated workflow {workflow_instance.workflow_id} with core orchestration")

        except Exception as e:
            logger.error(f"Error coordinating with core orchestration: {str(e)}")
            raise

    def get_workflow_performance_metrics(self) -> Dict[str, Any]:
        """Get overall workflow performance metrics"""

        # This would typically aggregate metrics from multiple workflows
        # For now, return placeholder metrics
        return {
            "total_workflows_processed": 0,
            "average_execution_time": 0.0,
            "success_rate": 100.0,
            "efficiency_score": 0.85,
            "quality_score": 0.90,
            "workflow_types": {
                "query_processing": {"count": 0, "avg_time": 0.0},
                "video_processing": {"count": 0, "avg_time": 0.0},
                "document_processing": {"count": 0, "avg_time": 0.0},
                "causation_analysis": {"count": 0, "avg_time": 0.0}
            }
        }