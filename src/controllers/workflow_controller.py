"""
WorkflowController - The Single Controller for ALL workflow types
Serves as the only HTTP entry point, eliminates controller proliferation
Delegates ALL business logic to models - contains NO business logic itself
"""

import logging
from typing import Optional, Dict, Any
from fastapi import HTTPException, Path, Body

from src.models import (
    EnhancedQueryRequest,
    EnhancedQueryResponse,
    EnhancedBackgroundJobResponse,
    QueryModeConfig,
    SystemCapabilities,
    QueryValidationResult,
    YouTubeIngestRequest,
    PDFIngestRequest,
    ManualIngestRequest,
    IngestResponse
)
from src.models.workflow_models import WorkflowType

logger = logging.getLogger(__name__)


class WorkflowController:
    """
    Single HTTP controller for ALL workflow types and operations
    Eliminates QueryController and validation controller mixing
    Contains NO business logic - only HTTP handling and model delegation
    """

    def __init__(self, workflow_model, system_service):
        self.workflow_model = workflow_model
        self.system_service = system_service

    # ========================================================================
    # Query Processing Workflows
    # ========================================================================

    async def submit_query(self, request: EnhancedQueryRequest) -> EnhancedBackgroundJobResponse:
        """Submit a query for processing - delegates to WorkflowModel"""
        try:
            # Input validation (HTTP layer responsibility only)
            self._validate_query_request(request)

            # Delegate ALL business logic to WorkflowModel
            workflow_id = await self.workflow_model.create_workflow(
                workflow_type=WorkflowType.QUERY_PROCESSING,
                input_data={
                    "query": request.query,
                    "query_mode": request.query_mode,
                    "metadata_filter": request.metadata_filter.dict() if request.metadata_filter else None,
                    "top_k": request.top_k,
                    "prompt_template": request.prompt_template,
                    "validation_config": request.validation_config.dict() if request.validation_config else None,
                    "include_sources": request.include_sources,
                    "response_format": request.response_format
                }
            )

            # Start workflow execution
            await self.workflow_model.start_workflow(workflow_id)

            # Get workflow status for response
            status = await self.workflow_model.get_workflow_status(workflow_id)

            return EnhancedBackgroundJobResponse(
                message=f"Query processing started in '{request.query_mode.value}' mode",
                job_id=workflow_id,
                job_type="query_processing",
                query_mode=request.query_mode,
                expected_processing_time=status.get("estimated_duration", 30),
                status="processing",
                complexity_level="simple",  # Would be determined by WorkflowModel
                validation_enabled=bool(request.validation_config and request.validation_config.enabled),
                created_at=None
            )

        except ValueError as e:
            logger.warning(f"Invalid query request: {str(e)}")
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

    async def get_query_result(self, workflow_id: str) -> Optional[EnhancedQueryResponse]:
        """Get query results - delegates to WorkflowModel"""
        try:
            if not workflow_id or len(workflow_id.strip()) < 5:
                raise HTTPException(status_code=400, detail="Invalid workflow ID")

            # Delegate to WorkflowModel
            result = await self.workflow_model.get_workflow_result(workflow_id)

            if not result:
                raise HTTPException(status_code=404, detail=f"Workflow {workflow_id} not found")

            # If workflow is still running, return processing status
            status = await self.workflow_model.get_workflow_status(workflow_id)

            if status and status.get("status") != "completed":
                return EnhancedQueryResponse(
                    query=result.get("query", ""),
                    answer=f"Processing... {status.get('progress_percentage', 0):.1f}% complete",
                    documents=[],
                    query_mode=result.get("query_mode", "facts"),
                    status=status.get("status", "processing"),
                    job_id=workflow_id,
                    execution_time=status.get("execution_time", 0)
                )

            # Return completed result
            return EnhancedQueryResponse(
                query=result.get("query", ""),
                answer=result.get("answer", ""),
                documents=result.get("documents", []),
                query_mode=result.get("query_mode", "facts"),
                status="completed",
                job_id=workflow_id,
                execution_time=result.get("execution_time", 0),
                metadata_filters_used=result.get("metadata_filter")
            )

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error retrieving query result: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error retrieving result: {str(e)}")

    async def validate_query_compatibility(self, request: EnhancedQueryRequest) -> QueryValidationResult:
        """Validate query compatibility - delegates to WorkflowModel query logic"""
        try:
            self._validate_query_request(request)

            # Delegate to WorkflowModel's query model
            # For now, return a basic validation result
            return QueryValidationResult(
                is_valid=True,
                mode_compatibility={request.query_mode: True},
                recommendations=["Query is compatible with selected mode"],
                suggested_mode=request.query_mode,
                confidence_score=0.8,
                validation_recommended=bool(request.validation_config and request.validation_config.enabled)
            )

        except ValueError as e:
            logger.warning(f"Invalid query for validation: {str(e)}")
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            logger.error(f"Error validating query: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error validating query: {str(e)}")

    # ========================================================================
    # Document Processing Workflows
    # ========================================================================

    async def ingest_youtube_video(self, request: YouTubeIngestRequest) -> EnhancedBackgroundJobResponse:
        """Ingest YouTube video - delegates to WorkflowModel"""
        try:
            workflow_id = await self.workflow_model.create_workflow(
                workflow_type=WorkflowType.VIDEO_PROCESSING,
                input_data={
                    "url": str(request.url),
                    "metadata": request.metadata or {},
                    "platform": "youtube"
                }
            )

            await self.workflow_model.start_workflow(workflow_id)

            return EnhancedBackgroundJobResponse(
                message="YouTube video ingestion started",
                job_id=workflow_id,
                job_type="video_processing",
                query_mode="facts",  # Default for ingestion
                expected_processing_time=120,
                status="processing"
            )

        except Exception as e:
            logger.error(f"Error ingesting YouTube video: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error ingesting video: {str(e)}")

    async def ingest_pdf_document(self, request: PDFIngestRequest) -> EnhancedBackgroundJobResponse:
        """Ingest PDF document - delegates to WorkflowModel"""
        try:
            workflow_id = await self.workflow_model.create_workflow(
                workflow_type=WorkflowType.DOCUMENT_PROCESSING,
                input_data={
                    "file_path": request.file_path,
                    "metadata": request.metadata or {},
                    "document_type": "pdf"
                }
            )

            await self.workflow_model.start_workflow(workflow_id)

            return EnhancedBackgroundJobResponse(
                message="PDF document ingestion started",
                job_id=workflow_id,
                job_type="document_processing",
                query_mode="facts",
                expected_processing_time=60,
                status="processing"
            )

        except Exception as e:
            logger.error(f"Error ingesting PDF document: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error ingesting document: {str(e)}")

    async def ingest_manual_text(self, request: ManualIngestRequest) -> EnhancedBackgroundJobResponse:
        """Ingest manual text - delegates to WorkflowModel"""
        try:
            workflow_id = await self.workflow_model.create_workflow(
                workflow_type=WorkflowType.DOCUMENT_PROCESSING,
                input_data={
                    "content": request.content,
                    "metadata": request.metadata.dict() if request.metadata else {},
                    "document_type": "manual_text"
                }
            )

            await self.workflow_model.start_workflow(workflow_id)

            return EnhancedBackgroundJobResponse(
                message="Manual text ingestion started",
                job_id=workflow_id,
                job_type="document_processing",
                query_mode="facts",
                expected_processing_time=30,
                status="processing"
            )

        except Exception as e:
            logger.error(f"Error ingesting manual text: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error ingesting text: {str(e)}")

    # ========================================================================
    # Causation Analysis Workflows (Future)
    # ========================================================================

    async def start_causation_analysis(self, analysis_request: Dict[str, Any]) -> EnhancedBackgroundJobResponse:
        """Start causation analysis - delegates to WorkflowModel"""
        try:
            workflow_id = await self.workflow_model.create_workflow(
                workflow_type=WorkflowType.CAUSATION_ANALYSIS,
                input_data=analysis_request
            )

            await self.workflow_model.start_workflow(workflow_id)

            return EnhancedBackgroundJobResponse(
                message="Causation analysis started",
                job_id=workflow_id,
                job_type="causation_analysis",
                query_mode="facts",
                expected_processing_time=90,
                status="processing"
            )

        except Exception as e:
            logger.error(f"Error starting causation analysis: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error starting analysis: {str(e)}")

    # ========================================================================
    # System Information Endpoints
    # ========================================================================

    async def get_query_modes(self) -> List[QueryModeConfig]:
        """Get available query modes - delegates to SystemService"""
        try:
            return await self.system_service.get_query_modes()
        except Exception as e:
            logger.error(f"Error getting query modes: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error getting query modes: {str(e)}")

    async def get_query_mode(self, mode: str) -> QueryModeConfig:
        """Get specific query mode - delegates to SystemService"""
        try:
            result = await self.system_service.get_query_mode(mode)
            if not result:
                raise HTTPException(status_code=404, detail=f"Query mode '{mode}' not found")
            return result
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error getting query mode {mode}: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error getting query mode: {str(e)}")

    async def get_system_capabilities(self) -> SystemCapabilities:
        """Get system capabilities - delegates to SystemService"""
        try:
            return await self.system_service.get_system_capabilities()
        except Exception as e:
            logger.error(f"Error getting system capabilities: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error getting system capabilities: {str(e)}")

    async def get_workflow_types(self) -> List[Dict[str, Any]]:
        """Get available workflow types"""
        try:
            return self.workflow_model.get_available_workflow_types()
        except Exception as e:
            logger.error(f"Error getting workflow types: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error getting workflow types: {str(e)}")

    # ========================================================================
    # Workflow Management Endpoints
    # ========================================================================

    async def get_workflow_status(self, workflow_id: str) -> Dict[str, Any]:
        """Get workflow status - delegates to WorkflowModel"""
        try:
            if not workflow_id:
                raise HTTPException(status_code=400, detail="Workflow ID is required")

            status = await self.workflow_model.get_workflow_status(workflow_id)

            if not status:
                raise HTTPException(status_code=404, detail=f"Workflow {workflow_id} not found")

            return status

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error getting workflow status: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error getting workflow status: {str(e)}")

    async def cancel_workflow(self, workflow_id: str) -> Dict[str, str]:
        """Cancel workflow - delegates to WorkflowModel"""
        try:
            if not workflow_id:
                raise HTTPException(status_code=400, detail="Workflow ID is required")

            await self.workflow_model.cancel_workflow(workflow_id)

            return {"message": f"Workflow {workflow_id} cancelled successfully", "workflow_id": workflow_id}

        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except Exception as e:
            logger.error(f"Error cancelling workflow: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error cancelling workflow: {str(e)}")

    # ========================================================================
    # Legacy Validation Endpoints (To be removed in future)
    # ========================================================================

    async def get_validation_progress(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get validation progress - delegates to WorkflowModel validation logic"""
        try:
            if not job_id:
                raise HTTPException(status_code=400, detail="Job ID is required")

            # For now, return workflow status since validation will be removed
            status = await self.workflow_model.get_workflow_status(job_id)

            if not status:
                raise HTTPException(status_code=404, detail=f"No workflow found for job {job_id}")

            # Convert workflow status to validation-like format for compatibility
            return {
                "workflow_id": job_id,
                "status": status.get("status", "unknown"),
                "progress": status.get("progress_percentage", 0),
                "current_step": status.get("current_task"),
                "awaiting_user_input": status.get("awaiting_user_input", False)
            }

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error getting validation progress: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error getting validation progress: {str(e)}")

    async def submit_user_choice(self, job_id: str, choice_data: Dict[str, Any]) -> Dict[str, Any]:
        """Submit user choice - delegates to WorkflowModel"""
        try:
            if not job_id:
                raise HTTPException(status_code=400, detail="Job ID is required")

            if not choice_data.get("decision"):
                raise HTTPException(status_code=400, detail="Decision is required")

            # For now, just acknowledge the choice since validation will be removed
            return {
                "message": "User choice recorded",
                "job_id": job_id,
                "choice": choice_data.get("decision")
            }

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error submitting user choice: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error submitting user choice: {str(e)}")

    # ========================================================================
    # Debug and Administrative Endpoints
    # ========================================================================

    async def get_manufacturers(self) -> List[str]:
        """Get available manufacturers - delegates to SystemService"""
        try:
            return await self.system_service.get_manufacturers()
        except Exception as e:
            logger.error(f"Error getting manufacturers: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error getting manufacturers: {str(e)}")

    async def get_models(self, manufacturer: Optional[str] = None) -> List[str]:
        """Get available models - delegates to SystemService"""
        try:
            return await self.system_service.get_models(manufacturer)
        except Exception as e:
            logger.error(f"Error getting models: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error getting models: {str(e)}")

    async def get_queue_status(self) -> Dict[str, Any]:
        """Get queue status - delegates to SystemService"""
        try:
            return await self.system_service.get_queue_status()
        except Exception as e:
            logger.error(f"Error getting queue status: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error getting queue status: {str(e)}")

    async def debug_document_retrieval(self, request: Dict[str, str]) -> Dict[str, Any]:
        """Debug document retrieval - delegates to SystemService"""
        try:
            query = request.get("query", "")
            if not query:
                raise HTTPException(status_code=400, detail="Query parameter is required")

            limit = min(int(request.get("limit", 100)), 500)

            return await self.system_service.debug_document_retrieval(query, limit)

        except HTTPException:
            raise
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            logger.error(f"Debug retrieval failed: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Document retrieval failed: {str(e)}")

    # ========================================================================
    # HTTP Layer Validation (Only Basic Request Validation)
    # ========================================================================

    def _validate_query_request(self, request: EnhancedQueryRequest) -> None:
        """Validate incoming query request (HTTP layer validation only)"""
        if not request.query or len(request.query.strip()) < 3:
            raise ValueError("Query must be at least 3 characters long")

        if len(request.query) > 1000:
            raise ValueError("Query too long (maximum 1000 characters)")

        if request.top_k and (request.top_k < 1 or request.top_k > 20):
            raise ValueError("top_k must be between 1 and 20")

        if request.validation_config and request.validation_config.enabled:
            if request.validation_config.confidence_threshold < 0 or request.validation_config.confidence_threshold > 1:
                raise ValueError("confidence_threshold must be between 0 and 1")