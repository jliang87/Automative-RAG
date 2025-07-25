import logging
from typing import Optional, List, Dict, Any
from fastapi import HTTPException, Path

from src.models import (  # Updated import
    EnhancedQueryRequest,
    EnhancedQueryResponse,
    EnhancedBackgroundJobResponse,
    QueryModeConfig,
    SystemCapabilities,
    QueryValidationResult
)
from src.services.query_service import QueryService
from src.services.validation_service import ValidationService
from src.services.system_service import SystemService

logger = logging.getLogger(__name__)


class QueryController:
    """
    Controller for query-related HTTP endpoints.
    Handles only HTTP concerns - delegates business logic to services.
    """

    def __init__(self,
                 query_service: QueryService,
                 validation_service: ValidationService,
                 system_service: SystemService):
        self.query_service = query_service
        self.validation_service = validation_service
        self.system_service = system_service

    async def submit_query(self, request: EnhancedQueryRequest) -> EnhancedBackgroundJobResponse:
        """Submit a query for processing with optional validation."""
        try:
            # Input validation (HTTP layer responsibility)
            self._validate_query_request(request)

            # Delegate to service layer
            return await self.query_service.process_query(request)

        except ValueError as e:
            logger.warning(f"Invalid query request: {str(e)}")
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

    async def get_query_result(self, job_id: str) -> Optional[EnhancedQueryResponse]:
        """Get query results with integrated validation data."""
        try:
            if not job_id or len(job_id.strip()) < 5:
                raise HTTPException(status_code=400, detail="Invalid job ID")

            result = await self.query_service.get_query_result(job_id)

            if not result:
                raise HTTPException(status_code=404, detail=f"Job with ID {job_id} not found")

            return result

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error retrieving query result: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error retrieving result: {str(e)}")

    async def get_query_modes(self) -> List[QueryModeConfig]:
        """Get available query modes and their configurations."""
        try:
            return await self.system_service.get_query_modes()
        except Exception as e:
            logger.error(f"Error getting query modes: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error getting query modes: {str(e)}")

    async def get_query_mode(self, mode: str) -> QueryModeConfig:
        """Get configuration for a specific query mode."""
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
        """Get system capabilities and current status."""
        try:
            return await self.system_service.get_system_capabilities()
        except Exception as e:
            logger.error(f"Error getting system capabilities: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error getting system capabilities: {str(e)}")

    async def validate_query_for_modes(self, request: EnhancedQueryRequest) -> QueryValidationResult:
        """Validate a query for mode compatibility and suggest validation type."""
        try:
            self._validate_query_request(request)
            return await self.query_service.validate_query_compatibility(request)
        except ValueError as e:
            logger.warning(f"Invalid query for validation: {str(e)}")
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            logger.error(f"Error validating query: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error validating query: {str(e)}")

    def _validate_query_request(self, request: EnhancedQueryRequest) -> None:
        """Validate incoming query request (HTTP layer validation)."""
        if not request.query or len(request.query.strip()) < 3:
            raise ValueError("Query must be at least 3 characters long")

        if len(request.query) > 1000:
            raise ValueError("Query too long (maximum 1000 characters)")

        if request.top_k and (request.top_k < 1 or request.top_k > 20):
            raise ValueError("top_k must be between 1 and 20")

        if request.validation_config and request.validation_config.enabled:
            if request.validation_config.confidence_threshold < 0 or request.validation_config.confidence_threshold > 1:
                raise ValueError("confidence_threshold must be between 0 and 1")


class ValidationController:
    """
    Controller for validation-related HTTP endpoints.
    Handles validation workflow management endpoints.
    """

    def __init__(self, validation_service: ValidationService):
        self.validation_service = validation_service

    async def get_validation_progress(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get validation progress for a job."""
        try:
            if not job_id:
                raise HTTPException(status_code=400, detail="Job ID is required")

            result = await self.validation_service.get_validation_progress(job_id)

            if not result:
                raise HTTPException(status_code=404, detail=f"No validation workflow found for job {job_id}")

            return result

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error getting validation progress: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error getting validation progress: {str(e)}")

    async def submit_user_choice(self, job_id: str, choice_data: Dict[str, Any]) -> Dict[str, Any]:
        """Submit user choice for validation workflow."""
        try:
            if not job_id:
                raise HTTPException(status_code=400, detail="Job ID is required")

            if not choice_data.get("decision"):
                raise HTTPException(status_code=400, detail="Decision is required")

            return await self.validation_service.submit_user_choice(job_id, choice_data)

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error submitting user choice: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error submitting user choice: {str(e)}")

    async def restart_validation(self, job_id: str, restart_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Restart validation workflow from beginning or specific step."""
        try:
            if not job_id:
                raise HTTPException(status_code=400, detail="Job ID is required")

            return await self.validation_service.restart_validation_workflow(job_id, restart_data)

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error restarting validation: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error restarting validation: {str(e)}")

    async def cancel_validation(self, job_id: str) -> Dict[str, str]:
        """Cancel validation workflow for a job."""
        try:
            if not job_id:
                raise HTTPException(status_code=400, detail="Job ID is required")

            return await self.validation_service.cancel_validation_workflow(job_id)

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error cancelling validation: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error cancelling validation: {str(e)}")


class SystemController:
    """
    Controller for system information endpoints.
    """

    def __init__(self, system_service: SystemService):
        self.system_service = system_service

    async def get_manufacturers(self) -> List[str]:
        """Get a list of available manufacturers."""
        try:
            return await self.system_service.get_manufacturers()
        except Exception as e:
            logger.error(f"Error getting manufacturers: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error getting manufacturers: {str(e)}")

    async def get_models(self, manufacturer: Optional[str] = None) -> List[str]:
        """Get a list of available models, optionally filtered by manufacturer."""
        try:
            return await self.system_service.get_models(manufacturer)
        except Exception as e:
            logger.error(f"Error getting models: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error getting models: {str(e)}")

    async def get_queue_status(self) -> Dict[str, Any]:
        """Get status of the job chain queue system."""
        try:
            return await self.system_service.get_queue_status()
        except Exception as e:
            logger.error(f"Error getting queue status: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error getting queue status: {str(e)}")

    async def debug_document_retrieval(self, request: Dict[str, str]) -> Dict[str, Any]:
        """Debug endpoint to retrieve documents from vector store for browsing."""
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