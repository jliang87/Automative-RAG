from typing import Dict, List, Optional, Any
from fastapi import APIRouter, Path, Query as FastAPIQuery, Depends

from src.models.schema import (
    EnhancedQueryRequest,
    EnhancedQueryResponse,
    EnhancedBackgroundJobResponse,
    QueryModeConfig,
    SystemCapabilities,
    QueryValidationResult,
    QueryMode
)
from src.controllers.query_controller import QueryController, ValidationController, SystemController

# Fixed imports - use individual controller dependencies
from src.api.dependencies import get_query_controller, get_validation_controller, get_system_controller

router = APIRouter()

# ============================================================================
# Main Query Endpoints
# ============================================================================

@router.post("/", response_model=EnhancedBackgroundJobResponse)
async def submit_query_with_slash(
    request: EnhancedQueryRequest,
    query_controller: QueryController = Depends(get_query_controller)
) -> EnhancedBackgroundJobResponse:
    """Submit a query for processing with optional validation."""
    return await query_controller.submit_query(request)


@router.post("", response_model=EnhancedBackgroundJobResponse)
async def submit_query_without_slash(
    request: EnhancedQueryRequest,
    query_controller: QueryController = Depends(get_query_controller)
) -> EnhancedBackgroundJobResponse:
    """Submit a query for processing with optional validation."""
    return await query_controller.submit_query(request)


@router.get("/{job_id}", response_model=Optional[EnhancedQueryResponse])
async def get_query_result(
    job_id: str = Path(..., description="Job ID"),
    query_controller: QueryController = Depends(get_query_controller)
) -> Optional[EnhancedQueryResponse]:
    """Get query results with integrated validation data."""
    return await query_controller.get_query_result(job_id)


# ============================================================================
# Validation Sub-Resource Endpoints
# ============================================================================

@router.get("/{job_id}/validation", response_model=Optional[Dict[str, Any]])
async def get_validation_progress(
    job_id: str = Path(..., description="Job ID"),
    validation_controller: ValidationController = Depends(get_validation_controller)
) -> Optional[Dict[str, Any]]:
    """Get validation progress for a job."""
    return await validation_controller.get_validation_progress(job_id)


@router.post("/{job_id}/validation/user-choice", response_model=Dict[str, Any])
async def submit_user_choice_for_validation(
    job_id: str = Path(..., description="Job ID"),
    choice_data: Dict[str, Any],
    validation_controller: ValidationController = Depends(get_validation_controller)
) -> Dict[str, Any]:
    """Submit user choice for validation workflow."""
    return await validation_controller.submit_user_choice(job_id, choice_data)


@router.post("/{job_id}/validation/restart", response_model=Dict[str, Any])
async def restart_validation_workflow(
    job_id: str = Path(..., description="Job ID"),
    restart_data: Optional[Dict[str, Any]] = None,
    validation_controller: ValidationController = Depends(get_validation_controller)
) -> Dict[str, Any]:
    """Restart validation workflow from beginning or specific step."""
    return await validation_controller.restart_validation(job_id, restart_data)


@router.delete("/{job_id}/validation", response_model=Dict[str, str])
async def cancel_validation_workflow(
    job_id: str = Path(..., description="Job ID"),
    validation_controller: ValidationController = Depends(get_validation_controller)
) -> Dict[str, str]:
    """Cancel validation workflow for a job."""
    return await validation_controller.cancel_validation(job_id)


# ============================================================================
# System Information Endpoints
# ============================================================================

@router.get("/modes", response_model=List[QueryModeConfig])
async def get_query_modes(
    query_controller: QueryController = Depends(get_query_controller)
) -> List[QueryModeConfig]:
    """Get available query modes and their configurations."""
    return await query_controller.get_query_modes()


@router.get("/modes/{mode}", response_model=QueryModeConfig)
async def get_query_mode(
    mode: QueryMode,
    query_controller: QueryController = Depends(get_query_controller)
) -> QueryModeConfig:
    """Get configuration for a specific query mode."""
    return await query_controller.get_query_mode(mode.value)


@router.get("/capabilities", response_model=SystemCapabilities)
async def get_system_capabilities(
    query_controller: QueryController = Depends(get_query_controller)
) -> SystemCapabilities:
    """Get system capabilities and current status."""
    return await query_controller.get_system_capabilities()


@router.post("/validate", response_model=QueryValidationResult)
async def validate_query_for_modes(
    request: EnhancedQueryRequest,
    query_controller: QueryController = Depends(get_query_controller)
) -> QueryValidationResult:
    """Validate a query for mode compatibility and suggest validation type."""
    return await query_controller.validate_query_for_modes(request)


# ============================================================================
# Legacy/Utility Endpoints
# ============================================================================

@router.get("/manufacturers", response_model=List[str])
async def get_manufacturers(
    system_controller: SystemController = Depends(get_system_controller)
) -> List[str]:
    """Get a list of available manufacturers."""
    return await system_controller.get_manufacturers()


@router.get("/models", response_model=List[str])
async def get_models(
    manufacturer: Optional[str] = None,
    system_controller: SystemController = Depends(get_system_controller)
) -> List[str]:
    """Get a list of available models, optionally filtered by manufacturer."""
    return await system_controller.get_models(manufacturer)


@router.get("/queue-status", response_model=Dict[str, Any])
async def get_queue_status(
    system_controller: SystemController = Depends(get_system_controller)
) -> Dict[str, Any]:
    """Get status of the job chain queue system."""
    return await system_controller.get_queue_status()


@router.post("/debug-retrieval", response_model=Dict[str, Any])
async def debug_document_retrieval(
    request: Dict[str, str],
    system_controller: SystemController = Depends(get_system_controller)
) -> Dict[str, Any]:
    """Debug endpoint to retrieve documents from vector store for browsing."""
    return await system_controller.debug_document_retrieval(request)