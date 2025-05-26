# src/api/routers/query.py (Updated for Job Chain)

import uuid
from typing import Dict, List, Optional, Any

from fastapi import APIRouter, Depends, HTTPException
import logging

from src.core.background.job_chain import job_chain, JobType
from src.core.background.job_tracker import job_tracker
from src.models.schema import QueryRequest, QueryResponse, BackgroundJobResponse

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/", response_model=BackgroundJobResponse)
async def query(
        request: QueryRequest,
) -> BackgroundJobResponse:
    """
    Asynchronous query that returns a job ID for later polling.
    All processing happens using the job chain system.
    """
    try:
        # Generate a unique job ID
        job_id = str(uuid.uuid4())

        # Create a job record
        job_tracker.create_job(
            job_id=job_id,
            job_type="llm_inference",
            metadata={
                "query": request.query,
                "metadata_filter": request.metadata_filter,
                "top_k": getattr(request, "top_k", 5)
            }
        )

        # Start the job chain
        job_chain.start_job_chain(
            job_id=job_id,
            job_type=JobType.LLM_INFERENCE,
            initial_data={
                "query": request.query,
                "metadata_filter": request.metadata_filter
            }
        )

        return BackgroundJobResponse(
            message="Query is processing in the background",
            job_id=job_id,
            job_type="llm_inference",
            status="processing",
        )
    except Exception as e:
        logger.error(f"Error starting query processing: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing query: {str(e)}",
        )


@router.get("/results/{job_id}", response_model=Optional[QueryResponse])
async def get_query_result(job_id: str) -> Optional[QueryResponse]:
    """
    Get the result of an asynchronous query.
    """
    job_data = job_tracker.get_job(job_id)

    if not job_data:
        raise HTTPException(
            status_code=404,
            detail=f"Job with ID {job_id} not found"
        )

    status = job_data.get("status", "")
    metadata = job_data.get("metadata", {})
    result = job_data.get("result", {})

    # Parse result if it's a JSON string
    if isinstance(result, str):
        try:
            import json
            result = json.loads(result)
        except:
            result = {"answer": result}

    # Return appropriate response based on job status
    if status == "completed":
        return QueryResponse(
            query=result.get("query", metadata.get("query", "")),
            answer=result.get("answer", ""),
            documents=result.get("documents", []),
            metadata_filters_used=metadata.get("metadata_filter"),
            execution_time=result.get("execution_time", 0),
            status=status,
            job_id=job_id
        )
    elif status == "failed":
        return QueryResponse(
            query=metadata.get("query", ""),
            answer=f"Error: {job_data.get('error', 'Unknown error')}",
            documents=[],
            metadata_filters_used=metadata.get("metadata_filter"),
            execution_time=0,
            status=status,
            job_id=job_id
        )
    else:
        # Still processing
        chain_status = job_chain.get_job_chain_status(job_id)
        progress_msg = "Processing query..."
        if chain_status:
            current_task = chain_status.get("current_task", "unknown")
            progress_msg = f"Processing query... (Current: {current_task})"

        return QueryResponse(
            query=metadata.get("query", ""),
            answer=progress_msg,
            documents=[],
            metadata_filters_used=metadata.get("metadata_filter"),
            execution_time=0,
            status=status,
            job_id=job_id
        )


@router.get("/manufacturers", response_model=List[str])
async def get_manufacturers() -> List[str]:
    """Get a list of available manufacturers."""
    return [
        "Toyota", "Honda", "Ford", "Chevrolet", "BMW", "Mercedes",
        "Audi", "Volkswagen", "Nissan", "Hyundai", "Kia", "Subaru"
    ]


@router.get("/models", response_model=List[str])
async def get_models(manufacturer: Optional[str] = None) -> List[str]:
    """Get a list of available models, optionally filtered by manufacturer."""
    if manufacturer == "Toyota":
        return ["Camry", "Corolla", "RAV4", "Highlander", "Tacoma"]
    elif manufacturer == "Honda":
        return ["Civic", "Accord", "CR-V", "Pilot", "Odyssey"]
    elif manufacturer == "Ford":
        return ["Mustang", "F-150", "Escape", "Explorer", "Edge"]
    elif manufacturer == "BMW":
        return ["3 Series", "5 Series", "X3", "X5", "i4"]
    else:
        return ["Model S", "Model 3", "Model X", "Model Y", "Cybertruck"]


@router.get("/categories", response_model=List[str])
async def get_categories() -> List[str]:
    """Get a list of available vehicle categories."""
    return [
        "sedan", "suv", "truck", "sports", "minivan",
        "coupe", "convertible", "hatchback", "wagon"
    ]


@router.get("/engine-types", response_model=List[str])
async def get_engine_types() -> List[str]:
    """Get a list of available engine types."""
    return [
        "gasoline", "diesel", "electric", "hybrid", "hydrogen"
    ]


@router.get("/transmission-types", response_model=List[str])
async def get_transmission_types() -> List[str]:
    """Get a list of available transmission types."""
    return [
        "automatic", "manual", "cvt", "dct"
    ]


@router.get("/queue-status", response_model=Dict[str, Any])
async def get_queue_status() -> Dict[str, Any]:
    """Get status of the job chain queue system."""
    try:
        return job_chain.get_queue_status()
    except Exception as e:
        logger.error(f"Error getting queue status: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error getting queue status: {str(e)}"
        )