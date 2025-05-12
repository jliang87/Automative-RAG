import time
import json
import uuid
from typing import Dict, List, Optional, Union

from fastapi import APIRouter, Depends, HTTPException
import torch
from typing import Dict, Any

from src.api.dependencies import get_llm, get_retriever
from src.core.retriever import HybridRetriever
from src.core.llm import LocalLLM
from src.models.schema import QueryRequest, QueryResponse, BackgroundJobResponse

router = APIRouter()


@router.post("/", response_model=QueryResponse)
async def query(
        request: QueryRequest,
        retriever: HybridRetriever = Depends(get_retriever),
) -> QueryResponse:
    """
    Query the RAG system with hybrid search, ColBERT reranking, and local DeepSeek LLM.
    Uses asynchronous processing for the GPU-intensive parts.
    """
    start_time = time.time()

    # Get top documents using the retriever
    documents, _ = retriever.retrieve(
        query=request.query,
        metadata_filter=request.metadata_filter,
        rerank=False,  # Don't rerank yet - will be done in background task
    )

    # Generate a unique job ID
    job_id = str(uuid.uuid4())

    # Create a job record
    from src.core.background_tasks import job_tracker
    job_tracker.create_job(
        job_id=job_id,
        job_type="llm_inference",
        metadata={
            "query": request.query,
            "metadata_filter": request.metadata_filter
        }
    )

    # Prepare documents for serialization
    serializable_docs = []
    for doc, score in documents:
        serializable_docs.append({
            "content": doc.page_content,
            "metadata": doc.metadata,
            "relevance_score": score
        })

    # Send to background task for reranking and inference
    from src.core.background_tasks import perform_llm_inference
    perform_llm_inference.send(job_id, request.query, serializable_docs, request.metadata_filter)

    # Create response indicating task is in progress
    execution_time = time.time() - start_time
    response = QueryResponse(
        query=request.query,
        answer="Your query is being processed. Please check back in a moment.",
        documents=[],
        metadata_filters_used=request.metadata_filter,
        execution_time=execution_time,
        status="processing",
        job_id=job_id
    )

    return response


@router.post("/async", response_model=BackgroundJobResponse)
async def query_async(
        request: QueryRequest,
        retriever: HybridRetriever = Depends(get_retriever)
) -> BackgroundJobResponse:
    """
    Asynchronous query that returns a job ID for later polling.

    Args:
        request: Query request with query text and optional metadata filters

    Returns:
        Background job response with job ID
    """
    # Generate a unique job ID
    job_id = str(uuid.uuid4())

    # Get top documents using the retriever (this is quick)
    documents, _ = retriever.retrieve(
        query=request.query,
        metadata_filter=request.metadata_filter,
        rerank=True,
    )

    # Prepare documents for serialization
    serializable_docs = []
    for doc, score in documents:
        serializable_docs.append({
            "content": doc.page_content,
            "metadata": doc.metadata,
            "relevance_score": score
        })

    # Create a job record
    from src.core.background_tasks import job_tracker
    job_tracker.create_job(
        job_id=job_id,
        job_type="llm_inference",
        metadata={
            "query": request.query,
            "metadata_filter": request.metadata_filter
        }
    )

    # Start the background job without waiting for completion
    from src.core.background_tasks import perform_llm_inference
    perform_llm_inference.send(job_id, request.query, serializable_docs, request.metadata_filter)

    # Return job ID immediately
    return BackgroundJobResponse(
        message="Query is processing in the background",
        job_id=job_id,
        job_type="llm_inference",
        status="pending",
    )


@router.get("/results/{job_id}", response_model=Optional[QueryResponse])
async def get_query_result(job_id: str) -> Optional[QueryResponse]:
    """
    Get the result of an asynchronous query.

    Args:
        job_id: Job ID to check for results

    Returns:
        Query response if completed, or status update
    """
    from src.core.background_tasks import job_tracker

    job_data = job_tracker.get_job(job_id)

    if not job_data:
        raise HTTPException(
            status_code=404,
            detail=f"Job with ID {job_id} not found"
        )

    status = job_data.get("status", "")

    # Return appropriate response based on job status
    if status == "completed":
        # Job completed successfully
        result = job_data.get("result", {})

        if isinstance(result, str):
            # Try to parse JSON string if needed
            try:
                result = json.loads(result)
            except:
                result = {"answer": result}

        # Construct response
        return QueryResponse(
            query=job_data.get("metadata", {}).get("query", ""),
            answer=result.get("answer", ""),
            documents=result.get("documents", []),
            metadata_filters_used=job_data.get("metadata", {}).get("metadata_filter"),
            execution_time=result.get("execution_time", 0),
            status=status
        )
    elif status == "failed":
        # Return error response with status
        return QueryResponse(
            query=job_data.get("metadata", {}).get("query", ""),
            answer=f"Error: {job_data.get('error', 'Unknown error')}",
            documents=[],
            metadata_filters_used=job_data.get("metadata", {}).get("metadata_filter"),
            execution_time=0,
            status=status
        )
    else:
        # Still processing - return status only
        return QueryResponse(
            query=job_data.get("metadata", {}).get("query", ""),
            answer=f"Processing query... (Status: {status})",
            documents=[],
            metadata_filters_used=job_data.get("metadata", {}).get("metadata_filter"),
            execution_time=0,
            status=status
        )


@router.get("/manufacturers", response_model=List[str])
async def get_manufacturers(
        retriever: HybridRetriever = Depends(get_retriever),
) -> List[str]:
    """Get a list of available manufacturers."""
    # In a real implementation, this would query the vector store
    # for unique values of the manufacturer field
    return [
        "Toyota", "Honda", "Ford", "Chevrolet", "BMW", "Mercedes",
        "Audi", "Volkswagen", "Nissan", "Hyundai", "Kia", "Subaru"
    ]


@router.get("/models", response_model=List[str])
async def get_models(
        manufacturer: Optional[str] = None,
        retriever: HybridRetriever = Depends(get_retriever),
) -> List[str]:
    """Get a list of available models, optionally filtered by manufacturer."""
    # In a real implementation, this would query the vector store
    # for unique values of the model field, filtered by manufacturer
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


@router.get("/llm-info", response_model=Dict[str, Any])
async def get_llm_info(
        llm: LocalLLM = Depends(get_llm)
) -> Dict[str, Any]:
    """Get information about the local LLM configuration."""
    info = {
        "model_name": llm.model_path,
        "device": llm.device,
        "temperature": llm.temperature,
        "max_tokens": llm.max_tokens,
        "quantization": "4-bit" if llm.use_4bit else "8-bit" if llm.use_8bit else "none",
        "torch_dtype": str(llm.torch_dtype),
    }

    # Add VRAM usage if on GPU
    if llm.device.startswith("cuda") and torch.cuda.is_available():
        info["vram_usage"] = f"{torch.cuda.memory_allocated() / (1024 ** 3):.2f} GB"

    return info


@router.get("/queue-status", response_model=Dict[str, Any])
async def get_priority_queue_status():
    """Get status of the priority queue system."""
    # Import the actor to get queue status
    from src.core.background_tasks import get_priority_queue_status

    # Call the actor and get the result
    result = get_priority_queue_status.send()

    # Get the result with a timeout
    try:
        queue_status = result.get(block=True, timeout=10)
        return queue_status
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error getting priority queue status: {str(e)}"
        )
