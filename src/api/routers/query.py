import time
from typing import Dict, List, Optional, Union

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from src.api.dependencies import get_local_llm, get_retriever
from src.core.hybrid_retriever import HybridRetriever
from src.core.local_llm import LocalDeepSeekLLM
from src.models.schemas import QueryRequest, QueryResponse

router = APIRouter()


@router.post("/", response_model=QueryResponse)
async def query(
    request: QueryRequest,
    retriever: HybridRetriever = Depends(get_retriever),
    llm: LocalDeepSeekLLM = Depends(get_local_llm),
) -> QueryResponse:
    """
    Query the RAG system with hybrid search, ColBERT reranking, and local DeepSeek LLM.
    
    Args:
        request: Query request with query text and optional metadata filters
        
    Returns:
        Query response with answer and retrieved documents
    """
    start_time = time.time()
    
    # Get top documents using the retriever
    documents, _ = retriever.retrieve(
        query=request.query,
        metadata_filter=request.metadata_filter,
        rerank=True,
    )
    
    # Generate answer using the local LLM
    answer, sources = llm.answer_query_with_sources(
        query=request.query,
        documents=documents,
        metadata_filter=request.metadata_filter,
    )
    
    # Format retrieved documents for response
    formatted_documents = []
    for doc, score in documents:
        formatted_doc = {
            "id": doc.metadata.get("id", ""),
            "content": doc.page_content,
            "metadata": doc.metadata,
            "relevance_score": score,
        }
        formatted_documents.append(formatted_doc)
    
    execution_time = time.time() - start_time
    
    # Create response
    response = QueryResponse(
        query=request.query,
        answer=answer,
        documents=formatted_documents,
        metadata_filters_used=request.metadata_filter,
        execution_time=execution_time,
    )
    
    return response


@router.get("/manufacturers", response_model=List[str])
async def get_manufacturers(
    retriever: HybridRetriever = Depends(get_retriever),
) -> List[str]:
    """
    Get a list of available manufacturers.
    
    Returns:
        List of manufacturer names
    """
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
    """
    Get a list of available models, optionally filtered by manufacturer.
    
    Args:
        manufacturer: Optional manufacturer to filter by
        
    Returns:
        List of model names
    """
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
    """
    Get a list of available vehicle categories.
    
    Returns:
        List of categories
    """
    return [
        "sedan", "suv", "truck", "sports", "minivan", 
        "coupe", "convertible", "hatchback", "wagon"
    ]


@router.get("/engine-types", response_model=List[str])
async def get_engine_types() -> List[str]:
    """
    Get a list of available engine types.
    
    Returns:
        List of engine types
    """
    return [
        "gasoline", "diesel", "electric", "hybrid", "hydrogen"
    ]


@router.get("/transmission-types", response_model=List[str])
async def get_transmission_types() -> List[str]:
    """
    Get a list of available transmission types.
    
    Returns:
        List of transmission types
    """
    return [
        "automatic", "manual", "cvt", "dct"
    ]


@router.get("/llm-info", response_model=Dict[str, any])
async def get_llm_info(
    llm: LocalDeepSeekLLM = Depends(get_local_llm)
) -> Dict[str, any]:
    """
    Get information about the local LLM configuration.
    
    Returns:
        Dictionary with LLM information
    """
    info = {
        "model_name": llm.model_name,
        "device": llm.device,
        "temperature": llm.temperature,
        "max_tokens": llm.max_tokens,
        "quantization": "4-bit" if llm.use_4bit else "8-bit" if llm.use_8bit else "none",
        "torch_dtype": str(llm.torch_dtype),
    }
    
    # Add VRAM usage if on GPU
    if llm.device.startswith("cuda") and torch.cuda.is_available():
        info["vram_usage"] = f"{torch.cuda.memory_allocated() / (1024**3):.2f} GB"
        
    return info