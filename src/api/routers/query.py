import uuid
from typing import Dict, List, Optional, Any

from fastapi import APIRouter, Depends, HTTPException
from src.utils.unicode_handler import decode_unicode_in_dict
import logging
import numpy as np

from src.core.background.job_chain import job_chain, JobType
from src.core.background.job_tracker import job_tracker
from src.models.schema import QueryRequest, QueryResponse, BackgroundJobResponse

logger = logging.getLogger(__name__)

router = APIRouter()


async def handle_query_logic(request: QueryRequest) -> BackgroundJobResponse:
    """Common query handling function"""
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


@router.post("/", response_model=BackgroundJobResponse)
async def query_with_slash(request: QueryRequest) -> BackgroundJobResponse:
    """Query endpoint with trailing slash"""
    return await handle_query_logic(request)


@router.post("", response_model=BackgroundJobResponse)
async def query_without_slash(request: QueryRequest) -> BackgroundJobResponse:
    """Query endpoint without trailing slash"""
    return await handle_query_logic(request)


@router.get("/results/{job_id}", response_model=Optional[QueryResponse])
async def get_query_result(job_id: str) -> Optional[QueryResponse]:
    """Get the result of an asynchronous query."""
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
        # Get documents and clean them
        documents = result.get("documents", [])

        # Clean document format for UI
        cleaned_documents = []
        for doc_data in documents:
            if isinstance(doc_data, dict):
                # Clean the metadata to avoid null values
                doc_metadata = doc_data.get("metadata", {})

                cleaned_metadata = {}
                for key, value in doc_metadata.items():
                    if isinstance(value, (np.floating, np.float32, np.float64)):
                        cleaned_metadata[key] = float(value)
                    elif isinstance(value, (np.integer, np.int32, np.int64)):
                        cleaned_metadata[key] = int(value)
                    elif isinstance(value, np.ndarray):
                        cleaned_metadata[key] = value.tolist()
                    elif isinstance(value, str):
                        # CRITICAL FIX: Decode Unicode escape sequences
                        from src.utils.unicode_handler import decode_unicode_escapes
                        cleaned_metadata[key] = decode_unicode_escapes(value)
                    else:
                        cleaned_metadata[key] = value

                cleaned_doc = {
                    "id": doc_data.get("id", ""),
                    "content": doc_data.get("content", ""),
                    "metadata": cleaned_metadata,
                    "relevance_score": doc_data.get("relevance_score", 0.0)
                }
                cleaned_documents.append(cleaned_doc)

        # Clean the answer to remove any parsing artifacts
        answer = result.get("answer", "")

        # Remove common LLM thinking artifacts
        if answer.startswith("</think>\n\n"):
            answer = answer.replace("</think>\n\n", "").strip()
        if answer.startswith("<think>") and "</think>" in answer:
            # Remove entire thinking section
            answer = answer.split("</think>")[-1].strip()

        return QueryResponse(
            query=result.get("query", metadata.get("query", "")),
            answer=answer,
            documents=cleaned_documents,
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


# Add this to your src/api/routers/query.py for debugging
@router.post("/debug-retrieval", response_model=Dict[str, Any])
async def debug_document_retrieval(
        request: Dict[str, str]
):
    """Debug endpoint to see what documents are retrieved for a query"""
    try:
        query = request.get("query", "")

        # Import here to avoid circular imports
        from src.core.vectorstore import QdrantStore
        from src.config.settings import settings
        from qdrant_client import QdrantClient

        # Initialize vector store in metadata-only mode
        qdrant_client = QdrantClient(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
        )

        vector_store = QdrantStore(
            client=qdrant_client,
            collection_name=settings.qdrant_collection,
            embedding_function=None,  # Metadata-only mode
        )

        # Search by content (simple text matching)
        all_docs = []
        scroll_result = vector_store.client.scroll(
            collection_name=settings.qdrant_collection,
            limit=100,
            with_payload=True,
            with_vectors=False
        )

        points = scroll_result[0]
        matching_docs = []

        for point in points:
            content = point.payload.get("page_content", "")
            metadata = point.payload.get("metadata", {})

            # Check if any query terms match the content
            query_terms = query.split()
            matches = []
            for term in query_terms:
                if term in content:
                    matches.append(term)

            if matches:
                matching_docs.append({
                    "id": str(point.id),
                    "content": content,
                    "metadata": metadata,
                    "matched_terms": matches,
                    "content_preview": content[:200] + "..." if len(content) > 200 else content
                })

        return {
            "query": query,
            "total_documents_in_collection": len(points),
            "matching_documents": len(matching_docs),
            "documents": matching_docs
        }

    except Exception as e:
        logger.error(f"Debug retrieval error: {str(e)}")
        return {"error": str(e)}


# Also add this test endpoint to simulate the full query pipeline
@router.post("/debug-full-query", response_model=Dict[str, Any])
async def debug_full_query_pipeline(
        request: Dict[str, str]
):
    """Debug the full query pipeline to see what the LLM receives"""
    try:
        query = request.get("query", "")

        # Step 1: Get documents (simulate retrieval)
        from src.core.vectorstore import QdrantStore
        from src.config.settings import settings
        from qdrant_client import QdrantClient

        qdrant_client = QdrantClient(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
        )

        vector_store = QdrantStore(
            client=qdrant_client,
            collection_name=settings.qdrant_collection,
            embedding_function=None,
        )

        # Find matching documents
        scroll_result = vector_store.client.scroll(
            collection_name=settings.qdrant_collection,
            limit=100,
            with_payload=True,
            with_vectors=False
        )

        points = scroll_result[0]
        matching_docs = []

        for point in points:
            content = point.payload.get("page_content", "")
            metadata = point.payload.get("metadata", {})

            # Simple matching
            if any(term in content for term in query.split()):
                matching_docs.append({
                    "content": content,
                    "metadata": metadata,
                    "relevance_score": 0.8  # Mock score
                })

        # Step 2: Format context like LLM would receive
        context_parts = []
        for i, doc in enumerate(matching_docs[:3]):  # Top 3 docs
            metadata = doc["metadata"]
            source_type = metadata.get("source", "unknown")
            title = metadata.get("title", f"Document {i + 1}")

            # Format source information
            if source_type == "bilibili":
                source_info = f"Source {i + 1}: Bilibili - '{title}'"
                if "url" in metadata:
                    source_info += f" ({metadata['url']})"
            else:
                source_info = f"Source {i + 1}: {title}"

            # Add manufacturer and model if available
            manufacturer = metadata.get("manufacturer")
            model = metadata.get("model")
            year = metadata.get("year")

            if manufacturer or model or year:
                source_info += " - "
                if manufacturer:
                    source_info += manufacturer
                if model:
                    source_info += f" {model}"
                if year:
                    source_info += f" ({year})"

            # Format content block
            content_block = f"{source_info}\n{doc['content']}\n"
            context_parts.append(content_block)

        formatted_context = "\n\n".join(context_parts)

        return {
            "query": query,
            "documents_found": len(matching_docs),
            "context_for_llm": formatted_context,
            "raw_documents": matching_docs
        }

    except Exception as e:
        logger.error(f"Debug full query error: {str(e)}")
        return {"error": str(e)}