from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.openapi.utils import get_openapi

from contextlib import asynccontextmanager
import logging

from src.api.routers import auth, ingest, query, system
from src.api.routers.model import router as model_router
from src.api.dependencies import load_all_components

# Define API metadata
API_TITLE = "Automotive Specs RAG API - Unified Query System"
API_DESCRIPTION = """
API for automotive specifications retrieval with unified query processing.

## Unified Query System

This API now uses a **unified query system** where all queries go through the enhanced query endpoint:

### Query Modes:
- **Facts (Default)**: Direct verification of vehicle specifications - *replaces normal queries*
- **Features**: Evaluate whether to add new features  
- **Tradeoffs**: Analyze pros/cons of design choices
- **Scenarios**: Assess performance in user scenarios
- **Debate**: Multi-perspective discussions
- **Quotes**: Extract user reviews and feedback

### Migration Notes:
- Normal query endpoints have been retired
- Facts mode serves as the default for direct specification queries
- All queries now return enhanced response format with mode metadata
- Better consistency and user experience

### Key Features:
- Job chain processing with dedicated GPU workers
- Automatic hallucination detection
- Mode-aware document retrieval
- Tesla T4 optimized inference
"""
API_VERSION = "2.0.0"  # UPDATED: Major version bump for unified system

# Configure logging
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load only necessary components when the FastAPI server starts."""
    logger.info("🚀 Starting API service with unified query system...")

    try:
        # Initialize only the necessary components
        load_all_components()
        logger.info("✅ API components loaded successfully!")
        logger.info("🔄 Unified query system active - Facts mode is default")
    except Exception as e:
        logger.error(f"❌ Error during API initialization: {str(e)}")

    yield  # Application runs

    logger.info("🛑 Shutting down API service... Cleaning up resources!")


# Create FastAPI app
app = FastAPI(
    title=API_TITLE,
    description=API_DESCRIPTION,
    version=API_VERSION,
    docs_url=None,
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers WITHOUT authentication dependencies
app.include_router(
    auth.router,
    prefix="/auth",
    tags=["Authentication"],
)

app.include_router(
    ingest.router,
    prefix="/ingest",
    tags=["Ingestion"],
    # NO authentication required
)

app.include_router(
    query.router,
    prefix="/query",
    tags=["Unified Query System"],  # UPDATED: Reflects unified system
    # NO authentication required
)

app.include_router(system.router, prefix="/system", tags=["System"])

app.include_router(
    model_router,
    prefix="/model",
    tags=["Model Management"]
)


# Root endpoint
@app.get("/", tags=["Root"])
async def root():
    return {
        "message": "Automotive Specs RAG API - Unified Query System",
        "version": "2.0.0",
        "docs": "/docs",
        "architecture": "dedicated_gpu_workers",
        "authentication": "disabled",
        "query_system": "unified_enhanced",
        "default_mode": "facts",
        "migration_info": {
            "normal_queries": "retired",
            "enhanced_queries": "now_main_system",
            "facts_mode": "default_for_direct_queries",
            "breaking_changes": "normal_query_endpoints_removed"
        }
    }


# Custom docs endpoint
@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui_html():
    return get_swagger_ui_html(
        openapi_url="/openapi.json",
        title=API_TITLE + " - Swagger UI",
        oauth2_redirect_url=None,
    )


@app.get("/openapi.json", include_in_schema=False)
async def get_open_api_endpoint():
    return get_openapi(
        title=API_TITLE,
        version=API_VERSION,
        description=API_DESCRIPTION,
        routes=app.routes,
    )


# Health check endpoint
@app.get("/health", tags=["Health"])
async def health_check():
    """Simple health check that returns basic status information."""
    # Check basic Redis connectivity
    redis_ok = False
    try:
        from src.api.dependencies import redis_client
        if redis_client and redis_client.ping():
            redis_ok = True
    except Exception:
        pass

    # Check basic Qdrant connectivity
    qdrant_ok = False
    try:
        from src.api.dependencies import qdrant_client
        if qdrant_client:
            qdrant_client.get_collections()
            qdrant_ok = True
    except Exception:
        pass

    # Check job chain system
    job_chain_ok = False
    try:
        from src.core.orchestration.job_chain import job_chain
        queue_status = job_chain.get_queue_status()
        job_chain_ok = True
    except Exception:
        pass

    return {
        "status": "healthy" if redis_ok and job_chain_ok else "degraded",
        "mode": "dedicated_gpu_workers",
        "architecture": "event_driven",
        "authentication": "disabled",
        "query_system": "unified_enhanced",  # NEW
        "default_query_mode": "facts",       # NEW
        "version": "2.0.0",                  # NEW
        "components": {
            "redis": "connected" if redis_ok else "error",
            "qdrant": "connected" if qdrant_ok else "error",
            "job_chain": "active" if job_chain_ok else "error"
        }
    }


# Enhanced job chains endpoint for UI support
@app.get("/job-chains", tags=["Job Chains"])
async def get_job_chains_overview():
    """Get comprehensive overview of the job chain system."""
    try:
        from src.core.orchestration.job_chain import job_chain
        from src.core.orchestration.job_tracker import job_tracker

        # Get queue status (for 系统信息.py)
        queue_status = job_chain.get_queue_status()

        # Get job statistics (for both pages)
        job_stats = job_tracker.count_jobs_by_status()

        # Get recent jobs (for 后台任务.py)
        recent_jobs = job_tracker.get_all_jobs(limit=10)

        # Format recent jobs for UI consumption
        formatted_recent_jobs = []
        for job in recent_jobs:
            formatted_job = {
                "job_id": job.get("job_id", ""),
                "job_type": job.get("job_type", ""),
                "status": job.get("status", ""),
                "created_at": job.get("created_at", 0),
                "updated_at": job.get("updated_at", 0)
            }

            # Add progress info if available
            progress_info = job.get("progress_info", {})
            if progress_info:
                formatted_job["progress"] = progress_info.get("progress")
                formatted_job["progress_message"] = progress_info.get("message", "")

            # Add unified system metadata
            metadata = job.get("metadata", {})
            if isinstance(metadata, dict):
                formatted_job["query_mode"] = metadata.get("query_mode", "facts")
                formatted_job["unified_system"] = metadata.get("unified_system", True)

            formatted_recent_jobs.append(formatted_job)

        return {
            "queue_status": queue_status,
            "job_statistics": job_stats,
            "recent_jobs": formatted_recent_jobs,
            "system_info": {
                "architecture": "dedicated_gpu_workers",
                "self_triggering": True,
                "auto_queue_management": True,
                "total_jobs": job_stats.get("total", 0),
                "query_system": "unified_enhanced",  # NEW
                "default_mode": "facts"              # NEW
            }
        }
    except Exception as e:
        logger.error(f"Error getting job chains overview: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error getting job chains overview: {str(e)}"
        )


@app.get("/job-chains/{job_id}", tags=["Job Chains"])
async def get_job_chain_details(job_id: str):
    """Get detailed information about a specific job chain."""
    try:
        from src.core.orchestration.job_chain import job_chain
        from src.core.orchestration.job_tracker import job_tracker

        # Get job chain status
        chain_status = job_chain.get_job_chain_status(job_id)

        # Get job tracker information
        job_data = job_tracker.get_job(job_id)

        if not job_data:
            raise HTTPException(404, f"Job {job_id} not found")

        # Combine chain and tracker data
        combined_data = {
            "job_id": job_id,
            "status": job_data.get("status", "unknown"),
            "job_type": job_data.get("job_type", ""),
            "created_at": job_data.get("created_at", 0),
            "updated_at": job_data.get("updated_at", 0),
            "metadata": job_data.get("metadata", {}),
            "result": job_data.get("result", {}),
            "error": job_data.get("error"),
        }

        # Add unified system information
        metadata = combined_data.get("metadata", {})
        if isinstance(metadata, dict):
            combined_data["query_mode"] = metadata.get("query_mode", "facts")
            combined_data["unified_system"] = metadata.get("unified_system", True)
            combined_data["mode_name"] = metadata.get("mode_name", "车辆规格查询")

        # Add progress information
        progress_info = job_data.get("progress_info", {})
        if progress_info:
            combined_data["progress_info"] = progress_info

        # Add job chain specific info if available
        if chain_status:
            combined_data["job_chain"] = chain_status
            combined_data["progress_percentage"] = chain_status.get("progress_percentage", 0)
            combined_data["current_task"] = chain_status.get("current_task")
            combined_data["total_steps"] = chain_status.get("total_steps", 0)

        return combined_data

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting job chain details: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error getting job chain details: {str(e)}"
        )


# Worker status endpoint (for 系统信息.py)
@app.get("/workers/status", tags=["Workers"])
async def get_worker_status():
    """Get worker status for system monitoring."""
    try:
        from src.api.dependencies import get_redis_client
        from src.core.background.worker_status import get_worker_status_for_ui

        redis_client = get_redis_client()
        return get_worker_status_for_ui(redis_client)

    except Exception as e:
        logger.error(f"Error getting worker status: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error getting worker status: {e}"
        )


# NEW: System migration info endpoint
@app.get("/migration", tags=["Migration"])
async def get_migration_info():
    """Get information about the unified system migration."""
    return {
        "migration_status": "completed",
        "version": "2.0.0",
        "changes": {
            "removed": [
                "Normal query endpoints (/query with QueryRequest)",
                "Separate enhanced query endpoint (/query/enhanced)"
            ],
            "unified": [
                "Single query endpoint (/query with UnifiedQueryRequest)",
                "Facts mode as default (replaces normal queries)",
                "All responses in unified format"
            ],
            "benefits": [
                "Simplified API surface",
                "Consistent response format",
                "Intuitive default behavior",
                "Better user experience"
            ]
        },
        "compatibility": {
            "breaking_changes": True,
            "legacy_models": "deprecated_but_aliased",
            "migration_path": "update_to_unified_models"
        },
        "query_modes": {
            "facts": {
                "role": "default_mode",
                "replaces": "normal_queries",
                "description": "Direct verification of specifications"
            },
            "others": "enhanced_analysis_modes"
        }
    }


# NEW: Default query mode endpoint
@app.get("/query/default-mode", tags=["Unified Query System"])
async def get_default_query_mode():
    """Get the default query mode for the unified system."""
    return {
        "default_mode": "facts",
        "mode_name": "车辆规格查询",
        "description": "Direct verification of vehicle specifications",
        "replaces": "normal_queries",
        "icon": "📌",
        "complexity": "simple",
        "estimated_time": 10
    }