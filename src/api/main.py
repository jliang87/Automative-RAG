# src/api/main.py (Updated for Job Chain)

from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.openapi.utils import get_openapi

from contextlib import asynccontextmanager
import logging

from src.api.routers import auth, ingest, query, system
from src.api.routers.model import router as model_router
from src.config.settings import settings
from src.api.dependencies import get_token_header, load_all_components

# Define API metadata
API_TITLE = "Automotive Specs RAG API"
API_DESCRIPTION = "API for automotive specifications retrieval with job chain processing"
API_VERSION = "0.2.0"

# Configure logging
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load only necessary components when the FastAPI server starts."""
    logger.info("🚀 Starting API service with job chain system...")

    try:
        # Initialize only the necessary components
        load_all_components()
        logger.info("✅ API components loaded successfully!")
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

# Include routers
app.include_router(
    auth.router,
    prefix="/auth",
    tags=["Authentication"],
)

app.include_router(
    ingest.router,
    prefix="/ingest",
    tags=["Ingestion"],
    dependencies=[Depends(get_token_header)] if settings.api_auth_enabled else [],
)

app.include_router(
    query.router,
    prefix="/query",
    tags=["Query"],
    dependencies=[Depends(get_token_header)] if settings.api_auth_enabled else [],
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
        "message": "Automotive Specs RAG API with Job Chain Processing",
        "version": "0.2.0",
        "docs": "/docs",
        "architecture": "event_driven_job_chains"
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
        from src.core.background.job_chain import job_chain
        queue_status = job_chain.get_queue_status()
        job_chain_ok = True
    except Exception:
        pass

    return {
        "status": "healthy" if redis_ok and qdrant_ok and job_chain_ok else "degraded",
        "mode": "job_chain",
        "architecture": "event_driven",
        "components": {
            "redis": "connected" if redis_ok else "error",
            "qdrant": "connected" if qdrant_ok else "error",
            "job_chain": "active" if job_chain_ok else "error"
        }
    }


# Additional job chain specific endpoints
@app.get("/job-chains", tags=["Job Chains"])
async def get_job_chains_overview():
    """Get an overview of the job chain system."""
    try:
        from src.core.background.job_chain import job_chain
        from src.core.background.job_tracker import job_tracker

        # Get queue status
        queue_status = job_chain.get_queue_status()

        # Get job statistics
        job_stats = job_tracker.count_jobs_by_status()

        # Get recent jobs
        recent_jobs = job_tracker.get_all_jobs(limit=10)

        return {
            "queue_status": queue_status,
            "job_statistics": job_stats,
            "recent_jobs": recent_jobs,
            "system_info": {
                "architecture": "event_driven_job_chains",
                "self_triggering": True,
                "no_polling": True,
                "automatic_recovery": True
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
        from src.core.background.job_chain import job_chain
        from src.core.background.job_tracker import job_tracker

        # Get job chain status
        chain_status = job_chain.get_job_chain_status(job_id)
        if not chain_status:
            raise HTTPException(404, f"Job chain {job_id} not found")

        # Get job tracker information
        job_data = job_tracker.get_job(job_id)

        return {
            "job_chain": chain_status,
            "job_data": job_data,
            "combined_view": {
                "job_id": job_id,
                "status": job_data.get("status") if job_data else "unknown",
                "progress": chain_status.get("progress_percentage", 0),
                "current_task": chain_status.get("current_task"),
                "total_steps": chain_status.get("total_steps", 0),
                "step_timings": chain_status.get("step_timings", {})
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting job chain details: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error getting job chain details: {str(e)}"
        )