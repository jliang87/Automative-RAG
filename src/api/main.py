from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.openapi.utils import get_openapi

from contextlib import asynccontextmanager
import logging

from src.api.routers import auth, ingest, query, system
from src.config.settings import settings
from src.api.dependencies import get_token_header, load_all_components

# when uncommented, run without reload to debug
# import pydevd_pycharm
# pydevd_pycharm.settrace(
#     'localhost',
#     port=5678,
#     stdoutToServer=True,
#     stderrToServer=True,
#     suspend=True  # Pause execution
# )

# Define API metadata
API_TITLE = "Automotive Specs RAG API"
API_DESCRIPTION = "API for automotive specifications retrieval augmented generation with late interaction retrieval"
API_VERSION = "0.1.0"

# Configure logging
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load only necessary components when the FastAPI server starts."""
    logger.info("🚀 Starting API service... Loading minimal components (no GPU models)")

    try:
        # Initialize only the necessary components
        load_all_components()
        logger.info("✅ API components loaded successfully!")
    except Exception as e:
        logger.error(f"❌ Error during API initialization: {str(e)}")
        # We still allow the app to start even if there are initialization errors
        # Individual endpoints will handle their dependencies

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
    CORSMiddleware, # type: ignore
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


# Root endpoint
@app.get("/", tags=["Root"])
async def root():
    return {
        "message": "Automotive Specs RAG API",
        "version": "0.1.0",
        "docs": "/docs",
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

    return {
        "status": "healthy" if redis_ok and qdrant_ok else "degraded",
        "mode": "api-only",
        "components": {
            "redis": "connected" if redis_ok else "error",
            "qdrant": "connected" if qdrant_ok else "error"
        }
    }