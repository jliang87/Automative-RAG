from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.openapi.utils import get_openapi

from contextlib import asynccontextmanager

from src.api.routers import auth, ingest, query
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

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load all components once when the FastAPI server starts."""
    print("🚀 Starting application... Loading all components")
    # Initialize all components at startup
    load_all_components()
    print("✅ All components loaded successfully!")
    yield  # Application runs
    print("🛑 Shutting down FastAPI... Cleaning up resources!")

# Create FastAPI app
app = FastAPI(
    title="Automotive Specs RAG API",
    description="API for automotive specifications retrieval augmented generation with late interaction retrieval",
    version="0.1.0",
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
        title=app.title + " - Swagger UI",
        oauth2_redirect_url=app.swagger_ui_oauth2_redirect_url,
    )

@app.get("/openapi.json", include_in_schema=False)
async def get_open_api_endpoint():
    return get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )

# Health check endpoint
@app.get("/health", tags=["Health"])
async def health_check():
    return {"status": "healthy"}
