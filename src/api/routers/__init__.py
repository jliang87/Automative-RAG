# Exports all routers for easy importing
from .auth import router as auth_router
from .ingest import router as ingest_router
from .query import router as query_router
from .system import router as system_router
from .model import router as model_router

__all__ = ["auth_router", "ingest_router", "query_router", "system_router", "model_router"]