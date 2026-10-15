from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.config import settings
from src.api.routes.health import router as health_router
from src.api.routes.analyze import router as analyze_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan context manager for startup and shutdown routines."""
    # Ensure ephemeral storage directory exists
    settings.storage_path.mkdir(parents=True, exist_ok=True)
    yield


def create_app() -> FastAPI:
    """Application factory for TrueFrame AI Backend."""
    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description="TrueFrame AI — Digital Media Forensics & Deepfake Detection API",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # CORS configuration for decoupled frontend communication
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Register API Routers under /api/v1 prefix
    app.include_router(health_router, prefix="/api/v1")
    app.include_router(analyze_router, prefix="/api/v1")

    @app.get("/", tags=["Root"])
    async def root():
        return {
            "name": settings.app_name,
            "version": settings.app_version,
            "documentation": "/docs",
            "health": "/api/v1/health/live",
            "endpoints": {
                "analyze_image": "/api/v1/analyze/image",
            },
        }

    return app


app = create_app()
