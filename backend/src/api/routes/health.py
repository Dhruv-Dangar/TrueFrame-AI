from fastapi import APIRouter
from src.config import settings

router = APIRouter(prefix="/health", tags=["Health & Observability"])


@router.get("/live", summary="Liveness Probe")
async def liveness_probe():
    """Verify that the FastAPI ASGI server process is alive and responding."""
    return {
        "status": "ok",
        "service": settings.app_name,
        "version": settings.app_version,
    }


@router.get("/ready", summary="Readiness Probe")
async def readiness_probe():
    """Verify system readiness including storage paths and configuration."""
    storage_ready = settings.storage_path.exists()
    return {
        "status": "ready" if storage_ready else "degraded",
        "environment": settings.app_env,
        "active_model": settings.active_model,
        "inference_device": settings.inference_device,
        "storage_ready": storage_ready,
    }
