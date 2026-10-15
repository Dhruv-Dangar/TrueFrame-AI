from fastapi import APIRouter, UploadFile, File, HTTPException, status
from src.schemas.analysis import AnalysisResponse
from src.services.forensic_pipeline import forensic_pipeline
from src.config import settings

router = APIRouter(prefix="/analyze", tags=["Forensic Analysis"])

ALLOWED_IMAGE_MIMES = {"image/jpeg", "image/png", "image/webp"}


@router.post(
    "/image",
    response_model=AnalysisResponse,
    status_code=status.HTTP_200_OK,
    summary="Analyze Single Image for Facial Manipulation",
)
async def analyze_image(
    file: UploadFile = File(..., description="JPEG, PNG, or WebP evidentiary image file"),
) -> AnalysisResponse:
    """
    Submits an image for forensic deepfake analysis.
    
    1. Verifies SHA-256 cryptographic seal.
    2. Validates image resolution and compression boundaries.
    3. Detects face and extracts 1.3x expanded crop.
    4. Executes calibrated EfficientNet-B0 inference on CPU.
    5. Generates Eigen-CAM visual saliency heatmap overlay.
    """
    # 1. Validate MIME type
    if file.content_type not in ALLOWED_IMAGE_MIMES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported file type '{file.content_type}'. Supported: JPEG, PNG, WebP.",
        )

    # 2. Read bytes and enforce size limits
    max_bytes = settings.max_image_size_mb * 1024 * 1024
    content = await file.read()
    if len(content) > max_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Image size exceeds maximum limit of {settings.max_image_size_mb} MB.",
        )

    if len(content) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty (0 bytes).",
        )

    try:
        response = forensic_pipeline.analyze_image_bytes(
            raw_bytes=content,
            filename=file.filename or "unknown_upload.jpg",
        )
        return response
    except ValueError as val_err:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(val_err),
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference error during analysis: {str(exc)}",
        )
