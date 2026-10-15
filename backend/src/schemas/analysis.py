from typing import Optional, Dict, Any
from pydantic import BaseModel, Field


class BoundingBox(BaseModel):
    """Coordinates of a detected face region."""
    x: int
    y: int
    width: int
    height: int


class FaceDetectionResult(BaseModel):
    """Result of facial detection and landmark verification."""
    face_detected: bool
    confidence: float = Field(ge=0.0, le=1.0)
    bounding_box: Optional[BoundingBox] = None
    faces_count: int = 0
    message: str


class ForensicScore(BaseModel):
    """Calibrated authenticity and manipulation likelihood estimation."""
    manipulated_probability: float = Field(ge=0.0, le=1.0)
    authentic_probability: float = Field(ge=0.0, le=1.0)
    raw_logit: float
    uncertainty_margin: float
    assessment_label: str
    confidence_tier: str
    calibration_method: str = "Temperature Scaling (T=1.2)"


class EvidentiaryQuality(BaseModel):
    """Quality and compression metrics of submitted evidentiary media."""
    resolution: str
    width: int
    height: int
    aspect_ratio: float
    quality_grade: str
    compression_warning: bool


class ModelCardMetadata(BaseModel):
    """Audit metadata describing model lineage and configuration."""
    model_name: str
    backbone: str
    parameters_count: int
    input_resolution: str
    device: str
    explainability_method: str
    training_domain: str = "FaceForensics++ (c23 / c40) Reference Baseline"


class AnalysisResponse(BaseModel):
    """Complete, cryptographically sealed forensic analysis report."""
    investigation_id: str
    media_sha256: str
    timestamp_utc: str
    filename: str
    file_size_bytes: int
    face_detection: FaceDetectionResult
    forensic_score: Optional[ForensicScore] = None
    evidentiary_quality: EvidentiaryQuality
    saliency_heatmap_base64: Optional[str] = None
    model_card: ModelCardMetadata
    disclaimer: str
