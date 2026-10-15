import io
import hashlib
import uuid
from datetime import datetime, timezone
from PIL import Image

from src.schemas.analysis import (
    AnalysisResponse,
    EvidentiaryQuality,
)
from src.services.face_detector import face_detector_service
from src.services.model_engine import model_engine
from src.services.explainability import explainability_service


class ForensicPipelineService:
    """
    End-to-End Forensic Analysis Coordinator.
    Orchestrates SHA-256 sealing, evidentiary quality validation,
    face alignment, calibrated inference, and visual explainability.
    """

    @staticmethod
    def analyze_image_bytes(raw_bytes: bytes, filename: str) -> AnalysisResponse:
        """
        Processes uploaded raw image bytes through the complete forensic pipeline.
        """
        investigation_id = str(uuid.uuid4())
        media_sha256 = hashlib.sha256(raw_bytes).hexdigest()
        timestamp_utc = datetime.now(timezone.utc).isoformat()
        file_size_bytes = len(raw_bytes)

        # 1. Parse Image & Extract Evidentiary Quality Metrics
        try:
            image = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
        except Exception as e:
            raise ValueError(f"Invalid or corrupted image format: {str(e)}")

        w, h = image.size
        aspect_ratio = round(w / float(h), 2) if h > 0 else 1.0

        # Quality grading based on resolution
        if min(w, h) < 160:
            quality_grade = "Severely Degraded (Low Resolution)"
            compression_warning = True
        elif min(w, h) < 320:
            quality_grade = "Sub-Optimal Resolution"
            compression_warning = True
        else:
            quality_grade = "High Evidentiary Integrity"
            compression_warning = False

        evidentiary_quality = EvidentiaryQuality(
            resolution=f"{w}x{h}",
            width=w,
            height=h,
            aspect_ratio=aspect_ratio,
            quality_grade=quality_grade,
            compression_warning=compression_warning,
        )

        # 2. Extract & Align Primary Face
        face_crop, face_detection = face_detector_service.extract_face(image)

        # 3. Model Inference & Explainability (Only if face is detected)
        forensic_score = None
        saliency_heatmap_base64 = None

        if face_detection.face_detected and face_crop is not None:
            score, feature_maps = model_engine.predict(face_crop)
            forensic_score = score

            if feature_maps is not None:
                heatmap_b64, _ = explainability_service.generate_eigen_cam(
                    feature_maps=feature_maps,
                    face_image=face_crop,
                )
                saliency_heatmap_base64 = heatmap_b64

        disclaimer = (
            "TrueFrame AI is a forensic diagnostic aid. Assessment scores represent statistical "
            "likelihoods conditioned on training distributions. Findings should be cross-examined with "
            "source provenance and chain-of-custody metadata."
        )

        return AnalysisResponse(
            investigation_id=investigation_id,
            media_sha256=media_sha256,
            timestamp_utc=timestamp_utc,
            filename=filename,
            file_size_bytes=file_size_bytes,
            face_detection=face_detection,
            forensic_score=forensic_score,
            evidentiary_quality=evidentiary_quality,
            saliency_heatmap_base64=saliency_heatmap_base64,
            model_card=model_engine.get_model_card(),
            disclaimer=disclaimer,
        )


forensic_pipeline = ForensicPipelineService()
