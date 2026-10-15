from pathlib import Path
from typing import Tuple, Optional, Dict, Any
import numpy as np
from PIL import Image
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from src.schemas.analysis import BoundingBox, FaceDetectionResult

# Model weights path for BlazeFace short range model
MODEL_PATH = Path("models/weights/blaze_face_short_range.tflite")


class FaceDetectionService:
    """CPU-optimized Face Detection & Alignment service using MediaPipe."""

    def __init__(self, model_path: Path = MODEL_PATH):
        self.model_path = model_path
        self._detector: Optional[vision.FaceDetector] = None
        self._initialize_detector()

    def _initialize_detector(self) -> None:
        """Initialize MediaPipe FaceDetector instance."""
        if not self.model_path.exists():
            # If not yet downloaded, service will flag absence gracefully
            return

        base_options = python.BaseOptions(model_asset_path=str(self.model_path))
        options = vision.FaceDetectorOptions(
            base_options=base_options,
            min_detection_confidence=0.5,
        )
        self._detector = vision.FaceDetector.create_from_options(options)

    def extract_face(
        self,
        pil_image: Image.Image,
        margin_scale: float = 1.3,
        target_size: Tuple[int, int] = (224, 224),
    ) -> Tuple[Optional[Image.Image], FaceDetectionResult]:
        """
        Detects primary face, applies 1.3x margin to capture perimeter blending
        artifacts, and resizes to target tensor dimensions.
        """
        image_rgb = pil_image.convert("RGB")
        width, height = image_rgb.size
        np_image = np.array(image_rgb)

        if self._detector is None:
            # Fallback if model weights are not present
            return None, FaceDetectionResult(
                face_detected=False,
                confidence=0.0,
                faces_count=0,
                message="Face detection model asset not initialized.",
            )

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np_image)
        detection_result = self._detector.detect(mp_image)

        if not detection_result.detections:
            return None, FaceDetectionResult(
                face_detected=False,
                confidence=0.0,
                faces_count=0,
                message="No human facial region detected with confidence >= 0.50.",
            )

        # Select primary detection with highest confidence score
        primary = max(
            detection_result.detections,
            key=lambda d: d.categories[0].score if d.categories else 0.0,
        )
        score = float(primary.categories[0].score) if primary.categories else 0.5
        bbox = primary.bounding_box

        # Compute expanded bounding box with 1.3x margin to capture facial seams
        center_x = bbox.origin_x + bbox.width / 2.0
        center_y = bbox.origin_y + bbox.height / 2.0
        expanded_w = bbox.width * margin_scale
        expanded_h = bbox.height * margin_scale

        x_min = max(0, int(center_x - expanded_w / 2.0))
        y_min = max(0, int(center_y - expanded_h / 2.0))
        x_max = min(width, int(center_x + expanded_w / 2.0))
        y_max = min(height, int(center_y + expanded_h / 2.0))

        # Crop and resize
        cropped_face = image_rgb.crop((x_min, y_min, x_max, y_max))
        aligned_face = cropped_face.resize(target_size, Image.Resampling.BILINEAR)

        result_info = FaceDetectionResult(
            face_detected=True,
            confidence=round(score, 4),
            bounding_box=BoundingBox(
                x=bbox.origin_x,
                y=bbox.origin_y,
                width=bbox.width,
                height=bbox.height,
            ),
            faces_count=len(detection_result.detections),
            message=f"Face detected with {score * 100:.1f}% confidence.",
        )

        return aligned_face, result_info


# Global singleton instance for face detection
face_detector_service = FaceDetectionService()
