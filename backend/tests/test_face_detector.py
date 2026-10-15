import numpy as np
from PIL import Image
from src.services.face_detector import face_detector_service


def test_face_detector_no_face():
    """Verify face detector handles non-face images gracefully without hallucinating detections."""
    # Blank black image
    blank_image = Image.fromarray(np.zeros((300, 300, 3), dtype=np.uint8))

    crop, result = face_detector_service.extract_face(blank_image)

    assert crop is None
    assert result.face_detected is False
    assert result.confidence == 0.0
    assert result.bounding_box is None
    assert "No human facial region" in result.message
