import io
import numpy as np
from PIL import Image
from fastapi.testclient import TestClient
from src.main import app

client = TestClient(app)


def test_analyze_image_valid_request():
    """Verify POST /api/v1/analyze/image processes an evidentiary image and returns full report."""
    # Generate synthetic image in memory (600x600 for standard high-resolution)
    img = Image.fromarray(np.uint8(np.random.randint(50, 200, (600, 600, 3))))
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG")
    buffer.seek(0)

    response = client.post(
        "/api/v1/analyze/image",
        files={"file": ("test_evidence.jpg", buffer, "image/jpeg")},
    )

    assert response.status_code == 200
    data = response.json()

    # Validate forensic sealing
    assert "investigation_id" in data
    assert len(data["media_sha256"]) == 64  # SHA-256 length
    assert data["filename"] == "test_evidence.jpg"

    # Validate evidentiary quality assessment
    assert data["evidentiary_quality"]["resolution"] == "600x600"
    assert data["evidentiary_quality"]["compression_warning"] is False

    # Validate model card metadata
    assert data["model_card"]["backbone"] == "EfficientNet-B0"
    assert data["model_card"]["device"] == "cpu"


def test_analyze_image_invalid_mime():
    """Verify non-image uploads are rejected with HTTP 415."""
    text_buffer = io.BytesIO(b"This is not a forensic image")

    response = client.post(
        "/api/v1/analyze/image",
        files={"file": ("document.txt", text_buffer, "text/plain")},
    )

    assert response.status_code == 415
    assert "Unsupported file type" in response.json()["detail"]


def test_analyze_image_empty_file():
    """Verify empty uploads are rejected with HTTP 400."""
    empty_buffer = io.BytesIO(b"")

    response = client.post(
        "/api/v1/analyze/image",
        files={"file": ("empty.jpg", empty_buffer, "image/jpeg")},
    )

    assert response.status_code == 400
    assert "empty" in response.json()["detail"].lower()
