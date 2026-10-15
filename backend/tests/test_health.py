from fastapi.testclient import TestClient
from src.main import app

client = TestClient(app)


def test_root_endpoint():
    """Verify root discovery endpoint returns metadata."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "TrueFrame AI Backend"
    assert "version" in data
    assert "documentation" in data


def test_health_liveness():
    """Verify liveness probe returns HTTP 200 and ok status."""
    response = client.get("/api/v1/health/live")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "version" in data


def test_health_readiness():
    """Verify readiness probe returns configuration and storage status."""
    response = client.get("/api/v1/health/ready")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ("ready", "degraded")
    assert "active_model" in data
    assert "inference_device" in data
