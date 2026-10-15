import numpy as np
import torch
from PIL import Image
from src.services.model_engine import model_engine
from src.services.explainability import explainability_service


def test_model_engine_inference_shape_and_score():
    """Verify that model engine produces calibrated probability and hooks feature maps."""
    # Create synthetic RGB image (224, 224, 3)
    dummy_image = Image.fromarray(np.uint8(np.random.randint(50, 200, (224, 224, 3))))

    score, feature_maps = model_engine.predict(dummy_image)

    # Validate probability bounds
    assert 0.0 <= score.manipulated_probability <= 1.0
    assert 0.0 <= score.authentic_probability <= 1.0
    assert abs(score.manipulated_probability + score.authentic_probability - 1.0) < 1e-4

    # Validate uncertainty and categorical label
    assert score.uncertainty_margin > 0.0
    assert score.assessment_label in (
        "Suspicious Manipulation Detected",
        "Likely Authentic",
        "Inconclusive / Ambiguous Artifacts",
    )

    # Validate feature maps hooked from final convolutional layer
    assert feature_maps is not None
    assert feature_maps.ndim == 4
    assert feature_maps.shape[1] == 1280  # EfficientNet-B0 top feature channels


def test_eigen_cam_generation():
    """Verify Eigen-CAM produces a base64 overlay and normalized heatmap."""
    dummy_image = Image.fromarray(np.uint8(np.random.randint(50, 200, (224, 224, 3))))
    # Synthetic activation tensor (1, 1280, 7, 7)
    dummy_features = torch.randn(1, 1280, 7, 7)

    b64_str, raw_cam = explainability_service.generate_eigen_cam(
        feature_maps=dummy_features,
        face_image=dummy_image,
    )

    assert b64_str.startswith("data:image/jpeg;base64,")
    assert raw_cam.shape == (224, 224)
    assert 0.0 <= np.min(raw_cam) <= np.max(raw_cam) <= 1.0
