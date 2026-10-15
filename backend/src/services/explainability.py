import base64
import io
from typing import Tuple, Optional
import numpy as np
import cv2
import torch
from PIL import Image


class ExplainabilityService:
    """
    Eigen-CAM explainability generator.
    Produces mathematically sound visual activation saliency maps
    from the final convolutional feature maps without requiring backward gradients.
    """

    @staticmethod
    def generate_eigen_cam(
        feature_maps: torch.Tensor,
        face_image: Image.Image,
        alpha: float = 0.55,
    ) -> Tuple[str, np.ndarray]:
        """
        Computes Eigen-CAM by projecting activation channels along their 1st principal component.
        
        Args:
            feature_maps: Tensor of shape (1, C, H, W) from the final convolutional layer.
            face_image: The aligned face crop PIL Image (224, 224).
            alpha: Saliency map blending opacity factor.
            
        Returns:
            Tuple of (base64_encoded_overlay_image, raw_normalized_heatmap_array).
        """
        # Detach and convert features to 3D numpy array (C, H, W)
        features = feature_maps.squeeze(0).detach().cpu().numpy()
        c, h, w = features.shape

        # Reshape to (C, H*W) for Singular Value Decomposition
        flat_features = features.reshape(c, -1)
        flat_features = flat_features - np.mean(flat_features, axis=1, keepdims=True)

        # 1st Principal Component via SVD
        try:
            u, s, vh = np.linalg.svd(flat_features, full_matrices=False)
            first_component = u[:, 0]
            # Project channels onto 1st principal component
            cam = np.dot(first_component, features.reshape(c, -1)).reshape(h, w)
        except Exception:
            # Fallback to channel-wise mean if SVD fails on degenerate tensors
            cam = np.mean(features, axis=0)

        # Apply ReLU to keep positive activations
        cam = np.maximum(cam, 0)

        # Normalize to [0, 1]
        cam_min, cam_max = np.min(cam), np.max(cam)
        if cam_max - cam_min > 1e-6:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        # Resize heatmap to match face crop dimensions and clip interpolation bounds
        target_w, target_h = face_image.size
        heatmap_resized = cv2.resize(cam, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
        heatmap_resized = np.clip(heatmap_resized, 0.0, 1.0)
        heatmap_uint8 = np.uint8(255 * heatmap_resized)

        # Apply JET colormap for high-contrast forensic visualization
        heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        heatmap_colored_rgb = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

        # Alpha blend with original face crop
        original_rgb = np.array(face_image.convert("RGB"))
        blended = np.uint8(alpha * heatmap_colored_rgb + (1.0 - alpha) * original_rgb)

        # Encode blended image to Base64 JPEG
        pil_blended = Image.fromarray(blended)
        buffer = io.BytesIO()
        pil_blended.save(buffer, format="JPEG", quality=90)
        base64_str = f"data:image/jpeg;base64,{base64.b64encode(buffer.getvalue()).decode('utf-8')}"

        return base64_str, heatmap_resized


explainability_service = ExplainabilityService()
