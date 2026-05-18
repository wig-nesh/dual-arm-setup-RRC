import cv2
import numpy as np
import requests
import json
import io
from typing import Dict, Optional, Tuple


class FFSClient:
    """
    Client to communicate with a remote Fast Foundation Stereo (FFS) inference server.
    """

    def __init__(self, server_url: str, timeout: float = 10.0):
        self.server_url = server_url.rstrip("/")
        self.timeout = timeout
        self.endpoint = f"{self.server_url}/ffs_depth"

    def _encode_image(self, image: np.ndarray, ext: str = ".png") -> bytes:
        """Encodes an image to bytes."""
        success, encoded = cv2.imencode(ext, image)
        if not success:
            raise ValueError(f"Failed to encode image to {ext}.")
        return encoded.tobytes()

    def _deserialize_outputs(self, response_content: bytes) -> np.ndarray:
        """Deserializes compressed NPZ response containing the depth map."""
        try:
            buf = io.BytesIO(response_content)
            with np.load(buf, allow_pickle=True) as data:
                if 'depth' in data.files:
                    return data['depth']
                else:
                    # Return the first array if 'depth' isn't explicitly named
                    return data[data.files[0]]
        except Exception as e:
            raise ValueError(f"Failed to decode NPZ response from server: {e}")

    def predict_depth(
        self, 
        left_ir: np.ndarray, 
        right_ir: np.ndarray, 
        rgb: np.ndarray,
        K_ir: np.ndarray,
        baseline: float
    ) -> np.ndarray:
        """
        Predict depth map using FFS.

        Args:
            left_ir: Left infrared image.
            right_ir: Right infrared image.
            rgb: RGB image.
            K_ir: 3x3 intrinsic matrix of the IR cameras.
            baseline: Baseline between left and right IR cameras in meters.

        Returns:
            depth: NumPy array of shape (H, W) and dtype float32 (meters).
        """
        left_bytes = self._encode_image(left_ir, ".png")
        right_bytes = self._encode_image(right_ir, ".png")
        rgb_bytes = self._encode_image(rgb, ".jpg")

        files = {
            "left": ("left.png", left_bytes, "image/png"),
            "right": ("right.png", right_bytes, "image/png"),
            "rgb": ("rgb.jpg", rgb_bytes, "image/jpeg"),
        }

        data = {
            "K_ir": json.dumps(K_ir.tolist()),
            "baseline": baseline
        }

        try:
            response = requests.post(
                self.endpoint, files=files, data=data, timeout=self.timeout
            )
            response.raise_for_status()
            return self._deserialize_outputs(response.content)
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"FFS server request failed: {e}")
