import cv2
import numpy as np
import requests
import json
import io
from typing import Tuple, Optional


class SAMClient:
    """
    Client to communicate with a remote SAM inference server.
    """

    def __init__(self, server_url: str, timeout: float = 10.0):
        self.server_url = server_url.rstrip("/")
        self.timeout = timeout
        self.endpoint = f"{self.server_url}/sam_predict"

    def _encode_image(self, image: np.ndarray) -> bytes:
        """Encodes a BGR image to JPEG bytes."""
        success, encoded = cv2.imencode(".jpg", image)
        if not success:
            raise ValueError("Failed to encode image to JPEG.")
        return encoded.tobytes()

    def _decode_mask(self, response_content: bytes) -> np.ndarray:
        """Decodes PNG binary response to a uint8 mask."""
        nparr = np.frombuffer(response_content, np.uint8)
        mask = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError("Failed to decode mask from server response.")
        return mask

    def _post_request(self, image_bytes: bytes, data: dict) -> np.ndarray:
        """Sends a multipart/form-data POST request to the server."""
        files = {"image": ("image.jpg", image_bytes, "image/jpeg")}

        try:
            response = requests.post(
                self.endpoint, files=files, data=data, timeout=self.timeout
            )
            response.raise_for_status()
            return self._decode_mask(response.content)
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"SAM server request failed: {e}")

    def predict_from_point(
        self, image: np.ndarray, point_xy: Tuple[int, int]
    ) -> np.ndarray:
        """
        Predict a mask from a single keypoint prompt.
        Args:
            image: BGR image array.
            point_xy: (x, y) coordinates.
        """
        img_bytes = self._encode_image(image)
        data = {"point": json.dumps(list(point_xy))}
        return self._post_request(img_bytes, data)

    def predict_from_box(
        self, image: np.ndarray, bbox_xyxy: Tuple[int, int, int, int]
    ) -> np.ndarray:
        """
        Predict a mask from a bounding box prompt.
        Args:
            image: BGR image array.
            bbox_xyxy: (x1, y1, x2, y2) coordinates.
        """
        img_bytes = self._encode_image(image)
        data = {"bbox": json.dumps(list(bbox_xyxy))}
        return self._post_request(img_bytes, data)


if __name__ == "__main__":
    # Example usage for testing
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--img", type=str, help="Path to image")
    parser.add_argument(
        "--url", type=str, default="http://localhost:8000", help="Server URL"
    )
    args = parser.parse_args()

    if args.img and os.path.exists(args.img):
        client = SAMClient(args.url)
        img = cv2.imread(args.img)

        # Simple point prompt at center of image for demo
        h, w = img.shape[:2]
        center = (w // 2, h // 2)

        print(f"Requesting mask for point {center}...")
        try:
            mask = client.predict_from_point(img, center)

            # Visualization
            vis = img.copy()
            vis[mask > 0] = [0, 255, 0]  # Green tint for mask
            cv2.circle(vis, center, 5, (0, 0, 255), -1)  # Red dot for prompt

            cv2.imshow("SAM Prediction", vis)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except Exception as e:
            print(f"Error: {e}")
    else:
        print("Please provide a valid image path via --img to run the demo.")
