import numpy as np
import requests
import io
import os
from typing import Dict


class ModelClient:
    """
    Client to communicate with a remote pointcloud inference server using NPZ serialization.
    """

    def __init__(self, server_url: str, timeout: float = 10.0):
        self.server_url = server_url.rstrip("/")
        self.timeout = timeout
        self.endpoint = f"{self.server_url}/predict"

    def _serialize_inputs(self, points: np.ndarray) -> bytes:
        """Serializes pointcloud to a compressed NPZ in memory."""
        if not isinstance(points, np.ndarray):
            raise TypeError("Points must be a numpy array.")

        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(f"Points must have shape (N, 3), got {points.shape}")

        if points.dtype != np.float32:
            # We convert instead of raising to be helpful, but keeping requirements in mind
            points = points.astype(np.float32)

        buf = io.BytesIO()
        np.savez_compressed(buf, points=points)
        return buf.getvalue()

    def _deserialize_outputs(self, response_content: bytes) -> Dict[str, np.ndarray]:
        """Deserializes compressed NPZ response into a dictionary of arrays."""
        try:
            buf = io.BytesIO(response_content)
            with np.load(buf, allow_pickle=True) as data:
                # np.load returns a lazy-loading NpzFile; convert to dict to pull into memory
                result = {}
                for key in data.files:
                    val = data[key]
                    # If it's a pickle object, try to extract arrays
                    if isinstance(val, np.ndarray):
                        result[key] = val
                    else:
                        # Convert to numpy array if possible
                        try:
                            result[key] = np.array(val)
                        except:
                            pass
                return result
        except Exception as e:
            raise ValueError(f"Failed to decode NPZ response from server: {e}")

    def _post_request(
        self, data_bytes: bytes, platform_height: float
    ) -> Dict[str, np.ndarray]:
        """Sends a multipart/form-data POST request with the NPZ file."""
        files = {"data": ("cloud.npz", data_bytes, "application/octet-stream")}
        data = {"platform_height": str(platform_height)}

        try:
            response = requests.post(
                self.endpoint, files=files, data=data, timeout=self.timeout
            )
            response.raise_for_status()
            return self._deserialize_outputs(response.content)
        except requests.exceptions.HTTPError as e:
            # For 422 errors, server may still return unfiltered grasps in NPZ format
            if response.status_code == 422:
                try:
                    return self._deserialize_outputs(response.content)
                except Exception:
                    pass
            # Try to extract the JSON error message from the server
            error_msg = str(e)
            try:
                server_info = response.json()
                if "error" in server_info:
                    error_msg += f" | Server Error: {server_info['error']}"
            except Exception:
                pass
            raise RuntimeError(f"Pointcloud server request failed: {error_msg}")
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Pointcloud server request failed: {e}")

    def predict(
        self, points: np.ndarray, platform_height: float = 0.08
    ) -> Dict[str, np.ndarray]:
        """
        Sends pointcloud to server and returns model outputs.

        Args:
            points: NumPy array of shape (N, 3) and dtype float32.
            platform_height: The height of the platform in meters.

        Returns:
            Dictionary mapping output names (grasps, scores, etc.) to NumPy arrays.
        """
        data_bytes = self._serialize_inputs(points)
        return self._post_request(data_bytes, platform_height)


if __name__ == "__main__":
    # Small demo
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--url", type=str, default="http://localhost:8000", help="Server URL"
    )
    args = parser.parse_args()

    client = ModelClient(args.url)

    # Generate a random pointcloud (e.g., 2048 points)
    print("Generating random pointcloud (2048, 3)...")
    test_points = np.random.rand(2048, 3).astype(np.float32)

    print(f"Sending request to {client.endpoint}...")
    try:
        # Note: This will fail unless you have a server running
        results = client.predict(test_points)

        print("\nSuccess! Received outputs:")
        for key, arr in results.items():
            print(f" - {key}: shape {arr.shape}, dtype {arr.dtype}")

    except Exception as e:
        print(f"\nRequest failed (expected if no server is running):")
        print(f"Error: {e}")
