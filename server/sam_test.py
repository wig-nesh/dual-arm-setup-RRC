# DEPENDENCIES NOT HANDLED!!! figure it out yourself on server

import json
import traceback

import cv2
import numpy as np
from flask import Flask, request, Response, jsonify
from ultralytics import SAM

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------

MODEL_PATH = "sam2.1_l.pt"

app = Flask(__name__)

print(f"[INFO] Loading SAM model: {MODEL_PATH}")
model = SAM(MODEL_PATH)
print("[INFO] SAM model loaded")


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def decode_image(file_storage) -> np.ndarray:
    """
    Decode uploaded image into BGR numpy array.
    """
    image_bytes = file_storage.read()

    np_arr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError("Failed to decode uploaded image")

    return image


def encode_mask_png(mask: np.ndarray) -> bytes:
    """
    Encode uint8 mask to PNG bytes.
    """
    success, encoded = cv2.imencode(".png", mask)

    if not success:
        raise ValueError("Failed to encode mask")

    return encoded.tobytes()


def run_sam(
        image_bgr,
        point=None,
        bbox=None,
    ):
    """
    Run SAM inference and return binary uint8 mask.
    """

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    kwargs = {
        "source": image_rgb,
    }

    if point is not None:
        # point format: [x, y]
        kwargs["points"] = [point]

    if bbox is not None:
        # bbox format: [x1, y1, x2, y2]
        kwargs["bboxes"] = [bbox]

    results = model.predict(**kwargs)

    if len(results) == 0 or results[0].masks is None:
        raise RuntimeError("SAM returned no masks")

    masks = results[0].masks.data.cpu().numpy()

    if len(masks) == 0:
        raise RuntimeError("SAM returned empty masks")

    # combine all masks
    combined_mask = np.any(masks, axis=0).astype(np.uint8) * 255

    return combined_mask


# -----------------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------------

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/sam_predict", methods=["POST"])
def sam_predict():
    try:
        # ---------------------------------------------------------------------
        # Validate image
        # ---------------------------------------------------------------------

        if "image" not in request.files:
            return jsonify({"error": "Missing image file"}), 400

        image_file = request.files["image"]
        image = decode_image(image_file)

        # ---------------------------------------------------------------------
        # Parse prompts
        # ---------------------------------------------------------------------

        point = None
        bbox = None

        if "point" in request.form:
            point = json.loads(request.form["point"])

            if not (
                isinstance(point, list)
                and len(point) == 2
            ):
                return jsonify({"error": "Invalid point format"}), 400

        if "bbox" in request.form:
            bbox = json.loads(request.form["bbox"])

            if not (
                isinstance(bbox, list)
                and len(bbox) == 4
            ):
                return jsonify({"error": "Invalid bbox format"}), 400

        if point is None and bbox is None:
            return jsonify(
                {"error": "Provide either point or bbox"}
            ), 400

        # ---------------------------------------------------------------------
        # Run inference
        # ---------------------------------------------------------------------

        mask = run_sam(
            image_bgr=image,
            point=point,
            bbox=bbox,
        )

        # ---------------------------------------------------------------------
        # Return PNG mask
        # ---------------------------------------------------------------------

        mask_png = encode_mask_png(mask)

        return Response(
            mask_png,
            mimetype="image/png",
        )

    except Exception as e:
        traceback.print_exc()

        return jsonify(
            {
                "error": str(e),
                "traceback": traceback.format_exc(),
            }
        ), 500


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=8000,
        threaded=True,
    )
