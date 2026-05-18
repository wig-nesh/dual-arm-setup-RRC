# ffs_server.py

import io
import os
import json
import traceback
import tempfile

import cv2
import imageio
import numpy as np
import torch

from flask import Flask, request, Response, jsonify

from omegaconf import OmegaConf
from core.utils.utils import InputPadder

from Utils import (
    AMP_DTYPE,
    set_logging_format,
    set_seed,
)

# =============================================================================
# CONFIG
# =============================================================================

MODEL_PATH = "weights/23-36-37/model_best_bp2_serialize.pth"

VALID_ITERS = 16
MAX_DISP = 256
SCALE = 1.0

# =============================================================================
# APP
# =============================================================================

app = Flask(__name__)

# =============================================================================
# INIT
# =============================================================================

set_logging_format()
set_seed(0)

torch.autograd.set_grad_enabled(False)

print(f"[INFO] Loading FFS model: {MODEL_PATH}")

model = torch.load(
    MODEL_PATH,
    map_location="cpu",
    weights_only=False
)

model.args.valid_iters = VALID_ITERS
model.args.max_disp = MAX_DISP

model.cuda().eval()

print("[INFO] FFS model loaded")

# =============================================================================
# UTILITIES
# =============================================================================

def decode_image(file_storage, flags=cv2.IMREAD_UNCHANGED):

    image_bytes = file_storage.read()

    np_arr = np.frombuffer(image_bytes, np.uint8)

    image = cv2.imdecode(np_arr, flags)

    if image is None:
        raise ValueError("Failed to decode image")

    return image


def serialize_depth(depth):

    buf = io.BytesIO()

    np.savez_compressed(
        buf,
        depth=depth.astype(np.float32)
    )

    return buf.getvalue()


# =============================================================================
# CORE INFERENCE
# =============================================================================

@torch.no_grad()
def run_ffs_depth(
    left_ir,
    right_ir,
    rgb,
    K_ir,
    baseline,
):

    # -------------------------------------------------------------------------
    # MATCH ORIGINAL SCRIPT
    # -------------------------------------------------------------------------

    img0 = left_ir
    img1 = right_ir

    if len(img0.shape) == 2:
        img0 = np.tile(img0[..., None], (1, 1, 3))
        img1 = np.tile(img1[..., None], (1, 1, 3))

    img0 = img0[..., :3]
    img1 = img1[..., :3]

    H, W = img0.shape[:2]

    img0 = cv2.resize(
        img0,
        fx=SCALE,
        fy=SCALE,
        dsize=None
    )

    img1 = cv2.resize(
        img1,
        dsize=(img0.shape[1], img0.shape[0])
    )

    H, W = img0.shape[:2]

    # -------------------------------------------------------------------------
    # TORCH
    # -------------------------------------------------------------------------

    img0_t = (
        torch.as_tensor(img0)
        .cuda()
        .float()[None]
        .permute(0, 3, 1, 2)
    )

    img1_t = (
        torch.as_tensor(img1)
        .cuda()
        .float()[None]
        .permute(0, 3, 1, 2)
    )

    padder = InputPadder(
        img0_t.shape,
        divis_by=32,
        force_square=False
    )

    img0_t, img1_t = padder.pad(
        img0_t,
        img1_t
    )

    # -------------------------------------------------------------------------
    # FORWARD
    # -------------------------------------------------------------------------

    with torch.amp.autocast(
        "cuda",
        enabled=True,
        dtype=AMP_DTYPE
    ):

        disp = model.forward(
            img0_t,
            img1_t,
            iters=VALID_ITERS,
            test_mode=True,
            optimize_build_volume='pytorch1'
        )

    # -------------------------------------------------------------------------
    # UNPAD
    # -------------------------------------------------------------------------

    disp = padder.unpad(
        disp.float()
    )

    disp = (
        disp.data
        .cpu()
        .numpy()
        .reshape(H, W)
        .clip(0, None)
    )

    # -------------------------------------------------------------------------
    # REMOVE INVISIBLE
    # -------------------------------------------------------------------------

    yy, xx = np.meshgrid(
        np.arange(disp.shape[0]),
        np.arange(disp.shape[1]),
        indexing='ij'
    )

    us_right = xx - disp

    invalid = us_right < 0

    disp[invalid] = np.inf

    # -------------------------------------------------------------------------
    # DEPTH
    # -------------------------------------------------------------------------

    K_ir = K_ir.copy()

    K_ir[:2] *= SCALE

    depth = K_ir[0, 0] * baseline / disp

    depth = depth.astype(np.float32)

    return depth


# =============================================================================
# ROUTE
# =============================================================================

@app.route("/ffs_depth", methods=["POST"])
def ffs_depth():

    try:

        # ---------------------------------------------------------------------
        # VALIDATE INPUTS
        # ---------------------------------------------------------------------

        required_files = ["left", "right", "rgb"]

        for key in required_files:
            if key not in request.files:
                return jsonify({
                    "error": f"Missing file: {key}"
                }), 400

        if "K_ir" not in request.form:
            return jsonify({
                "error": "Missing K_ir"
            }), 400

        if "baseline" not in request.form:
            return jsonify({
                "error": "Missing baseline"
            }), 400

        # ---------------------------------------------------------------------
        # LOAD IMAGES
        # ---------------------------------------------------------------------

        left_ir = decode_image(
            request.files["left"]
        )

        right_ir = decode_image(
            request.files["right"]
        )

        rgb = decode_image(
            request.files["rgb"],
            flags=cv2.IMREAD_COLOR
        )

        # ---------------------------------------------------------------------
        # LOAD PARAMS
        # ---------------------------------------------------------------------

        K_ir = np.array(
            json.loads(request.form["K_ir"]),
            dtype=np.float32
        )

        baseline = float(
            request.form["baseline"]
        )

        # ---------------------------------------------------------------------
        # INFERENCE
        # ---------------------------------------------------------------------

        depth = run_ffs_depth(
            left_ir=left_ir,
            right_ir=right_ir,
            rgb=rgb,
            K_ir=K_ir,
            baseline=baseline,
        )

        # ---------------------------------------------------------------------
        # RESPONSE
        # ---------------------------------------------------------------------

        response_bytes = serialize_depth(
            depth
        )

        return Response(
            response_bytes,
            mimetype="application/octet-stream"
        )

    except Exception as e:

        traceback.print_exc()

        return jsonify({
            "error": str(e),
            "traceback": traceback.format_exc(),
        }), 500


# =============================================================================
# HEALTH
# =============================================================================

@app.route("/health", methods=["GET"])
def health():

    return jsonify({
        "status": "ok"
    })


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=8000,
        threaded=True,
    )
