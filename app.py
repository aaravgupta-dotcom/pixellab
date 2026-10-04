"""PixelLab Web Application Server
Flask server providing REST APIs for filter metadata, batch filter execution,
step-by-step pixel arithmetic traces, and raw ndarray region inspection.
"""

import os
import io
import time
import base64
import numpy as np
from PIL import Image
from flask import Flask, render_template, request, jsonify
from filters import FILTER_REGISTRY, get_filters_metadata

app = Flask(__name__, template_folder="templates", static_folder="static")
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024  # 32MB max payload


def decode_image_base64(data_uri: str) -> np.ndarray:
    """Decode a base64 Data URI or raw base64 string into a uint8 NumPy ndarray."""
    if "," in data_uri:
        data_uri = data_uri.split(",", 1)[1]
    raw_bytes = base64.b64decode(data_uri)
    pil_img = Image.open(io.BytesIO(raw_bytes))
    # Convert palette / grayscale images to RGBA
    if pil_img.mode not in ("RGB", "RGBA"):
        pil_img = pil_img.convert("RGBA")
    return np.array(pil_img, dtype=np.uint8)


def encode_image_base64(arr: np.ndarray, format: str = "PNG") -> str:
    """Encode a uint8 NumPy array into a base64 Data URI using Pillow."""
    pil_img = Image.fromarray(arr)
    buffer = io.BytesIO()
    pil_img.save(buffer, format=format)
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    mime = "image/png" if format.upper() == "PNG" else "image/jpeg"
    return f"data:{mime};base64,{encoded}"


@app.route("/")
def index():
    """Serve the main PixelLab studio web application."""
    return render_template("index.html")


@app.route("/api/filters", methods=["GET"])
def api_filters():
    """Return complete metadata for all 50 filters including parameters, docstrings, and source code."""
    metadata = get_filters_metadata()
    return jsonify({"status": "ok", "count": len(metadata), "filters": metadata})


@app.route("/api/apply", methods=["POST"])
def api_apply():
    """Apply an ordered stack of filters using authoritative NumPy operations."""
    req_json = request.get_json(force=True)
    if not req_json or "image" not in req_json:
        return jsonify({"error": "Missing 'image' parameter"}), 400

    try:
        arr = decode_image_base64(req_json["image"])
    except Exception as e:
        return jsonify({"error": f"Failed to decode image: {str(e)}"}), 400

    # Downscale working copy if long side exceeds 1600px
    H, W = arr.shape[:2]
    max_dim = max(H, W)
    if max_dim > 1600:
        scale = 1600.0 / max_dim
        new_w = int(round(W * scale))
        new_h = int(round(H * scale))
        pil_tmp = Image.fromarray(arr)
        arr = np.array(pil_tmp.resize((new_w, new_h), Image.Resampling.LANCZOS), dtype=np.uint8)

    filter_stack = req_json.get("filters", [])
    start_time = time.perf_counter()

    current_arr = arr
    for item in filter_stack:
        fid = item.get("filter_id")
        if fid not in FILTER_REGISTRY:
            continue
        params = item.get("params", {})
        fn = FILTER_REGISTRY[fid]["fn"]
        try:
            current_arr = fn(current_arr, **params)
        except Exception as e:
            return jsonify({"error": f"Filter '{fid}' execution failed: {str(e)}"}), 500

    elapsed_ms = (time.perf_counter() - start_time) * 1000.0

    output_format = req_json.get("format", "PNG").upper()
    data_uri = encode_image_base64(current_arr, format=output_format)

    return jsonify(
        {
            "status": "ok",
            "image": data_uri,
            "timing_ms": round(elapsed_ms, 2),
            "shape": list(current_arr.shape),
            "dtype": str(current_arr.dtype),
        }
    )


@app.route("/api/trace", methods=["POST"])
def api_trace():
    """Compute step-by-step arithmetic trace for a specific pixel and filter."""
    req_json = request.get_json(force=True)
    if not req_json:
        return jsonify({"error": "Empty request body"}), 400

    fid = req_json.get("filter_id")
    if fid not in FILTER_REGISTRY:
        return jsonify({"error": f"Unknown filter: {fid}"}), 400

    x = int(req_json.get("x", 0))
    y = int(req_json.get("y", 0))
    params = req_json.get("params", {})

    r, g, b, a = 180, 120, 60, 255
    if "image" in req_json:
        try:
            arr = decode_image_base64(req_json["image"])
            H, W = arr.shape[:2]
            clamped_y = max(0, min(H - 1, y))
            clamped_x = max(0, min(W - 1, x))
            pix = arr[clamped_y, clamped_x]
            r = int(pix[0])
            g = int(pix[1])
            b = int(pix[2])
            a = int(pix[3]) if len(pix) > 3 else 255
        except Exception:
            pass

    trace_fn = FILTER_REGISTRY[fid]["trace_fn"]
    trace_data = trace_fn(r, g, b, a, params)

    return jsonify(
        {
            "status": "ok",
            "filter_id": fid,
            "filter_name": FILTER_REGISTRY[fid]["name"],
            "x": x,
            "y": y,
            "input_pixel": [r, g, b, a],
            "formula": trace_data.get("formula", ""),
            "steps": trace_data.get("steps", []),
            "output_pixel": trace_data.get("output", [r, g, b, a]),
        }
    )


@app.route("/api/pixels", methods=["POST"])
def api_pixels():
    """Retrieve raw ndarray subregion values, memory footprint, and channel statistics."""
    req_json = request.get_json(force=True)
    if not req_json or "image" not in req_json:
        return jsonify({"error": "Missing 'image' parameter"}), 400

    try:
        arr = decode_image_base64(req_json["image"])
    except Exception as e:
        return jsonify({"error": f"Failed to decode image: {str(e)}"}), 400

    region = req_json.get("region", {"x": 0, "y": 0, "width": 16, "height": 16})
    rx = max(0, int(region.get("x", 0)))
    ry = max(0, int(region.get("y", 0)))
    rw = min(16, max(1, int(region.get("width", 16))))
    rh = min(16, max(1, int(region.get("height", 16))))

    H, W = arr.shape[:2]
    end_x = min(W, rx + rw)
    end_y = min(H, ry + rh)

    sub = arr[ry:end_y, rx:end_x]

    # Convert to json serializable 3D matrix
    pixels_matrix = []
    for y_idx in range(sub.shape[0]):
        row = []
        for x_idx in range(sub.shape[1]):
            val = [int(c) for c in sub[y_idx, x_idx]]
            row.append(val)
        pixels_matrix.append(row)

    total_bytes = arr.nbytes

    return jsonify(
        {
            "status": "ok",
            "shape": list(arr.shape),
            "dtype": str(arr.dtype),
            "total_bytes": total_bytes,
            "memory_mb": round(total_bytes / (1024 * 1024), 2),
            "region": {"x": rx, "y": ry, "width": sub.shape[1], "height": sub.shape[0]},
            "pixels": pixels_matrix,
        }
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    print(f"Starting PixelLab Image Filter Studio on http://127.0.0.1:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)
