"""Deterministic Filter Parity Comparison (Python/NumPy vs JavaScript)
Verifies that JS and NumPy filter outputs differ by at most 1 unit on uint8 scale.
"""

import json
import subprocess
import numpy as np
from filters import FILTER_REGISTRY

# Deterministic tone & color filters to test for exact numerical parity
DETERMINISTIC_FILTERS = [
    ("brightness", {"offset": 20}),
    ("contrast", {"factor": 1.4}),
    ("exposure", {"ev": 0.5}),
    ("saturation", {"factor": 1.5}),
    ("grayscale", {"mode": "rec709"}),
    ("invert", {"amount": 1.0}),
    ("sepia", {"intensity": 0.8}),
    ("solarize", {"threshold": 0.5}),
    ("posterize", {"levels": 4}),
]

def run_parity():
    # Create 8x8 synthetic image
    H, W = 8, 8
    img = np.zeros((H, W, 4), dtype=np.uint8)
    for y in range(H):
        for x in range(W):
            img[y, x] = [x * 32, y * 32, (x + y) * 16, 255]

    report = []
    all_passed = True

    for fid, params in DETERMINISTIC_FILTERS:
        entry = FILTER_REGISTRY[fid]
        py_out = entry["fn"](img, **params)

        # Call JS via node
        node_script = f"""
        global.ImageData = class ImageData {{
          constructor(w, h) {{ this.width = w; this.height = h; this.data = new Uint8ClampedArray(w * h * 4); }}
        }};
        const {{ JS_FILTERS }} = require('./static/js/filters.js');
        const img = new ImageData({W}, {H});
        const pyIn = {json.dumps(img.flatten().tolist())};
        for (let i = 0; i < pyIn.length; i++) img.data[i] = pyIn[i];
        const res = JS_FILTERS['{fid}'](img, {json.dumps(params)});
        process.stdout.write(JSON.stringify(Array.from(res.data)));
        """
        proc = subprocess.run(["node", "-e", node_script], capture_output=True, text=True, check=True)
        js_raw = json.loads(proc.stdout)
        js_out = np.array(js_raw, dtype=np.uint8).reshape((H, W, 4))

        # Check maximum absolute difference
        diff = np.abs(py_out.astype(np.int32) - js_out.astype(np.int32))
        max_diff = np.max(diff)
        passed = max_diff <= 1

        report.append({
            "filter": fid,
            "params": params,
            "max_abs_diff": int(max_diff),
            "status": "PASS" if passed else "FAIL"
        })
        if not passed:
            all_passed = False

    print("=== PIXELLAB NUMERICAL PARITY REPORT (JS vs NUMPY) ===")
    for row in report:
        print(f"[{row['status']}] Filter: {row['filter']:<15} | Max Diff: {row['max_abs_diff']} level(s) | Params: {row['params']}")
    print(f"Overall Parity Status: {'PASSED (<= 1 level per channel)' if all_passed else 'FAILED'}")
    return all_passed

if __name__ == "__main__":
    success = run_parity()
    if not success:
        exit(1)
