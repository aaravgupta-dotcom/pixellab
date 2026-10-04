"""Artistic and Stylization Filters (Filters 33 to 40)
Authoritative NumPy implementations for painting, sketching, halftoning, and procedural geometry.
"""

import numpy as np
from filters.helpers import to_float32, to_uint8, convolve2d, rgb_to_luminance, _make_gaussian_kernel


def filter_oil_paint(img: np.ndarray, radius: int = 2) -> np.ndarray:
    """Kuwahara non-linear smoothing filter creating oil painting strokes.

    Calculates mean and variance in four overlapping spatial quadrants around each pixel,
    assigning the mean color of the quadrant possessing minimal variance to preserve sharp edges.
    """
    f = to_float32(img)
    r = max(1, min(int(radius), 4))
    m = r + 1
    k_full = 2 * r + 1

    kernels = []
    for y_start, x_start in [(0, 0), (0, r), (r, 0), (r, r)]:
        q_k = np.zeros((k_full, k_full), dtype=np.float32)
        q_k[y_start : y_start + m, x_start : x_start + m] = 1.0 / (m * m)
        kernels.append(q_k)

    luma = rgb_to_luminance(f[:, :, :3], mode="rec709")
    luma_sq = luma**2

    variances = []
    means_rgb = []
    for q_k in kernels:
        m_luma = convolve2d(luma, q_k)
        m_sq = convolve2d(luma_sq, q_k)
        var = np.maximum(0.0, m_sq - m_luma**2)
        variances.append(var)

        q_rgb = np.zeros_like(f[:, :, :3])
        for c in range(3):
            q_rgb[:, :, c] = convolve2d(f[:, :, c], q_k)
        means_rgb.append(q_rgb)

    variances = np.stack(variances, axis=-1)
    best_q = np.argmin(variances, axis=-1)

    means_rgb = np.stack(means_rgb, axis=-1)
    out_rgb = np.take_along_axis(means_rgb, best_q[:, :, None, None], axis=-1).squeeze(-1)

    out = f.copy()
    out[:, :, :3] = np.clip(out_rgb, 0.0, 1.0)
    return to_uint8(out)


def trace_oil_paint(r: int, g: int, b: int, a: int, params: dict) -> dict:
    rad = int(params.get("radius", 2))
    steps = [
        f"Kuwahara Filter (radius={rad}): 4 spatial quadrants Q1, Q2, Q3, Q4",
        "Quadrants: top-left, top-right, bottom-left, bottom-right",
        "Calculated local variance σ² for each quadrant",
        "Selected quadrant with minimal variance to preserve directional boundaries",
    ]
    return {
        "formula": "I_out(x, y) = Mean(Q_min_variance)",
        "steps": steps,
        "output": [r, g, b, a],
    }


def filter_pencil_sketch(
    img: np.ndarray, contrast: float = 1.5, blend: float = 1.0
) -> np.ndarray:
    """Color dodge blend between grayscale base and inverted Gaussian blurred mask."""
    f = to_float32(img)
    luma = rgb_to_luminance(f[:, :, :3], mode="rec709")
    inv_luma = 1.0 - luma

    k = _make_gaussian_kernel(radius=4, sigma=3.0)
    blurred_inv = convolve2d(inv_luma, k)

    # Color dodge: base / (1 - mask)
    dodge = np.clip(luma / np.maximum(1.0 - blurred_inv, 1e-4), 0.0, 1.0)

    # Boost contrast
    c = float(contrast)
    sketch = np.clip((dodge - 0.5) * c + 0.5, 0.0, 1.0)

    # Blend with original
    t = float(blend)
    sketch_3d = np.repeat(sketch[:, :, None], 3, axis=-1)
    out_rgb = (1.0 - t) * f[:, :, :3] + t * sketch_3d

    out = f.copy()
    out[:, :, :3] = out_rgb
    return to_uint8(out)


def trace_pencil_sketch(r: int, g: int, b: int, a: int, params: dict) -> dict:
    c = float(params.get("contrast", 1.5))
    steps = [
        "1. Converted to grayscale luminance L",
        "2. Inverted grayscale: L_inv = 1.0 − L",
        "3. Gaussian blurred inverted mask (σ=3.0)",
        "4. Applied Color Dodge formula: L / (1.0 − L_inv_blur)",
        f"5. Contrast scaling: (Dodge − 0.5) × {c:.2f} + 0.5",
    ]
    return {
        "formula": "Sketch = clamp((L / (1 − Blur(1−L)) − 0.5) × contrast + 0.5, 0, 1)",
        "steps": steps,
        "output": [r, g, b, a],
    }


def filter_cross_hatch(img: np.ndarray, density: int = 5, angle: float = 45.0) -> np.ndarray:
    """Multi-layer ink pen cross-hatching responding to tone density thresholds."""
    f = to_float32(img)
    H, W = f.shape[:2]
    luma = rgb_to_luminance(f[:, :, :3], mode="rec709")

    d = max(2, int(density))
    grid_y, grid_x = np.indices((H, W), dtype=np.float32)

    # 4 Hatch patterns
    h1 = ((grid_x + grid_y) % d < 1.0).astype(np.float32)
    h2 = ((grid_x - grid_y) % d < 1.0).astype(np.float32)
    h3 = (grid_y % d < 1.0).astype(np.float32)
    h4 = (grid_x % d < 1.0).astype(np.float32)

    ink = np.ones((H, W), dtype=np.float32)

    # Very dark: all 4 hatches
    ink[luma < 0.2] = 1.0 - (h1[luma < 0.2] + h2[luma < 0.2] + h3[luma < 0.2] + h4[luma < 0.2]).clip(0, 1)
    # Dark: 3 hatches
    m = (luma >= 0.2) & (luma < 0.4)
    ink[m] = 1.0 - (h1[m] + h2[m] + h3[m]).clip(0, 1)
    # Midtones: 2 hatches
    m = (luma >= 0.4) & (luma < 0.6)
    ink[m] = 1.0 - (h1[m] + h2[m]).clip(0, 1)
    # Light midtones: 1 hatch
    m = (luma >= 0.6) & (luma < 0.8)
    ink[m] = 1.0 - h1[m]
    # Highlights: clean paper (1.0)
    ink[luma >= 0.8] = 1.0

    out = f.copy()
    out[:, :, :3] = np.repeat(ink[:, :, None], 3, axis=-1)
    return to_uint8(out)


def trace_cross_hatch(r: int, g: int, b: int, a: int, params: dict) -> dict:
    d = int(params.get("density", 5))
    luma = (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255.0
    steps = [
        f"Luminance L = {luma:.2f}, Grid pitch = {d}px",
        "Tone bucket activates diagonal (+45°), orthogonal (−45°), and cross strokes",
        f"Level: {'Quad Hatch' if luma < 0.2 else 'Triple Hatch' if luma < 0.4 else 'Double Hatch' if luma < 0.6 else 'Single Hatch' if luma < 0.8 else 'Blank Paper'}",
    ]
    return {
        "formula": "Ink strokes activated progressively across luminance intervals [0.2, 0.4, 0.6, 0.8]",
        "steps": steps,
        "output": [r, g, b, a],
    }


def filter_pointillism(
    img: np.ndarray, dot_size: int = 6, density: float = 80.0, seed: int = 42
) -> np.ndarray:
    """Neo-impressionist stippling dots colored by localized neighborhood chromatic samples."""
    f = to_float32(img)
    H, W = f.shape[:2]
    rng = np.random.default_rng(int(seed))

    step = max(2, int(dot_size))
    # Base background: warm off-white canvas
    canvas = np.full((H, W, 3), 0.95, dtype=np.float32)

    # Grid sampling points with jitter
    for y in range(0, H, step):
        for x in range(0, W, step):
            if rng.random() * 100.0 > float(density):
                continue
            jy = int(np.clip(y + rng.integers(-step // 2, step // 2 + 1), 0, H - 1))
            jx = int(np.clip(x + rng.integers(-step // 2, step // 2 + 1), 0, W - 1))
            color = f[jy, jx, :3]

            rad = max(1, step // 2)
            y_min, y_max = max(0, jy - rad), min(H, jy + rad + 1)
            x_min, x_max = max(0, jx - rad), min(W, jx + rad + 1)

            gy, gx = np.ogrid[y_min - jy : y_max - jy, x_min - jx : x_max - jx]
            mask = gx * gx + gy * gy <= rad * rad
            canvas[y_min:y_max, x_min:x_max][mask] = color

    out = f.copy()
    out[:, :, :3] = canvas
    return to_uint8(out)


def trace_pointillism(r: int, g: int, b: int, a: int, params: dict) -> dict:
    ds = int(params.get("dot_size", 6))
    s = int(params.get("seed", 42))
    steps = [
        f"Dot size = {ds}px, Seed = {s}",
        "Image discretized into dot clusters",
        "Circles rendered with sampled local palette onto canvas substrate",
    ]
    return {
        "formula": "Disk(radius=d/2) stamped at jittered lattice nodes",
        "steps": steps,
        "output": [r, g, b, a],
    }


def filter_mosaic_tiles(img: np.ndarray, tile_size: int = 12, grout: int = 1) -> np.ndarray:
    """Grid binning mosaic with dark mortar grout boundary division."""
    f = to_float32(img)
    H, W = f.shape[:2]
    ts = max(2, int(tile_size))
    gr = max(0, int(grout))

    grid_y, grid_x = np.indices((H, W), dtype=np.int32)
    tile_y = (grid_y // ts) * ts + (ts // 2)
    tile_x = (grid_x // ts) * ts + (ts // 2)

    tile_y = np.clip(tile_y, 0, H - 1)
    tile_x = np.clip(tile_x, 0, W - 1)

    mosaic = f[tile_y, tile_x, :3].copy()

    if gr > 0:
        border_mask = ((grid_x % ts) < gr) | ((grid_y % ts) < gr)
        mosaic[border_mask] = 0.15  # dark cement grout

    out = f.copy()
    out[:, :, :3] = mosaic
    return to_uint8(out)


def trace_mosaic_tiles(r: int, g: int, b: int, a: int, params: dict) -> dict:
    ts = int(params.get("tile_size", 12))
    gr = int(params.get("grout", 1))
    steps = [
        f"Tile cell dimensions = {ts}×{ts}px, Grout thickness = {gr}px",
        "Pixel coordinates mapped to block center (x_cell, y_cell)",
        f"Mortar line check: (x mod {ts} < {gr}) or (y mod {ts} < {gr})",
    ]
    return {
        "formula": "CellColor = I(⌊x/S⌋·S + S/2, ⌊y/S⌋·S + S/2); Grout = #262626",
        "steps": steps,
        "output": [r, g, b, a],
    }


def filter_crystallize(img: np.ndarray, cell_size: int = 16, seed: int = 42) -> np.ndarray:
    """Voronoi nearest-neighbor cell tessellation."""
    f = to_float32(img)
    H, W = f.shape[:2]
    cs = max(4, int(cell_size))
    rng = np.random.default_rng(int(seed))

    # Grid of seed points
    cols = (W + cs - 1) // cs + 1
    rows = (H + cs - 1) // cs + 1

    # Jitter per cell
    jitter_x = rng.uniform(0.1, 0.9, size=(rows, cols)) * cs
    jitter_y = rng.uniform(0.1, 0.9, size=(rows, cols)) * cs

    cell_x0 = np.arange(cols) * cs
    cell_y0 = np.arange(rows) * cs
    px = np.clip(cell_x0[None, :] + jitter_x, 0, W - 1).astype(np.float32)
    py = np.clip(cell_y0[:, None] + jitter_y, 0, H - 1).astype(np.float32)

    # For each pixel, find nearest seed
    grid_y, grid_x = np.indices((H, W), dtype=np.float32)
    cell_i = np.clip((grid_y // cs).astype(np.int32), 0, rows - 1)
    cell_j = np.clip((grid_x // cs).astype(np.int32), 0, cols - 1)

    min_dist_sq = np.full((H, W), 1e9, dtype=np.float32)
    best_sx = np.zeros((H, W), dtype=np.int32)
    best_sy = np.zeros((H, W), dtype=np.int32)

    # Check 3x3 neighbor grid cells
    for di in [-1, 0, 1]:
        ni = np.clip(cell_i + di, 0, rows - 1)
        for dj in [-1, 0, 1]:
            nj = np.clip(cell_j + dj, 0, cols - 1)
            sx = px[ni, nj]
            sy = py[ni, nj]
            d2 = (grid_x - sx) ** 2 + (grid_y - sy) ** 2
            closer = d2 < min_dist_sq
            min_dist_sq[closer] = d2[closer]
            best_sx[closer] = sx[closer].astype(np.int32)
            best_sy[closer] = sy[closer].astype(np.int32)

    out_rgb = f[best_sy, best_sx, :3]
    out = f.copy()
    out[:, :, :3] = out_rgb
    return to_uint8(out)


def trace_crystallize(r: int, g: int, b: int, a: int, params: dict) -> dict:
    cs = int(params.get("cell_size", 16))
    steps = [
        f"Voronoi crystal lattice cell size = {cs}px",
        "Displaced seed points generated with pseudo-random jitter",
        "Pixel assigned color of nearest Euclidean seed site",
    ]
    return {
        "formula": "Site = argmin_k ||(x, y) − Seed_k||²; out = I_in(Site)",
        "steps": steps,
        "output": [r, g, b, a],
    }


def filter_stained_glass(
    img: np.ndarray, cell_size: int = 16, border_width: int = 2, seed: int = 42
) -> np.ndarray:
    """Voronoi tessellation with dark lead came edge boundaries between cells."""
    f = to_float32(img)
    H, W = f.shape[:2]
    cs = max(4, int(cell_size))
    bw = max(1, int(border_width))
    rng = np.random.default_rng(int(seed))

    cols = (W + cs - 1) // cs + 1
    rows = (H + cs - 1) // cs + 1

    jitter_x = rng.uniform(0.1, 0.9, size=(rows, cols)) * cs
    jitter_y = rng.uniform(0.1, 0.9, size=(rows, cols)) * cs

    cell_x0 = np.arange(cols) * cs
    cell_y0 = np.arange(rows) * cs
    px = np.clip(cell_x0[None, :] + jitter_x, 0, W - 1).astype(np.float32)
    py = np.clip(cell_y0[:, None] + jitter_y, 0, H - 1).astype(np.float32)

    grid_y, grid_x = np.indices((H, W), dtype=np.float32)
    cell_i = np.clip((grid_y // cs).astype(np.int32), 0, rows - 1)
    cell_j = np.clip((grid_x // cs).astype(np.int32), 0, cols - 1)

    d1 = np.full((H, W), 1e9, dtype=np.float32)
    d2 = np.full((H, W), 1e9, dtype=np.float32)
    best_sx = np.zeros((H, W), dtype=np.int32)
    best_sy = np.zeros((H, W), dtype=np.int32)

    for di in [-1, 0, 1]:
        ni = np.clip(cell_i + di, 0, rows - 1)
        for dj in [-1, 0, 1]:
            nj = np.clip(cell_j + dj, 0, cols - 1)
            sx = px[ni, nj]
            sy = py[ni, nj]
            dist = np.sqrt((grid_x - sx) ** 2 + (grid_y - sy) ** 2)

            # update d1 and d2
            update_first = dist < d1
            update_second = (~update_first) & (dist < d2)

            d2[update_first] = d1[update_first]
            d1[update_first] = dist[update_first]
            best_sx[update_first] = sx[update_first].astype(np.int32)
            best_sy[update_first] = sy[update_first].astype(np.int32)

            d2[update_second] = dist[update_second]

    out_rgb = f[best_sy, best_sx, :3].copy()
    # Dark lead came boundary where d2 - d1 is small
    border = (d2 - d1) <= float(bw)
    out_rgb[border] = 0.08  # lead came outline

    out = f.copy()
    out[:, :, :3] = out_rgb
    return to_uint8(out)


def trace_stained_glass(r: int, g: int, b: int, a: int, params: dict) -> dict:
    cs = int(params.get("cell_size", 16))
    bw = int(params.get("border_width", 2))
    steps = [
        f"Voronoi crystal lattice cell size = {cs}px, Lead came width = {bw}px",
        "Computed 1st and 2nd nearest neighbor distances d1 and d2",
        "Boundary check: (d2 − d1) ≤ width → colored with dark lead came #141414",
    ]
    return {
        "formula": "Border = (d2 − d1 ≤ width) ? LeadCame : NearestCellColor",
        "steps": steps,
        "output": [r, g, b, a],
    }


def filter_linocut(img: np.ndarray, threshold: float = 0.5, grain: float = 0.3) -> np.ndarray:
    """High-contrast relief printmaking combining thresholding with directional woodcut grain."""
    f = to_float32(img)
    H, W = f.shape[:2]
    luma = rgb_to_luminance(f[:, :, :3], mode="rec709")

    th = float(threshold)
    gr = float(grain)

    # Directional woodcut line modulation
    grid_y, grid_x = np.indices((H, W), dtype=np.float32)
    lines = 0.5 * (np.sin(grid_y * 0.8 + np.sin(grid_x * 0.1) * 3.0) + 1.0)
    modulated_luma = luma + gr * (lines - 0.5)

    cut = np.where(modulated_luma >= th, 0.95, 0.1)  # Warm paper vs deep ink
    out = f.copy()
    out[:, :, :3] = np.repeat(cut[:, :, None], 3, axis=-1)
    return to_uint8(out)


def trace_linocut(r: int, g: int, b: int, a: int, params: dict) -> dict:
    th = float(params.get("threshold", 0.5))
    gr = float(params.get("grain", 0.3))
    luma = (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255.0
    steps = [
        f"Input luminance L = {luma:.2f}",
        f"Woodcut relief threshold = {th:.2f}, Texture grain weight = {gr:.2f}",
        f"Modulated tone evaluated against cutting gouge threshold",
    ]
    return {
        "formula": "Cut = (L + grain × SinLines ≥ threshold) ? Paper : ReliefInk",
        "steps": steps,
        "output": [r, g, b, a],
    }
