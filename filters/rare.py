"""Rare and Unique Signature Filters (Filters 41 to 50)
Authoritative NumPy implementations for high-concept creative technologist transformations:
Thermal, Cyanotype, Risograph, CMYK Misregistration, Chromatic Aberration,
Pixel Sort, Datamosh Glitch, VHS Tracking, Bayer & Floyd-Steinberg Dither, and Kirlian Aura.
"""

import numpy as np
from filters.helpers import (
    to_float32,
    to_uint8,
    convolve2d,
    convolve_rgb,
    rgb_to_luminance,
    remap_bilinear,
    _make_gaussian_kernel,
)


def filter_thermal_vision(img: np.ndarray, contrast: float = 1.0) -> np.ndarray:
    """False-color thermal imaging mapping luminance to an infrared heat palette.

    Palette stops: Black (0.0) -> Indigo (0.25) -> Flame Red (0.5) -> Solar Yellow (0.75) -> White (1.0).
    """
    f = to_float32(img)
    luma = rgb_to_luminance(f[:, :, :3], mode="rec709")

    c = float(contrast)
    luma = np.clip((luma - 0.5) * c + 0.5, 0.0, 1.0)

    # Gradient stops
    stops = np.array([0.0, 0.25, 0.5, 0.75, 1.0], dtype=np.float32)
    palette = np.array(
        [
            [0.0, 0.0, 0.04],       # Deep black / dark violet
            [0.2, 0.0, 0.5],        # Indigo
            [0.85, 0.08, 0.08],     # Flame red
            [1.0, 0.85, 0.0],       # Solar yellow
            [1.0, 1.0, 1.0],        # White heat
        ],
        dtype=np.float32,
    )

    out_rgb = np.zeros((luma.shape[0], luma.shape[1], 3), dtype=np.float32)
    for ch in range(3):
        out_rgb[:, :, ch] = np.interp(luma, stops, palette[:, ch])

    out = f.copy()
    out[:, :, :3] = out_rgb
    return to_uint8(out)


def trace_thermal_vision(r: int, g: int, b: int, a: int, params: dict) -> dict:
    luma = (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255.0
    steps = [
        f"Luminance L = {luma:.3f}",
        "Heat map progression: Black(0.0) → Indigo(0.25) → Red(0.5) → Yellow(0.75) → White(1.0)",
        f"Interpolated thermal spectrum for L={luma:.3f}",
    ]
    return {
        "formula": "RGB = PiecewiseLinear(L, [Black, Indigo, Red, Yellow, White])",
        "steps": steps,
        "output": [r, g, b, a],
    }


def filter_cyanotype(
    img: np.ndarray, blue_intensity: float = 1.0, contrast: float = 1.2, grain: float = 0.2
) -> np.ndarray:
    """Historical Prussian blue photographic sun-print with paper fiber grain."""
    f = to_float32(img)
    H, W = f.shape[:2]
    luma = rgb_to_luminance(f[:, :, :3], mode="rec709")

    # High contrast transfer curve
    c = float(contrast)
    toned = np.clip((luma - 0.5) * c + 0.5, 0.0, 1.0)

    # Paper base: warm unbleached paper #EDE5D5
    paper = np.array([0.93, 0.90, 0.83], dtype=np.float32)
    # Prussian blue ink: deep ferric ferrocyanide #0B3356
    prussian = np.array([0.04, 0.20, 0.38], dtype=np.float32)

    # Invert tone: shadows receive dense Prussian blue, highlights remain bare paper
    t = toned[:, :, None]
    base = paper * t + prussian * (1.0 - t) * float(blue_intensity)

    # Paper grain
    if grain > 0:
        gr = float(grain)
        rng = np.random.default_rng(101)
        noise = rng.normal(0.0, 0.04 * gr, size=(H, W, 1)).astype(np.float32)
        base = np.clip(base + noise, 0.0, 1.0)

    out = f.copy()
    out[:, :, :3] = np.clip(base, 0.0, 1.0)
    return to_uint8(out)


def trace_cyanotype(r: int, g: int, b: int, a: int, params: dict) -> dict:
    luma = (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255.0
    steps = [
        f"Base Luminance L = {luma:.3f}",
        "Prussian Blue Ink (Ferric Ferrocyanide) mapped to dark density (1 − L)",
        "Raw Paper Base (#EDE5D5) preserved in highlights",
    ]
    return {
        "formula": "RGB = Paper × L + PrussianBlue × (1 − L)",
        "steps": steps,
        "output": [r, g, b, a],
    }


def filter_risograph(
    img: np.ndarray, misregistration: float = 3.0, grain: float = 0.5, seed: int = 42
) -> np.ndarray:
    """Dual-drum Risograph press: Fluorescent Pink + Teal drums with mechanical plate shift and grain."""
    f = to_float32(img)
    H, W = f.shape[:2]
    rng = np.random.default_rng(int(seed))

    luma = rgb_to_luminance(f[:, :, :3], mode="rec709")

    # Drum 1: Fluorescent Pink (dense in warm/red/mid tones)
    pink_drum = np.clip(1.0 - f[:, :, 1], 0.0, 1.0)  # Inverted green channel
    # Drum 2: Federal Blue / Teal (dense in cool/shadows)
    teal_drum = np.clip(1.0 - luma, 0.0, 1.0)

    # Shift Drum 2 spatially (misregistration plate offset)
    m = float(misregistration)
    dx = int(round(m))
    dy = int(round(m * 0.5))

    teal_shifted = np.zeros_like(teal_drum)
    if dy >= 0 and dx >= 0:
        teal_shifted[dy:, dx:] = teal_drum[: H - dy, : W - dx]
    else:
        teal_shifted = teal_drum

    # Halftone grain
    gr = float(grain)
    noise = rng.uniform(-0.15 * gr, 0.15 * gr, size=(H, W)).astype(np.float32)
    pink_drum = np.clip(pink_drum + noise, 0.0, 1.0)
    teal_shifted = np.clip(teal_shifted + noise, 0.0, 1.0)

    # Inks
    pink_ink = np.array([0.98, 0.15, 0.45], dtype=np.float32)  # Fluorescent Pink
    teal_ink = np.array([0.05, 0.48, 0.55], dtype=np.float32)  # Teal
    paper = np.array([0.94, 0.92, 0.86], dtype=np.float32)    # Warm Paper

    # Subtractive overprint multiplication
    p_layer = 1.0 - (1.0 - pink_ink) * pink_drum[:, :, None]
    t_layer = 1.0 - (1.0 - teal_ink) * teal_shifted[:, :, None]
    combined = paper * p_layer * t_layer

    out = f.copy()
    out[:, :, :3] = np.clip(combined, 0.0, 1.0)
    return to_uint8(out)


def trace_risograph(r: int, g: int, b: int, a: int, params: dict) -> dict:
    m = float(params.get("misregistration", 3.0))
    steps = [
        "Color separated into 2 physical ink drums:",
        "Drum 1: Fluorescent Pink (#FA2673)",
        f"Drum 2: Teal (#0D7A8C) with plate registration offset Δ=({m:.1f}px, {m*0.5:.1f}px)",
        "Subtractive ink multiply onto unbleached substrate",
    ]
    return {
        "formula": "Paper × (1 − PinkDrum × (1−PinkInk)) × (1 − TealDrum_shifted × (1−TealInk))",
        "steps": steps,
        "output": [r, g, b, a],
    }


def filter_newsprint_cmyk(
    img: np.ndarray, dot_scale: float = 4.0, misregistration: float = 2.0
) -> np.ndarray:
    """CMYK four-color process newsprint with rotated screen angles and mechanical plate shift."""
    f = to_float32(img)
    H, W = f.shape[:2]
    rgb = f[:, :, :3]

    # Convert RGB to CMYK
    k_comp = 1.0 - np.max(rgb, axis=-1)
    denom = np.maximum(1.0 - k_comp, 1e-4)
    c_comp = (1.0 - rgb[:, :, 0] - k_comp) / denom
    m_comp = (1.0 - rgb[:, :, 1] - k_comp) / denom
    y_comp = (1.0 - rgb[:, :, 2] - k_comp) / denom

    # Rotated dot screens
    scale = max(2.0, float(dot_scale))
    grid_y, grid_x = np.indices((H, W), dtype=np.float32)

    def screen(angle_deg, density, ox=0, oy=0):
        rad = np.radians(angle_deg)
        rot_x = (grid_x + ox) * np.cos(rad) - (grid_y + oy) * np.sin(rad)
        rot_y = (grid_x + ox) * np.sin(rad) + (grid_y + oy) * np.cos(rad)
        pat = 0.5 * (np.sin(rot_x * 2.0 * np.pi / scale) * np.cos(rot_y * 2.0 * np.pi / scale) + 1.0)
        return (density > pat).astype(np.float32)

    m = float(misregistration)
    # Standard screen angles: Cyan 15°, Magenta 75°, Yellow 0°, Black 45°
    dot_c = screen(15.0, c_comp, ox=m, oy=0)
    dot_m = screen(75.0, m_comp, ox=0, oy=m)
    dot_y = screen(0.0, y_comp, ox=-m, oy=0)
    dot_k = screen(45.0, k_comp, ox=0, oy=-m)

    # Reconstruct subtractive CMYK to RGB
    r_out = (1.0 - dot_c) * (1.0 - dot_k)
    g_out = (1.0 - dot_m) * (1.0 - dot_k)
    b_out = (1.0 - dot_y) * (1.0 - dot_k)

    # Modulate with warm newsprint paper tint
    paper = np.array([0.96, 0.94, 0.88], dtype=np.float32)
    out_rgb = np.stack([r_out, g_out, b_out], axis=-1) * paper

    out = f.copy()
    out[:, :, :3] = np.clip(out_rgb, 0.0, 1.0)
    return to_uint8(out)


def trace_newsprint_cmyk(r: int, g: int, b: int, a: int, params: dict) -> dict:
    scale = float(params.get("dot_scale", 4.0))
    m = float(params.get("misregistration", 2.0))
    steps = [
        "Separated RGB into subtractive Cyan, Magenta, Yellow, Key (Black)",
        f"Screen angles: C(15°), M(75°), Y(0°), K(45°) with frequency scale {scale:.1f}px",
        f"Applied mechanical plate misregistration displacement = {m:.1f}px",
    ]
    return {
        "formula": "RGB = (1 − Screen_C)·(1 − Screen_M)·(1 − Screen_Y)·(1 − Screen_K) × Paper",
        "steps": steps,
        "output": [r, g, b, a],
    }


def filter_chromatic_aberration(
    img: np.ndarray, shift: float = 6.0, angle: float = 0.0
) -> np.ndarray:
    """Optical chromatic aberration: radial red/blue channel spatial divergence."""
    f = to_float32(img)
    H, W = f.shape[:2]
    cx = (W - 1.0) / 2.0
    cy = (H - 1.0) / 2.0

    s = float(shift)
    rad = np.radians(float(angle))

    grid_y, grid_x = np.indices((H, W), dtype=np.float32)
    dx = grid_x - cx
    dy = grid_y - cy
    dist = np.sqrt(dx**2 + dy**2)
    max_d = max(cx, cy)
    norm_d = dist / max(max_d, 1.0)

    # Shift along angle plus radial component
    shift_r = s * norm_d
    rx = grid_x + shift_r * np.cos(rad)
    ry = grid_y + shift_r * np.sin(rad)

    bx = grid_x - shift_r * np.cos(rad)
    by = grid_y - shift_r * np.sin(rad)

    red_channel = remap_bilinear(f[:, :, 0:1], rx, ry, mode="reflect")
    blue_channel = remap_bilinear(f[:, :, 2:3], bx, by, mode="reflect")

    out = f.copy()
    out[:, :, 0] = red_channel[:, :, 0]
    # Green remains untouched at center optical axis
    out[:, :, 2] = blue_channel[:, :, 0]
    return to_uint8(out)


def trace_chromatic_aberration(r: int, g: int, b: int, a: int, params: dict) -> dict:
    s = float(params.get("shift", 6.0))
    ang = float(params.get("angle", 0.0))
    steps = [
        f"Dispersion shift = {s:.1f}px at angle {ang:.0f}°",
        "Red channel displaced along +vector",
        "Green channel anchored at optical axis (undistorted)",
        "Blue channel displaced along −vector",
    ]
    return {
        "formula": "R'(x + Δ, y + Δ), G'(x, y), B'(x − Δ, y − Δ)",
        "steps": steps,
        "output": [r, g, b, a],
    }


def filter_pixel_sort(
    img: np.ndarray,
    threshold_low: float = 0.25,
    threshold_high: float = 0.8,
    direction: str = "horizontal",
) -> np.ndarray:
    """ASDF Pixel Sort: run-length sorting of contiguous pixel intervals by luminance."""
    f = to_float32(img)
    H, W = f.shape[:2]
    luma = rgb_to_luminance(f[:, :, :3], mode="rec709")

    th_l = float(threshold_low)
    th_h = float(threshold_high)
    mask = (luma >= th_l) & (luma <= th_h)

    out = f.copy()

    if direction == "vertical":
        # Process columns
        for x in range(W):
            col_mask = mask[:, x]
            col_luma = luma[:, x]
            col_rgb = out[:, x, :3]

            in_run = False
            start = 0
            for y in range(H):
                if col_mask[y] and not in_run:
                    in_run = True
                    start = y
                elif not col_mask[y] and in_run:
                    in_run = False
                    if y - start > 1:
                        order = np.argsort(col_luma[start:y])
                        col_rgb[start:y] = col_rgb[start:y][order]
            if in_run and H - start > 1:
                order = np.argsort(col_luma[start:H])
                col_rgb[start:H] = col_rgb[start:H][order]
            out[:, x, :3] = col_rgb
    else:
        # Process rows
        for y in range(H):
            row_mask = mask[y, :]
            row_luma = luma[y, :]
            row_rgb = out[y, :, :3]

            in_run = False
            start = 0
            for x in range(W):
                if row_mask[x] and not in_run:
                    in_run = True
                    start = x
                elif not row_mask[x] and in_run:
                    in_run = False
                    if x - start > 1:
                        order = np.argsort(row_luma[start:x])
                        row_rgb[start:x] = row_rgb[start:x][order]
            if in_run and W - start > 1:
                order = np.argsort(row_luma[start:W])
                row_rgb[start:W] = row_rgb[start:W][order]
            out[y, :, :3] = row_rgb

    return to_uint8(out)


def trace_pixel_sort(r: int, g: int, b: int, a: int, params: dict) -> dict:
    tl = float(params.get("threshold_low", 0.25))
    th = float(params.get("threshold_high", 0.8))
    luma = (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255.0
    qualifies = tl <= luma <= th
    steps = [
        f"Pixel Luminance L = {luma:.2f}",
        f"Sort Interval Gate: [{tl:.2f}, {th:.2f}] → {'Active Sort Run' if qualifies else 'Locked Boundary'}",
        "Contiguous spans sorted monotonically by brightness",
    ]
    return {
        "formula": "SortSpan(I) for contiguous segments where threshold_low ≤ L ≤ threshold_high",
        "steps": steps,
        "output": [r, g, b, a],
    }


def filter_datamosh(
    img: np.ndarray,
    block_size: int = 16,
    shift_strength: float = 20.0,
    tear_prob: float = 0.15,
    seed: int = 42,
) -> np.ndarray:
    """Compression macroblock corruption, motion vector tearing, and channel glitch."""
    f = to_float32(img)
    H, W = f.shape[:2]
    rng = np.random.default_rng(int(seed))

    bs = max(8, int(block_size))
    s = float(shift_strength)

    out = f.copy()

    # Macroblock displacement
    for y in range(0, H - bs, bs):
        for x in range(0, W - bs, bs):
            if rng.random() < 0.25:
                dx = int(rng.integers(-int(s), int(s) + 1))
                dy = int(rng.integers(-int(s // 2), int(s // 2) + 1))
                src_y = np.clip(y + dy, 0, H - bs)
                src_x = np.clip(x + dx, 0, W - bs)
                out[y : y + bs, x : x + bs] = f[src_y : src_y + bs, src_x : src_x + bs]

    # Horizontal scanline tears
    tp = float(tear_prob)
    for y in range(H):
        if rng.random() < tp:
            shift_x = int(rng.integers(-int(s * 1.5), int(s * 1.5) + 1))
            out[y, :] = np.roll(out[y, :], shift_x, axis=0)
            # Channel aberration on torn line
            if rng.random() < 0.5:
                out[y, :, 0] = np.roll(out[y, :, 0], int(s // 2), axis=0)

    return to_uint8(out)


def trace_datamosh(r: int, g: int, b: int, a: int, params: dict) -> dict:
    bs = int(params.get("block_size", 16))
    s = float(params.get("shift_strength", 20.0))
    steps = [
        f"Macroblock grid = {bs}×{bs}px, Shift magnitude = ±{s:.0f}px",
        "P-frame motion vector displacement & sync tear simulation",
        "Deterministic PRNG seed preserves glitch topology",
    ]
    return {
        "formula": "Block(x, y) ← Block(x + Δx, y + Δy) on corrupted macroblock slots",
        "steps": steps,
        "output": [r, g, b, a],
    }


def filter_vhs_tracking(
    img: np.ndarray, tracking_noise: float = 0.5, chroma_shift: float = 4.0, seed: int = 42
) -> np.ndarray:
    """Analog magnetic tape emulation: RF chroma bleed, line jitter, and bottom head-switch band."""
    f = to_float32(img)
    H, W = f.shape[:2]
    rng = np.random.default_rng(int(seed))

    tn = float(tracking_noise)
    cs = int(round(float(chroma_shift)))

    out = f.copy()

    # 1. Horizontal scanline sync jitter
    for y in range(H):
        jitter = int(round(np.sin(y * 0.1) * 2.0 + rng.normal(0, 0.8)))
        if jitter != 0:
            out[y, :] = np.roll(out[y, :], jitter, axis=0)

    # 2. Chroma delay bleed (red channel delayed relative to blue)
    if cs > 0:
        out[:, :, 0] = np.roll(out[:, :, 0], cs, axis=1)
        out[:, :, 2] = np.roll(out[:, :, 2], -cs // 2, axis=1)

    # 3. Bottom-edge head switching noise band (bottom 8% of frame)
    band_h = max(4, int(H * 0.08))
    static_band = rng.uniform(0.0, 1.0, size=(band_h, W, 1)).astype(np.float32)
    # Severe horizontal phase tear in tracking band
    for by in range(band_h):
        y_coord = H - band_h + by
        tear = int(rng.integers(-25, 25))
        out[y_coord, :] = np.roll(out[y_coord, :], tear, axis=0)

    blend_static = np.clip(tn * 1.5, 0.0, 1.0)
    out[H - band_h : H, :, :3] = (
        (1.0 - blend_static) * out[H - band_h : H, :, :3] + blend_static * static_band
    )

    return to_uint8(out)


def trace_vhs_tracking(r: int, g: int, b: int, a: int, params: dict) -> dict:
    cs = float(params.get("chroma_shift", 4.0))
    tn = float(params.get("tracking_noise", 0.5))
    steps = [
        f"Analog chroma delay shift = {cs:.1f}px (U/V RF subcarrier delay)",
        "Line-by-line sync timebase jitter",
        f"Helical scan head-switching tracking band at frame bottom (noise={tn:.2f})",
    ]
    return {
        "formula": "ChromaShift(R, B) + ScanlineJitter(y) + HeadSwitchRoll(bottom 8%)",
        "steps": steps,
        "output": [r, g, b, a],
    }


def filter_dither(img: np.ndarray, method: str = "bayer", levels: int = 4) -> np.ndarray:
    """Ordered Bayer matrix dithering or sequential Floyd-Steinberg error diffusion."""
    f = to_float32(img)
    H, W = f.shape[:2]
    n = max(2, int(levels))
    step = 1.0 / (n - 1)

    if method == "floyd_steinberg":
        # Floyd-Steinberg is inherently sequential as documented in the requirements
        out = f.copy()
        # Working buffer
        buf = out[:, :, :3].copy()
        for y in range(H):
            for x in range(W):
                old_val = buf[y, x]
                new_val = np.round(old_val / step) * step
                new_val = np.clip(new_val, 0.0, 1.0)
                buf[y, x] = new_val
                err = old_val - new_val

                if x + 1 < W:
                    buf[y, x + 1] += err * (7.0 / 16.0)
                if y + 1 < H:
                    if x - 1 >= 0:
                        buf[y + 1, x - 1] += err * (3.0 / 16.0)
                    buf[y + 1, x] += err * (5.0 / 16.0)
                    if x + 1 < W:
                        buf[y + 1, x + 1] += err * (1.0 / 16.0)
        out[:, :, :3] = np.clip(buf, 0.0, 1.0)
        return to_uint8(out)
    else:
        # Bayer 4x4 matrix (fully vectorized)
        bayer_4x4 = (
            np.array(
                [
                    [0, 8, 2, 10],
                    [12, 4, 14, 6],
                    [3, 11, 1, 9],
                    [15, 7, 13, 5],
                ],
                dtype=np.float32,
            )
            / 16.0
            - 0.5
        )

        grid_y, grid_x = np.indices((H, W))
        bayer_matrix = bayer_4x4[grid_y % 4, grid_x % 4][:, :, None]

        dithered = f[:, :, :3] + bayer_matrix * step
        quantized = np.clip(np.round(dithered / step) * step, 0.0, 1.0)

        out = f.copy()
        out[:, :, :3] = quantized
        return to_uint8(out)


def trace_dither(r: int, g: int, b: int, a: int, params: dict) -> dict:
    method = str(params.get("method", "bayer"))
    n = int(params.get("levels", 4))
    steps = [
        f"Quantization depth = {n} steps (interval = {255.0/(n-1):.1f})",
        f"Algorithm: {'Bayer 4×4 Spatial Ordered Dither' if method == 'bayer' else 'Floyd-Steinberg Error Diffusion'}",
        "Preserves perceived tonal gradients in reduced bit-depth palettes",
    ]
    return {
        "formula": "Bayer: Q(I + M_4x4 · Δ); Floyd-Steinberg: Propagate error [7/16, 3/16, 5/16, 1/16]",
        "steps": steps,
        "output": [r, g, b, a],
    }


def filter_kirlian_aura(
    img: np.ndarray, glow_radius: int = 5, intensity: float = 1.5, hue: float = 180.0
) -> np.ndarray:
    """High-voltage electro-photographic corona discharge glow along structural contours."""
    f = to_float32(img)
    luma = rgb_to_luminance(f[:, :, :3], mode="rec709")

    # Edge extraction
    kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32) / 4.0
    ky = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32) / 4.0
    gx = convolve2d(luma, kx)
    gy = convolve2d(luma, ky)
    edges = np.sqrt(gx**2 + gy**2)

    # Blur edges to create radiant corona aura
    r = max(1, int(glow_radius))
    k_glow = _make_gaussian_kernel(radius=r, sigma=max(r / 1.5, 0.5))
    aura = convolve2d(edges, k_glow) * float(intensity)

    # Aura chromatic color tint
    h_rad = np.radians(float(hue))
    tint = np.array(
        [
            0.5 + 0.5 * np.cos(h_rad),
            0.5 + 0.5 * np.cos(h_rad - 2.094),
            0.5 + 0.5 * np.cos(h_rad + 2.094),
        ],
        dtype=np.float32,
    )

    glow_rgb = aura[:, :, None] * tint[None, None, :]

    # Composite aura onto darkened high-contrast base
    darkened = f[:, :, :3] * 0.4
    composite = np.clip(darkened + glow_rgb, 0.0, 1.0)

    out = f.copy()
    out[:, :, :3] = composite
    return to_uint8(out)


def trace_kirlian_aura(r: int, g: int, b: int, a: int, params: dict) -> dict:
    rad = int(params.get("glow_radius", 5))
    h = float(params.get("hue", 180.0))
    steps = [
        "1. Gradient contour extraction via Sobel operator",
        f"2. Gaussian halo bloom dilation (radius={rad}px)",
        f"3. Spectral tint projection at hue angle {h:.0f}°",
        "4. Additive synthesis onto darkened photographic substrate",
    ]
    return {
        "formula": "I_out = 0.4·Base + Bloom(Sobel(L), σ) × Tint(Hue)",
        "steps": steps,
        "output": [r, g, b, a],
    }
