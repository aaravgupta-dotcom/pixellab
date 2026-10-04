"""Color and Channel Filters (Filters 9 to 17)
Authoritative NumPy implementations for color space operations and channel manipulation.
"""

import numpy as np
from filters.helpers import to_float32, to_uint8, rgb_to_luminance, rgb_to_hsv, hsv_to_rgb


def filter_saturation(img: np.ndarray, factor: float = 1.0) -> np.ndarray:
    """Scale color saturation relative to perceptual luminance.

    Math:
        L = 0.2126*R + 0.7152*G + 0.0722*B
        I_out = np.clip(L + (I_in - L) * factor, 0.0, 1.0)
    """
    f = to_float32(img)
    rgb = f[:, :, :3]
    luma = rgb_to_luminance(rgb, mode="rec709")[:, :, None]
    out = np.clip(luma + (rgb - luma) * float(factor), 0.0, 1.0)
    if f.shape[2] == 4:
        out = np.concatenate([out, f[:, :, 3:4]], axis=-1)
    return to_uint8(out)


def trace_saturation(r: int, g: int, b: int, a: int, params: dict) -> dict:
    factor = float(params.get("factor", 1.0))
    luma = (0.2126 * r + 0.7152 * g + 0.0722 * b)
    steps = [
        f"Luminance L = 0.2126×{r} + 0.7152×{g} + 0.0722×{b} = {luma:.1f}",
        f"Saturation multiplier = {factor:.2f}",
    ]
    out_rgb = []
    for name, val in [("Red", r), ("Green", g), ("Blue", b)]:
        calc = luma + (val - luma) * factor
        clamped = max(0.0, min(255.0, calc))
        final_val = int(round(clamped))
        steps.append(f"{name}: {luma:.1f} + ({val} − {luma:.1f}) × {factor:.2f} = {calc:.1f} → {final_val}")
        out_rgb.append(final_val)

    return {
        "formula": "I_out = clamp(L + (I_in − L) × factor, 0, 255)",
        "steps": steps,
        "output": [out_rgb[0], out_rgb[1], out_rgb[2], a],
    }


def filter_hue_rotate(img: np.ndarray, degrees: float = 0.0) -> np.ndarray:
    """Rotate image hue angle around the color wheel in HSV space.

    Math:
        H_out = (H_in + degrees) % 360
    """
    f = to_float32(img)
    hsv = rgb_to_hsv(f[:, :, :3])
    hsv[:, :, 0] = (hsv[:, :, 0] + float(degrees)) % 360.0
    rgb = hsv_to_rgb(hsv)
    if f.shape[2] == 4:
        rgb = np.concatenate([rgb, f[:, :, 3:4]], axis=-1)
    return to_uint8(rgb)


def trace_hue_rotate(r: int, g: int, b: int, a: int, params: dict) -> dict:
    degrees = float(params.get("degrees", 0.0))
    rn, gn, bn = r / 255.0, g / 255.0, b / 255.0
    max_c, min_c = max(rn, gn, bn), min(rn, gn, bn)
    delta = max_c - min_c
    if delta < 1e-5:
        h = 0.0
    elif max_c == rn:
        h = (60.0 * ((gn - bn) / delta) + 360.0) % 360.0
    elif max_c == gn:
        h = (60.0 * ((bn - rn) / delta) + 120.0) % 360.0
    else:
        h = (60.0 * ((rn - gn) / delta) + 240.0) % 360.0

    new_h = (h + degrees) % 360.0
    # calculate new rgb
    v = max_c
    s = 0.0 if max_c < 1e-5 else delta / max_c
    c = v * s
    x = c * (1.0 - abs((new_h / 60.0) % 2.0 - 1.0))
    m = v - c
    seg = int(new_h / 60.0) % 6
    lut = [[c, x, 0], [x, c, 0], [0, c, x], [0, x, c], [x, 0, c], [c, 0, x]]
    nr, ng, nb = [int(round(min(1.0, max(0.0, val + m)) * 255.0)) for val in lut[seg]]

    steps = [
        f"Input HSV: H={h:.1f}°, S={s:.2f}, V={v:.2f}",
        f"Rotated Hue: ({h:.1f}° + {degrees:.1f}°) mod 360 = {new_h:.1f}°",
        f"Converted back to RGB: ({nr}, {ng}, {nb})",
    ]
    return {
        "formula": "H' = (H + degrees) mod 360",
        "steps": steps,
        "output": [nr, ng, nb, a],
    }


def filter_grayscale(img: np.ndarray, mode: str = "rec709") -> np.ndarray:
    """Convert image to grayscale with 3 selectable weighting standards.

    Modes:
        - rec709: (0.2126 R + 0.7152 G + 0.0722 B)
        - rec601: (0.2990 R + 0.5870 G + 0.1140 B)
        - average: (0.3333 R + 0.3333 G + 0.3333 B)
    """
    f = to_float32(img)
    luma = rgb_to_luminance(f[:, :, :3], mode=mode)[:, :, None]
    out = np.repeat(luma, 3, axis=-1)
    if f.shape[2] == 4:
        out = np.concatenate([out, f[:, :, 3:4]], axis=-1)
    return to_uint8(out)


def trace_grayscale(r: int, g: int, b: int, a: int, params: dict) -> dict:
    mode = str(params.get("mode", "rec709"))
    if mode == "rec601":
        weights = (0.2990, 0.5870, 0.1140)
        desc = "Rec.601 (NTSC/SDTV)"
    elif mode == "average":
        weights = (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)
        desc = "Unweighted Mean"
    else:
        weights = (0.2126, 0.7152, 0.0722)
        desc = "Rec.709 (sRGB/HDTV)"

    val = weights[0] * r + weights[1] * g + weights[2] * b
    final_val = int(round(max(0.0, min(255.0, val))))
    steps = [
        f"Standard: {desc}",
        f"L = {weights[0]:.4f}×{r} + {weights[1]:.4f}×{g} + {weights[2]:.4f}×{b} = {val:.2f} → {final_val}",
    ]
    return {
        "formula": f"L = {weights[0]:.3f}×R + {weights[1]:.3f}×G + {weights[2]:.3f}×B",
        "steps": steps,
        "output": [final_val, final_val, final_val, a],
    }


def filter_sepia(img: np.ndarray, intensity: float = 1.0) -> np.ndarray:
    """Warm nostalgic sepia tone matrix transformation.

    Math:
        R' = 0.393*R + 0.769*G + 0.189*B
        G' = 0.349*R + 0.686*G + 0.168*B
        B' = 0.272*R + 0.534*G + 0.131*B
        out = (1 - intensity)*in + intensity*sepia
    """
    f = to_float32(img)
    rgb = f[:, :, :3]
    matrix = np.array(
        [
            [0.393, 0.769, 0.189],
            [0.349, 0.686, 0.168],
            [0.272, 0.534, 0.131],
        ],
        dtype=np.float32,
    )
    sepia = np.tensordot(rgb, matrix, axes=([2], [1]))
    sepia = np.clip(sepia, 0.0, 1.0)
    t = float(intensity)
    out = np.clip((1.0 - t) * rgb + t * sepia, 0.0, 1.0)
    if f.shape[2] == 4:
        out = np.concatenate([out, f[:, :, 3:4]], axis=-1)
    return to_uint8(out)


def trace_sepia(r: int, g: int, b: int, a: int, params: dict) -> dict:
    intensity = float(params.get("intensity", 1.0))
    sr = 0.393 * r + 0.769 * g + 0.189 * b
    sg = 0.349 * r + 0.686 * g + 0.168 * b
    sb = 0.272 * r + 0.534 * g + 0.131 * b

    fr = int(round(max(0.0, min(255.0, (1.0 - intensity) * r + intensity * sr))))
    fg = int(round(max(0.0, min(255.0, (1.0 - intensity) * g + intensity * sg))))
    fb = int(round(max(0.0, min(255.0, (1.0 - intensity) * b + intensity * sb))))

    steps = [
        f"Sepia Red   = 0.393×{r} + 0.769×{g} + 0.189×{b} = {sr:.1f} → blend({intensity:.2f}) → {fr}",
        f"Sepia Green = 0.349×{r} + 0.686×{g} + 0.168×{b} = {sg:.1f} → blend({intensity:.2f}) → {fg}",
        f"Sepia Blue  = 0.272×{r} + 0.534×{g} + 0.131×{b} = {sb:.1f} → blend({intensity:.2f}) → {fb}",
    ]
    return {
        "formula": "RGB' = (1 - t) × RGB + t × (Matrix × RGB)",
        "steps": steps,
        "output": [fr, fg, fb, a],
    }


def filter_invert(img: np.ndarray, amount: float = 1.0) -> np.ndarray:
    """Invert channel values: out = 1.0 - in (with blend amount)."""
    f = to_float32(img)
    t = float(amount)
    out = f.copy()
    out[:, :, :3] = (1.0 - t) * f[:, :, :3] + t * (1.0 - f[:, :, :3])
    return to_uint8(out)


def trace_invert(r: int, g: int, b: int, a: int, params: dict) -> dict:
    amount = float(params.get("amount", 1.0))
    out_rgb = []
    steps = []
    for name, val in [("Red", r), ("Green", g), ("Blue", b)]:
        inv = 255 - val
        calc = (1.0 - amount) * val + amount * inv
        final_val = int(round(calc))
        steps.append(f"{name}: 255 − {val} = {inv} → blend({amount:.2f}) → {final_val}")
        out_rgb.append(final_val)
    return {
        "formula": "I_out = (1 − t) × I_in + t × (255 − I_in)",
        "steps": steps,
        "output": [out_rgb[0], out_rgb[1], out_rgb[2], a],
    }


def filter_posterize(img: np.ndarray, levels: int = 4) -> np.ndarray:
    """Quantize color depth into a discrete number of tonal steps.

    Math:
        I_out = np.floor(I_in * (levels - 1) + 0.5) / (levels - 1)
    """
    f = to_float32(img)
    n = max(int(levels), 2)
    out = f.copy()
    out[:, :, :3] = np.floor(f[:, :, :3] * (n - 1) + 0.5) / (n - 1)
    return to_uint8(out)


def trace_posterize(r: int, g: int, b: int, a: int, params: dict) -> dict:
    n = max(int(params.get("levels", 4)), 2)
    step_size = 255.0 / (n - 1)
    steps = [f"Posterize into {n} tonal levels (bucket interval = {step_size:.1f})"]
    out_rgb = []
    for name, val in [("Red", r), ("Green", g), ("Blue", b)]:
        bucket = int(round(val / step_size))
        final_val = int(round(bucket * step_size))
        steps.append(f"{name}: round({val} / {step_size:.1f}) = {bucket} × {step_size:.1f} = {final_val}")
        out_rgb.append(final_val)
    return {
        "formula": "I_out = round(I_in / (255 / (levels - 1))) × (255 / (levels - 1))",
        "steps": steps,
        "output": [out_rgb[0], out_rgb[1], out_rgb[2], a],
    }


def filter_solarize(img: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Sabattier effect: partial tone reversal for values exceeding a threshold.

    Math:
        I_out = np.where(I_in > threshold, 1.0 - I_in, I_in)
    """
    f = to_float32(img)
    th = float(threshold)
    rgb = f[:, :, :3]
    out_rgb = np.where(rgb > th, 1.0 - rgb, rgb)
    out = f.copy()
    out[:, :, :3] = out_rgb
    return to_uint8(out)


def trace_solarize(r: int, g: int, b: int, a: int, params: dict) -> dict:
    th = float(params.get("threshold", 0.5))
    th_byte = th * 255.0
    steps = [f"Threshold = {th:.2f} ({th_byte:.1f} in uint8)"]
    out_rgb = []
    for name, val in [("Red", r), ("Green", g), ("Blue", b)]:
        if val > th_byte:
            final_val = 255 - val
            steps.append(f"{name}: {val} > {th_byte:.1f} → invert: 255 − {val} = {final_val}")
        else:
            final_val = val
            steps.append(f"{name}: {val} ≤ {th_byte:.1f} → retain: {final_val}")
        out_rgb.append(final_val)
    return {
        "formula": "I_out = (I_in > threshold) ? (255 − I_in) : I_in",
        "steps": steps,
        "output": [out_rgb[0], out_rgb[1], out_rgb[2], a],
    }


def filter_channel_mixer(
    img: np.ndarray,
    rr: float = 1.0,
    rg: float = 0.0,
    rb: float = 0.0,
    gr: float = 0.0,
    gg: float = 1.0,
    gb: float = 0.0,
    br: float = 0.0,
    bg: float = 0.0,
    bb: float = 1.0,
) -> np.ndarray:
    """Full 3x3 channel re-mixing matrix."""
    f = to_float32(img)
    matrix = np.array(
        [[float(rr), float(rg), float(rb)], [float(gr), float(gg), float(gb)], [float(br), float(bg), float(bb)]],
        dtype=np.float32,
    )
    mixed = np.tensordot(f[:, :, :3], matrix, axes=([2], [1]))
    out = f.copy()
    out[:, :, :3] = np.clip(mixed, 0.0, 1.0)
    return to_uint8(out)


def trace_channel_mixer(r: int, g: int, b: int, a: int, params: dict) -> dict:
    rr = float(params.get("rr", 1.0))
    rg = float(params.get("rg", 0.0))
    rb = float(params.get("rb", 0.0))
    gr = float(params.get("gr", 0.0))
    gg = float(params.get("gg", 1.0))
    gb = float(params.get("gb", 0.0))
    br = float(params.get("br", 0.0))
    bg = float(params.get("bg", 0.0))
    bb = float(params.get("bb", 1.0))

    calc_r = rr * r + rg * g + rb * b
    calc_g = gr * r + gg * g + gb * b
    calc_b = br * r + bg * g + bb * b

    fr = int(round(max(0.0, min(255.0, calc_r))))
    fg = int(round(max(0.0, min(255.0, calc_g))))
    fb = int(round(max(0.0, min(255.0, calc_b))))

    steps = [
        f"Red'   = {rr:.2f}×{r} + {rg:.2f}×{g} + {rb:.2f}×{b} = {calc_r:.1f} → {fr}",
        f"Green' = {gr:.2f}×{r} + {gg:.2f}×{g} + {gb:.2f}×{b} = {calc_g:.1f} → {fg}",
        f"Blue'  = {br:.2f}×{r} + {bg:.2f}×{g} + {bb:.2f}×{b} = {calc_b:.1f} → {fb}",
    ]
    return {
        "formula": "RGB' = clamp(Matrix × RGB, 0, 255)",
        "steps": steps,
        "output": [fr, fg, fb, a],
    }


def filter_selective_color(
    img: np.ndarray, target_hue: float = 0.0, tolerance: float = 30.0
) -> np.ndarray:
    """Isolate a selected hue within a tolerance band; convert all other pixels to grayscale.

    Math:
        dist = min(|H - target|, 360 - |H - target|)
        weight = clamp(1.0 - (dist / tolerance), 0.0, 1.0)
        I_out = weight * RGB + (1 - weight) * Luminance
    """
    f = to_float32(img)
    rgb = f[:, :, :3]
    hsv = rgb_to_hsv(rgb)
    h = hsv[:, :, 0]

    th = float(target_hue) % 360.0
    tol = max(float(tolerance), 1.0)

    diff = np.abs(h - th)
    dist = np.minimum(diff, 360.0 - diff)
    weight = np.clip(1.0 - (dist / tol), 0.0, 1.0)[:, :, None]

    luma = rgb_to_luminance(rgb, mode="rec709")[:, :, None]
    out_rgb = weight * rgb + (1.0 - weight) * luma

    out = f.copy()
    out[:, :, :3] = np.clip(out_rgb, 0.0, 1.0)
    return to_uint8(out)


def trace_selective_color(r: int, g: int, b: int, a: int, params: dict) -> dict:
    target_hue = float(params.get("target_hue", 0.0)) % 360.0
    tol = max(float(params.get("tolerance", 30.0)), 1.0)

    rn, gn, bn = r / 255.0, g / 255.0, b / 255.0
    max_c, min_c = max(rn, gn, bn), min(rn, gn, bn)
    delta = max_c - min_c
    if delta < 1e-5:
        h = 0.0
    elif max_c == rn:
        h = (60.0 * ((gn - bn) / delta) + 360.0) % 360.0
    elif max_c == gn:
        h = (60.0 * ((bn - rn) / delta) + 120.0) % 360.0
    else:
        h = (60.0 * ((rn - gn) / delta) + 240.0) % 360.0

    diff = abs(h - target_hue)
    dist = min(diff, 360.0 - diff)
    weight = max(0.0, min(1.0, 1.0 - (dist / tol)))

    luma = 0.2126 * r + 0.7152 * g + 0.0722 * b
    fr = int(round(weight * r + (1.0 - weight) * luma))
    fg = int(round(weight * g + (1.0 - weight) * luma))
    fb = int(round(weight * b + (1.0 - weight) * luma))

    steps = [
        f"Pixel Hue H = {h:.1f}°, Target Hue = {target_hue:.1f}°",
        f"Circular angular distance = {dist:.1f}° (tolerance = {tol:.1f}°)",
        f"Retention weight = {weight:.2f}, Grayscale Luma = {luma:.1f}",
        f"Blended output: R={fr}, G={fg}, B={fb}",
    ]
    return {
        "formula": "weight = clamp(1 − (|H − target| / tol), 0, 1); out = weight × RGB + (1 − weight) × L",
        "steps": steps,
        "output": [fr, fg, fb, a],
    }
