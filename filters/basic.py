"""Tone and Light Filters (Filters 1 to 8)
Authoritative NumPy implementations for foundational photometric transformations.
"""

import numpy as np
from filters.helpers import to_float32, to_uint8, rgb_to_luminance, rgb_to_hsv, hsv_to_rgb


def filter_brightness(img: np.ndarray, offset: float = 0.0) -> np.ndarray:
    """Adjust image brightness by adding a linear channel offset.

    Math:
        I_out = np.clip(I_in + (offset / 100.0), 0.0, 1.0)
    """
    f = to_float32(img)
    delta = offset / 100.0
    out = np.clip(f[:, :, :3] + delta, 0.0, 1.0)
    if f.shape[2] == 4:
        out = np.concatenate([out, f[:, :, 3:4]], axis=-1)
    return to_uint8(out)


def trace_brightness(r: int, g: int, b: int, a: int, params: dict) -> dict:
    offset = float(params.get("offset", 0.0))
    delta = offset / 100.0
    steps = []
    out_rgb = []
    for name, val in [("Red", r), ("Green", g), ("Blue", b)]:
        norm = val / 255.0
        res_norm = norm + delta
        clamped_norm = max(0.0, min(1.0, res_norm))
        final_val = int(round(clamped_norm * 255.0))
        steps.append(
            f"{name}: ({val}/255 = {norm:.3f}) + {delta:+.2f} = {res_norm:.3f} → clamp[0, 1] → {clamped_norm:.3f} × 255 = {final_val}"
        )
        out_rgb.append(final_val)
    return {
        "formula": "I_out = clamp((I_in / 255 + offset / 100) * 255, 0, 255)",
        "steps": steps,
        "output": [out_rgb[0], out_rgb[1], out_rgb[2], a],
    }


def filter_contrast(img: np.ndarray, factor: float = 1.0) -> np.ndarray:
    """Scale pixel contrast around midpoint 0.5 (128 in uint8).

    Math:
        I_out = np.clip((I_in - 0.5) * factor + 0.5, 0.0, 1.0)
    """
    f = to_float32(img)
    out = np.clip((f[:, :, :3] - 0.5) * factor + 0.5, 0.0, 1.0)
    if f.shape[2] == 4:
        out = np.concatenate([out, f[:, :, 3:4]], axis=-1)
    return to_uint8(out)


def trace_contrast(r: int, g: int, b: int, a: int, params: dict) -> dict:
    factor = float(params.get("factor", 1.0))
    steps = []
    out_rgb = []
    for name, val in [("Red", r), ("Green", g), ("Blue", b)]:
        calc = (val - 128.0) * factor + 128.0
        clamped = max(0.0, min(255.0, calc))
        final_val = int(round(clamped))
        steps.append(
            f"{name}: ({val} − 128) × {factor:.2f} + 128 = {calc:.1f} → clamp[0, 255] → {final_val}"
        )
        out_rgb.append(final_val)
    return {
        "formula": "I_out = clamp((I_in − 128) × factor + 128, 0, 255)",
        "steps": steps,
        "output": [out_rgb[0], out_rgb[1], out_rgb[2], a],
    }


def filter_gamma(img: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    """Non-linear power law gamma correction.

    Math:
        I_out = np.clip(I_in ** (1.0 / max(gamma, 0.01)), 0.0, 1.0)
    """
    f = to_float32(img)
    inv_gamma = 1.0 / max(float(gamma), 0.01)
    out = np.clip(f[:, :, :3] ** inv_gamma, 0.0, 1.0)
    if f.shape[2] == 4:
        out = np.concatenate([out, f[:, :, 3:4]], axis=-1)
    return to_uint8(out)


def trace_gamma(r: int, g: int, b: int, a: int, params: dict) -> dict:
    gamma = max(float(params.get("gamma", 1.0)), 0.01)
    inv_g = 1.0 / gamma
    steps = []
    out_rgb = []
    for name, val in [("Red", r), ("Green", g), ("Blue", b)]:
        norm = val / 255.0
        calc_norm = norm ** inv_g
        clamped = max(0.0, min(1.0, calc_norm))
        final_val = int(round(clamped * 255.0))
        steps.append(
            f"{name}: ({val}/255 = {norm:.3f}) ^ (1 / {gamma:.2f} = {inv_g:.3f}) = {calc_norm:.3f} → {final_val}"
        )
        out_rgb.append(final_val)
    return {
        "formula": "I_out = clamp((I_in / 255) ^ (1 / gamma) × 255, 0, 255)",
        "steps": steps,
        "output": [out_rgb[0], out_rgb[1], out_rgb[2], a],
    }


def filter_exposure(img: np.ndarray, ev: float = 0.0) -> np.ndarray:
    """Photographic exposure stops adjustment using power of 2.

    Math:
        I_out = np.clip(I_in * (2.0 ** ev), 0.0, 1.0)
    """
    f = to_float32(img)
    mult = 2.0 ** float(ev)
    out = np.clip(f[:, :, :3] * mult, 0.0, 1.0)
    if f.shape[2] == 4:
        out = np.concatenate([out, f[:, :, 3:4]], axis=-1)
    return to_uint8(out)


def trace_exposure(r: int, g: int, b: int, a: int, params: dict) -> dict:
    ev = float(params.get("ev", 0.0))
    mult = 2.0 ** ev
    steps = []
    out_rgb = []
    for name, val in [("Red", r), ("Green", g), ("Blue", b)]:
        calc = val * mult
        clamped = max(0.0, min(255.0, calc))
        final_val = int(round(clamped))
        steps.append(
            f"{name}: {val} × 2^({ev:.2f}) = {val} × {mult:.3f} = {calc:.1f} → clamp → {final_val}"
        )
        out_rgb.append(final_val)
    return {
        "formula": "I_out = clamp(I_in × 2^EV, 0, 255)",
        "steps": steps,
        "output": [out_rgb[0], out_rgb[1], out_rgb[2], a],
    }


def filter_highlights_shadows(img: np.ndarray, shadows: float = 0.0, highlights: float = 0.0) -> np.ndarray:
    """Independently recover or deepen shadow zones and highlight zones.

    Math:
        L = luminance(RGB)
        shadow_weight = (1.0 - L)^2
        highlight_weight = L^2
        I_out = I_in + shadow_weight * (shadows / 100) + highlight_weight * (highlights / 100)
    """
    f = to_float32(img)
    rgb = f[:, :, :3]
    luma = rgb_to_luminance(rgb, mode="rec709")[:, :, None]

    s_weight = (1.0 - luma) ** 2.0
    h_weight = luma ** 2.0

    s_delta = (shadows / 100.0) * s_weight
    h_delta = (highlights / 100.0) * h_weight

    out = np.clip(rgb + s_delta + h_delta, 0.0, 1.0)
    if f.shape[2] == 4:
        out = np.concatenate([out, f[:, :, 3:4]], axis=-1)
    return to_uint8(out)


def trace_highlights_shadows(r: int, g: int, b: int, a: int, params: dict) -> dict:
    shadows = float(params.get("shadows", 0.0))
    highlights = float(params.get("highlights", 0.0))
    luma = (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255.0
    s_weight = (1.0 - luma) ** 2.0
    h_weight = luma ** 2.0
    s_delta = (shadows / 100.0) * s_weight * 255.0
    h_delta = (highlights / 100.0) * h_weight * 255.0

    steps = [
        f"Luminance L = (0.2126×{r} + 0.7152×{g} + 0.0722×{b})/255 = {luma:.3f}",
        f"Shadow weight (1-L)^2 = {s_weight:.3f}, delta = {s_delta:+.1f}",
        f"Highlight weight L^2 = {h_weight:.3f}, delta = {h_delta:+.1f}",
    ]
    out_rgb = []
    for name, val in [("Red", r), ("Green", g), ("Blue", b)]:
        calc = val + s_delta + h_delta
        clamped = max(0.0, min(255.0, calc))
        final_val = int(round(clamped))
        steps.append(f"{name}: {val} + ({s_delta:+.1f}) + ({h_delta:+.1f}) = {calc:.1f} → clamp → {final_val}")
        out_rgb.append(final_val)

    return {
        "formula": "I_out = clamp(I_in + (1-L)^2 × (shadows/100) + L^2 × (highlights/100), 0, 255)",
        "steps": steps,
        "output": [out_rgb[0], out_rgb[1], out_rgb[2], a],
    }


def filter_temperature(img: np.ndarray, warmth: float = 0.0) -> np.ndarray:
    """Adjust color temperature towards warm amber (+warmth) or cool cyan (-warmth).

    Math:
        R_out = R + (warmth / 100) * 0.2
        B_out = B - (warmth / 100) * 0.2
    """
    f = to_float32(img)
    delta = (warmth / 100.0) * 0.2
    out = f.copy()
    out[:, :, 0] = np.clip(out[:, :, 0] + delta, 0.0, 1.0)
    out[:, :, 2] = np.clip(out[:, :, 2] - delta, 0.0, 1.0)
    return to_uint8(out)


def trace_temperature(r: int, g: int, b: int, a: int, params: dict) -> dict:
    warmth = float(params.get("warmth", 0.0))
    delta = (warmth / 100.0) * 0.2 * 255.0
    new_r = int(round(max(0.0, min(255.0, r + delta))))
    new_b = int(round(max(0.0, min(255.0, b - delta))))
    steps = [
        f"Warmth delta = ({warmth}/100) × 0.2 × 255 = {delta:+.1f}",
        f"Red: {r} + {delta:+.1f} = {r + delta:.1f} → {new_r}",
        f"Green: {g} (unchanged) → {g}",
        f"Blue: {b} - {delta:+.1f} = {b - delta:.1f} → {new_b}",
    ]
    return {
        "formula": "R' = R + warmth × 0.2, B' = B − warmth × 0.2",
        "steps": steps,
        "output": [new_r, g, new_b, a],
    }


def filter_tint(img: np.ndarray, tint: float = 0.0) -> np.ndarray:
    """Adjust green-magenta color balance tint.

    Math:
        G_out = G - (tint / 100) * 0.2  (positive tint boosts magenta by suppressing green)
    """
    f = to_float32(img)
    delta = (tint / 100.0) * 0.2
    out = f.copy()
    out[:, :, 1] = np.clip(out[:, :, 1] - delta, 0.0, 1.0)
    return to_uint8(out)


def trace_tint(r: int, g: int, b: int, a: int, params: dict) -> dict:
    tint = float(params.get("tint", 0.0))
    delta = (tint / 100.0) * 0.2 * 255.0
    new_g = int(round(max(0.0, min(255.0, g - delta))))
    steps = [
        f"Tint offset on Green channel = -({tint}/100) × 0.2 × 255 = {-delta:+.1f}",
        f"Red: {r} (unchanged)",
        f"Green: {g} - {delta:+.1f} = {g - delta:.1f} → {new_g}",
        f"Blue: {b} (unchanged)",
    ]
    return {
        "formula": "G' = clamp(G − (tint / 100) × 0.2, 0, 255)",
        "steps": steps,
        "output": [r, new_g, b, a],
    }


def filter_vibrance(img: np.ndarray, amount: float = 0.0) -> np.ndarray:
    """Smart saturation boost prioritizing muted colors while protecting saturated tones.

    Math:
        max_c = max(R, G, B), min_c = min(R, G, B)
        current_sat = (max_c - min_c) / (max_c + 1e-5)
        weight = (1.0 - current_sat) * (amount / 100)
        I_out = I_in + (I_in - avg_luma) * weight
    """
    f = to_float32(img)
    rgb = f[:, :, :3]
    max_c = np.max(rgb, axis=-1, keepdims=True)
    min_c = np.min(rgb, axis=-1, keepdims=True)
    delta = max_c - min_c
    sat = delta / (max_c + 1e-6)

    factor = (amount / 100.0) * (1.0 - sat)
    luma = rgb_to_luminance(rgb, mode="rec709")[:, :, None]

    out = np.clip(rgb + (rgb - luma) * factor, 0.0, 1.0)
    if f.shape[2] == 4:
        out = np.concatenate([out, f[:, :, 3:4]], axis=-1)
    return to_uint8(out)


def trace_vibrance(r: int, g: int, b: int, a: int, params: dict) -> dict:
    amount = float(params.get("amount", 0.0))
    rn, gn, bn = r / 255.0, g / 255.0, b / 255.0
    max_c = max(rn, gn, bn)
    min_c = min(rn, gn, bn)
    sat = (max_c - min_c) / (max_c + 1e-6)
    factor = (amount / 100.0) * (1.0 - sat)
    luma = 0.2126 * rn + 0.7152 * gn + 0.0722 * bn

    steps = [
        f"Input normalized: R={rn:.3f}, G={gn:.3f}, B={bn:.3f}",
        f"Max={max_c:.3f}, Min={min_c:.3f} → Saturation={sat:.3f}",
        f"Vibrance weighting (1 - sat) = {1.0 - sat:.3f} → factor={factor:+.3f}",
        f"Luma L={luma:.3f}",
    ]
    out_rgb = []
    for name, norm, val in [("Red", rn, r), ("Green", gn, g), ("Blue", bn, b)]:
        calc_norm = norm + (norm - luma) * factor
        clamped = max(0.0, min(1.0, calc_norm))
        final_val = int(round(clamped * 255.0))
        steps.append(f"{name}: {val} + ({norm:.3f} - {luma:.3f}) × {factor:+.3f} = {calc_norm:.3f} → {final_val}")
        out_rgb.append(final_val)

    return {
        "formula": "I_out = clamp(I_in + (I_in − L) × (amount / 100) × (1 − Sat), 0, 255)",
        "steps": steps,
        "output": [out_rgb[0], out_rgb[1], out_rgb[2], a],
    }
