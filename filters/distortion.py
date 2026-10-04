"""Distortion and Warping Filters (Filters 26 to 32)
Authoritative NumPy implementations for geometric remaps and polar coordinate transforms.
"""

import numpy as np
from filters.helpers import to_float32, to_uint8, remap_bilinear


def filter_swirl(img: np.ndarray, angle: float = 180.0, radius: float = 0.8) -> np.ndarray:
    """Twist pixels around the image center with quadratic radial falloff.

    Math:
        r = sqrt((x - cx)^2 + (y - cy)^2)
        theta' = theta + angle * (1 - r / R)^2  for r < R
    """
    f = to_float32(img)
    H, W = f.shape[:2]
    cx = (W - 1.0) / 2.0
    cy = (H - 1.0) / 2.0
    max_r = min(cx, cy) * max(float(radius), 0.05)

    grid_y, grid_x = np.indices((H, W), dtype=np.float32)
    dx = grid_x - cx
    dy = grid_y - cy
    r = np.sqrt(dx**2 + dy**2)
    theta = np.arctan2(dy, dx)

    ang_rad = np.radians(float(angle))
    factor = np.clip(1.0 - (r / max_r), 0.0, 1.0) ** 2.0
    new_theta = theta + ang_rad * factor

    map_x = cx + r * np.cos(new_theta)
    map_y = cy + r * np.sin(new_theta)

    out = remap_bilinear(f, map_x, map_y, mode="reflect")
    return to_uint8(out)


def trace_swirl(r: int, g: int, b: int, a: int, params: dict) -> dict:
    ang = float(params.get("angle", 180.0))
    rad = float(params.get("radius", 0.8))
    steps = [
        f"Center pivot (cx, cy) with active radius = {rad*100:.0f}%",
        f"Twist angle = {ang:.1f}° at center, decaying to 0° at boundary",
        "Polar coordinates rotated: θ' = θ + Δθ × (1 − r/R)²",
        "Sampled with bilinear interpolation",
    ]
    return {
        "formula": "θ' = θ + angle × (1 − r / R)²",
        "steps": steps,
        "output": [r, g, b, a],
    }


def filter_ripple(
    img: np.ndarray, amplitude: float = 10.0, frequency: float = 15.0
) -> np.ndarray:
    """Simulate water surface wave propagation using dual sinusoidal coordinate offsets."""
    f = to_float32(img)
    H, W = f.shape[:2]
    grid_y, grid_x = np.indices((H, W), dtype=np.float32)

    amp = float(amplitude)
    freq = float(frequency)
    kx = (2.0 * np.pi * freq) / max(H, 1)
    ky = (2.0 * np.pi * freq) / max(W, 1)

    map_x = grid_x + amp * np.sin(grid_y * kx)
    map_y = grid_y + amp * np.cos(grid_x * ky)

    out = remap_bilinear(f, map_x, map_y, mode="reflect")
    return to_uint8(out)


def trace_ripple(r: int, g: int, b: int, a: int, params: dict) -> dict:
    amp = float(params.get("amplitude", 10.0))
    freq = float(params.get("frequency", 15.0))
    steps = [
        f"Wave Amplitude A = {amp:.1f}px, Wave Frequency f = {freq:.1f}",
        "x' = x + A · sin(2π · f · y / H)",
        "y' = y + A · cos(2π · f · x / W)",
        "Smooth sinusoidal wave refraction",
    ]
    return {
        "formula": "x' = x + A·sin(ω·y), y' = y + A·cos(ω·x)",
        "steps": steps,
        "output": [r, g, b, a],
    }


def filter_fisheye(img: np.ndarray, strength: float = 0.5) -> np.ndarray:
    """Radial barrel / pincushion optical lens curvature."""
    f = to_float32(img)
    H, W = f.shape[:2]
    cx = (W - 1.0) / 2.0
    cy = (H - 1.0) / 2.0
    r_max = np.sqrt(cx**2 + cy**2)

    grid_y, grid_x = np.indices((H, W), dtype=np.float32)
    dx = grid_x - cx
    dy = grid_y - cy
    r = np.sqrt(dx**2 + dy**2)
    theta = np.arctan2(dy, dx)

    r_norm = r / max(r_max, 1e-5)
    s = float(strength)
    # Distortion power law
    r_dist = r * (1.0 + s * (r_norm**2)) / (1.0 + max(0.0, s))

    map_x = cx + r_dist * np.cos(theta)
    map_y = cy + r_dist * np.sin(theta)

    out = remap_bilinear(f, map_x, map_y, mode="reflect")
    return to_uint8(out)


def trace_fisheye(r: int, g: int, b: int, a: int, params: dict) -> dict:
    s = float(params.get("strength", 0.5))
    steps = [
        f"Lens curvature strength = {s:.2f}",
        "Normalized radial distance r_norm = r / R_max",
        "Displacement curve: r_dist = r · (1 + s · r_norm²)",
        "Peripheral compression creating wide-angle barrel effect",
    ]
    return {
        "formula": "r' = r × (1 + strength × (r / R_max)²)",
        "steps": steps,
        "output": [r, g, b, a],
    }


def filter_pinch_bulge(
    img: np.ndarray, strength: float = 0.5, radius: float = 0.8
) -> np.ndarray:
    """Localized radial expansion (bulge) or contraction (pinch)."""
    f = to_float32(img)
    H, W = f.shape[:2]
    cx = (W - 1.0) / 2.0
    cy = (H - 1.0) / 2.0
    max_r = min(cx, cy) * max(float(radius), 0.05)

    grid_y, grid_x = np.indices((H, W), dtype=np.float32)
    dx = grid_x - cx
    dy = grid_y - cy
    r = np.sqrt(dx**2 + dy**2)

    s = float(strength)
    power = 1.0 + s if s >= 0 else 1.0 / (1.0 - s + 1e-5)

    inside = r < max_r
    r_mapped = r.copy()
    r_mapped[inside] = max_r * ((r[inside] / max_r) ** power)

    ratio = np.ones_like(r)
    non_zero = r > 1e-5
    ratio[non_zero] = r_mapped[non_zero] / r[non_zero]

    map_x = cx + dx * ratio
    map_y = cy + dy * ratio

    out = remap_bilinear(f, map_x, map_y, mode="reflect")
    return to_uint8(out)


def trace_pinch_bulge(r: int, g: int, b: int, a: int, params: dict) -> dict:
    s = float(params.get("strength", 0.5))
    rad = float(params.get("radius", 0.8))
    steps = [
        f"Strength = {s:+.2f} ({'Bulge' if s >= 0 else 'Pinch'}), Radius = {rad*100:.0f}%",
        "Mapped radial distance r' = R × (r / R)^p",
        "Power coefficient controls center expansion/contraction",
    ]
    return {
        "formula": "r' = R × (r / R)^(1 + strength)",
        "steps": steps,
        "output": [r, g, b, a],
    }


def filter_kaleidoscope(img: np.ndarray, segments: int = 6) -> np.ndarray:
    """Reflective radial symmetry creating mirror facets."""
    f = to_float32(img)
    H, W = f.shape[:2]
    cx = (W - 1.0) / 2.0
    cy = (H - 1.0) / 2.0

    grid_y, grid_x = np.indices((H, W), dtype=np.float32)
    dx = grid_x - cx
    dy = grid_y - cy
    r = np.sqrt(dx**2 + dy**2)
    theta = np.arctan2(dy, dx) % (2.0 * np.pi)

    n = max(2, int(segments))
    sector = (2.0 * np.pi) / n

    folded_theta = np.abs((theta % sector) - (sector / 2.0))

    map_x = cx + r * np.cos(folded_theta)
    map_y = cy + r * np.sin(folded_theta)

    out = remap_bilinear(f, map_x, map_y, mode="reflect")
    return to_uint8(out)


def trace_kaleidoscope(r: int, g: int, b: int, a: int, params: dict) -> dict:
    n = max(2, int(params.get("segments", 6)))
    steps = [
        f"Kaleidoscope segments = {n}",
        f"Sector angle α = 360° / {n} = {360.0/n:.1f}°",
        "Folded angle: θ' = |(θ mod α) − α/2|",
        "Mirrored reflection within each angular wedge",
    ]
    return {
        "formula": "θ' = |(θ mod (2π/N)) − π/N|",
        "steps": steps,
        "output": [r, g, b, a],
    }


def filter_tiny_planet(img: np.ndarray, scale: float = 1.0) -> np.ndarray:
    """Stereographic polar coordinate projection wrapping horizon into a sphere."""
    f = to_float32(img)
    H, W = f.shape[:2]
    cx = (W - 1.0) / 2.0
    cy = (H - 1.0) / 2.0
    r_max = min(cx, cy) / max(float(scale), 0.1)

    grid_y, grid_x = np.indices((H, W), dtype=np.float32)
    dx = grid_x - cx
    dy = grid_y - cy
    r = np.sqrt(dx**2 + dy**2)
    theta = np.arctan2(dy, dx)

    # u maps angle [-pi, pi] to [0, W - 1]
    map_x = ((theta + np.pi) / (2.0 * np.pi)) * (W - 1)
    # v maps radial distance [0, r_max] to [0, H - 1]
    map_y = (r / r_max) * (H - 1)

    out = remap_bilinear(f, map_x, map_y, mode="reflect")
    return to_uint8(out)


def trace_tiny_planet(r: int, g: int, b: int, a: int, params: dict) -> dict:
    sc = float(params.get("scale", 1.0))
    steps = [
        f"Polar Stereographic Projection (scale={sc:.2f})",
        "Converts Cartesian coordinates (x, y) into polar (r, θ)",
        "x' = (θ + π) / 2π × (W − 1)",
        "y' = (r / R_max) × (H − 1)",
    ]
    return {
        "formula": "u = (θ + π)/2π × W, v = (r / R) × H",
        "steps": steps,
        "output": [r, g, b, a],
    }


def filter_mirror_tunnel(img: np.ndarray, repeat: int = 3, zoom: float = 1.5) -> np.ndarray:
    """Recursive concentric infinity mirror reflections."""
    f = to_float32(img)
    H, W = f.shape[:2]
    cx = (W - 1.0) / 2.0
    cy = (H - 1.0) / 2.0

    grid_y, grid_x = np.indices((H, W), dtype=np.float32)
    dx = grid_x - cx
    dy = grid_y - cy

    n = max(1, int(repeat))
    z = float(zoom) ** n

    map_x = cx + dx * z
    map_y = cy + dy * z

    out = remap_bilinear(f, map_x, map_y, mode="reflect")
    return to_uint8(out)


def trace_mirror_tunnel(r: int, g: int, b: int, a: int, params: dict) -> dict:
    n = int(params.get("repeat", 3))
    z = float(params.get("zoom", 1.5))
    steps = [
        f"Recursive mirror iterations = {n}, zoom per bounce = {z:.2f}",
        f"Total scale expansion = {z**n:.2f}",
        "Mirrored reflection at boundaries creates infinite regression",
    ]
    return {
        "formula": "x' = cx + (x − cx) × zoom^n (with mirror bounce)",
        "steps": steps,
        "output": [r, g, b, a],
    }
