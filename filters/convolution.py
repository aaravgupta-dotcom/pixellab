"""Convolution and Detail Filters (Filters 18 to 25)
Authoritative NumPy implementations for spatial 2D filtering, blurring, sharpening, and edge detection.
"""

import numpy as np
from filters.helpers import to_float32, to_uint8, convolve2d, convolve_rgb, rgb_to_luminance, _make_gaussian_kernel


def filter_gaussian_blur(img: np.ndarray, radius: int = 2) -> np.ndarray:
    """Separable 2D Gaussian blur spatial smoothing.

    Math:
        G(x, y) = exp(-(x^2 + y^2) / (2 * sigma^2))
        I_out = convolve2d(I_in, G)
    """
    f = to_float32(img)
    k = _make_gaussian_kernel(radius=radius)
    out = convolve_rgb(f, k)
    return to_uint8(out)


def trace_gaussian_blur(r: int, g: int, b: int, a: int, params: dict) -> dict:
    rad = int(params.get("radius", 2))
    k = _make_gaussian_kernel(radius=rad)
    center_weight = k[rad, rad]
    steps = [
        f"Gaussian kernel size: {2*rad+1}×{2*rad+1} (radius={rad})",
        f"Center weight G(0, 0) = {center_weight:.4f}",
        f"Kernel sum = {np.sum(k):.4f} (normalized)",
        f"Pixel computation: weighted sum over {2*rad+1}×{2*rad+1} neighborhood around (x, y)",
    ]
    return {
        "formula": "I_out(x, y) = ∑∑ I_in(x+i, y+j) × G(i, j)",
        "steps": steps,
        "output": [r, g, b, a],
    }


def filter_box_blur(img: np.ndarray, radius: int = 2) -> np.ndarray:
    """Normalized uniform box blur convolution.

    Math:
        K[i, j] = 1.0 / ((2*r + 1)^2)
        I_out = convolve2d(I_in, K)
    """
    f = to_float32(img)
    r = max(1, int(radius))
    size = 2 * r + 1
    k = np.ones((size, size), dtype=np.float32) / (size * size)
    out = convolve_rgb(f, k)
    return to_uint8(out)


def trace_box_blur(r: int, g: int, b: int, a: int, params: dict) -> dict:
    rad = int(params.get("radius", 2))
    size = 2 * rad + 1
    count = size * size
    w = 1.0 / count
    steps = [
        f"Box blur window: {size}×{size} = {count} pixels",
        f"Uniform cell weight = 1 / {count} = {w:.5f}",
        "Local mean calculated by summing neighborhood and dividing by cell count",
    ]
    return {
        "formula": f"I_out(x, y) = (1/{count}) × ∑_(i,j) I_in(x+i, y+j)",
        "steps": steps,
        "output": [r, g, b, a],
    }


def filter_motion_blur(img: np.ndarray, distance: int = 7, angle: float = 0.0) -> np.ndarray:
    """Linear directional motion blur simulation.

    Math:
        K is a line segment of length D oriented at angle θ, normalized to sum to 1.
    """
    f = to_float32(img)
    d = max(1, int(distance))
    rad = np.radians(float(angle))
    size = 2 * d + 1
    k = np.zeros((size, size), dtype=np.float32)
    center = d
    for step in np.linspace(-d, d, num=2 * d + 1):
        dx = int(round(step * np.cos(rad)))
        dy = int(round(step * np.sin(rad)))
        x_idx = np.clip(center + dx, 0, size - 1)
        y_idx = np.clip(center + dy, 0, size - 1)
        k[y_idx, x_idx] += 1.0

    k_sum = np.sum(k)
    if k_sum > 0:
        k /= k_sum
    else:
        k[center, center] = 1.0

    out = convolve_rgb(f, k)
    return to_uint8(out)


def trace_motion_blur(r: int, g: int, b: int, a: int, params: dict) -> dict:
    dist = int(params.get("distance", 7))
    ang = float(params.get("angle", 0.0))
    steps = [
        f"Motion distance: {dist}px at {ang:.1f}°",
        f"Line kernel samples: {2 * dist + 1} points along vector [cos({ang:.0f}°), sin({ang:.0f}°)]",
        "Result is directional average along velocity line",
    ]
    return {
        "formula": "I_out(x, y) = (1/N) × ∑_(k) I_in(x + k·cos θ, y + k·sin θ)",
        "steps": steps,
        "output": [r, g, b, a],
    }


def filter_sharpen(img: np.ndarray, amount: float = 1.0) -> np.ndarray:
    """Laplacian high-pass detail boost sharpening.

    Math:
        K = [ [0, -s, 0], [-s, 1 + 4s, -s], [0, -s, 0] ]
    """
    f = to_float32(img)
    s = float(amount)
    k = np.array([[0.0, -s, 0.0], [-s, 1.0 + 4.0 * s, -s], [0.0, -s, 0.0]], dtype=np.float32)
    out = convolve_rgb(f, k)
    return to_uint8(np.clip(out, 0.0, 1.0))


def trace_sharpen(r: int, g: int, b: int, a: int, params: dict) -> dict:
    s = float(params.get("amount", 1.0))
    steps = [
        f"Sharpen strength s = {s:.2f}",
        f"3×3 Kernel: center = {1.0 + 4.0 * s:.2f}, cardinal neighbors = {-s:.2f}",
        "High frequencies amplified by subtracting second derivative (Laplacian)",
    ]
    return {
        "formula": "I_out = I_in + s × (4·I_in − (I_up + I_down + I_left + I_right))",
        "steps": steps,
        "output": [r, g, b, a],
    }


def filter_unsharp_mask(
    img: np.ndarray, radius: int = 2, amount: float = 1.5, threshold: float = 0.02
) -> np.ndarray:
    """Classic photographic unsharp mask: original + amount * (original - blurred)."""
    f = to_float32(img)
    k = _make_gaussian_kernel(radius=radius)
    blurred = convolve_rgb(f, k)
    diff = f[:, :, :3] - blurred[:, :, :3]
    mask = (np.abs(diff) >= float(threshold)).astype(np.float32)
    out_rgb = f[:, :, :3] + float(amount) * diff * mask
    out = f.copy()
    out[:, :, :3] = np.clip(out_rgb, 0.0, 1.0)
    return to_uint8(out)


def trace_unsharp_mask(r: int, g: int, b: int, a: int, params: dict) -> dict:
    amt = float(params.get("amount", 1.5))
    rad = int(params.get("radius", 2))
    th = float(params.get("threshold", 0.02))
    steps = [
        f"Gaussian blur baseline radius = {rad}px",
        f"High pass difference: Δ = Original − Blurred",
        f"Threshold gate: |Δ| ≥ {th:.3f} to avoid magnifying flat noise",
        f"Sharpened value = Original + {amt:.2f} × Δ",
    ]
    return {
        "formula": "I_out = I_orig + amount × (I_orig − I_blur) if |diff| ≥ threshold",
        "steps": steps,
        "output": [r, g, b, a],
    }


def filter_emboss(img: np.ndarray, strength: float = 1.5, angle: float = 135.0) -> np.ndarray:
    """Directional derivative emboss producing pseudo-3D relief shading.

    Math:
        K is 3×3 directional gradient matrix biased by 0.5 (neutral gray 128).
    """
    f = to_float32(img)
    s = float(strength)
    rad = np.radians(float(angle))
    dx = np.cos(rad) * s
    dy = np.sin(rad) * s

    k = np.array(
        [
            [-dx - dy, -dy, dx - dy],
            [-dx, 0.0, dx],
            [-dx + dy, dy, dx + dy],
        ],
        dtype=np.float32,
    )

    luma = rgb_to_luminance(f[:, :, :3], mode="rec709")
    deriv = convolve2d(luma, k)
    embossed = np.clip(deriv + 0.5, 0.0, 1.0)[:, :, None]
    out = np.repeat(embossed, 3, axis=-1)
    if f.shape[2] == 4:
        out = np.concatenate([out, f[:, :, 3:4]], axis=-1)
    return to_uint8(out)


def trace_emboss(r: int, g: int, b: int, a: int, params: dict) -> dict:
    s = float(params.get("strength", 1.5))
    ang = float(params.get("angle", 135.0))
    steps = [
        f"Directional angle = {ang:.0f}°, strength = {s:.2f}",
        "Directional gradient computed across 3×3 neighborhood",
        "Added +128 (0.5) neutral gray offset to reveal shadows and highlights",
    ]
    return {
        "formula": "I_out = clamp(∇_θ(Luminance) + 128, 0, 255)",
        "steps": steps,
        "output": [128, 128, 128, a],
    }


def filter_sobel_edge(
    img: np.ndarray, threshold: float = 0.1, mode: str = "magnitude"
) -> np.ndarray:
    """Sobel edge detection measuring horizontal, vertical, or total gradient magnitude.

    Math:
        G_x = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]] * L
        G_y = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]] * L
        G = sqrt(G_x^2 + G_y^2)
    """
    f = to_float32(img)
    luma = rgb_to_luminance(f[:, :, :3], mode="rec709")

    kx = np.array([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], dtype=np.float32) / 4.0
    ky = np.array([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]], dtype=np.float32) / 4.0

    gx = convolve2d(luma, kx)
    gy = convolve2d(luma, ky)

    if mode == "horizontal":
        edge = np.abs(gx)
    elif mode == "vertical":
        edge = np.abs(gy)
    else:  # magnitude
        edge = np.sqrt(gx**2 + gy**2)

    th = float(threshold)
    edge = np.where(edge >= th, edge, 0.0)
    edge = np.clip(edge, 0.0, 1.0)[:, :, None]

    out = np.repeat(edge, 3, axis=-1)
    if f.shape[2] == 4:
        out = np.concatenate([out, f[:, :, 3:4]], axis=-1)
    return to_uint8(out)


def trace_sobel_edge(r: int, g: int, b: int, a: int, params: dict) -> dict:
    mode = str(params.get("mode", "magnitude"))
    th = float(params.get("threshold", 0.1))
    steps = [
        "Horizontal gradient G_x = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]] / 4",
        "Vertical gradient   G_y = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]] / 4",
        f"Mode: {mode}, Threshold gate: {th:.2f}",
        "Edge magnitude = √(G_x² + G_y²)",
    ]
    return {
        "formula": "G = √(G_x² + G_y²); out = (G ≥ threshold) ? G : 0",
        "steps": steps,
        "output": [0, 0, 0, a],
    }


def filter_laplacian_edge(img: np.ndarray, strength: float = 1.0) -> np.ndarray:
    """Second-derivative Laplacian operator detecting omnidirectional zero-crossings."""
    f = to_float32(img)
    luma = rgb_to_luminance(f[:, :, :3], mode="rec709")
    k = np.array([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    lap = np.abs(convolve2d(luma, k)) * float(strength)
    edge = np.clip(lap, 0.0, 1.0)[:, :, None]
    out = np.repeat(edge, 3, axis=-1)
    if f.shape[2] == 4:
        out = np.concatenate([out, f[:, :, 3:4]], axis=-1)
    return to_uint8(out)


def trace_laplacian_edge(r: int, g: int, b: int, a: int, params: dict) -> dict:
    s = float(params.get("strength", 1.0))
    steps = [
        "Laplacian kernel: [[0, 1, 0], [1, -4, 1], [0, 1, 0]]",
        f"Strength multiplier = {s:.2f}",
        "Measures rate of change of gradient (second spatial derivative Δ²f)",
    ]
    return {
        "formula": "Δ²L = |(L_top + L_bottom + L_left + L_right − 4·L_center)| × strength",
        "steps": steps,
        "output": [0, 0, 0, a],
    }
