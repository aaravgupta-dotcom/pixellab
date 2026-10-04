"""PixelLab Image Processing Utilities
Authoritative NumPy numerical helper routines for image filtering,
convolution, color conversion, coordinate mapping, and data type clamping.
"""

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def to_float32(img: np.ndarray) -> np.ndarray:
    """Convert an image array (uint8 or float) to float32 normalized in [0.0, 1.0]."""
    if img.dtype == np.uint8:
        return img.astype(np.float32) / 255.0
    return np.clip(img.astype(np.float32), 0.0, 1.0)


def to_uint8(img: np.ndarray) -> np.ndarray:
    """Convert a float32 image array in [0.0, 1.0] to uint8 clamped in [0, 255]."""
    return np.clip(np.round(img * 255.0), 0, 255).astype(np.uint8)


def _make_gaussian_kernel(radius: int, sigma: float = None) -> np.ndarray:
    """Generate a normalized 2D Gaussian convolution kernel."""
    r = max(1, int(radius))
    if sigma is None or sigma <= 0:
        s = max(r / 2.0, 0.5)
    else:
        s = float(sigma)
    coords = np.arange(-r, r + 1, dtype=np.float32)
    x, y = np.meshgrid(coords, coords)
    kernel = np.exp(-(x**2 + y**2) / (2.0 * s**2))
    return kernel / np.sum(kernel)


def convolve2d(channel: np.ndarray, kernel: np.ndarray, mode: str = "reflect") -> np.ndarray:
    """Vectorized 2D spatial convolution on a single channel using sliding_window_view.

    Args:
        channel: 2D array of shape (H, W) in float32.
        kernel: 2D array of shape (Kh, Kw) in float32. Kh and Kw must be odd.
        mode: Boundary padding mode ('reflect', 'edge', 'constant', etc.)

    Returns:
        Convolved 2D array of shape (H, W) in float32.
    """
    kh, kw = kernel.shape
    pad_h = kh // 2
    pad_w = kw // 2
    padded = np.pad(channel, ((pad_h, pad_h), (pad_w, pad_w)), mode=mode)
    windows = sliding_window_view(padded, (kh, kw))
    # windows shape: (H, W, kh, kw)
    return np.einsum("ijab,ab->ij", windows, kernel)


def convolve_rgb(img: np.ndarray, kernel: np.ndarray, mode: str = "reflect") -> np.ndarray:
    """Convolve each channel of an RGB/RGBA image with a 2D kernel."""
    out = np.zeros_like(img, dtype=np.float32)
    # Process color channels (leaving alpha channel unchanged if present)
    channels_to_process = min(3, img.shape[2])
    for c in range(channels_to_process):
        out[:, :, c] = convolve2d(img[:, :, c], kernel, mode=mode)
    if img.shape[2] == 4:
        out[:, :, 3] = img[:, :, 3]
    return out


def rgb_to_luminance(rgb: np.ndarray, mode: str = "rec709") -> np.ndarray:
    """Compute 2D luminance array from RGB image array using specified standard weights.

    Weights:
        - rec709: (0.2126 R + 0.7152 G + 0.0722 B) - modern sRGB / HDTV
        - rec601: (0.2990 R + 0.5870 G + 0.1140 B) - legacy SDTV / NTSC
        - average: (0.3333 R + 0.3333 G + 0.3333 B) - unweighted mean
    """
    if mode == "rec601":
        weights = np.array([0.2990, 0.5870, 0.1140], dtype=np.float32)
    elif mode == "average":
        weights = np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0], dtype=np.float32)
    else:  # rec709
        weights = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)

    return np.tensordot(rgb[:, :, :3], weights, axes=([2], [0]))


def rgb_to_hsv(rgb: np.ndarray) -> np.ndarray:
    """Vectorized conversion from RGB [0, 1] to HSV (H in [0, 360], S in [0, 1], V in [0, 1])."""
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    max_c = np.maximum(np.maximum(r, g), b)
    min_c = np.minimum(np.minimum(r, g), b)
    delta = max_c - min_c

    # Value
    v = max_c

    # Saturation
    s = np.zeros_like(v)
    non_zero = max_c > 1e-6
    s[non_zero] = delta[non_zero] / max_c[non_zero]

    # Hue
    h = np.zeros_like(v)
    d_mask = delta > 1e-6

    # When r is max
    r_mask = d_mask & (max_c == r)
    h[r_mask] = (60.0 * ((g[r_mask] - b[r_mask]) / delta[r_mask]) + 360.0) % 360.0

    # When g is max
    g_mask = d_mask & (max_c == g) & ~r_mask
    h[g_mask] = (60.0 * ((b[g_mask] - r[g_mask]) / delta[g_mask]) + 120.0) % 360.0

    # When b is max
    b_mask = d_mask & (max_c == b) & ~r_mask & ~g_mask
    h[b_mask] = (60.0 * ((r[b_mask] - g[b_mask]) / delta[b_mask]) + 240.0) % 360.0

    hsv = np.stack([h, s, v], axis=-1)
    if rgb.shape[2] == 4:
        hsv = np.concatenate([hsv, rgb[:, :, 3:4]], axis=-1)
    return hsv


def hsv_to_rgb(hsv: np.ndarray) -> np.ndarray:
    """Vectorized conversion from HSV (H in [0, 360], S in [0, 1], V in [0, 1]) to RGB [0, 1]."""
    h = hsv[:, :, 0] % 360.0
    s = np.clip(hsv[:, :, 1], 0.0, 1.0)
    v = np.clip(hsv[:, :, 2], 0.0, 1.0)

    c = v * s
    x = c * (1.0 - np.abs((h / 60.0) % 2.0 - 1.0))
    m = v - c

    h_segment = (h / 60.0).astype(np.int32) % 6

    rgb = np.zeros((h.shape[0], h.shape[1], 3), dtype=np.float32)

    seg0 = h_segment == 0
    rgb[seg0] = np.stack([c[seg0], x[seg0], np.zeros_like(c[seg0])], axis=-1)

    seg1 = h_segment == 1
    rgb[seg1] = np.stack([x[seg1], c[seg1], np.zeros_like(c[seg1])], axis=-1)

    seg2 = h_segment == 2
    rgb[seg2] = np.stack([np.zeros_like(c[seg2]), c[seg2], x[seg2]], axis=-1)

    seg3 = h_segment == 3
    rgb[seg3] = np.stack([np.zeros_like(c[seg3]), x[seg3], c[seg3]], axis=-1)

    seg4 = h_segment == 4
    rgb[seg4] = np.stack([x[seg4], np.zeros_like(c[seg4]), c[seg4]], axis=-1)

    seg5 = h_segment == 5
    rgb[seg5] = np.stack([c[seg5], np.zeros_like(c[seg5]), x[seg5]], axis=-1)

    rgb += m[..., None]
    rgb = np.clip(rgb, 0.0, 1.0)

    if hsv.shape[2] == 4:
        rgb = np.concatenate([rgb, hsv[:, :, 3:4]], axis=-1)
    return rgb


def remap_bilinear(img: np.ndarray, map_x: np.ndarray, map_y: np.ndarray, mode: str = "reflect") -> np.ndarray:
    """Vectorized bilinear remapping of image coordinates.

    Args:
        img: float32 array of shape (H, W, C).
        map_x: float32 array of shape (H, W) with target x-coordinates.
        map_y: float32 array of shape (H, W) with target y-coordinates.
        mode: 'reflect' or 'clamp'
    """
    H, W = img.shape[:2]

    x0 = np.floor(map_x).astype(np.int32)
    x1 = x0 + 1
    y0 = np.floor(map_y).astype(np.int32)
    y1 = y0 + 1

    wx = (map_x - x0)[..., None]
    wy = (map_y - y0)[..., None]

    if mode == "reflect":
        def reflect_coord(coords, max_val):
            # Safe reflection for arbitrary indices
            period = 2 * (max_val - 1)
            if period <= 0:
                return np.zeros_like(coords)
            c = np.abs(coords) % period
            return np.where(c >= max_val, period - c, c)

        x0_c = reflect_coord(x0, W)
        x1_c = reflect_coord(x1, W)
        y0_c = reflect_coord(y0, H)
        y1_c = reflect_coord(y1, H)
    else:
        x0_c = np.clip(x0, 0, W - 1)
        x1_c = np.clip(x1, 0, W - 1)
        y0_c = np.clip(y0, 0, H - 1)
        y1_c = np.clip(y1, 0, H - 1)

    ia = img[y0_c, x0_c]
    ib = img[y0_c, x1_c]
    ic = img[y1_c, x0_c]
    id_ = img[y1_c, x1_c]

    top = ia * (1.0 - wx) + ib * wx
    bottom = ic * (1.0 - wx) + id_ * wx
    return top * (1.0 - wy) + bottom * wy
