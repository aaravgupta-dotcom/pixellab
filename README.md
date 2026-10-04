# PixelLab — Image Filter Studio & Numerical Laboratory

**PixelLab** is an analog darkroom / risograph-styled image filter studio and educational developer environment. It features dual twin pipelines: instant 60fps canvas-based JavaScript filter twins on `<canvas>` ImageData, paired with authoritative, vectorized Python/NumPy implementations on the backend.

An advanced **"Inside the Pixel"** tab exposes the exact mathematics behind every pixel calculation, step-by-step arithmetic traces with live numbers, animated spatial convolution visualizers, and interactive pixel arithmetic tutorials.

---

## Tech Stack

- **Backend**: Python 3.11+, Flask 3.1+, NumPy 2.x for authoritative numerical image processing, Pillow for decoding and encoding image files.
- **Frontend**: Plain HTML5, CSS3, and vanilla modern JavaScript (ES2022). Zero UI kits, zero Tailwind, zero React.
- **Typography**: Fraunces & IBM Plex Sans for editorial interface typography, JetBrains Mono for code and numeric arrays.
- **Aesthetic**: Risograph print shop / analog darkroom design system (`--paper: #ECE4D3`, `--ink: #1C1A17`, `--turmeric: #E3A41B`, `--vermilion: #D8442A`, `--verdigris: #2E7D6F`, hard 1.5px outlines, offset drop-shadows `4px 4px 0 var(--ink)`, graph-paper notebook).

---

## Project Structure

```
pixellab/
├── app.py                      # Flask backend API (/api/filters, /api/apply, /api/trace, /api/pixels)
├── filters/
│   ├── __init__.py             # FILTER_REGISTRY & metadata extractor
│   ├── helpers.py              # Vectorized convolve2d (sliding_window_view), color transforms, remap_bilinear
│   ├── basic.py                # Tone & Light filters (1–8)
│   ├── color.py                # Color & Channel filters (9–17)
│   ├── convolution.py          # Spatial Convolution & Detail filters (18–25)
│   ├── distortion.py           # Geometric Warping & Polar distortion filters (26–32)
│   ├── stylize.py              # Artistic & Painting filters (33–40)
│   └── rare.py                 # Signature & Unique filters (41–50)
├── static/
│   ├── css/style.css           # Risograph darkroom design system & typography
│   ├── js/filters.js           # Exact JavaScript twins for all 50 filters
│   ├── js/app.js               # Canvas pipeline, filter stack, zoom/pan, before/after split, undo/redo
│   ├── js/inspector.js         # Pixel inspector, 11x11 loupe, RGB/HSL edit, brush, region select, histogram
│   └── js/advanced.js          # "Inside the Pixel" tab: numeric trace, convolution visualizer, code diff, heatmap
├── templates/
│   └── index.html              # Studio interface markup
├── tests/
│   ├── test_filters.py         # Automated pytest suite for all 50 NumPy filters and shapes
│   ├── test_parity.js          # Node.js runner verifying all 50 JS filters
│   └── test_parity_comparison.py # Parity check verifying max difference <= 1 level per channel
├── requirements.txt            # Python dependencies
└── README.md
```

---

## Quickstart & Running

### 1. Installation
Clone or navigate to the directory and install dependencies:
```bash
pip install -r requirements.txt
```

### 2. Run the Studio
```bash
python app.py
```
Open your browser to:
```
http://127.0.0.1:5001
```

### 3. Run Automated Tests
```bash
# Verify all 50 NumPy filters return correct shape and uint8 dtype
pytest tests/test_filters.py -v

# Verify parity between JavaScript twins and NumPy backend (<= 1 level difference)
python tests/test_parity_comparison.py
```

---

## Complete Filter Library (All 50 Filters)

### Tone and Light (8)
1. **Brightness**: Linear channel offset $I_{\text{out}} = \text{clamp}(I_{\text{in}} + \Delta)$.
2. **Contrast**: Center-pivot scaling $I_{\text{out}} = \text{clamp}((I_{\text{in}} - 128) \times \text{factor} + 128)$.
3. **Gamma**: Power law midtone correction $I_{\text{out}} = I_{\text{in}}^{1/\gamma}$.
4. **Exposure**: Photographic stop scaling $I_{\text{out}} = \text{clamp}(I_{\text{in}} \times 2^{\text{EV}})$.
5. **Highlights & Shadows**: Independent recovery curves weighted by $(1-L)^2$ and $L^2$.
6. **Temperature**: Color temperature warmth (amber $+R/-B$) vs coolness $(-R/+B)$.
7. **Tint**: Green/Magenta balance adjustment (suppress/boost Green channel).
8. **Vibrance**: Saturation boost weighted inversely by existing saturation $(1 - \text{Sat})$.

### Color (9)
9. **Saturation**: Radial chromatic purity scaling relative to luminance $L$.
10. **Hue Rotate**: Circular angle rotation across the 360° color wheel.
11. **Grayscale**: 3 selectable weighting standards (Rec.709, Rec.601, and Average).
12. **Sepia**: Historical silver-gelatin warm tone matrix transformation.
13. **Invert**: Darkroom negative reversal $I_{\text{out}} = 255 - I_{\text{in}}$.
14. **Posterize**: Uniform tonal step quantization into $N$ distinct color buckets.
15. **Solarize**: Sabattier tone reversal above an exposure threshold.
16. **Channel Mixer**: Full $3 \times 3$ RGB cross-channel remap matrix.
17. **Selective Color**: Isolates target hue within an angular tolerance band, desaturating the remainder.

### Convolution and Detail (8)
18. **Gaussian Blur**: 2D separable Gaussian spatial smoothing kernel.
19. **Box Blur**: Uniform normalized square window mean filtering.
20. **Motion Blur**: Directional linear convolution oriented at angle $\theta$.
21. **Sharpen**: High-pass Laplacian edge detail boost.
22. **Unsharp Mask**: Subtracts Gaussian blurred copy from original with threshold gate.
23. **Emboss**: Directional 3D surface relief derivative biased by neutral gray 128.
24. **Sobel Edge**: First-derivative gradient magnitude $\sqrt{G_x^2 + G_y^2}$.
25. **Laplacian Edge**: Second-derivative omnidirectional edge detection.

### Distortion (7)
26. **Swirl**: Archimedean vortex with quadratic radial falloff.
27. **Ripple / Water**: Dual sinusoidal harmonic wave refraction.
28. **Fisheye**: Hemispherical wide-angle barrel lens curvature.
29. **Pinch / Bulge**: Localized center-weighted expansion or contraction field.
30. **Kaleidoscope**: Multi-faceted radial mirror symmetry.
31. **Tiny Planet**: Stereographic polar coordinate wrap.
32. **Mirror Tunnel**: Concentric infinite regression reflections.

### Stylize (8)
33. **Oil Paint (Kuwahara)**: Minimal variance quadrant smoothing preserving crisp boundaries.
34. **Pencil Sketch**: Color dodge blend between grayscale base and inverted blurred mask.
35. **Cross-hatch**: Multi-layered ink pen cross strokes responding to luminance thresholds.
36. **Pointillism**: Neo-impressionist stippling dots colored by sampled local pigments.
37. **Mosaic Tiles**: Tessellated ceramic blocks separated by dark mortar grout.
38. **Crystallize (Voronoi)**: Polygonal nearest-neighbor facet quantization.
39. **Stained Glass**: Voronoi crystals framed by dark lead came outlines.
40. **Linocut / Woodcut**: Relief printmaking combining thresholding with directional woodcut grain.

### Rare and Unique (10) — Signature Set
41. **Thermal Vision**: False-color gradient mapping (Black -> Indigo -> Flame Red -> Solar Yellow -> White).
42. **Cyanotype**: Prussian blue architectural print with unbleached paper grain highlights.
43. **Risograph**: Dual-drum Fluorescent Pink + Teal inks with misregistered plate alignment.
44. **Newsprint CMYK**: Process color halftone screens (15°, 75°, 0°, 45°) with plate shifts.
45. **Chromatic Aberration**: Transverse optical dispersion displacing Red and Blue channels.
46. **Pixel Sort**: Run-length sorting of contiguous pixel spans by luminance.
47. **Glitch / Datamosh**: Video compression macroblock tears, sync loss, and channel jitter.
48. **VHS Tracking**: Analog tape RF chroma delay, line jitter, and bottom head-switch static.
49. **Bayer & Floyd-Steinberg Dither**: Switchable spatial matrix vs sequential error diffusion.
50. **Kirlian Aura**: High-voltage bio-luminescent corona discharge glowing along contours.

---

## How to Add a New Filter in Under 10 Lines

Adding a new filter requires registering the Python function and registering the JavaScript twin:

### In Python (`filters/basic.py` or new module):
```python
def filter_myfilter(img: np.ndarray, strength: float = 0.5) -> np.ndarray:
    """Multiplies channels by a constant gain factor."""
    f = to_float32(img)
    return to_uint8(np.clip(f * (1.0 + float(strength)), 0.0, 1.0))
```

### In `filters/__init__.py`:
```python
FILTER_REGISTRY["myfilter"] = {
    "fn": filter_myfilter,
    "trace_fn": lambda r, g, b, a, p: {"formula": "I_out = clamp(I_in * (1 + strength))", "steps": [f"Gain: {r} * {1+float(p.get('strength',0.5)):.2f}"], "output": [min(255, int(r * 1.5)), g, b, a]},
    "category": "tone", "name": "Gain Boost", "description": "Boosts channel gain.", "math": "I * (1 + s)",
    "params": [{"id": "strength", "name": "Strength", "type": "range", "min": 0, "max": 2, "default": 0.5, "step": 0.1}]
}
```

### In JavaScript (`static/js/filters.js`):
```javascript
JS_FILTERS.myfilter = (src, p) => {
  const out = new ImageData(src.width, src.height);
  const s = 1.0 + (parseFloat(p.strength) || 0.5);
  for (let i = 0; i < src.data.length; i += 4) {
    out.data[i] = Math.min(255, src.data[i] * s);
    out.data[i+1] = Math.min(255, src.data[i+1] * s);
    out.data[i+2] = Math.min(255, src.data[i+2] * s);
    out.data[i+3] = src.data[i+3];
  }
  return out;
};
```

---

## API Specification

- `GET /api/filters`: Returns full registry metadata, parameters, descriptions, math formulas, docstrings, and introspected source code.
- `POST /api/apply`: Accepts JSON `{ image: "<base64>", filters: [{ filter_id, params }] }`, executes filter stack in NumPy, returns `{ image, timing_ms, shape, dtype }`.
- `POST /api/trace`: Accepts `{ image, x, y, filter_id, params }`, returns arithmetic trace showing substituted numbers and step breakdown.
- `POST /api/pixels`: Accepts `{ image, region: { x, y, width, height } }`, returns raw `ndarray` pixel values, memory footprint, and channel statistics.
