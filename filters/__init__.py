"""PixelLab Filter Registry
Consolidates all 50 filters across 6 categories with metadata, docstrings,
and source code introspection for both processing and developer education.
"""

import inspect
from filters.basic import (
    filter_brightness, trace_brightness,
    filter_contrast, trace_contrast,
    filter_gamma, trace_gamma,
    filter_exposure, trace_exposure,
    filter_highlights_shadows, trace_highlights_shadows,
    filter_temperature, trace_temperature,
    filter_tint, trace_tint,
    filter_vibrance, trace_vibrance,
)
from filters.color import (
    filter_saturation, trace_saturation,
    filter_hue_rotate, trace_hue_rotate,
    filter_grayscale, trace_grayscale,
    filter_sepia, trace_sepia,
    filter_invert, trace_invert,
    filter_posterize, trace_posterize,
    filter_solarize, trace_solarize,
    filter_channel_mixer, trace_channel_mixer,
    filter_selective_color, trace_selective_color,
)
from filters.convolution import (
    filter_gaussian_blur, trace_gaussian_blur,
    filter_box_blur, trace_box_blur,
    filter_motion_blur, trace_motion_blur,
    filter_sharpen, trace_sharpen,
    filter_unsharp_mask, trace_unsharp_mask,
    filter_emboss, trace_emboss,
    filter_sobel_edge, trace_sobel_edge,
    filter_laplacian_edge, trace_laplacian_edge,
)
from filters.distortion import (
    filter_swirl, trace_swirl,
    filter_ripple, trace_ripple,
    filter_fisheye, trace_fisheye,
    filter_pinch_bulge, trace_pinch_bulge,
    filter_kaleidoscope, trace_kaleidoscope,
    filter_tiny_planet, trace_tiny_planet,
    filter_mirror_tunnel, trace_mirror_tunnel,
)
from filters.stylize import (
    filter_oil_paint, trace_oil_paint,
    filter_pencil_sketch, trace_pencil_sketch,
    filter_cross_hatch, trace_cross_hatch,
    filter_pointillism, trace_pointillism,
    filter_mosaic_tiles, trace_mosaic_tiles,
    filter_crystallize, trace_crystallize,
    filter_stained_glass, trace_stained_glass,
    filter_linocut, trace_linocut,
)
from filters.rare import (
    filter_thermal_vision, trace_thermal_vision,
    filter_cyanotype, trace_cyanotype,
    filter_risograph, trace_risograph,
    filter_newsprint_cmyk, trace_newsprint_cmyk,
    filter_chromatic_aberration, trace_chromatic_aberration,
    filter_pixel_sort, trace_pixel_sort,
    filter_datamosh, trace_datamosh,
    filter_vhs_tracking, trace_vhs_tracking,
    filter_dither, trace_dither,
    filter_kirlian_aura, trace_kirlian_aura,
)

FILTER_REGISTRY = {
    # ---------------- 1. Tone and light (8) ----------------
    "brightness": {
        "fn": filter_brightness,
        "trace_fn": trace_brightness,
        "category": "tone",
        "name": "Brightness",
        "description": "Lifts or drops overall image illumination by adding a uniform channel offset.",
        "math": "I_out = clamp(I_in + offset / 100, 0, 1)",
        "params": [
            {"id": "offset", "name": "Offset", "type": "range", "min": -100, "max": 100, "default": 15, "step": 1, "tooltip": "Offset in percentage"}
        ]
    },
    "contrast": {
        "fn": filter_contrast,
        "trace_fn": trace_contrast,
        "category": "tone",
        "name": "Contrast",
        "description": "Expands or compresses tonal range around neutral middle gray (128).",
        "math": "I_out = clamp((I_in − 0.5) × factor + 0.5, 0, 1)",
        "params": [
            {"id": "factor", "name": "Factor", "type": "range", "min": 0.0, "max": 3.0, "default": 1.3, "step": 0.05, "tooltip": "Contrast scaling factor"}
        ]
    },
    "gamma": {
        "fn": filter_gamma,
        "trace_fn": trace_gamma,
        "category": "tone",
        "name": "Gamma",
        "description": "Applies non-linear power-law curve to alter midtone values without clipping limits.",
        "math": "I_out = I_in ^ (1 / gamma)",
        "params": [
            {"id": "gamma", "name": "Gamma", "type": "range", "min": 0.2, "max": 3.0, "default": 1.2, "step": 0.05, "tooltip": "Power curve exponent"}
        ]
    },
    "exposure": {
        "fn": filter_exposure,
        "trace_fn": trace_exposure,
        "category": "tone",
        "name": "Exposure",
        "description": "Simulates camera lens exposure adjustment measured in photographic stops.",
        "math": "I_out = clamp(I_in × 2^EV, 0, 1)",
        "params": [
            {"id": "ev", "name": "EV Stops", "type": "range", "min": -3.0, "max": 3.0, "default": 0.5, "step": 0.1, "tooltip": "Exposure Value stops"}
        ]
    },
    "highlights_shadows": {
        "fn": filter_highlights_shadows,
        "trace_fn": trace_highlights_shadows,
        "category": "tone",
        "name": "Highlights & Shadows",
        "description": "Independently lifts dark shadows or suppresses bright highlights.",
        "math": "I_out = I_in + (1−L)² × (shadows/100) + L² × (highlights/100)",
        "params": [
            {"id": "shadows", "name": "Shadows", "type": "range", "min": -100, "max": 100, "default": 25, "step": 1, "tooltip": "Shadow zone lift"},
            {"id": "highlights", "name": "Highlights", "type": "range", "min": -100, "max": 100, "default": -20, "step": 1, "tooltip": "Highlight zone pull"}
        ]
    },
    "temperature": {
        "fn": filter_temperature,
        "trace_fn": trace_temperature,
        "category": "tone",
        "name": "Temperature",
        "description": "Warms colors toward golden amber or cools toward winter cyan.",
        "math": "R' = R + warmth × 0.2, B' = B − warmth × 0.2",
        "params": [
            {"id": "warmth", "name": "Warmth", "type": "range", "min": -100, "max": 100, "default": 30, "step": 1, "tooltip": "Amber warmth vs cyan cool"}
        ]
    },
    "tint": {
        "fn": filter_tint,
        "trace_fn": trace_tint,
        "category": "tone",
        "name": "Tint",
        "description": "Balances green and magenta chromatic shifts.",
        "math": "G' = clamp(G − tint × 0.2, 0, 1)",
        "params": [
            {"id": "tint", "name": "Tint", "type": "range", "min": -100, "max": 100, "default": 20, "step": 1, "tooltip": "Magenta (+) vs Green (-)"}
        ]
    },
    "vibrance": {
        "fn": filter_vibrance,
        "trace_fn": trace_vibrance,
        "category": "tone",
        "name": "Vibrance",
        "description": "Boosts muted colors gently while protecting already-saturated skin tones.",
        "math": "I_out = I_in + (I_in − L) × (amount / 100) × (1 − Sat)",
        "params": [
            {"id": "amount", "name": "Amount", "type": "range", "min": -100, "max": 100, "default": 40, "step": 1, "tooltip": "Smart saturation boost"}
        ]
    },

    # ---------------- 2. Color (9) ----------------
    "saturation": {
        "fn": filter_saturation,
        "trace_fn": trace_saturation,
        "category": "color",
        "name": "Saturation",
        "description": "Intensifies or strips color purity across all channels equally.",
        "math": "I_out = clamp(L + (I_in − L) × factor, 0, 1)",
        "params": [
            {"id": "factor", "name": "Factor", "type": "range", "min": 0.0, "max": 3.0, "default": 1.4, "step": 0.05, "tooltip": "Color purity scale"}
        ]
    },
    "hue_rotate": {
        "fn": filter_hue_rotate,
        "trace_fn": trace_hue_rotate,
        "category": "color",
        "name": "Hue Rotate",
        "description": "Spins all color hues around the 360-degree color wheel.",
        "math": "H' = (H + degrees) mod 360",
        "params": [
            {"id": "degrees", "name": "Angle", "type": "range", "min": 0, "max": 360, "default": 90, "step": 1, "tooltip": "Wheel rotation in degrees"}
        ]
    },
    "grayscale": {
        "fn": filter_grayscale,
        "trace_fn": trace_grayscale,
        "category": "color",
        "name": "Grayscale",
        "description": "Strips chromatic color using selectable photometric weighting standards.",
        "math": "L = w_R · R + w_G · G + w_B · B",
        "params": [
            {"id": "mode", "name": "Standard", "type": "select", "options": ["rec709", "rec601", "average"], "default": "rec709", "tooltip": "Rec.709 (HDTV), Rec.601 (SDTV), or Average"}
        ]
    },
    "sepia": {
        "fn": filter_sepia,
        "trace_fn": trace_sepia,
        "category": "color",
        "name": "Sepia",
        "description": "Applies a vintage warm silver-gelatin photographic tint.",
        "math": "RGB' = (1−t)·RGB + t·(Matrix × RGB)",
        "params": [
            {"id": "intensity", "name": "Intensity", "type": "range", "min": 0.0, "max": 1.0, "default": 0.8, "step": 0.05, "tooltip": "Sepia wash strength"}
        ]
    },
    "invert": {
        "fn": filter_invert,
        "trace_fn": trace_invert,
        "category": "color",
        "name": "Invert",
        "description": "Reverses all color channels like a darkroom negative film strip.",
        "math": "I_out = 1.0 − I_in",
        "params": [
            {"id": "amount", "name": "Amount", "type": "range", "min": 0.0, "max": 1.0, "default": 1.0, "step": 0.05, "tooltip": "Negative inversion blend"}
        ]
    },
    "posterize": {
        "fn": filter_posterize,
        "trace_fn": trace_posterize,
        "category": "color",
        "name": "Posterize",
        "description": "Quantizes continuous color gradients into discrete flat tonal bands.",
        "math": "I_out = ⌊I_in · (N−1) + 0.5⌋ / (N−1)",
        "params": [
            {"id": "levels", "name": "Tonal Levels", "type": "range", "min": 2, "max": 16, "default": 4, "step": 1, "tooltip": "Number of color steps per channel"}
        ]
    },
    "solarize": {
        "fn": filter_solarize,
        "trace_fn": trace_solarize,
        "category": "color",
        "name": "Solarize",
        "description": "Inverts only values exceeding a threshold (Sabattier darkroom effect).",
        "math": "I_out = (I_in > threshold) ? (1 − I_in) : I_in",
        "params": [
            {"id": "threshold", "name": "Threshold", "type": "range", "min": 0.1, "max": 0.9, "default": 0.5, "step": 0.05, "tooltip": "Inversion inflection point"}
        ]
    },
    "channel_mixer": {
        "fn": filter_channel_mixer,
        "trace_fn": trace_channel_mixer,
        "category": "color",
        "name": "Channel Mixer",
        "description": "Cross-blends red, green, and blue channels via linear matrix coefficients.",
        "math": "RGB' = Matrix_3x3 × RGB",
        "params": [
            {"id": "rr", "name": "Red from Red", "type": "range", "min": 0.0, "max": 2.0, "default": 1.0, "step": 0.1, "tooltip": "R -> R"},
            {"id": "rg", "name": "Red from Green", "type": "range", "min": 0.0, "max": 2.0, "default": 0.0, "step": 0.1, "tooltip": "G -> R"},
            {"id": "rb", "name": "Red from Blue", "type": "range", "min": 0.0, "max": 2.0, "default": 0.2, "step": 0.1, "tooltip": "B -> R"},
            {"id": "gr", "name": "Green from Red", "type": "range", "min": 0.0, "max": 2.0, "default": 0.0, "step": 0.1, "tooltip": "R -> G"},
            {"id": "gg", "name": "Green from Green", "type": "range", "min": 0.0, "max": 2.0, "default": 1.0, "step": 0.1, "tooltip": "G -> G"},
            {"id": "gb", "name": "Green from Blue", "type": "range", "min": 0.0, "max": 2.0, "default": 0.0, "step": 0.1, "tooltip": "B -> G"},
            {"id": "br", "name": "Blue from Red", "type": "range", "min": 0.0, "max": 2.0, "default": 0.3, "step": 0.1, "tooltip": "R -> B"},
            {"id": "bg", "name": "Blue from Green", "type": "range", "min": 0.0, "max": 2.0, "default": 0.0, "step": 0.1, "tooltip": "G -> B"},
            {"id": "bb", "name": "Blue from Blue", "type": "range", "min": 0.0, "max": 2.0, "default": 0.8, "step": 0.1, "tooltip": "B -> B"}
        ]
    },
    "selective_color": {
        "fn": filter_selective_color,
        "trace_fn": trace_selective_color,
        "category": "color",
        "name": "Selective Color",
        "description": "Preserves a chosen color hue while desaturating everything else to monochrome.",
        "math": "out = weight · RGB + (1 − weight) · L",
        "params": [
            {"id": "target_hue", "name": "Target Hue", "type": "range", "min": 0, "max": 360, "default": 0, "step": 1, "tooltip": "Target color angle (0=Red, 120=Green, 240=Blue)"},
            {"id": "tolerance", "name": "Tolerance", "type": "range", "min": 5, "max": 120, "default": 35, "step": 1, "tooltip": "Hue angular window"}
        ]
    },

    # ---------------- 3. Convolution and detail (8) ----------------
    "gaussian_blur": {
        "fn": filter_gaussian_blur,
        "trace_fn": trace_gaussian_blur,
        "category": "convolution",
        "name": "Gaussian Blur",
        "description": "Smooths fine noise and details using a bell-shaped Gaussian distribution kernel.",
        "math": "G(x, y) = exp(−(x² + y²) / (2σ²))",
        "params": [
            {"id": "radius", "name": "Radius", "type": "range", "min": 1, "max": 8, "default": 2, "step": 1, "tooltip": "Gaussian window extent"}
        ]
    },
    "box_blur": {
        "fn": filter_box_blur,
        "trace_fn": trace_box_blur,
        "category": "convolution",
        "name": "Box Blur",
        "description": "Averages every pixel with equal weights across a square neighborhood window.",
        "math": "K[i, j] = 1 / (2r + 1)²",
        "params": [
            {"id": "radius", "name": "Radius", "type": "range", "min": 1, "max": 8, "default": 2, "step": 1, "tooltip": "Box window radius"}
        ]
    },
    "motion_blur": {
        "fn": filter_motion_blur,
        "trace_fn": trace_motion_blur,
        "category": "convolution",
        "name": "Motion Blur",
        "description": "Simulates camera or subject high-speed movement along a linear directional angle.",
        "math": "Line kernel aligned to vector [cos θ, sin θ]",
        "params": [
            {"id": "distance", "name": "Distance", "type": "range", "min": 2, "max": 16, "default": 7, "step": 1, "tooltip": "Length of motion trail"},
            {"id": "angle", "name": "Angle", "type": "range", "min": 0, "max": 180, "default": 0, "step": 5, "tooltip": "Angle of motion vector"}
        ]
    },
    "sharpen": {
        "fn": filter_sharpen,
        "trace_fn": trace_sharpen,
        "category": "convolution",
        "name": "Sharpen",
        "description": "Enhances edge contrast and fine texture details by subtracting second derivatives.",
        "math": "K = [[0, −s, 0], [−s, 1 + 4s, −s], [0, −s, 0]]",
        "params": [
            {"id": "amount", "name": "Amount", "type": "range", "min": 0.2, "max": 3.0, "default": 1.0, "step": 0.1, "tooltip": "High frequency boost"}
        ]
    },
    "unsharp_mask": {
        "fn": filter_unsharp_mask,
        "trace_fn": trace_unsharp_mask,
        "category": "convolution",
        "name": "Unsharp Mask",
        "description": "Classic darkroom technique: amplifies the difference between original and blurred copy.",
        "math": "out = orig + amount · (orig − blur) if |diff| ≥ threshold",
        "params": [
            {"id": "radius", "name": "Radius", "type": "range", "min": 1, "max": 4, "default": 2, "step": 1, "tooltip": "Blur radius"},
            {"id": "amount", "name": "Amount", "type": "range", "min": 0.5, "max": 3.0, "default": 1.5, "step": 0.1, "tooltip": "Boost coefficient"},
            {"id": "threshold", "name": "Threshold", "type": "range", "min": 0.0, "max": 0.1, "default": 0.02, "step": 0.01, "tooltip": "Noise threshold gate"}
        ]
    },
    "emboss": {
        "fn": filter_emboss,
        "trace_fn": trace_emboss,
        "category": "convolution",
        "name": "Emboss",
        "description": "Produces a tactile 3D bas-relief stamp effect along directional lighting angles.",
        "math": "out = clamp(∇_θ(Luminance) + 128, 0, 255)",
        "params": [
            {"id": "strength", "name": "Strength", "type": "range", "min": 0.5, "max": 3.0, "default": 1.5, "step": 0.1, "tooltip": "Relief depth"},
            {"id": "angle", "name": "Light Angle", "type": "range", "min": 0, "max": 360, "default": 135, "step": 15, "tooltip": "Sun illumination angle"}
        ]
    },
    "sobel_edge": {
        "fn": filter_sobel_edge,
        "trace_fn": trace_sobel_edge,
        "category": "convolution",
        "name": "Sobel Edge",
        "description": "Calculates spatial luminance gradients using discrete 3×3 Sobel differential operators.",
        "math": "G = √(G_x² + G_y²)",
        "params": [
            {"id": "threshold", "name": "Threshold", "type": "range", "min": 0.02, "max": 0.5, "default": 0.1, "step": 0.02, "tooltip": "Noise floor cutoff"},
            {"id": "mode", "name": "Mode", "type": "select", "options": ["magnitude", "horizontal", "vertical"], "default": "magnitude", "tooltip": "Gradient component"}
        ]
    },
    "laplacian_edge": {
        "fn": filter_laplacian_edge,
        "trace_fn": trace_laplacian_edge,
        "category": "convolution",
        "name": "Laplacian Edge",
        "description": "Detects high-frequency rapid transitions in all directions using second spatial derivatives.",
        "math": "Δ²L = 4·Center − (Up + Down + Left + Right)",
        "params": [
            {"id": "strength", "name": "Strength", "type": "range", "min": 0.5, "max": 3.0, "default": 1.2, "step": 0.1, "tooltip": "Edge line amplifier"}
        ]
    },

    # ---------------- 4. Distortion (7) ----------------
    "swirl": {
        "fn": filter_swirl,
        "trace_fn": trace_swirl,
        "category": "distortion",
        "name": "Swirl",
        "description": "Twists image pixels in an Archimedean vortex with quadratic radial falloff.",
        "math": "θ' = θ + angle · (1 − r/R)²",
        "params": [
            {"id": "angle", "name": "Angle", "type": "range", "min": -360, "max": 360, "default": 120, "step": 10, "tooltip": "Vortex rotation angle"},
            {"id": "radius", "name": "Radius", "type": "range", "min": 0.2, "max": 1.0, "default": 0.8, "step": 0.05, "tooltip": "Vortex boundary reach"}
        ]
    },
    "ripple": {
        "fn": filter_ripple,
        "trace_fn": trace_ripple,
        "category": "distortion",
        "name": "Ripple / Water",
        "description": "Refracts pixels across sinusoidal harmonic waves simulating water ripples.",
        "math": "x' = x + A·sin(ω·y), y' = y + A·cos(ω·x)",
        "params": [
            {"id": "amplitude", "name": "Amplitude", "type": "range", "min": 2, "max": 30, "default": 10, "step": 1, "tooltip": "Wave height displacement"},
            {"id": "frequency", "name": "Frequency", "type": "range", "min": 5, "max": 40, "default": 15, "step": 1, "tooltip": "Wave crest pitch"}
        ]
    },
    "fisheye": {
        "fn": filter_fisheye,
        "trace_fn": trace_fisheye,
        "category": "distortion",
        "name": "Fisheye",
        "description": "Bends perspective outward into an extreme ultra-wide curved barrel lens projection.",
        "math": "r' = r · (1 + strength · (r / R_max)²)",
        "params": [
            {"id": "strength", "name": "Strength", "type": "range", "min": -0.8, "max": 1.5, "default": 0.6, "step": 0.05, "tooltip": "Curvature curve"}
        ]
    },
    "pinch_bulge": {
        "fn": filter_pinch_bulge,
        "trace_fn": trace_pinch_bulge,
        "category": "distortion",
        "name": "Pinch / Bulge",
        "description": "Expands or sucks pixels into a localized central radial field.",
        "math": "r' = R · (r / R)^(1 + strength)",
        "params": [
            {"id": "strength", "name": "Strength", "type": "range", "min": -0.8, "max": 1.2, "default": 0.5, "step": 0.05, "tooltip": "Positive=Bulge, Negative=Pinch"},
            {"id": "radius", "name": "Radius", "type": "range", "min": 0.2, "max": 1.0, "default": 0.75, "step": 0.05, "tooltip": "Active circular radius"}
        ]
    },
    "kaleidoscope": {
        "fn": filter_kaleidoscope,
        "trace_fn": trace_kaleidoscope,
        "category": "distortion",
        "name": "Kaleidoscope",
        "description": "Mirrors pie-slice triangular segments radially into symmetrical stained-glass reflections.",
        "math": "θ' = |(θ mod (2π/N)) − π/N|",
        "params": [
            {"id": "segments", "name": "Segments", "type": "range", "min": 2, "max": 16, "default": 6, "step": 1, "tooltip": "Number of radial mirror symmetry planes"}
        ]
    },
    "tiny_planet": {
        "fn": filter_tiny_planet,
        "trace_fn": trace_tiny_planet,
        "category": "distortion",
        "name": "Tiny Planet",
        "description": "Wraps panoramic Cartesian coordinates into a circular stereographic polar world.",
        "math": "u = (θ + π)/2π · W, v = (r/R) · H",
        "params": [
            {"id": "scale", "name": "Scale", "type": "range", "min": 0.5, "max": 2.0, "default": 1.0, "step": 0.05, "tooltip": "Polar projection zoom"}
        ]
    },
    "mirror_tunnel": {
        "fn": filter_mirror_tunnel,
        "trace_fn": trace_mirror_tunnel,
        "category": "distortion",
        "name": "Mirror Tunnel",
        "description": "Repeats concentric inward reflections simulating standing between two facing mirrors.",
        "math": "x' = cx + (x − cx) · zoom^n",
        "params": [
            {"id": "repeat", "name": "Repeat", "type": "range", "min": 1, "max": 6, "default": 3, "step": 1, "tooltip": "Number of reflections"},
            {"id": "zoom", "name": "Zoom", "type": "range", "min": 1.1, "max": 2.2, "default": 1.5, "step": 0.05, "tooltip": "Zoom factor per bounce"}
        ]
    },

    # ---------------- 5. Stylize (8) ----------------
    "oil_paint": {
        "fn": filter_oil_paint,
        "trace_fn": trace_oil_paint,
        "category": "stylize",
        "name": "Oil Paint (Kuwahara)",
        "description": "Preserves edges while simplifying brush regions via minimal variance quadrants.",
        "math": "out = Mean(Quadrant with min variance)",
        "params": [
            {"id": "radius", "name": "Brush Radius", "type": "range", "min": 1, "max": 4, "default": 2, "step": 1, "tooltip": "Kuwahara stroke width"}
        ]
    },
    "pencil_sketch": {
        "fn": filter_pencil_sketch,
        "trace_fn": trace_pencil_sketch,
        "category": "stylize",
        "name": "Pencil Sketch",
        "description": "Simulates graphite pencil line work using an inverted blurred color-dodge blend.",
        "math": "out = L / (1 − Blur(1−L))",
        "params": [
            {"id": "contrast", "name": "Pencil Contrast", "type": "range", "min": 0.8, "max": 2.5, "default": 1.5, "step": 0.1, "tooltip": "Graphite stroke density"},
            {"id": "blend", "name": "Blend", "type": "range", "min": 0.2, "max": 1.0, "default": 0.9, "step": 0.05, "tooltip": "Sketch vs original photo"}
        ]
    },
    "cross_hatch": {
        "fn": filter_cross_hatch,
        "trace_fn": trace_cross_hatch,
        "category": "stylize",
        "name": "Cross-hatch",
        "description": "Builds shading from intersecting pen strokes drawn at stepped tonal thresholds.",
        "math": "Iterative line layers enabled by decreasing luminance",
        "params": [
            {"id": "density", "name": "Hatch Pitch", "type": "range", "min": 3, "max": 10, "default": 5, "step": 1, "tooltip": "Pen line distance spacing"}
        ]
    },
    "pointillism": {
        "fn": filter_pointillism,
        "trace_fn": trace_pointillism,
        "category": "stylize",
        "name": "Pointillism",
        "description": "Deconstructs scene into pure circular stippling dots of local sampled pigments.",
        "math": "Disk dots stamped at jittered lattice nodes",
        "params": [
            {"id": "dot_size", "name": "Dot Size", "type": "range", "min": 3, "max": 12, "default": 6, "step": 1, "tooltip": "Diameter of stipple points"},
            {"id": "density", "name": "Coverage", "type": "range", "min": 40, "max": 100, "default": 85, "step": 5, "tooltip": "Density percentage"},
            {"id": "seed", "name": "Seed", "type": "seed", "default": 42, "tooltip": "Random distribution seed"}
        ]
    },
    "mosaic_tiles": {
        "fn": filter_mosaic_tiles,
        "trace_fn": trace_mosaic_tiles,
        "category": "stylize",
        "name": "Mosaic Tiles",
        "description": "Assembles image from tessellated ceramic squares separated by dark mortar grout.",
        "math": "CellColor = I(center(x, y)); Grout at grid edges",
        "params": [
            {"id": "tile_size", "name": "Tile Size", "type": "range", "min": 6, "max": 32, "default": 14, "step": 1, "tooltip": "Square tile dimensions"},
            {"id": "grout", "name": "Grout Width", "type": "range", "min": 0, "max": 3, "default": 1, "step": 1, "tooltip": "Mortar joint thickness"}
        ]
    },
    "crystallize": {
        "fn": filter_crystallize,
        "trace_fn": trace_crystallize,
        "category": "stylize",
        "name": "Crystallize (Voronoi)",
        "description": "Groups pixels into organic polygonal facets based on nearest Euclidean seed sites.",
        "math": "Site = argmin_k ||(x, y) − Seed_k||²",
        "params": [
            {"id": "cell_size", "name": "Cell Size", "type": "range", "min": 8, "max": 32, "default": 16, "step": 1, "tooltip": "Polygon crystal diameter"},
            {"id": "seed", "name": "Seed", "type": "seed", "default": 42, "tooltip": "Voronoi seed distribution"}
        ]
    },
    "stained_glass": {
        "fn": filter_stained_glass,
        "trace_fn": trace_stained_glass,
        "category": "stylize",
        "name": "Stained Glass",
        "description": "Voronoi translucent polygonal crystals framed by dark decorative lead came outlines.",
        "math": "Border = (d2 − d1 ≤ width) ? Lead : CellColor",
        "params": [
            {"id": "cell_size", "name": "Cell Size", "type": "range", "min": 8, "max": 32, "default": 16, "step": 1, "tooltip": "Glass pane size"},
            {"id": "border_width", "name": "Lead Width", "type": "range", "min": 1, "max": 4, "default": 2, "step": 1, "tooltip": "Lead came border thickness"},
            {"id": "seed", "name": "Seed", "type": "seed", "default": 42, "tooltip": "Voronoi seed distribution"}
        ]
    },
    "linocut": {
        "fn": filter_linocut,
        "trace_fn": trace_linocut,
        "category": "stylize",
        "name": "Linocut / Woodcut",
        "description": "Simulates hand-carved relief printing with sharp gouged wood-grain textures.",
        "math": "Cut = (L + grain·SinTexture ≥ threshold) ? Paper : ReliefInk",
        "params": [
            {"id": "threshold", "name": "Threshold", "type": "range", "min": 0.2, "max": 0.8, "default": 0.5, "step": 0.05, "tooltip": "Inking threshold"},
            {"id": "grain", "name": "Wood Grain", "type": "range", "min": 0.0, "max": 0.8, "default": 0.35, "step": 0.05, "tooltip": "Chisel gouge texture intensity"}
        ]
    },

    # ---------------- 6. Rare and unique (10) ----------------
    "thermal_vision": {
        "fn": filter_thermal_vision,
        "trace_fn": trace_thermal_vision,
        "category": "rare",
        "name": "Thermal Vision",
        "description": "Maps luminance to a FLIR heat-signature spectrum (black, indigo, red, yellow, white).",
        "math": "RGB = PiecewiseLinear(L, [Black, Indigo, Red, Yellow, White])",
        "params": [
            {"id": "contrast", "name": "Heat Contrast", "type": "range", "min": 0.5, "max": 2.5, "default": 1.2, "step": 0.1, "tooltip": "Thermal gradient spread"}
        ]
    },
    "cyanotype": {
        "fn": filter_cyanotype,
        "trace_fn": trace_cyanotype,
        "category": "rare",
        "name": "Cyanotype",
        "description": "Prussian blue architectural sun-print with unbleached paper grain highlights.",
        "math": "RGB = Paper × L + PrussianBlue × (1 − L)",
        "params": [
            {"id": "blue_intensity", "name": "Ink Density", "type": "range", "min": 0.5, "max": 1.5, "default": 1.0, "step": 0.05, "tooltip": "Prussian blue concentration"},
            {"id": "contrast", "name": "Paper Contrast", "type": "range", "min": 0.8, "max": 2.0, "default": 1.2, "step": 0.1, "tooltip": "Sunlight exposure curve"},
            {"id": "grain", "name": "Paper Fiber", "type": "range", "min": 0.0, "max": 0.6, "default": 0.25, "step": 0.05, "tooltip": "Paper fiber grain noise"}
        ]
    },
    "risograph": {
        "fn": filter_risograph,
        "trace_fn": trace_risograph,
        "category": "rare",
        "name": "Risograph",
        "description": "Dual-drum press: Fluorescent Pink + Teal inks with misregistered plate alignment and grain.",
        "math": "Paper × (1 − PinkDrum × (1−Pink)) × (1 − TealDrum_shifted × (1−Teal))",
        "params": [
            {"id": "misregistration", "name": "Plate Shift", "type": "range", "min": 0.0, "max": 8.0, "default": 3.0, "step": 0.5, "tooltip": "Mechanical drum misregistration offset"},
            {"id": "grain", "name": "Screen Grain", "type": "range", "min": 0.1, "max": 1.0, "default": 0.5, "step": 0.05, "tooltip": "Halftone screen ink texture"},
            {"id": "seed", "name": "Seed", "type": "seed", "default": 42, "tooltip": "Drum grain texture seed"}
        ]
    },
    "newsprint_cmyk": {
        "fn": filter_newsprint_cmyk,
        "trace_fn": trace_newsprint_cmyk,
        "category": "rare",
        "name": "Newsprint CMYK",
        "description": "Offset press: rotated halftone screens (15°, 75°, 0°, 45°) with plate misregistration.",
        "math": "RGB = (1−Screen_C)·(1−Screen_M)·(1−Screen_Y)·(1−Screen_K) × Newsprint",
        "params": [
            {"id": "dot_scale", "name": "Halftone Dot Pitch", "type": "range", "min": 2.0, "max": 8.0, "default": 4.0, "step": 0.5, "tooltip": "Screen frequency pitch"},
            {"id": "misregistration", "name": "Plate Shift", "type": "range", "min": 0.0, "max": 6.0, "default": 2.0, "step": 0.5, "tooltip": "Color plate spatial displacement"}
        ]
    },
    "chromatic_aberration": {
        "fn": filter_chromatic_aberration,
        "trace_fn": trace_chromatic_aberration,
        "category": "rare",
        "name": "Chromatic Aberration",
        "description": "Shifts red and blue channels apart radially like a cheap analog lens.",
        "math": "R'(x + Δ, y + Δ), G'(x, y), B'(x − Δ, y − Δ)",
        "params": [
            {"id": "shift", "name": "Displacement", "type": "range", "min": 1, "max": 20, "default": 7, "step": 1, "tooltip": "Radial channel divergence"},
            {"id": "angle", "name": "Direction Angle", "type": "range", "min": 0, "max": 360, "default": 45, "step": 5, "tooltip": "Dispersion angle"}
        ]
    },
    "pixel_sort": {
        "fn": filter_pixel_sort,
        "trace_fn": trace_pixel_sort,
        "category": "rare",
        "name": "Pixel Sort",
        "description": "Sorts contiguous runs of pixels by brightness within a threshold window.",
        "math": "SortSpan(I) for contiguous segments where threshold_low ≤ L ≤ threshold_high",
        "params": [
            {"id": "threshold_low", "name": "Low Threshold", "type": "range", "min": 0.05, "max": 0.5, "default": 0.25, "step": 0.05, "tooltip": "Start sorting above this luma"},
            {"id": "threshold_high", "name": "High Threshold", "type": "range", "min": 0.5, "max": 0.95, "default": 0.8, "step": 0.05, "tooltip": "Stop sorting above this luma"},
            {"id": "direction", "name": "Direction", "type": "select", "options": ["horizontal", "vertical"], "default": "horizontal", "tooltip": "Sorting direction axis"}
        ]
    },
    "datamosh": {
        "fn": filter_datamosh,
        "trace_fn": trace_datamosh,
        "category": "rare",
        "name": "Glitch / Datamosh",
        "description": "Simulates digital video compression block tears, channel jitter, and sync loss.",
        "math": "Block(x, y) ← Block(x + Δx, y + Δy) on corrupted macroblock slots",
        "params": [
            {"id": "block_size", "name": "Block Size", "type": "range", "min": 8, "max": 32, "default": 16, "step": 4, "tooltip": "MPEG macroblock size"},
            {"id": "shift_strength", "name": "Displacement", "type": "range", "min": 5, "max": 40, "default": 20, "step": 2, "tooltip": "Motion vector shift distance"},
            {"id": "tear_prob", "name": "Tear Frequency", "type": "range", "min": 0.05, "max": 0.4, "default": 0.15, "step": 0.05, "tooltip": "Scanline horizontal slice probability"},
            {"id": "seed", "name": "Seed", "type": "seed", "default": 42, "tooltip": "Glitch sequence seed"}
        ]
    },
    "vhs_tracking": {
        "fn": filter_vhs_tracking,
        "trace_fn": trace_vhs_tracking,
        "category": "rare",
        "name": "VHS Tracking",
        "description": "Magnetic video tape decay: line jitter, chroma bleed, and bottom-edge head switching noise.",
        "math": "ChromaShift(R, B) + ScanlineJitter(y) + HeadSwitchRoll(bottom 8%)",
        "params": [
            {"id": "tracking_noise", "name": "Noise Static", "type": "range", "min": 0.1, "max": 1.0, "default": 0.5, "step": 0.05, "tooltip": "RF static amplitude at tape bottom"},
            {"id": "chroma_shift", "name": "Chroma Bleed", "type": "range", "min": 0, "max": 12, "default": 5, "step": 1, "tooltip": "Subcarrier color delay"},
            {"id": "seed", "name": "Seed", "type": "seed", "default": 42, "tooltip": "Tape noise seed"}
        ]
    },
    "dither": {
        "fn": filter_dither,
        "trace_fn": trace_dither,
        "category": "rare",
        "name": "Bayer & Floyd Dither",
        "description": "Quantizes bit-depth using ordered Bayer matrices or error-diffusion diffusion.",
        "math": "Bayer: Q(I + M_4x4 · Δ); Floyd-Steinberg: Propagate error [7/16, 3/16, 5/16, 1/16]",
        "params": [
            {"id": "method", "name": "Method", "type": "select", "options": ["bayer", "floyd_steinberg"], "default": "bayer", "tooltip": "Bayer 4×4 Ordered vs Floyd-Steinberg Error Diffusion"},
            {"id": "levels", "name": "Palette Depth", "type": "range", "min": 2, "max": 8, "default": 4, "step": 1, "tooltip": "Tonal quantization levels"}
        ]
    },
    "kirlian_aura": {
        "fn": filter_kirlian_aura,
        "trace_fn": trace_kirlian_aura,
        "category": "rare",
        "name": "Kirlian Aura",
        "description": "High-voltage bio-luminescent corona discharge glowing along high-contrast silhouettes.",
        "math": "out = 0.4·Base + Bloom(Sobel(L), σ) × Tint(Hue)",
        "params": [
            {"id": "glow_radius", "name": "Corona Radius", "type": "range", "min": 2, "max": 10, "default": 5, "step": 1, "tooltip": "Bloom radiation spread"},
            {"id": "intensity", "name": "Intensity", "type": "range", "min": 0.5, "max": 3.0, "default": 1.6, "step": 0.1, "tooltip": "Aura luminance power"},
            {"id": "hue", "name": "Aura Color", "type": "range", "min": 0, "max": 360, "default": 185, "step": 5, "tooltip": "Spectral discharge hue (185=Cyan, 300=Violet, 60=Gold)"}
        ]
    }
}


def get_filters_metadata() -> dict:
    """Inspect and compile registry metadata including source code and docstrings for developer tab."""
    metadata = {}
    for fid, entry in FILTER_REGISTRY.items():
        fn = entry["fn"]
        doc = fn.__doc__ or ""
        try:
            source = inspect.getsource(fn)
        except Exception:
            source = ""

        metadata[fid] = {
            "id": fid,
            "name": entry["name"],
            "category": entry["category"],
            "description": entry["description"],
            "math": entry["math"],
            "docstring": doc.strip(),
            "source_code": source,
            "params": entry["params"],
        }
    return metadata
