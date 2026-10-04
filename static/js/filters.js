/**
 * PixelLab JavaScript Filter Twins (All 50 Filters)
 * High-performance browser-side canvas ImageData implementations for real-time 60fps slider previews.
 * Numerically aligned with authoritative NumPy backend operations.
 */

const JS_FILTERS = (() => {
  // Helper: clone ImageData
  function cloneImageData(src) {
    const copy = new ImageData(src.width, src.height);
    copy.data.set(src.data);
    return copy;
  }

  // Helper: clamped byte
  function clamp(val) {
    return val < 0 ? 0 : val > 255 ? 255 : Math.round(val);
  }

  // PRNG for seeded filters
  function createPRNG(seed) {
    let s = (seed || 42) % 2147483647;
    if (s <= 0) s += 2147483646;
    return function() {
      s = (s * 16807) % 2147483647;
      return (s - 1) / 2147483646;
    };
  }

  // 2D Spatial Convolution on ImageData
  function convolve(src, kernel, kh, kw) {
    const w = src.width;
    const h = src.height;
    const out = new ImageData(w, h);
    const srcData = src.data;
    const outData = out.data;
    const padY = Math.floor(kh / 2);
    const padX = Math.floor(kw / 2);

    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        let r = 0, g = 0, b = 0;
        let ki = 0;
        for (let ky = -padY; ky <= padY; ky++) {
          let py = y + ky;
          // reflect boundary
          if (py < 0) py = -py;
          else if (py >= h) py = 2 * (h - 1) - py;
          py = Math.max(0, Math.min(h - 1, py));

          for (let kx = -padX; kx <= padX; kx++) {
            let px = x + kx;
            if (px < 0) px = -px;
            else if (px >= w) px = 2 * (w - 1) - px;
            px = Math.max(0, Math.min(w - 1, px));

            const idx = (py * w + px) * 4;
            const weight = kernel[ki++];
            r += srcData[idx] * weight;
            g += srcData[idx + 1] * weight;
            b += srcData[idx + 2] * weight;
          }
        }
        const oIdx = (y * w + x) * 4;
        outData[oIdx] = clamp(r);
        outData[oIdx + 1] = clamp(g);
        outData[oIdx + 2] = clamp(b);
        outData[oIdx + 3] = srcData[oIdx + 3];
      }
    }
    return out;
  }

  // Helper: make 2D Gaussian kernel
  function makeGaussianKernel(radius) {
    const r = Math.max(1, radius | 0);
    const sigma = Math.max(r / 2.0, 0.5);
    const size = 2 * r + 1;
    const kernel = new Float32Array(size * size);
    let sum = 0;
    let idx = 0;
    for (let y = -r; y <= r; y++) {
      for (let x = -r; x <= r; x++) {
        const val = Math.exp(-(x * x + y * y) / (2 * sigma * sigma));
        kernel[idx++] = val;
        sum += val;
      }
    }
    for (let i = 0; i < kernel.length; i++) kernel[i] /= sum;
    return { kernel, size };
  }

  // Bilinear coordinate sample
  function sampleBilinear(data, w, h, x, y) {
    // Reflect boundary
    function reflect(coord, maxVal) {
      if (maxVal <= 1) return 0;
      const period = 2 * (maxVal - 1);
      let c = Math.abs(coord) % period;
      if (c >= maxVal) c = period - c;
      return c;
    }
    const x0 = Math.floor(x);
    const y0 = Math.floor(y);
    const x1 = x0 + 1;
    const y1 = y0 + 1;
    const wx = x - x0;
    const wy = y - y0;

    const rx0 = reflect(x0, w);
    const rx1 = reflect(x1, w);
    const ry0 = reflect(y0, h);
    const ry1 = reflect(y1, h);

    const i00 = (ry0 * w + rx0) * 4;
    const i10 = (ry0 * w + rx1) * 4;
    const i01 = (ry1 * w + rx0) * 4;
    const i11 = (ry1 * w + rx1) * 4;

    const r = (data[i00] * (1 - wx) + data[i10] * wx) * (1 - wy) + (data[i01] * (1 - wx) + data[i11] * wx) * wy;
    const g = (data[i00 + 1] * (1 - wx) + data[i10 + 1] * wx) * (1 - wy) + (data[i01 + 1] * (1 - wx) + data[i11 + 1] * wx) * wy;
    const b = (data[i00 + 2] * (1 - wx) + data[i10 + 2] * wx) * (1 - wy) + (data[i01 + 2] * (1 - wx) + data[i11 + 2] * wx) * wy;
    const a = (data[i00 + 3] * (1 - wx) + data[i10 + 3] * wx) * (1 - wy) + (data[i01 + 3] * (1 - wx) + data[i11 + 3] * wx) * wy;

    return [r, g, b, a];
  }

  return {
    // ================= Category 1: Tone and Light (8) =================
    brightness: (src, p) => {
      const out = cloneImageData(src);
      const d = out.data;
      const delta = (parseFloat(p.offset) || 0) * 2.55;
      for (let i = 0; i < d.length; i += 4) {
        d[i] = clamp(d[i] + delta);
        d[i + 1] = clamp(d[i + 1] + delta);
        d[i + 2] = clamp(d[i + 2] + delta);
      }
      return out;
    },

    contrast: (src, p) => {
      const out = cloneImageData(src);
      const d = out.data;
      const f = parseFloat(p.factor) !== undefined ? parseFloat(p.factor) : 1.0;
      for (let i = 0; i < d.length; i += 4) {
        d[i] = clamp((d[i] - 128) * f + 128);
        d[i + 1] = clamp((d[i + 1] - 128) * f + 128);
        d[i + 2] = clamp((d[i + 2] - 128) * f + 128);
      }
      return out;
    },

    gamma: (src, p) => {
      const out = cloneImageData(src);
      const d = out.data;
      const g = Math.max(0.1, parseFloat(p.gamma) || 1.0);
      const invG = 1.0 / g;
      const lut = new Uint8Array(256);
      for (let i = 0; i < 256; i++) {
        lut[i] = clamp(Math.pow(i / 255.0, invG) * 255.0);
      }
      for (let i = 0; i < d.length; i += 4) {
        d[i] = lut[d[i]];
        d[i + 1] = lut[d[i + 1]];
        d[i + 2] = lut[d[i + 2]];
      }
      return out;
    },

    exposure: (src, p) => {
      const out = cloneImageData(src);
      const d = out.data;
      const ev = parseFloat(p.ev) || 0.0;
      const mult = Math.pow(2.0, ev);
      for (let i = 0; i < d.length; i += 4) {
        d[i] = clamp(d[i] * mult);
        d[i + 1] = clamp(d[i + 1] * mult);
        d[i + 2] = clamp(d[i + 2] * mult);
      }
      return out;
    },

    highlights_shadows: (src, p) => {
      const out = cloneImageData(src);
      const d = out.data;
      const sh = (parseFloat(p.shadows) || 0.0) / 100.0 * 255.0;
      const hl = (parseFloat(p.highlights) || 0.0) / 100.0 * 255.0;
      for (let i = 0; i < d.length; i += 4) {
        const luma = (0.2126 * d[i] + 0.7152 * d[i + 1] + 0.0722 * d[i + 2]) / 255.0;
        const sW = Math.pow(1.0 - luma, 2.0);
        const hW = Math.pow(luma, 2.0);
        const delta = sW * sh + hW * hl;
        d[i] = clamp(d[i] + delta);
        d[i + 1] = clamp(d[i + 1] + delta);
        d[i + 2] = clamp(d[i + 2] + delta);
      }
      return out;
    },

    temperature: (src, p) => {
      const out = cloneImageData(src);
      const d = out.data;
      const w = ((parseFloat(p.warmth) || 0.0) / 100.0) * 0.2 * 255.0;
      for (let i = 0; i < d.length; i += 4) {
        d[i] = clamp(d[i] + w);
        d[i + 2] = clamp(d[i + 2] - w);
      }
      return out;
    },

    tint: (src, p) => {
      const out = cloneImageData(src);
      const d = out.data;
      const t = ((parseFloat(p.tint) || 0.0) / 100.0) * 0.2 * 255.0;
      for (let i = 0; i < d.length; i += 4) {
        d[i + 1] = clamp(d[i + 1] - t);
      }
      return out;
    },

    vibrance: (src, p) => {
      const out = cloneImageData(src);
      const d = out.data;
      const amt = (parseFloat(p.amount) || 0.0) / 100.0;
      for (let i = 0; i < d.length; i += 4) {
        const r = d[i] / 255.0, g = d[i + 1] / 255.0, b = d[i + 2] / 255.0;
        const maxC = Math.max(r, g, b);
        const minC = Math.min(r, g, b);
        const sat = (maxC - minC) / (maxC + 1e-6);
        const factor = amt * (1.0 - sat);
        const luma = 0.2126 * r + 0.7152 * g + 0.0722 * b;
        d[i] = clamp((r + (r - luma) * factor) * 255.0);
        d[i + 1] = clamp((g + (g - luma) * factor) * 255.0);
        d[i + 2] = clamp((b + (b - luma) * factor) * 255.0);
      }
      return out;
    },

    // ================= Category 2: Color (9) =================
    saturation: (src, p) => {
      const out = cloneImageData(src);
      const d = out.data;
      const f = parseFloat(p.factor) !== undefined ? parseFloat(p.factor) : 1.0;
      for (let i = 0; i < d.length; i += 4) {
        const luma = 0.2126 * d[i] + 0.7152 * d[i + 1] + 0.0722 * d[i + 2];
        d[i] = clamp(luma + (d[i] - luma) * f);
        d[i + 1] = clamp(luma + (d[i + 1] - luma) * f);
        d[i + 2] = clamp(luma + (d[i + 2] - luma) * f);
      }
      return out;
    },

    hue_rotate: (src, p) => {
      const out = cloneImageData(src);
      const d = out.data;
      const deg = ((parseFloat(p.degrees) || 0) % 360 + 360) % 360;
      for (let i = 0; i < d.length; i += 4) {
        let r = d[i] / 255.0, g = d[i + 1] / 255.0, b = d[i + 2] / 255.0;
        const max = Math.max(r, g, b), min = Math.min(r, g, b);
        const delta = max - min;
        let h = 0, s = max === 0 ? 0 : delta / max, v = max;
        if (delta > 1e-5) {
          if (max === r) h = (60 * ((g - b) / delta) + 360) % 360;
          else if (max === g) h = (60 * ((b - r) / delta) + 120) % 360;
          else h = (60 * ((r - g) / delta) + 240) % 360;
        }
        h = (h + deg) % 360;
        const c = v * s;
        const x = c * (1 - Math.abs((h / 60) % 2 - 1));
        const m = v - c;
        let nr = 0, ng = 0, nb = 0;
        const seg = Math.floor(h / 60) % 6;
        if (seg === 0) [nr, ng, nb] = [c, x, 0];
        else if (seg === 1) [nr, ng, nb] = [x, c, 0];
        else if (seg === 2) [nr, ng, nb] = [0, c, x];
        else if (seg === 3) [nr, ng, nb] = [0, x, c];
        else if (seg === 4) [nr, ng, nb] = [x, 0, c];
        else [nr, ng, nb] = [c, 0, x];
        d[i] = clamp((nr + m) * 255.0);
        d[i + 1] = clamp((ng + m) * 255.0);
        d[i + 2] = clamp((nb + m) * 255.0);
      }
      return out;
    },

    grayscale: (src, p) => {
      const out = cloneImageData(src);
      const d = out.data;
      const mode = p.mode || "rec709";
      let wr = 0.2126, wg = 0.7152, wb = 0.0722;
      if (mode === "rec601") { wr = 0.2990; wg = 0.5870; wb = 0.1140; }
      else if (mode === "average") { wr = 1/3; wg = 1/3; wb = 1/3; }
      for (let i = 0; i < d.length; i += 4) {
        const val = clamp(wr * d[i] + wg * d[i + 1] + wb * d[i + 2]);
        d[i] = val; d[i + 1] = val; d[i + 2] = val;
      }
      return out;
    },

    sepia: (src, p) => {
      const out = cloneImageData(src);
      const d = out.data;
      const t = parseFloat(p.intensity) !== undefined ? parseFloat(p.intensity) : 1.0;
      for (let i = 0; i < d.length; i += 4) {
        const r = d[i], g = d[i + 1], b = d[i + 2];
        const sr = clamp(0.393 * r + 0.769 * g + 0.189 * b);
        const sg = clamp(0.349 * r + 0.686 * g + 0.168 * b);
        const sb = clamp(0.272 * r + 0.534 * g + 0.131 * b);
        d[i] = clamp((1 - t) * r + t * sr);
        d[i + 1] = clamp((1 - t) * g + t * sg);
        d[i + 2] = clamp((1 - t) * b + t * sb);
      }
      return out;
    },

    invert: (src, p) => {
      const out = cloneImageData(src);
      const d = out.data;
      const t = parseFloat(p.amount) !== undefined ? parseFloat(p.amount) : 1.0;
      for (let i = 0; i < d.length; i += 4) {
        d[i] = clamp((1 - t) * d[i] + t * (255 - d[i]));
        d[i + 1] = clamp((1 - t) * d[i + 1] + t * (255 - d[i + 1]));
        d[i + 2] = clamp((1 - t) * d[i + 2] + t * (255 - d[i + 2]));
      }
      return out;
    },

    posterize: (src, p) => {
      const out = cloneImageData(src);
      const d = out.data;
      const levels = Math.max(2, parseInt(p.levels) || 4);
      const step = 255.0 / (levels - 1);
      for (let i = 0; i < d.length; i += 4) {
        d[i] = clamp(Math.round(d[i] / step) * step);
        d[i + 1] = clamp(Math.round(d[i + 1] / step) * step);
        d[i + 2] = clamp(Math.round(d[i + 2] / step) * step);
      }
      return out;
    },

    solarize: (src, p) => {
      const out = cloneImageData(src);
      const d = out.data;
      const th = (parseFloat(p.threshold) || 0.5) * 255.0;
      for (let i = 0; i < d.length; i += 4) {
        if (d[i] > th) d[i] = 255 - d[i];
        if (d[i + 1] > th) d[i + 1] = 255 - d[i + 1];
        if (d[i + 2] > th) d[i + 2] = 255 - d[i + 2];
      }
      return out;
    },

    channel_mixer: (src, p) => {
      const out = cloneImageData(src);
      const d = out.data;
      const rr = parseFloat(p.rr) || 1, rg = parseFloat(p.rg) || 0, rb = parseFloat(p.rb) || 0;
      const gr = parseFloat(p.gr) || 0, gg = parseFloat(p.gg) || 1, gb = parseFloat(p.gb) || 0;
      const br = parseFloat(p.br) || 0, bg = parseFloat(p.bg) || 0, bb = parseFloat(p.bb) || 1;
      for (let i = 0; i < d.length; i += 4) {
        const r = d[i], g = d[i + 1], b = d[i + 2];
        d[i] = clamp(rr * r + rg * g + rb * b);
        d[i + 1] = clamp(gr * r + gg * g + gb * b);
        d[i + 2] = clamp(br * r + bg * g + bb * b);
      }
      return out;
    },

    selective_color: (src, p) => {
      const out = cloneImageData(src);
      const d = out.data;
      const target = (parseFloat(p.target_hue) || 0) % 360;
      const tol = Math.max(1, parseFloat(p.tolerance) || 30);
      for (let i = 0; i < d.length; i += 4) {
        const r = d[i] / 255.0, g = d[i + 1] / 255.0, b = d[i + 2] / 255.0;
        const max = Math.max(r, g, b), min = Math.min(r, g, b);
        const delta = max - min;
        let h = 0;
        if (delta > 1e-5) {
          if (max === r) h = (60 * ((g - b) / delta) + 360) % 360;
          else if (max === g) h = (60 * ((b - r) / delta) + 120) % 360;
          else h = (60 * ((r - g) / delta) + 240) % 360;
        }
        const diff = Math.abs(h - target);
        const dist = Math.min(diff, 360 - diff);
        const weight = Math.max(0, Math.min(1, 1 - dist / tol));
        const luma = 0.2126 * d[i] + 0.7152 * d[i + 1] + 0.0722 * d[i + 2];
        d[i] = clamp(weight * d[i] + (1 - weight) * luma);
        d[i + 1] = clamp(weight * d[i + 1] + (1 - weight) * luma);
        d[i + 2] = clamp(weight * d[i + 2] + (1 - weight) * luma);
      }
      return out;
    },

    // ================= Category 3: Convolution and Detail (8) =================
    gaussian_blur: (src, p) => {
      const { kernel, size } = makeGaussianKernel(p.radius || 2);
      return convolve(src, kernel, size, size);
    },

    box_blur: (src, p) => {
      const r = Math.max(1, parseInt(p.radius) || 2);
      const size = 2 * r + 1;
      const count = size * size;
      const kernel = new Float32Array(count).fill(1.0 / count);
      return convolve(src, kernel, size, size);
    },

    motion_blur: (src, p) => {
      const dist = Math.max(1, parseInt(p.distance) || 7);
      const ang = (parseFloat(p.angle) || 0) * Math.PI / 180.0;
      const size = 2 * dist + 1;
      const kernel = new Float32Array(size * size);
      const center = dist;
      let count = 0;
      for (let s = -dist; s <= dist; s++) {
        const dx = Math.round(s * Math.cos(ang));
        const dy = Math.round(s * Math.sin(ang));
        const x = Math.max(0, Math.min(size - 1, center + dx));
        const y = Math.max(0, Math.min(size - 1, center + dy));
        kernel[y * size + x] += 1.0;
        count++;
      }
      for (let i = 0; i < kernel.length; i++) kernel[i] /= count;
      return convolve(src, kernel, size, size);
    },

    sharpen: (src, p) => {
      const s = parseFloat(p.amount) || 1.0;
      const kernel = new Float32Array([
        0, -s, 0,
        -s, 1 + 4 * s, -s,
        0, -s, 0
      ]);
      return convolve(src, kernel, 3, 3);
    },

    unsharp_mask: (src, p) => {
      const rad = parseInt(p.radius) || 2;
      const amt = parseFloat(p.amount) || 1.5;
      const th = (parseFloat(p.threshold) || 0.02) * 255.0;
      const { kernel, size } = makeGaussianKernel(rad);
      const blurred = convolve(src, kernel, size, size);
      const out = cloneImageData(src);
      const d = out.data, bd = blurred.data, sd = src.data;
      for (let i = 0; i < d.length; i += 4) {
        for (let c = 0; c < 3; c++) {
          const diff = sd[i + c] - bd[i + c];
          d[i + c] = Math.abs(diff) >= th ? clamp(sd[i + c] + amt * diff) : sd[i + c];
        }
      }
      return out;
    },

    emboss: (src, p) => {
      const s = parseFloat(p.strength) || 1.5;
      const rad = ((parseFloat(p.angle) || 135) * Math.PI) / 180.0;
      const dx = Math.cos(rad) * s;
      const dy = Math.sin(rad) * s;
      const kernel = new Float32Array([
        -dx - dy, -dy, dx - dy,
        -dx, 0, dx,
        -dx + dy, dy, dx + dy
      ]);
      const w = src.width, h = src.height;
      const out = new ImageData(w, h);
      const sd = src.data, od = out.data;
      for (let y = 0; y < h; y++) {
        for (let x = 0; x < w; x++) {
          let sum = 0;
          let ki = 0;
          for (let ky = -1; ky <= 1; ky++) {
            const py = Math.max(0, Math.min(h - 1, y + ky));
            for (let kx = -1; kx <= 1; kx++) {
              const px = Math.max(0, Math.min(w - 1, x + kx));
              const idx = (py * w + px) * 4;
              const luma = 0.2126 * sd[idx] + 0.7152 * sd[idx + 1] + 0.0722 * sd[idx + 2];
              sum += luma * kernel[ki++];
            }
          }
          const val = clamp(sum + 128);
          const oIdx = (y * w + x) * 4;
          od[oIdx] = val; od[oIdx + 1] = val; od[oIdx + 2] = val; od[oIdx + 3] = sd[oIdx + 3];
        }
      }
      return out;
    },

    sobel_edge: (src, p) => {
      const th = (parseFloat(p.threshold) || 0.1) * 255.0;
      const mode = p.mode || "magnitude";
      const w = src.width, h = src.height;
      const out = new ImageData(w, h);
      const sd = src.data, od = out.data;
      const kx = [-1, 0, 1, -2, 0, 2, -1, 0, 1];
      const ky = [-1, -2, -1, 0, 0, 0, 1, 2, 1];

      for (let y = 0; y < h; y++) {
        for (let x = 0; x < w; x++) {
          let gx = 0, gy = 0, ki = 0;
          for (let dy = -1; dy <= 1; dy++) {
            const py = Math.max(0, Math.min(h - 1, y + dy));
            for (let dx = -1; dx <= 1; dx++) {
              const px = Math.max(0, Math.min(w - 1, x + dx));
              const idx = (py * w + px) * 4;
              const luma = 0.2126 * sd[idx] + 0.7152 * sd[idx + 1] + 0.0722 * sd[idx + 2];
              gx += luma * (kx[ki] / 4.0);
              gy += luma * (ky[ki] / 4.0);
              ki++;
            }
          }
          let edge = mode === "horizontal" ? Math.abs(gx) : mode === "vertical" ? Math.abs(gy) : Math.sqrt(gx * gx + gy * gy);
          if (edge < th) edge = 0;
          const val = clamp(edge);
          const oIdx = (y * w + x) * 4;
          od[oIdx] = val; od[oIdx + 1] = val; od[oIdx + 2] = val; od[oIdx + 3] = sd[oIdx + 3];
        }
      }
      return out;
    },

    laplacian_edge: (src, p) => {
      const s = parseFloat(p.strength) || 1.2;
      const kernel = new Float32Array([
        0, 1, 0,
        1, -4, 1,
        0, 1, 0
      ]);
      const w = src.width, h = src.height;
      const out = new ImageData(w, h);
      const sd = src.data, od = out.data;
      for (let y = 0; y < h; y++) {
        for (let x = 0; x < w; x++) {
          let sum = 0, ki = 0;
          for (let dy = -1; dy <= 1; dy++) {
            const py = Math.max(0, Math.min(h - 1, y + dy));
            for (let dx = -1; dx <= 1; dx++) {
              const px = Math.max(0, Math.min(w - 1, x + dx));
              const idx = (py * w + px) * 4;
              const luma = 0.2126 * sd[idx] + 0.7152 * sd[idx + 1] + 0.0722 * sd[idx + 2];
              sum += luma * kernel[ki++];
            }
          }
          const val = clamp(Math.abs(sum) * s);
          const oIdx = (y * w + x) * 4;
          od[oIdx] = val; od[oIdx + 1] = val; od[oIdx + 2] = val; od[oIdx + 3] = sd[oIdx + 3];
        }
      }
      return out;
    },

    // ================= Category 4: Distortion (7) =================
    swirl: (src, p) => {
      const w = src.width, h = src.height;
      const out = new ImageData(w, h);
      const sd = src.data, od = out.data;
      const cx = (w - 1) / 2, cy = (h - 1) / 2;
      const maxR = Math.min(cx, cy) * (parseFloat(p.radius) || 0.8);
      const ang = (parseFloat(p.angle) || 120) * Math.PI / 180.0;

      for (let y = 0; y < h; y++) {
        const dy = y - cy;
        for (let x = 0; x < w; x++) {
          const dx = x - cx;
          const r = Math.sqrt(dx * dx + dy * dy);
          let newTheta = Math.atan2(dy, dx);
          if (r < maxR) {
            const factor = Math.pow(1 - r / maxR, 2);
            newTheta += ang * factor;
          }
          const sx = cx + r * Math.cos(newTheta);
          const sy = cy + r * Math.sin(newTheta);
          const [pr, pg, pb, pa] = sampleBilinear(sd, w, h, sx, sy);
          const oIdx = (y * w + x) * 4;
          od[oIdx] = clamp(pr); od[oIdx + 1] = clamp(pg); od[oIdx + 2] = clamp(pb); od[oIdx + 3] = clamp(pa);
        }
      }
      return out;
    },

    ripple: (src, p) => {
      const w = src.width, h = src.height;
      const out = new ImageData(w, h);
      const sd = src.data, od = out.data;
      const amp = parseFloat(p.amplitude) || 10;
      const freq = parseFloat(p.frequency) || 15;
      const kx = (2 * Math.PI * freq) / Math.max(h, 1);
      const ky = (2 * Math.PI * freq) / Math.max(w, 1);

      for (let y = 0; y < h; y++) {
        for (let x = 0; x < w; x++) {
          const sx = x + amp * Math.sin(y * kx);
          const sy = y + amp * Math.cos(x * ky);
          const [pr, pg, pb, pa] = sampleBilinear(sd, w, h, sx, sy);
          const oIdx = (y * w + x) * 4;
          od[oIdx] = clamp(pr); od[oIdx + 1] = clamp(pg); od[oIdx + 2] = clamp(pb); od[oIdx + 3] = clamp(pa);
        }
      }
      return out;
    },

    fisheye: (src, p) => {
      const w = src.width, h = src.height;
      const out = new ImageData(w, h);
      const sd = src.data, od = out.data;
      const cx = (w - 1) / 2, cy = (h - 1) / 2;
      const maxR = Math.sqrt(cx * cx + cy * cy);
      const s = parseFloat(p.strength) || 0.6;

      for (let y = 0; y < h; y++) {
        const dy = y - cy;
        for (let x = 0; x < w; x++) {
          const dx = x - cx;
          const r = Math.sqrt(dx * dx + dy * dy);
          const theta = Math.atan2(dy, dx);
          const rNorm = r / maxR;
          const rDist = r * (1 + s * rNorm * rNorm) / (1 + Math.max(0, s));
          const sx = cx + rDist * Math.cos(theta);
          const sy = cy + rDist * Math.sin(theta);
          const [pr, pg, pb, pa] = sampleBilinear(sd, w, h, sx, sy);
          const oIdx = (y * w + x) * 4;
          od[oIdx] = clamp(pr); od[oIdx + 1] = clamp(pg); od[oIdx + 2] = clamp(pb); od[oIdx + 3] = clamp(pa);
        }
      }
      return out;
    },

    pinch_bulge: (src, p) => {
      const w = src.width, h = src.height;
      const out = new ImageData(w, h);
      const sd = src.data, od = out.data;
      const cx = (w - 1) / 2, cy = (h - 1) / 2;
      const maxR = Math.min(cx, cy) * (parseFloat(p.radius) || 0.75);
      const s = parseFloat(p.strength) || 0.5;
      const power = s >= 0 ? 1 + s : 1 / (1 - s + 1e-5);

      for (let y = 0; y < h; y++) {
        const dy = y - cy;
        for (let x = 0; x < w; x++) {
          const dx = x - cx;
          const r = Math.sqrt(dx * dx + dy * dy);
          let ratio = 1.0;
          if (r < maxR && r > 1e-4) {
            const rMapped = maxR * Math.pow(r / maxR, power);
            ratio = rMapped / r;
          }
          const sx = cx + dx * ratio;
          const sy = cy + dy * ratio;
          const [pr, pg, pb, pa] = sampleBilinear(sd, w, h, sx, sy);
          const oIdx = (y * w + x) * 4;
          od[oIdx] = clamp(pr); od[oIdx + 1] = clamp(pg); od[oIdx + 2] = clamp(pb); od[oIdx + 3] = clamp(pa);
        }
      }
      return out;
    },

    kaleidoscope: (src, p) => {
      const w = src.width, h = src.height;
      const out = new ImageData(w, h);
      const sd = src.data, od = out.data;
      const cx = (w - 1) / 2, cy = (h - 1) / 2;
      const n = Math.max(2, parseInt(p.segments) || 6);
      const sector = (2 * Math.PI) / n;

      for (let y = 0; y < h; y++) {
        const dy = y - cy;
        for (let x = 0; x < w; x++) {
          const dx = x - cx;
          const r = Math.sqrt(dx * dx + dy * dy);
          let theta = (Math.atan2(dy, dx) + 2 * Math.PI) % (2 * Math.PI);
          const folded = Math.abs((theta % sector) - sector / 2.0);
          const sx = cx + r * Math.cos(folded);
          const sy = cy + r * Math.sin(folded);
          const [pr, pg, pb, pa] = sampleBilinear(sd, w, h, sx, sy);
          const oIdx = (y * w + x) * 4;
          od[oIdx] = clamp(pr); od[oIdx + 1] = clamp(pg); od[oIdx + 2] = clamp(pb); od[oIdx + 3] = clamp(pa);
        }
      }
      return out;
    },

    tiny_planet: (src, p) => {
      const w = src.width, h = src.height;
      const out = new ImageData(w, h);
      const sd = src.data, od = out.data;
      const cx = (w - 1) / 2, cy = (h - 1) / 2;
      const scale = Math.max(0.1, parseFloat(p.scale) || 1.0);
      const maxR = Math.min(cx, cy) / scale;

      for (let y = 0; y < h; y++) {
        const dy = y - cy;
        for (let x = 0; x < w; x++) {
          const dx = x - cx;
          const r = Math.sqrt(dx * dx + dy * dy);
          const theta = Math.atan2(dy, dx);
          const sx = ((theta + Math.PI) / (2 * Math.PI)) * (w - 1);
          const sy = (r / maxR) * (h - 1);
          const [pr, pg, pb, pa] = sampleBilinear(sd, w, h, sx, sy);
          const oIdx = (y * w + x) * 4;
          od[oIdx] = clamp(pr); od[oIdx + 1] = clamp(pg); od[oIdx + 2] = clamp(pb); od[oIdx + 3] = clamp(pa);
        }
      }
      return out;
    },

    mirror_tunnel: (src, p) => {
      const w = src.width, h = src.height;
      const out = new ImageData(w, h);
      const sd = src.data, od = out.data;
      const cx = (w - 1) / 2, cy = (h - 1) / 2;
      const rep = Math.max(1, parseInt(p.repeat) || 3);
      const z = Math.pow(parseFloat(p.zoom) || 1.5, rep);

      for (let y = 0; y < h; y++) {
        const dy = y - cy;
        for (let x = 0; x < w; x++) {
          const dx = x - cx;
          const sx = cx + dx * z;
          const sy = cy + dy * z;
          const [pr, pg, pb, pa] = sampleBilinear(sd, w, h, sx, sy);
          const oIdx = (y * w + x) * 4;
          od[oIdx] = clamp(pr); od[oIdx + 1] = clamp(pg); od[oIdx + 2] = clamp(pb); od[oIdx + 3] = clamp(pa);
        }
      }
      return out;
    },

    // ================= Category 5: Stylize (8) =================
    oil_paint: (src, p) => {
      const rad = Math.max(1, Math.min(4, parseInt(p.radius) || 2));
      const w = src.width, h = src.height;
      const out = new ImageData(w, h);
      const sd = src.data, od = out.data;

      // 4 quadrants around (x, y)
      const offsets = [
        [-rad, 0, -rad, 0], // Q1
        [-rad, 0, 0, rad],  // Q2
        [0, rad, -rad, 0],  // Q3
        [0, rad, 0, rad]    // Q4
      ];

      for (let y = 0; y < h; y++) {
        for (let x = 0; x < w; x++) {
          let minVar = 1e9;
          let bestR = sd[(y * w + x) * 4], bestG = sd[(y * w + x) * 4 + 1], bestB = sd[(y * w + x) * 4 + 2];

          for (let q = 0; q < 4; q++) {
            const [y0, y1, x0, x1] = offsets[q];
            let sumR = 0, sumG = 0, sumB = 0, sumL = 0, sumLSq = 0, count = 0;

            for (let dy = y0; dy <= y1; dy++) {
              const py = Math.max(0, Math.min(h - 1, y + dy));
              for (let dx = x0; dx <= x1; dx++) {
                const px = Math.max(0, Math.min(w - 1, x + dx));
                const idx = (py * w + px) * 4;
                const r = sd[idx], g = sd[idx + 1], b = sd[idx + 2];
                const l = 0.2126 * r + 0.7152 * g + 0.0722 * b;
                sumR += r; sumG += g; sumB += b;
                sumL += l; sumLSq += l * l;
                count++;
              }
            }

            const meanL = sumL / count;
            const variance = sumLSq / count - meanL * meanL;
            if (variance < minVar) {
              minVar = variance;
              bestR = sumR / count;
              bestG = sumG / count;
              bestB = sumB / count;
            }
          }

          const oIdx = (y * w + x) * 4;
          od[oIdx] = clamp(bestR); od[oIdx + 1] = clamp(bestG); od[oIdx + 2] = clamp(bestB); od[oIdx + 3] = sd[oIdx + 3];
        }
      }
      return out;
    },

    pencil_sketch: (src, p) => {
      const c = parseFloat(p.contrast) || 1.5;
      const blend = parseFloat(p.blend) || 0.9;
      const w = src.width, h = src.height;
      const invGray = new ImageData(w, h);
      const sd = src.data, id = invGray.data;

      for (let i = 0; i < sd.length; i += 4) {
        const luma = 0.2126 * sd[i] + 0.7152 * sd[i + 1] + 0.0722 * sd[i + 2];
        const inv = 255 - luma;
        id[i] = inv; id[i + 1] = inv; id[i + 2] = inv; id[i + 3] = 255;
      }

      const { kernel, size } = makeGaussianKernel(4);
      const blurredInv = convolve(invGray, kernel, size, size);
      const bd = blurredInv.data;

      const out = new ImageData(w, h);
      const od = out.data;

      for (let i = 0; i < sd.length; i += 4) {
        const luma = (0.2126 * sd[i] + 0.7152 * sd[i + 1] + 0.0722 * sd[i + 2]) / 255.0;
        const bVal = bd[i] / 255.0;
        const dodge = Math.min(1.0, luma / Math.max(1e-4, 1.0 - bVal));
        const sketch = clamp(((dodge - 0.5) * c + 0.5) * 255.0);
        od[i] = clamp((1 - blend) * sd[i] + blend * sketch);
        od[i + 1] = clamp((1 - blend) * sd[i + 1] + blend * sketch);
        od[i + 2] = clamp((1 - blend) * sd[i + 2] + blend * sketch);
        od[i + 3] = sd[i + 3];
      }
      return out;
    },

    cross_hatch: (src, p) => {
      const d = Math.max(2, parseInt(p.density) || 5);
      const w = src.width, h = src.height;
      const out = new ImageData(w, h);
      const sd = src.data, od = out.data;

      for (let y = 0; y < h; y++) {
        for (let x = 0; x < w; x++) {
          const idx = (y * w + x) * 4;
          const luma = (0.2126 * sd[idx] + 0.7152 * sd[idx + 1] + 0.0722 * sd[idx + 2]) / 255.0;
          let ink = 1.0;

          const h1 = (x + y) % d < 1 ? 0 : 1;
          const h2 = (x - y + 10000 * d) % d < 1 ? 0 : 1;
          const h3 = y % d < 1 ? 0 : 1;
          const h4 = x % d < 1 ? 0 : 1;

          if (luma < 0.2) ink = Math.min(h1, h2, h3, h4);
          else if (luma < 0.4) ink = Math.min(h1, h2, h3);
          else if (luma < 0.6) ink = Math.min(h1, h2);
          else if (luma < 0.8) ink = h1;

          const val = clamp(ink * 255.0);
          od[idx] = val; od[idx + 1] = val; od[idx + 2] = val; od[idx + 3] = sd[idx + 3];
        }
      }
      return out;
    },

    pointillism: (src, p) => {
      const step = Math.max(2, parseInt(p.dot_size) || 6);
      const cov = parseFloat(p.density) || 85.0;
      const rng = createPRNG(p.seed || 42);
      const w = src.width, h = src.height;
      const out = new ImageData(w, h);
      const sd = src.data, od = out.data;

      // canvas base
      for (let i = 0; i < od.length; i += 4) {
        od[i] = 242; od[i + 1] = 240; od[i + 2] = 235; od[i + 3] = 255;
      }

      const rad = Math.max(1, Math.floor(step / 2));
      for (let y = 0; y < h; y += step) {
        for (let x = 0; x < w; x += step) {
          if (rng() * 100 > cov) continue;
          const jx = Math.max(0, Math.min(w - 1, x + Math.floor((rng() - 0.5) * step)));
          const jy = Math.max(0, Math.min(h - 1, y + Math.floor((rng() - 0.5) * step)));
          const sIdx = (jy * w + jx) * 4;
          const r = sd[sIdx], g = sd[sIdx + 1], b = sd[sIdx + 2];

          for (let dy = -rad; dy <= rad; dy++) {
            const py = jy + dy;
            if (py < 0 || py >= h) continue;
            for (let dx = -rad; dx <= rad; dx++) {
              const px = jx + dx;
              if (px < 0 || px >= w) continue;
              if (dx * dx + dy * dy <= rad * rad) {
                const idx = (py * w + px) * 4;
                od[idx] = r; od[idx + 1] = g; od[idx + 2] = b; od[idx + 3] = 255;
              }
            }
          }
        }
      }
      return out;
    },

    mosaic_tiles: (src, p) => {
      const ts = Math.max(2, parseInt(p.tile_size) || 14);
      const gr = Math.max(0, parseInt(p.grout) || 1);
      const w = src.width, h = src.height;
      const out = new ImageData(w, h);
      const sd = src.data, od = out.data;

      for (let y = 0; y < h; y++) {
        const ty = Math.min(h - 1, Math.floor(y / ts) * ts + Math.floor(ts / 2));
        const isGroutY = gr > 0 && y % ts < gr;
        for (let x = 0; x < w; x++) {
          const idx = (y * w + x) * 4;
          if (isGroutY || (gr > 0 && x % ts < gr)) {
            od[idx] = 38; od[idx + 1] = 38; od[idx + 2] = 38; od[idx + 3] = sd[idx + 3];
          } else {
            const tx = Math.min(w - 1, Math.floor(x / ts) * ts + Math.floor(ts / 2));
            const tIdx = (ty * w + tx) * 4;
            od[idx] = sd[tIdx]; od[idx + 1] = sd[tIdx + 1]; od[idx + 2] = sd[tIdx + 2]; od[idx + 3] = sd[idx + 3];
          }
        }
      }
      return out;
    },

    crystallize: (src, p) => {
      const cs = Math.max(4, parseInt(p.cell_size) || 16);
      const rng = createPRNG(p.seed || 42);
      const w = src.width, h = src.height;
      const cols = Math.ceil(w / cs) + 1;
      const rows = Math.ceil(h / cs) + 1;

      const seedsX = new Float32Array(rows * cols);
      const seedsY = new Float32Array(rows * cols);
      for (let i = 0; i < rows; i++) {
        for (let j = 0; j < cols; j++) {
          const idx = i * cols + j;
          seedsX[idx] = Math.max(0, Math.min(w - 1, j * cs + (0.1 + rng() * 0.8) * cs));
          seedsY[idx] = Math.max(0, Math.min(h - 1, i * cs + (0.1 + rng() * 0.8) * cs));
        }
      }

      const out = new ImageData(w, h);
      const sd = src.data, od = out.data;

      for (let y = 0; y < h; y++) {
        const ci = Math.floor(y / cs);
        for (let x = 0; x < w; x++) {
          const cj = Math.floor(x / cs);
          let minDist = 1e9, bestX = x, bestY = y;

          for (let di = -1; di <= 1; di++) {
            const ni = Math.max(0, Math.min(rows - 1, ci + di));
            for (let dj = -1; dj <= 1; dj++) {
              const nj = Math.max(0, Math.min(cols - 1, cj + dj));
              const sIdx = ni * cols + nj;
              const sx = seedsX[sIdx], sy = seedsY[sIdx];
              const d2 = (x - sx) * (x - sx) + (y - sy) * (y - sy);
              if (d2 < minDist) {
                minDist = d2;
                bestX = Math.round(sx);
                bestY = Math.round(sy);
              }
            }
          }

          const bIdx = (bestY * w + bestX) * 4;
          const oIdx = (y * w + x) * 4;
          od[oIdx] = sd[bIdx]; od[oIdx + 1] = sd[bIdx + 1]; od[oIdx + 2] = sd[bIdx + 2]; od[oIdx + 3] = sd[oIdx + 3];
        }
      }
      return out;
    },

    stained_glass: (src, p) => {
      const cs = Math.max(4, parseInt(p.cell_size) || 16);
      const bw = Math.max(1, parseInt(p.border_width) || 2);
      const rng = createPRNG(p.seed || 42);
      const w = src.width, h = src.height;
      const cols = Math.ceil(w / cs) + 1;
      const rows = Math.ceil(h / cs) + 1;

      const seedsX = new Float32Array(rows * cols);
      const seedsY = new Float32Array(rows * cols);
      for (let i = 0; i < rows; i++) {
        for (let j = 0; j < cols; j++) {
          const idx = i * cols + j;
          seedsX[idx] = Math.max(0, Math.min(w - 1, j * cs + (0.1 + rng() * 0.8) * cs));
          seedsY[idx] = Math.max(0, Math.min(h - 1, i * cs + (0.1 + rng() * 0.8) * cs));
        }
      }

      const out = new ImageData(w, h);
      const sd = src.data, od = out.data;

      for (let y = 0; y < h; y++) {
        const ci = Math.floor(y / cs);
        for (let x = 0; x < w; x++) {
          const cj = Math.floor(x / cs);
          let d1 = 1e9, d2 = 1e9, bestX = x, bestY = y;

          for (let di = -1; di <= 1; di++) {
            const ni = Math.max(0, Math.min(rows - 1, ci + di));
            for (let dj = -1; dj <= 1; dj++) {
              const nj = Math.max(0, Math.min(cols - 1, cj + dj));
              const sIdx = ni * cols + nj;
              const sx = seedsX[sIdx], sy = seedsY[sIdx];
              const dist = Math.sqrt((x - sx) * (x - sx) + (y - sy) * (y - sy));
              if (dist < d1) {
                d2 = d1;
                d1 = dist;
                bestX = Math.round(sx);
                bestY = Math.round(sy);
              } else if (dist < d2) {
                d2 = dist;
              }
            }
          }

          const oIdx = (y * w + x) * 4;
          if (d2 - d1 <= bw) {
            od[oIdx] = 20; od[oIdx + 1] = 20; od[oIdx + 2] = 20; od[oIdx + 3] = sd[oIdx + 3];
          } else {
            const bIdx = (bestY * w + bestX) * 4;
            od[oIdx] = sd[bIdx]; od[oIdx + 1] = sd[bIdx + 1]; od[oIdx + 2] = sd[bIdx + 2]; od[oIdx + 3] = sd[oIdx + 3];
          }
        }
      }
      return out;
    },

    linocut: (src, p) => {
      const th = parseFloat(p.threshold) || 0.5;
      const gr = parseFloat(p.grain) || 0.35;
      const w = src.width, h = src.height;
      const out = new ImageData(w, h);
      const sd = src.data, od = out.data;

      for (let y = 0; y < h; y++) {
        for (let x = 0; x < w; x++) {
          const idx = (y * w + x) * 4;
          const luma = (0.2126 * sd[idx] + 0.7152 * sd[idx + 1] + 0.0722 * sd[idx + 2]) / 255.0;
          const line = 0.5 * (Math.sin(y * 0.8 + Math.sin(x * 0.1) * 3.0) + 1.0);
          const mod = luma + gr * (line - 0.5);
          const ink = mod >= th ? 242 : 26;
          od[idx] = ink; od[idx + 1] = ink; od[idx + 2] = ink; od[idx + 3] = sd[idx + 3];
        }
      }
      return out;
    },

    // ================= Category 6: Rare and Unique (10) =================
    thermal_vision: (src, p) => {
      const c = parseFloat(p.contrast) || 1.2;
      const out = cloneImageData(src);
      const d = out.data;

      const palette = [
        [0, 0, 10],      // 0.00
        [51, 0, 128],    // 0.25
        [217, 20, 20],   // 0.50
        [255, 217, 0],   // 0.75
        [255, 255, 255]  // 1.00
      ];

      for (let i = 0; i < d.length; i += 4) {
        let luma = (0.2126 * d[i] + 0.7152 * d[i + 1] + 0.0722 * d[i + 2]) / 255.0;
        luma = Math.max(0, Math.min(1, (luma - 0.5) * c + 0.5));
        const pos = luma * 4.0;
        const idx = Math.min(3, Math.floor(pos));
        const t = pos - idx;
        const c0 = palette[idx], c1 = palette[idx + 1];
        d[i] = clamp(c0[0] + (c1[0] - c0[0]) * t);
        d[i + 1] = clamp(c0[1] + (c1[1] - c0[1]) * t);
        d[i + 2] = clamp(c0[2] + (c1[2] - c0[2]) * t);
      }
      return out;
    },

    cyanotype: (src, p) => {
      const blue = parseFloat(p.blue_intensity) || 1.0;
      const c = parseFloat(p.contrast) || 1.2;
      const gr = parseFloat(p.grain) || 0.25;
      const out = cloneImageData(src);
      const d = out.data;
      const rng = createPRNG(101);

      const paper = [237, 230, 212];
      const prussian = [10, 51, 97];

      for (let i = 0; i < d.length; i += 4) {
        let luma = (0.2126 * d[i] + 0.7152 * d[i + 1] + 0.0722 * d[i + 2]) / 255.0;
        luma = Math.max(0, Math.min(1, (luma - 0.5) * c + 0.5));
        const noise = gr > 0 ? (rng() - 0.5) * gr * 30 : 0;
        d[i] = clamp(paper[0] * luma + prussian[0] * (1 - luma) * blue + noise);
        d[i + 1] = clamp(paper[1] * luma + prussian[1] * (1 - luma) * blue + noise);
        d[i + 2] = clamp(paper[2] * luma + prussian[2] * (1 - luma) * blue + noise);
      }
      return out;
    },

    risograph: (src, p) => {
      const m = parseFloat(p.misregistration) || 3.0;
      const gr = parseFloat(p.grain) || 0.5;
      const rng = createPRNG(p.seed || 42);
      const w = src.width, h = src.height;
      const out = new ImageData(w, h);
      const sd = src.data, od = out.data;
      const dx = Math.round(m), dy = Math.round(m * 0.5);

      const pinkInk = [250, 38, 115];
      const tealInk = [13, 122, 140];
      const paper = [240, 235, 220];

      for (let y = 0; y < h; y++) {
        for (let x = 0; x < w; x++) {
          const idx = (y * w + x) * 4;
          const noise = (rng() - 0.5) * gr * 0.3;
          const pinkDrum = Math.max(0, Math.min(1, (255 - sd[idx + 1]) / 255.0 + noise));

          // Teal shifted
          const sy = Math.max(0, Math.min(h - 1, y - dy));
          const sx = Math.max(0, Math.min(w - 1, x - dx));
          const tIdx = (sy * w + sx) * 4;
          const lumaT = (0.2126 * sd[tIdx] + 0.7152 * sd[tIdx + 1] + 0.0722 * sd[tIdx + 2]) / 255.0;
          const tealDrum = Math.max(0, Math.min(1, (1.0 - lumaT) + noise));

          const pLayerR = 1.0 - (1.0 - pinkInk[0] / 255.0) * pinkDrum;
          const pLayerG = 1.0 - (1.0 - pinkInk[1] / 255.0) * pinkDrum;
          const pLayerB = 1.0 - (1.0 - pinkInk[2] / 255.0) * pinkDrum;

          const tLayerR = 1.0 - (1.0 - tealInk[0] / 255.0) * tealDrum;
          const tLayerG = 1.0 - (1.0 - tealInk[1] / 255.0) * tealDrum;
          const tLayerB = 1.0 - (1.0 - tealInk[2] / 255.0) * tealDrum;

          od[idx] = clamp(paper[0] * pLayerR * tLayerR);
          od[idx + 1] = clamp(paper[1] * pLayerG * tLayerG);
          od[idx + 2] = clamp(paper[2] * pLayerB * tLayerB);
          od[idx + 3] = sd[idx + 3];
        }
      }
      return out;
    },

    newsprint_cmyk: (src, p) => {
      const scale = Math.max(2.0, parseFloat(p.dot_scale) || 4.0);
      const m = parseFloat(p.misregistration) || 2.0;
      const w = src.width, h = src.height;
      const out = new ImageData(w, h);
      const sd = src.data, od = out.data;

      function dotScreen(angleDeg, density, px, py, ox, oy) {
        const rad = (angleDeg * Math.PI) / 180.0;
        const rx = (px + ox) * Math.cos(rad) - (py + oy) * Math.sin(rad);
        const ry = (px + ox) * Math.sin(rad) + (py + oy) * Math.cos(rad);
        const pat = 0.5 * (Math.sin((rx * 2 * Math.PI) / scale) * Math.cos((ry * 2 * Math.PI) / scale) + 1.0);
        return density > pat ? 1 : 0;
      }

      for (let y = 0; y < h; y++) {
        for (let x = 0; x < w; x++) {
          const idx = (y * w + x) * 4;
          const r = sd[idx] / 255.0, g = sd[idx + 1] / 255.0, b = sd[idx + 2] / 255.0;
          const k = 1.0 - Math.max(r, g, b);
          const denom = Math.max(1e-4, 1.0 - k);
          const c = (1.0 - r - k) / denom;
          const mComp = (1.0 - g - k) / denom;
          const yComp = (1.0 - b - k) / denom;

          const dotC = dotScreen(15, c, x, y, m, 0);
          const dotM = dotScreen(75, mComp, x, y, 0, m);
          const dotY = dotScreen(0, yComp, x, y, -m, 0);
          const dotK = dotScreen(45, k, x, y, 0, -m);

          const rOut = (1.0 - dotC) * (1.0 - dotK) * 245;
          const gOut = (1.0 - dotM) * (1.0 - dotK) * 240;
          const bOut = (1.0 - dotY) * (1.0 - dotK) * 225;

          od[idx] = clamp(rOut); od[idx + 1] = clamp(gOut); od[idx + 2] = clamp(bOut); od[idx + 3] = sd[idx + 3];
        }
      }
      return out;
    },

    chromatic_aberration: (src, p) => {
      const shift = parseFloat(p.shift) || 7.0;
      const ang = (parseFloat(p.angle) || 45) * Math.PI / 180.0;
      const w = src.width, h = src.height;
      const out = new ImageData(w, h);
      const sd = src.data, od = out.data;
      const cx = (w - 1) / 2, cy = (h - 1) / 2;
      const maxD = Math.max(cx, cy);

      for (let y = 0; y < h; y++) {
        const dy = y - cy;
        for (let x = 0; x < w; x++) {
          const dx = x - cx;
          const dist = Math.sqrt(dx * dx + dy * dy) / maxD;
          const s = shift * dist;
          const rx = x + s * Math.cos(ang), ry = y + s * Math.sin(ang);
          const bx = x - s * Math.cos(ang), by = y - s * Math.sin(ang);

          const [pr] = sampleBilinear(sd, w, h, rx, ry);
          const [, , pb] = sampleBilinear(sd, w, h, bx, by);
          const idx = (y * w + x) * 4;
          od[idx] = clamp(pr);
          od[idx + 1] = sd[idx + 1]; // Green stays anchored
          od[idx + 2] = clamp(pb);
          od[idx + 3] = sd[idx + 3];
        }
      }
      return out;
    },

    pixel_sort: (src, p) => {
      const tl = (parseFloat(p.threshold_low) || 0.25) * 255.0;
      const th = (parseFloat(p.threshold_high) || 0.8) * 255.0;
      const dir = p.direction || "horizontal";
      const out = cloneImageData(src);
      const d = out.data, w = out.width, h = out.height;

      if (dir === "vertical") {
        for (let x = 0; x < w; x++) {
          let inRun = false, start = 0;
          for (let y = 0; y < h; y++) {
            const idx = (y * w + x) * 4;
            const luma = 0.2126 * d[idx] + 0.7152 * d[idx + 1] + 0.0722 * d[idx + 2];
            const qualifies = luma >= tl && luma <= th;
            if (qualifies && !inRun) { inRun = true; start = y; }
            else if (!qualifies && inRun) {
              inRun = false;
              if (y - start > 1) sortRunCol(start, y);
            }
          }
          if (inRun && h - start > 1) sortRunCol(start, h);

          function sortRunCol(y0, y1) {
            const span = [];
            for (let y = y0; y < y1; y++) {
              const idx = (y * w + x) * 4;
              span.push({
                luma: 0.2126 * d[idx] + 0.7152 * d[idx + 1] + 0.0722 * d[idx + 2],
                r: d[idx], g: d[idx + 1], b: d[idx + 2], a: d[idx + 3]
              });
            }
            span.sort((a, b) => a.luma - b.luma);
            for (let y = y0; y < y1; y++) {
              const idx = (y * w + x) * 4;
              const item = span[y - y0];
              d[idx] = item.r; d[idx + 1] = item.g; d[idx + 2] = item.b; d[idx + 3] = item.a;
            }
          }
        }
      } else {
        for (let y = 0; y < h; y++) {
          let inRun = false, start = 0;
          for (let x = 0; x < w; x++) {
            const idx = (y * w + x) * 4;
            const luma = 0.2126 * d[idx] + 0.7152 * d[idx + 1] + 0.0722 * d[idx + 2];
            const qualifies = luma >= tl && luma <= th;
            if (qualifies && !inRun) { inRun = true; start = x; }
            else if (!qualifies && inRun) {
              inRun = false;
              if (x - start > 1) sortRunRow(start, x);
            }
          }
          if (inRun && w - start > 1) sortRunRow(start, w);

          function sortRunRow(x0, x1) {
            const span = [];
            for (let x = x0; x < x1; x++) {
              const idx = (y * w + x) * 4;
              span.push({
                luma: 0.2126 * d[idx] + 0.7152 * d[idx + 1] + 0.0722 * d[idx + 2],
                r: d[idx], g: d[idx + 1], b: d[idx + 2], a: d[idx + 3]
              });
            }
            span.sort((a, b) => a.luma - b.luma);
            for (let x = x0; x < x1; x++) {
              const idx = (y * w + x) * 4;
              const item = span[x - x0];
              d[idx] = item.r; d[idx + 1] = item.g; d[idx + 2] = item.b; d[idx + 3] = item.a;
            }
          }
        }
      }
      return out;
    },

    datamosh: (src, p) => {
      const bs = Math.max(8, parseInt(p.block_size) || 16);
      const s = parseFloat(p.shift_strength) || 20.0;
      const tp = parseFloat(p.tear_prob) || 0.15;
      const rng = createPRNG(p.seed || 42);
      const out = cloneImageData(src);
      const d = out.data, w = out.width, h = out.height;

      // Corrupt blocks
      for (let y = 0; y < h - bs; y += bs) {
        for (let x = 0; x < w - bs; x += bs) {
          if (rng() < 0.25) {
            const dx = Math.round((rng() - 0.5) * 2 * s);
            const dy = Math.round((rng() - 0.5) * s);
            const sy = Math.max(0, Math.min(h - bs, y + dy));
            const sx = Math.max(0, Math.min(w - bs, x + dx));
            for (let by = 0; by < bs; by++) {
              for (let bx = 0; bx < bs; bx++) {
                const sIdx = ((sy + by) * w + (sx + bx)) * 4;
                const dIdx = ((y + by) * w + (x + bx)) * 4;
                d[dIdx] = d[sIdx]; d[dIdx + 1] = d[sIdx + 1]; d[dIdx + 2] = d[sIdx + 2];
              }
            }
          }
        }
      }

      // Scanline tears
      for (let y = 0; y < h; y++) {
        if (rng() < tp) {
          const shiftX = Math.round((rng() - 0.5) * 3 * s);
          const rowCopy = new Uint8Array(w * 4);
          for (let x = 0; x < w; x++) {
            const idx = (y * w + x) * 4;
            rowCopy[x * 4] = d[idx]; rowCopy[x * 4 + 1] = d[idx + 1]; rowCopy[x * 4 + 2] = d[idx + 2]; rowCopy[x * 4 + 3] = d[idx + 3];
          }
          for (let x = 0; x < w; x++) {
            const sx = (x - shiftX + w * 100) % w;
            const idx = (y * w + x) * 4;
            d[idx] = rowCopy[sx * 4];
            d[idx + 1] = rowCopy[sx * 4 + 1];
            d[idx + 2] = rowCopy[sx * 4 + 2];
          }
        }
      }
      return out;
    },

    vhs_tracking: (src, p) => {
      const tn = parseFloat(p.tracking_noise) || 0.5;
      const cs = Math.round(parseFloat(p.chroma_shift) || 5);
      const rng = createPRNG(p.seed || 42);
      const out = cloneImageData(src);
      const d = out.data, w = out.width, h = out.height;

      // 1. Line jitter
      for (let y = 0; y < h; y++) {
        const jitter = Math.round(Math.sin(y * 0.1) * 2.0 + (rng() - 0.5) * 1.6);
        if (jitter !== 0) {
          const rowCopy = new Uint8Array(w * 4);
          for (let x = 0; x < w; x++) {
            const idx = (y * w + x) * 4;
            rowCopy[x * 4] = d[idx]; rowCopy[x * 4 + 1] = d[idx + 1]; rowCopy[x * 4 + 2] = d[idx + 2]; rowCopy[x * 4 + 3] = d[idx + 3];
          }
          for (let x = 0; x < w; x++) {
            const sx = (x - jitter + w * 100) % w;
            const idx = (y * w + x) * 4;
            d[idx] = rowCopy[sx * 4]; d[idx + 1] = rowCopy[sx * 4 + 1]; d[idx + 2] = rowCopy[sx * 4 + 2];
          }
        }
      }

      // 2. Chroma shift
      if (cs !== 0) {
        for (let y = 0; y < h; y++) {
          const rowR = new Uint8Array(w);
          for (let x = 0; x < w; x++) rowR[x] = d[(y * w + x) * 4];
          for (let x = 0; x < w; x++) {
            const sx = (x - cs + w * 10) % w;
            d[(y * w + x) * 4] = rowR[sx];
          }
        }
      }

      // 3. Bottom tracking static
      const bandH = Math.max(4, Math.floor(h * 0.08));
      const blendStatic = Math.min(1.0, tn * 1.5);
      for (let y = h - bandH; y < h; y++) {
        const tear = Math.round((rng() - 0.5) * 50);
        for (let x = 0; x < w; x++) {
          const idx = (y * w + x) * 4;
          const staticVal = Math.round(rng() * 255);
          d[idx] = clamp((1 - blendStatic) * d[idx] + blendStatic * staticVal);
          d[idx + 1] = clamp((1 - blendStatic) * d[idx + 1] + blendStatic * staticVal);
          d[idx + 2] = clamp((1 - blendStatic) * d[idx + 2] + blendStatic * staticVal);
        }
      }
      return out;
    },

    dither: (src, p) => {
      const method = p.method || "bayer";
      const levels = Math.max(2, parseInt(p.levels) || 4);
      const step = 255.0 / (levels - 1);
      const out = cloneImageData(src);
      const d = out.data, w = out.width, h = out.height;

      if (method === "floyd_steinberg") {
        // Sequential error diffusion
        const buf = new Float32Array(w * h * 3);
        for (let i = 0, bi = 0; i < d.length; i += 4, bi += 3) {
          buf[bi] = d[i]; buf[bi + 1] = d[i + 1]; buf[bi + 2] = d[i + 2];
        }
        for (let y = 0; y < h; y++) {
          for (let x = 0; x < w; x++) {
            const bi = (y * w + x) * 3;
            for (let c = 0; c < 3; c++) {
              const oldVal = buf[bi + c];
              const newVal = clamp(Math.round(oldVal / step) * step);
              buf[bi + c] = newVal;
              const err = oldVal - newVal;
              if (x + 1 < w) buf[(y * w + x + 1) * 3 + c] += err * (7 / 16.0);
              if (y + 1 < h) {
                if (x - 1 >= 0) buf[((y + 1) * w + x - 1) * 3 + c] += err * (3 / 16.0);
                buf[((y + 1) * w + x) * 3 + c] += err * (5 / 16.0);
                if (x + 1 < w) buf[((y + 1) * w + x + 1) * 3 + c] += err * (1 / 16.0);
              }
            }
          }
        }
        for (let i = 0, bi = 0; i < d.length; i += 4, bi += 3) {
          d[i] = clamp(buf[bi]); d[i + 1] = clamp(buf[bi + 1]); d[i + 2] = clamp(buf[bi + 2]);
        }
      } else {
        // Bayer 4x4 matrix
        const bayer4x4 = [
          0, 8, 2, 10,
          12, 4, 14, 6,
          3, 11, 1, 9,
          15, 7, 13, 5
        ];
        for (let y = 0; y < h; y++) {
          for (let x = 0; x < w; x++) {
            const idx = (y * w + x) * 4;
            const bVal = (bayer4x4[(y % 4) * 4 + (x % 4)] / 16.0 - 0.5) * step;
            d[idx] = clamp(Math.round((d[idx] + bVal) / step) * step);
            d[idx + 1] = clamp(Math.round((d[idx + 1] + bVal) / step) * step);
            d[idx + 2] = clamp(Math.round((d[idx + 2] + bVal) / step) * step);
          }
        }
      }
      return out;
    },

    kirlian_aura: (src, p) => {
      const rad = Math.max(1, parseInt(p.glow_radius) || 5);
      const intensity = parseFloat(p.intensity) || 1.6;
      const hue = parseFloat(p.hue) || 185;
      const w = src.width, h = src.height;

      // Sobel edge
      const edgeData = new ImageData(w, h);
      const sd = src.data, ed = edgeData.data;
      const kx = [-1, 0, 1, -2, 0, 2, -1, 0, 1];
      const ky = [-1, -2, -1, 0, 0, 0, 1, 2, 1];

      for (let y = 0; y < h; y++) {
        for (let x = 0; x < w; x++) {
          let gx = 0, gy = 0, ki = 0;
          for (let dy = -1; dy <= 1; dy++) {
            const py = Math.max(0, Math.min(h - 1, y + dy));
            for (let dx = -1; dx <= 1; dx++) {
              const px = Math.max(0, Math.min(w - 1, x + dx));
              const idx = (py * w + px) * 4;
              const luma = 0.2126 * sd[idx] + 0.7152 * sd[idx + 1] + 0.0722 * sd[idx + 2];
              gx += luma * (kx[ki] / 4.0);
              gy += luma * (ky[ki] / 4.0);
              ki++;
            }
          }
          const val = clamp(Math.sqrt(gx * gx + gy * gy));
          const idx = (y * w + x) * 4;
          ed[idx] = val; ed[idx + 1] = val; ed[idx + 2] = val; ed[idx + 3] = 255;
        }
      }

      // Blur edge to make halo bloom
      const { kernel, size } = makeGaussianKernel(rad);
      const bloom = convolve(edgeData, kernel, size, size);
      const bd = bloom.data;

      // Aura tint
      const hRad = (hue * Math.PI) / 180.0;
      const tr = 0.5 + 0.5 * Math.cos(hRad);
      const tg = 0.5 + 0.5 * Math.cos(hRad - 2.094);
      const tb = 0.5 + 0.5 * Math.cos(hRad + 2.094);

      const out = new ImageData(w, h);
      const od = out.data;

      for (let i = 0; i < sd.length; i += 4) {
        const darkR = sd[i] * 0.4;
        const darkG = sd[i + 1] * 0.4;
        const darkB = sd[i + 2] * 0.4;
        const bloomVal = bd[i] * intensity;
        od[i] = clamp(darkR + bloomVal * tr);
        od[i + 1] = clamp(darkG + bloomVal * tg);
        od[i + 2] = clamp(darkB + bloomVal * tb);
        od[i + 3] = sd[i + 3];
      }
      return out;
    }
  };
})();

// Export for node/test environment if available
if (typeof module !== "undefined" && module.exports) {
  module.exports = { JS_FILTERS };
}
