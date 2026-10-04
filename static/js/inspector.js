/**
 * PixelLab Pixel Inspector & Editor ("Edit Pixels")
 * Interactive loupe (11x11), pixel color readout, direct pixel manipulation,
 * precision brush & eraser, region selections (rect & lasso), color replacement,
 * and 4-channel live histogram.
 */

const PixelInspector = (() => {
  let activeTool = "inspect"; // 'inspect', 'brush', 'eraser', 'select_rect', 'select_lasso', 'eyedropper'
  let selectedCoord = { x: 0, y: 0 };
  let brushSize = 5;
  let brushHardness = 1.0;
  let brushOpacity = 1.0;
  let brushColor = "#E3A41B";
  let replaceTolerance = 30;

  // Selection mask state
  let selectionActive = false;
  let selectionMask = null; // Uint8Array(W * H) where 1 = selected
  let selectionPoints = []; // for lasso
  let rectStart = null;

  // Loupe canvas
  let loupeCanvas = null;
  let loupeCtx = null;
  let histCanvas = null;
  let histCtx = null;

  function init() {
    loupeCanvas = document.getElementById("loupe-canvas");
    if (loupeCanvas) {
      loupeCanvas.width = 11;
      loupeCanvas.height = 11;
      loupeCtx = loupeCanvas.getContext("2d");
      loupeCtx.imageSmoothingEnabled = false;
    }

    histCanvas = document.getElementById("histogram-canvas");
    if (histCanvas) {
      histCanvas.width = 256;
      histCanvas.height = 100;
      histCtx = histCanvas.getContext("2d");
    }

    setupEventListeners();
  }

  function setupEventListeners() {
    // Tool buttons
    document.querySelectorAll(".tool-btn").forEach(btn => {
      btn.addEventListener("click", () => {
        document.querySelectorAll(".tool-btn").forEach(b => b.classList.remove("active"));
        btn.classList.add("active");
        activeTool = btn.dataset.tool;
      });
    });

    // Brush controls
    const bSize = document.getElementById("brush-size");
    if (bSize) bSize.addEventListener("input", e => {
      brushSize = parseInt(e.target.value) || 5;
      const lbl = document.getElementById("brush-size-val");
      if (lbl) lbl.textContent = `${brushSize}px`;
    });

    const bHard = document.getElementById("brush-hardness");
    if (bHard) bHard.addEventListener("input", e => {
      brushHardness = (parseInt(e.target.value) || 100) / 100.0;
      const lbl = document.getElementById("brush-hardness-val");
      if (lbl) lbl.textContent = `${Math.round(brushHardness * 100)}%`;
    });

    const bOpac = document.getElementById("brush-opacity");
    if (bOpac) bOpac.addEventListener("input", e => {
      brushOpacity = (parseInt(e.target.value) || 100) / 100.0;
      const lbl = document.getElementById("brush-opacity-val");
      if (lbl) lbl.textContent = `${Math.round(brushOpacity * 100)}%`;
    });

    const bColor = document.getElementById("brush-color");
    if (bColor) bColor.addEventListener("input", e => {
      brushColor = e.target.value;
    });

    // Single pixel edit sliders
    ["edit-r", "edit-g", "edit-b", "edit-a"].forEach(id => {
      const el = document.getElementById(id);
      if (el) el.addEventListener("input", updateEditPreviewFromSliders);
    });

    const applyPixelBtn = document.getElementById("btn-apply-pixel");
    if (applyPixelBtn) {
      applyPixelBtn.addEventListener("click", applyPixelEdit);
    }

    // Replace color
    const btnReplace = document.getElementById("btn-replace-color");
    if (btnReplace) btnReplace.addEventListener("click", replaceColorUnderCursor);

    // Clear selection
    const btnClearSel = document.getElementById("btn-clear-selection");
    if (btnClearSel) btnClearSel.addEventListener("click", clearSelection);

    // Fill selection
    const btnFillSel = document.getElementById("btn-fill-selection");
    if (btnFillSel) btnFillSel.addEventListener("click", fillSelection);
  }

  // Convert RGB to HEX
  function rgbToHex(r, g, b) {
    return "#" + [r, g, b].map(x => x.toString(16).padStart(2, "0")).join("").toUpperCase();
  }

  // Convert RGB to HSL
  function rgbToHsl(r, g, b) {
    r /= 255; g /= 255; b /= 255;
    const max = Math.max(r, g, b), min = Math.min(r, g, b);
    let h = 0, s = 0, l = (max + min) / 2;
    if (max !== min) {
      const d = max - min;
      s = l > 0.5 ? d / (2 - max - min) : d / (max + min);
      if (max === r) h = (g - b) / d + (g < b ? 6 : 0);
      else if (max === g) h = (b - r) / d + 2;
      else h = (r - g) / d + 4;
      h = Math.round(h * 60);
    }
    return { h, s: Math.round(s * 100), l: Math.round(l * 100) };
  }

  // Update loupe HUD display
  function updateLoupe(imgCtx, imgW, imgH, cx, cy) {
    if (!loupeCtx) return;
    const startX = cx - 5;
    const startY = cy - 5;
    const sub = imgCtx.getImageData(Math.max(0, Math.min(imgW - 11, startX)), Math.max(0, Math.min(imgH - 11, startY)), 11, 11);
    loupeCtx.putImageData(sub, 0, 0);

    const coordsLabel = document.getElementById("loupe-coords");
    if (coordsLabel) {
      coordsLabel.textContent = `(${cx}, ${cy})`;
    }
  }

  // Inspect pixel at (x, y)
  function inspectPixel(x, y, imgCtx, imgW, imgH) {
    if (x < 0 || x >= imgW || y < 0 || y >= imgH) return;
    selectedCoord = { x, y };

    const pix = imgCtx.getImageData(x, y, 1, 1).data;
    const r = pix[0], g = pix[1], b = pix[2], a = pix[3];
    const hex = rgbToHex(r, g, b);
    const hsl = rgbToHsl(r, g, b);
    const luma = Math.round(0.2126 * r + 0.7152 * g + 0.0722 * b);

    // Update readout UI elements
    const set = (id, txt) => { const el = document.getElementById(id); if (el) el.textContent = txt; };
    set("readout-xy", `${x}, ${y}`);
    set("readout-rgba", `${r}, ${g}, ${b}, ${a}`);
    set("readout-hex", hex);
    set("readout-hsl", `${hsl.h}°, ${hsl.s}%, ${hsl.l}%`);
    set("readout-luma", `${luma} / 255`);

    const swatch = document.getElementById("pixel-color-swatch");
    if (swatch) swatch.style.backgroundColor = `rgba(${r}, ${g}, ${b}, ${a / 255})`;

    // Fill direct editor inputs
    const setVal = (id, val) => { const el = document.getElementById(id); if (el) el.value = val; };
    setVal("edit-r", r);
    setVal("edit-g", g);
    setVal("edit-b", b);
    setVal("edit-a", a);
    const colorPicker = document.getElementById("edit-color-picker");
    if (colorPicker) colorPicker.value = hex;

    updateEditPreviewFromSliders();
    updateLoupe(imgCtx, imgW, imgH, x, y);
  }

  function updateEditPreviewFromSliders() {
    const getVal = id => parseInt(document.getElementById(id)?.value) || 0;
    const r = getVal("edit-r"), g = getVal("edit-g"), b = getVal("edit-b"), a = getVal("edit-a");
    const swatch = document.getElementById("edit-preview-swatch");
    if (swatch) swatch.style.backgroundColor = `rgba(${r}, ${g}, ${b}, ${a / 255})`;
  }

  function applyPixelEdit() {
    const getVal = id => parseInt(document.getElementById(id)?.value) || 0;
    const r = getVal("edit-r"), g = getVal("edit-g"), b = getVal("edit-b"), a = getVal("edit-a");

    const cv = PixelLabApp.getMainCanvas();
    if (!cv) return;
    const ctx = cv.getContext("2d");
    const { x, y } = selectedCoord;

    PixelLabApp.recordHistory("Edit Single Pixel");

    const imgData = ctx.createImageData(1, 1);
    imgData.data[0] = r; imgData.data[1] = g; imgData.data[2] = b; imgData.data[3] = a;
    ctx.putImageData(imgData, x, y);

    inspectPixel(x, y, ctx, cv.width, cv.height);
    updateHistogram(ctx, cv.width, cv.height);
  }

  // Paint with pixel brush or eraser
  function paintBrush(cx, cy, isEraser = false) {
    const cv = PixelLabApp.getMainCanvas();
    if (!cv) return;
    const ctx = cv.getContext("2d");
    const origData = PixelLabApp.getOriginalImageData();
    const curData = ctx.getImageData(0, 0, cv.width, cv.height);
    const d = curData.data, od = origData.data;
    const w = cv.width, h = cv.height;

    const rgb = hexToRgb(brushColor);
    const rad = Math.max(1, brushSize);

    for (let dy = -rad; dy <= rad; dy++) {
      const py = cy + dy;
      if (py < 0 || py >= h) continue;
      for (let dx = -rad; dx <= rad; dx++) {
        const px = cx + dx;
        if (px < 0 || px >= w) continue;

        const dist = Math.sqrt(dx * dx + dy * dy);
        if (dist > rad) continue;

        // Selection mask check
        if (selectionActive && selectionMask && !selectionMask[py * w + px]) {
          continue;
        }

        // Hardness falloff
        let falloff = 1.0;
        if (brushHardness < 1.0 && rad > 1) {
          const innerRad = rad * brushHardness;
          if (dist > innerRad) {
            falloff = 1.0 - (dist - innerRad) / (rad - innerRad);
          }
        }
        const effectiveAlpha = brushOpacity * falloff;
        const idx = (py * w + px) * 4;

        if (isEraser) {
          // Restore from original
          d[idx] = Math.round((1 - effectiveAlpha) * d[idx] + effectiveAlpha * od[idx]);
          d[idx + 1] = Math.round((1 - effectiveAlpha) * d[idx + 1] + effectiveAlpha * od[idx + 1]);
          d[idx + 2] = Math.round((1 - effectiveAlpha) * d[idx + 2] + effectiveAlpha * od[idx + 2]);
          d[idx + 3] = Math.round((1 - effectiveAlpha) * d[idx + 3] + effectiveAlpha * od[idx + 3]);
        } else {
          // Stamp brush color
          d[idx] = Math.round((1 - effectiveAlpha) * d[idx] + effectiveAlpha * rgb.r);
          d[idx + 1] = Math.round((1 - effectiveAlpha) * d[idx + 1] + effectiveAlpha * rgb.g);
          d[idx + 2] = Math.round((1 - effectiveAlpha) * d[idx + 2] + effectiveAlpha * rgb.b);
        }
      }
    }
    ctx.putImageData(curData, 0, 0);
  }

  function hexToRgb(hex) {
    let clean = hex.replace("#", "");
    if (clean.length === 3) clean = clean.split("").map(c => c + c).join("");
    const num = parseInt(clean, 16);
    return { r: (num >> 16) & 255, g: (num >> 8) & 255, b: num & 255 };
  }

  // Replace color under cursor
  function replaceColorUnderCursor() {
    const cv = PixelLabApp.getMainCanvas();
    if (!cv) return;
    const ctx = cv.getContext("2d");
    const { x, y } = selectedCoord;
    const targetData = ctx.getImageData(x, y, 1, 1).data;
    const tr = targetData[0], tg = targetData[1], tb = targetData[2];

    const newRgb = hexToRgb(brushColor);
    const tol = parseInt(document.getElementById("replace-tolerance")?.value) || 30;

    PixelLabApp.recordHistory("Replace Color");

    const imgData = ctx.getImageData(0, 0, cv.width, cv.height);
    const d = imgData.data;
    for (let i = 0; i < d.length; i += 4) {
      const dist = Math.sqrt(
        Math.pow(d[i] - tr, 2) + Math.pow(d[i + 1] - tg, 2) + Math.pow(d[i + 2] - tb, 2)
      );
      if (dist <= tol) {
        d[i] = newRgb.r;
        d[i + 1] = newRgb.g;
        d[i + 2] = newRgb.b;
      }
    }
    ctx.putImageData(imgData, 0, 0);
    inspectPixel(x, y, ctx, cv.width, cv.height);
    updateHistogram(ctx, cv.width, cv.height);
  }

  // Clear region selection
  function clearSelection() {
    selectionActive = false;
    selectionMask = null;
    selectionPoints = [];
    rectStart = null;
    PixelLabApp.redrawOverlay();
  }

  // Fill region selection with brush color
  function fillSelection() {
    if (!selectionActive || !selectionMask) return;
    const cv = PixelLabApp.getMainCanvas();
    if (!cv) return;
    const ctx = cv.getContext("2d");
    const imgData = ctx.getImageData(0, 0, cv.width, cv.height);
    const d = imgData.data;
    const rgb = hexToRgb(brushColor);

    PixelLabApp.recordHistory("Fill Selection");

    for (let i = 0; i < selectionMask.length; i++) {
      if (selectionMask[i]) {
        const idx = i * 4;
        d[idx] = rgb.r;
        d[idx + 1] = rgb.g;
        d[idx + 2] = rgb.b;
      }
    }
    ctx.putImageData(imgData, 0, 0);
    updateHistogram(ctx, cv.width, cv.height);
  }

  // Generate 4-channel live histogram
  function updateHistogram(ctx, w, h) {
    if (!histCtx) return;
    const imgData = ctx.getImageData(0, 0, w, h);
    const d = imgData.data;

    const histR = new Uint32Array(256);
    const histG = new Uint32Array(256);
    const histB = new Uint32Array(256);
    const histL = new Uint32Array(256);

    for (let i = 0; i < d.length; i += 4) {
      const r = d[i], g = d[i + 1], b = d[i + 2];
      histR[r]++;
      histG[g]++;
      histB[b]++;
      const luma = Math.round(0.2126 * r + 0.7152 * g + 0.0722 * b);
      histL[luma]++;
    }

    // Find max frequency
    let maxFreq = 1;
    for (let i = 0; i < 256; i++) {
      maxFreq = Math.max(maxFreq, histR[i], histG[i], histB[i], histL[i]);
    }

    histCtx.clearRect(0, 0, 256, 100);

    // Draw channels
    function drawCurve(hist, color) {
      histCtx.strokeStyle = color;
      histCtx.lineWidth = 1.2;
      histCtx.beginPath();
      for (let x = 0; x < 256; x++) {
        const y = 100 - (hist[x] / maxFreq) * 95;
        if (x === 0) histCtx.moveTo(x, y);
        else histCtx.lineTo(x, y);
      }
      histCtx.stroke();
    }

    drawCurve(histR, "#D8442A"); // Red (Vermilion)
    drawCurve(histG, "#2E7D6F"); // Green (Verdigris)
    drawCurve(histB, "#2D68C4"); // Blue
    drawCurve(histL, "#E3A41B"); // Luma (Turmeric)
  }

  return {
    init,
    getActiveTool: () => activeTool,
    inspectPixel,
    updateLoupe,
    paintBrush,
    updateHistogram,
    getSelectionMask: () => (selectionActive ? selectionMask : null),
    setSelectionMask: mask => { selectionMask = mask; selectionActive = !!mask; },
    getSelectedCoord: () => selectedCoord,
  };
})();
