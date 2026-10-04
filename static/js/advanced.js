/**
 * PixelLab Advanced Developer Tab ("Inside the Pixel")
 * Technical notebook with step-by-step numeric pixel trace, animated convolution visualizer,
 * side-by-side code with built-in custom syntax highlighter, raw 16x16 ndarray inspector,
 * channel split view, before/after diff heatmap, performance benchmarks, and interactive arithmetic demos.
 */

const PixelAdvanced = (() => {
  let selectedFilterId = "contrast";
  let metadataCache = {};
  let currentTraceCoord = { x: 50, y: 50 };
  let currentDataMode = "image"; // 'image', 'raw', 'channels'

  function init() {
    loadFiltersMetadata();
    setupEventListeners();
    initArithmeticDemos();
  }

  async function loadFiltersMetadata() {
    try {
      const res = await fetch("/api/filters");
      const json = await res.json();
      if (json.status === "ok") {
        metadataCache = json.filters;
        populateFilterSelector();
        updateFilterDetails(selectedFilterId);
      }
    } catch (err) {
      console.error("Failed to load filter metadata:", err);
    }
  }

  function setupEventListeners() {
    const sel = document.getElementById("advanced-filter-select");
    if (sel) {
      sel.addEventListener("change", e => {
        selectedFilterId = e.target.value;
        updateFilterDetails(selectedFilterId);
        runPixelTrace();
      });
    }

    const btnRunTrace = document.getElementById("btn-run-trace");
    if (btnRunTrace) {
      btnRunTrace.addEventListener("click", runPixelTrace);
    }

    const btnAnimateConv = document.getElementById("btn-animate-conv");
    if (btnAnimateConv) {
      btnAnimateConv.addEventListener("click", runConvolutionAnimation);
    }

    const btnBench = document.getElementById("btn-run-benchmark");
    if (btnBench) {
      btnBench.addEventListener("click", runPerformanceBenchmark);
    }

    // Data view toggles
    document.querySelectorAll(".data-view-toggle").forEach(btn => {
      btn.addEventListener("click", () => {
        document.querySelectorAll(".data-view-toggle").forEach(b => b.classList.remove("active"));
        btn.classList.add("active");
        currentDataMode = btn.dataset.mode;
        renderDataView();
      });
    });

    const btnHeatmap = document.getElementById("btn-toggle-heatmap");
    if (btnHeatmap) {
      btnHeatmap.addEventListener("click", renderDiffHeatmap);
    }
  }

  function populateFilterSelector() {
    const sel = document.getElementById("advanced-filter-select");
    if (!sel) return;
    sel.innerHTML = "";
    for (const [fid, meta] of Object.entries(metadataCache)) {
      const opt = document.createElement("option");
      opt.value = fid;
      opt.textContent = `[${meta.category.toUpperCase()}] ${meta.name}`;
      if (fid === selectedFilterId) opt.selected = true;
      sel.appendChild(opt);
    }
  }

  function updateFilterDetails(fid) {
    const meta = metadataCache[fid];
    if (!meta) return;

    // "Why it looks like this" blurb
    const blurbEl = document.getElementById("adv-filter-blurb");
    if (blurbEl) {
      blurbEl.textContent = meta.description;
    }

    // Math explanation
    const mathEl = document.getElementById("adv-filter-math");
    if (mathEl) {
      mathEl.textContent = meta.math || "";
    }

    // Code comparison
    const pyCodeEl = document.getElementById("code-numpy");
    if (pyCodeEl) {
      pyCodeEl.innerHTML = highlightPython(meta.source_code || "# No source available");
    }

    const jsCodeEl = document.getElementById("code-js");
    if (jsCodeEl) {
      const jsFn = JS_FILTERS[fid]?.toString() || "// No JS implementation found";
      jsCodeEl.innerHTML = highlightJS(jsFn);
    }

    // Update Convolution Visualizer if kernel filter
    updateConvolutionGrid(fid);
  }

  // Built-in Lightweight Syntax Highlighter (No external CDNs)
  function highlightPython(code) {
    const tokens = [];
    let text = code;

    // 1. Strings (including triple quotes)
    text = text.replace(/("""[\s\S]*?"""|'''[\s\S]*?'''|"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*')/g, match => {
      const id = `___STR_${tokens.length}___`;
      tokens.push(`<span class="syn-string">${escapeHtml(match)}</span>`);
      return id;
    });

    // 2. Comments
    text = text.replace(/(#.*$)/gm, match => {
      const id = `___COM_${tokens.length}___`;
      tokens.push(`<span class="syn-comment">${escapeHtml(match)}</span>`);
      return id;
    });

    text = escapeHtml(text);

    // 3. Keywords
    const keywords = ["def", "return", "import", "from", "if", "else", "elif", "for", "in", "while", "class", "as", "try", "except"];
    const kwRegex = new RegExp(`\\b(${keywords.join("|")})\\b`, "g");
    text = text.replace(kwRegex, '<span class="syn-keyword">$1</span>');

    // 4. Numbers
    text = text.replace(/\b(\d+(?:\.\d+)?)\b/g, '<span class="syn-number">$1</span>');

    // 5. Builtins
    text = text.replace(/\b(np|numpy|to_float32|to_uint8|convolve2d|convolve_rgb|rgb_to_luminance)\b/g, '<span class="syn-builtin">$1</span>');

    // Restore tokens
    tokens.forEach((tok, i) => {
      text = text.replace(new RegExp(`___(STR|COM)_${i}___`, "g"), tok);
    });

    return text;
  }

  function highlightJS(code) {
    const tokens = [];
    let text = code;

    // Strings
    text = text.replace(/(`[\s\S]*?`|"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*')/g, match => {
      const id = `___STR_${tokens.length}___`;
      tokens.push(`<span class="syn-string">${escapeHtml(match)}</span>`);
      return id;
    });

    // Comments
    text = text.replace(/(\/\/.*$)/gm, match => {
      const id = `___COM_${tokens.length}___`;
      tokens.push(`<span class="syn-comment">${escapeHtml(match)}</span>`);
      return id;
    });

    text = escapeHtml(text);

    const keywords = ["const", "let", "var", "function", "return", "if", "else", "for", "while", "new"];
    const kwRegex = new RegExp(`\\b(${keywords.join("|")})\\b`, "g");
    text = text.replace(kwRegex, '<span class="syn-keyword">$1</span>');
    text = text.replace(/\b(\d+(?:\.\d+)?)\b/g, '<span class="syn-number">$1</span>');
    text = text.replace(/\b(Math|ImageData|clamp|convolve|sampleBilinear)\b/g, '<span class="syn-builtin">$1</span>');

    tokens.forEach((tok, i) => {
      text = text.replace(new RegExp(`___(STR|COM)_${i}___`, "g"), tok);
    });

    return text;
  }

  function escapeHtml(text) {
    return text.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  }

  // Step-by-Step Pixel Trace
  async function runPixelTrace() {
    const cv = PixelLabApp.getMainCanvas();
    if (!cv) return;
    const { x, y } = currentTraceCoord;

    const currentFilterParams = PixelLabApp.getActiveFilterParams(selectedFilterId) || {};

    const traceBox = document.getElementById("trace-steps-box");
    if (!traceBox) return;

    traceBox.innerHTML = "<div class='trace-step-line'>Calculating arithmetic trace...</div>";

    try {
      const res = await fetch("/api/trace", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          image: cv.toDataURL("image/png"),
          x,
          y,
          filter_id: selectedFilterId,
          params: currentFilterParams
        })
      });
      const data = await res.json();
      if (data.status === "ok") {
        let html = `
          <div class="trace-formula">Formula: ${escapeHtml(data.formula)}</div>
          <div style="margin-bottom: 8px;"><strong>Pixel Coordinates:</strong> (${data.x}, ${data.y}) | <strong>Input:</strong> RGBA(${data.input_pixel.join(", ")}) → <strong>Output:</strong> RGBA(${data.output_pixel.join(", ")})</div>
        `;
        data.steps.forEach((step, idx) => {
          html += `<div class="trace-step-line">Step ${idx + 1}: ${escapeHtml(step)}</div>`;
        });
        traceBox.innerHTML = html;
      }
    } catch (err) {
      traceBox.innerHTML = `<div class="trace-step-line" style="color: var(--vermilion);">Trace request error: ${escapeHtml(err.message)}</div>`;
    }
  }

  // Convolution Visualizer Setup
  function updateConvolutionGrid(fid) {
    const convContainer = document.getElementById("convolution-visualizer-area");
    if (!convContainer) return;

    const kernels = {
      gaussian_blur: [
        [0.06, 0.12, 0.06],
        [0.12, 0.25, 0.12],
        [0.06, 0.12, 0.06]
      ],
      sharpen: [
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
      ],
      sobel_edge: [
        [-0.25, 0, 0.25],
        [-0.5, 0, 0.5],
        [-0.25, 0, 0.25]
      ],
      emboss: [
        [-2, -1, 0],
        [-1, 1, 1],
        [0, 1, 2]
      ],
      box_blur: [
        [0.11, 0.11, 0.11],
        [0.11, 0.11, 0.11],
        [0.11, 0.11, 0.11]
      ],
      laplacian_edge: [
        [0, 1, 0],
        [1, -4, 1],
        [0, 1, 0]
      ]
    };

    const k = kernels[fid] || kernels["gaussian_blur"];
    renderKernelMatrix(k);
  }

  function renderKernelMatrix(matrix) {
    const grid = document.getElementById("kernel-matrix-grid");
    if (!grid) return;
    grid.innerHTML = "";
    grid.style.gridTemplateColumns = `repeat(${matrix.length}, 44px)`;

    for (let r = 0; r < matrix.length; r++) {
      for (let c = 0; c < matrix[r].length; c++) {
        const val = matrix[r][c];
        const cell = document.createElement("div");
        cell.className = "matrix-cell";
        if (r === Math.floor(matrix.length / 2) && c === Math.floor(matrix[r].length / 2)) {
          cell.classList.add("center-cell");
        }
        cell.textContent = typeof val === "number" ? val.toFixed(2).replace(/\.00$/, "") : val;
        grid.appendChild(cell);
      }
    }
  }

  function runConvolutionAnimation() {
    const cells = document.querySelectorAll("#kernel-matrix-grid .matrix-cell");
    const resultBox = document.getElementById("conv-result-cell");
    let step = 0;

    cells.forEach(c => c.classList.remove("highlight"));
    if (resultBox) resultBox.textContent = "...";

    const timer = setInterval(() => {
      if (step > 0 && step <= cells.length) {
        cells[step - 1].classList.remove("highlight");
      }
      if (step < cells.length) {
        cells[step].classList.add("highlight");
        step++;
      } else {
        clearInterval(timer);
        if (resultBox) {
          resultBox.textContent = "Sum & Clamped!";
          resultBox.style.backgroundColor = "var(--turmeric)";
          setTimeout(() => { resultBox.style.backgroundColor = ""; }, 1500);
        }
      }
    }, 120);
  }

  // Data View: Raw Array (16x16) & Channel Split
  async function renderDataView() {
    const cv = PixelLabApp.getMainCanvas();
    if (!cv) return;

    const tableArea = document.getElementById("raw-array-view-area");
    const channelsArea = document.getElementById("channels-view-area");
    if (!tableArea || !channelsArea) return;

    if (currentDataMode === "raw") {
      tableArea.style.display = "block";
      channelsArea.style.display = "none";
      await fetchRawArrayTable(cv);
    } else if (currentDataMode === "channels") {
      tableArea.style.display = "none";
      channelsArea.style.display = "flex";
      renderChannelSplit(cv);
    } else {
      tableArea.style.display = "none";
      channelsArea.style.display = "none";
    }
  }

  async function fetchRawArrayTable(canvas) {
    const tableArea = document.getElementById("raw-array-view-area");
    tableArea.innerHTML = "<div>Loading ndarray slice...</div>";
    try {
      const res = await fetch("/api/pixels", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          image: canvas.toDataURL("image/png"),
          region: { x: currentTraceCoord.x, y: currentTraceCoord.y, width: 8, height: 8 }
        })
      });
      const data = await res.json();
      if (data.status === "ok") {
        let html = `
          <div style="font-family: var(--font-mono); font-size: 0.75rem; margin-bottom: 6px;">
            Shape: [${data.shape.join(", ")}] | dtype: ${data.dtype} | Buffer: ${data.memory_mb} MB (${data.total_bytes.toLocaleString()} bytes)
          </div>
          <table class="raw-data-table"><thead><tr><th>y \\ x</th>
        `;
        for (let col = 0; col < data.region.width; col++) {
          html += `<th>${data.region.x + col}</th>`;
        }
        html += `</tr></thead><tbody>`;

        data.pixels.forEach((row, ri) => {
          html += `<tr><th>${data.region.y + ri}</th>`;
          row.forEach(pixel => {
            const hex = "#" + pixel.slice(0, 3).map(v => v.toString(16).padStart(2, "0")).join("");
            html += `<td style="background-color: ${hex}22;" title="RGBA(${pixel.join(", ")})">${pixel[0]},${pixel[1]},${pixel[2]}</td>`;
          });
          html += `</tr>`;
        });
        html += `</tbody></table>`;
        tableArea.innerHTML = html;
      }
    } catch (err) {
      tableArea.innerHTML = `<div style="color: var(--vermilion);">Failed to fetch raw array: ${escapeHtml(err.message)}</div>`;
    }
  }

  function renderChannelSplit(canvas) {
    const channelsArea = document.getElementById("channels-view-area");
    channelsArea.innerHTML = "";
    const ctx = canvas.getContext("2d");
    const imgData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    const d = imgData.data;
    const w = canvas.width, h = canvas.height;

    ["Red", "Green", "Blue"].forEach((name, chIdx) => {
      const colDiv = document.createElement("div");
      colDiv.style.flex = "1";
      colDiv.style.textAlign = "center";

      const title = document.createElement("div");
      title.className = "tape-label";
      title.textContent = `${name} Channel`;
      title.style.marginBottom = "8px";

      const cv = document.createElement("canvas");
      cv.width = w; cv.height = h;
      cv.style.width = "100%";
      cv.style.border = "var(--rule)";
      cv.style.boxShadow = "var(--shadow-flat-sm)";
      cv.style.imageRendering = "pixelated";

      const chCtx = cv.getContext("2d");
      const chImg = chCtx.createImageData(w, h);
      const cd = chImg.data;
      for (let i = 0; i < d.length; i += 4) {
        const val = d[i + chIdx];
        cd[i] = chIdx === 0 ? val : 0;
        cd[i + 1] = chIdx === 1 ? val : 0;
        cd[i + 2] = chIdx === 2 ? val : 0;
        cd[i + 3] = 255;
      }
      chCtx.putImageData(chImg, 0, 0);

      colDiv.appendChild(title);
      colDiv.appendChild(cv);
      channelsArea.appendChild(colDiv);
    });
  }

  // Before/After Pixel Diff Heatmap
  function renderDiffHeatmap() {
    const cv = PixelLabApp.getMainCanvas();
    const origData = PixelLabApp.getOriginalImageData();
    if (!cv || !origData) return;

    const ctx = cv.getContext("2d");
    const curData = ctx.getImageData(0, 0, cv.width, cv.height);
    const d = curData.data, od = origData.data;

    const heatmapCanvas = document.getElementById("diff-heatmap-canvas");
    if (!heatmapCanvas) return;
    heatmapCanvas.width = cv.width;
    heatmapCanvas.height = cv.height;
    const hmCtx = heatmapCanvas.getContext("2d");
    const hmData = hmCtx.createImageData(cv.width, cv.height);
    const hd = hmData.data;

    let maxDiff = 1;
    for (let i = 0; i < d.length; i += 4) {
      const diff = Math.abs(d[i] - od[i]) + Math.abs(d[i + 1] - od[i + 1]) + Math.abs(d[i + 2] - od[i + 2]);
      if (diff > maxDiff) maxDiff = diff;
    }

    for (let i = 0; i < d.length; i += 4) {
      const diff = Math.abs(d[i] - od[i]) + Math.abs(d[i + 1] - od[i + 1]) + Math.abs(d[i + 2] - od[i + 2]);
      const norm = diff / maxDiff; // 0 to 1
      // Heatmap color: Blue (low) -> Green -> Yellow -> Red (high)
      hd[i] = Math.round(norm * 255);
      hd[i + 1] = Math.round((1 - Math.abs(norm - 0.5) * 2) * 255);
      hd[i + 2] = Math.round((1 - norm) * 255);
      hd[i + 3] = 255;
    }
    hmCtx.putImageData(hmData, 0, 0);

    const hmBox = document.getElementById("heatmap-display-box");
    if (hmBox) hmBox.style.display = "block";
  }

  // Performance Benchmarking (JS vs Python)
  async function runPerformanceBenchmark() {
    const cv = PixelLabApp.getMainCanvas();
    if (!cv) return;
    const benchOutput = document.getElementById("benchmark-results-box");
    if (!benchOutput) return;

    benchOutput.innerHTML = "Running benchmarks...";

    // 1. Benchmark JS
    const ctx = cv.getContext("2d");
    const testData = ctx.getImageData(0, 0, cv.width, cv.height);
    const params = PixelLabApp.getActiveFilterParams(selectedFilterId) || {};

    const t0 = performance.now();
    for (let run = 0; run < 3; run++) {
      JS_FILTERS[selectedFilterId](testData, params);
    }
    const jsDuration = (performance.now() - t0) / 3.0;

    // 2. Benchmark Python NumPy via /api/apply
    let pyDuration = 0;
    try {
      const res = await fetch("/api/apply", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          image: cv.toDataURL("image/png"),
          filters: [{ filter_id: selectedFilterId, params }]
        })
      });
      const data = await res.json();
      if (data.status === "ok") {
        pyDuration = data.timing_ms;
      }
    } catch (err) {
      console.warn("Python benchmark failed:", err);
    }

    const totalPixels = cv.width * cv.height;
    const memBytes = totalPixels * 4;

    benchOutput.innerHTML = `
      <div style="font-family: var(--font-mono); font-size: 0.8rem; line-height: 1.6;">
        <div><strong>Canvas Dimensions:</strong> ${cv.width} × ${cv.height} (${(totalPixels / 1e6).toFixed(2)} MP)</div>
        <div><strong>Buffer Memory:</strong> ${(memBytes / 1024 / 1024).toFixed(2)} MB (uint8 RGBA)</div>
        <div style="margin-top: 6px;">⚡ <strong>JavaScript (Canvas ImageData):</strong> <span style="color: var(--verdigris); font-weight: 700;">${jsDuration.toFixed(2)} ms</span></div>
        <div>🐍 <strong>Python / NumPy (Vectorized backend):</strong> <span style="color: var(--turmeric); font-weight: 700;">${pyDuration.toFixed(2)} ms</span></div>
        <div style="color: var(--ink-muted); font-size: 0.72rem; margin-top: 4px;">* JS runs instantly in client thread; NumPy includes round-trip server execution time.</div>
      </div>
    `;
  }

  // Interactive Pixel Arithmetic Educational Demos
  function initArithmeticDemos() {
    // 1. uint8 Overflow Demo
    const demo1Input = document.getElementById("demo-overflow-input");
    const demo1Add = document.getElementById("demo-overflow-add");
    const demo1Res = document.getElementById("demo-overflow-res");
    if (demo1Input && demo1Add && demo1Res) {
      const update1 = () => {
        const val = parseInt(demo1Input.value) || 200;
        const add = parseInt(demo1Add.value) || 80;
        const unmanaged = (val + add) & 255; // 8-bit wrap
        const clamped = Math.min(255, val + add);
        demo1Res.innerHTML = `Raw sum: ${val + add} → 8-bit wrap: <span style="color: var(--vermilion); font-weight: 700;">${unmanaged}</span> vs np.clip: <span style="color: var(--verdigris); font-weight: 700;">${clamped}</span>`;
      };
      demo1Input.addEventListener("input", update1);
      demo1Add.addEventListener("input", update1);
      update1();
    }

    // 2. Gamma vs Linear Light Demo
    const demoGammaVal = document.getElementById("demo-gamma-val");
    const demoGammaRes = document.getElementById("demo-gamma-res");
    if (demoGammaVal && demoGammaRes) {
      demoGammaVal.addEventListener("input", () => {
        const g = parseFloat(demoGammaVal.value) || 2.2;
        const c1 = 0.2, c2 = 0.8;
        const avgsRGB = (c1 + c2) / 2; // 0.5
        const lin1 = Math.pow(c1, g), lin2 = Math.pow(c2, g);
        const avgLin = (lin1 + lin2) / 2;
        const correctedsRGB = Math.pow(avgLin, 1.0 / g);
        demoGammaRes.innerHTML = `Naive blend: <strong>${(avgsRGB * 255).toFixed(0)}</strong> vs Gamma-correct linear blend: <strong>${(correctedsRGB * 255).toFixed(0)}</strong>`;
      });
    }

    // 3. Premultiplied Alpha Demo
    const demoAlpha = document.getElementById("demo-alpha-slider");
    const demoAlphaRes = document.getElementById("demo-alpha-res");
    if (demoAlpha && demoAlphaRes) {
      demoAlpha.addEventListener("input", () => {
        const a = (parseInt(demoAlpha.value) || 50) / 100.0;
        const r = 255;
        const straight = `Straight: RGB(${r}, 0, 0), A=${a.toFixed(2)}`;
        const premult = `Premultiplied: RGB(${Math.round(r * a)}, 0, 0), A=${a.toFixed(2)}`;
        demoAlphaRes.textContent = `${straight} | ${premult}`;
      });
    }
  }

  return {
    init,
    setTraceCoord: (x, y) => { currentTraceCoord = { x, y }; },
    updateFilterDetails
  };
})();
