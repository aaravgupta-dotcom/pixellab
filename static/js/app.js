/**
 * PixelLab Application Engine
 * Manages canvas pipeline, filter stack, non-destructive layer rendering,
 * undo/redo history, zoom/pan viewport, split-view slider, presets, and Python parity verification.
 */

const PixelLabApp = (() => {
  // Canvases
  let mainCanvas = null;
  let mainCtx = null;
  let originalImage = null; // Image object
  let originalImageData = null; // downscaled to max 1600px
  let workingImageData = null; // current filtered result

  // Viewport State
  let zoomLevel = 1.0;
  let panX = 0, panY = 0;
  let isPanning = false;
  let panStartX = 0, panStartY = 0;
  let isSpacePressed = false;
  let showPixelGrid = false;
  let showSplitView = false;
  let splitRatio = 0.5; // 0 to 1
  let isDraggingSplit = false;

  // Filter Stack & History
  let filterRegistry = {};
  let filterStack = []; // [ { id: uniqueId, filter_id: 'contrast', enabled: true, params: {} } ]
  let nextFilterUid = 1;
  let historyStack = [];
  let historyIndex = -1;
  const MAX_HISTORY = 30;

  // Render throttle
  let renderScheduled = false;

  function init() {
    mainCanvas = document.getElementById("main-canvas");
    mainCtx = mainCanvas.getContext("2d", { willReadFrequently: true });

    setupNavigation();
    setupCanvasInteractions();
    setupDropzone();
    setupHeaderButtons();
    setupShortcuts();

    // Load filter registry
    fetch("/api/filters")
      .then(res => res.json())
      .then(data => {
        if (data.status === "ok") {
          filterRegistry = data.filters;
          populateCategoryFilters("tone");
        }
      });

    // Check first-run onboarding
    if (!localStorage.getItem("pixellab_onboarded")) {
      showOnboardingModal();
    }

    // Load a default demo test pattern so studio is immediately usable
    generateDefaultPattern();
  }

  // Generate synthetic test image on start
  function generateDefaultPattern() {
    const W = 400, H = 400;
    mainCanvas.width = W;
    mainCanvas.height = H;

    // Draw vibrant darkroom / test card pattern
    const grad = mainCtx.createLinearGradient(0, 0, W, H);
    grad.addColorStop(0, "#D8442A"); // Vermilion
    grad.addColorStop(0.5, "#E3A41B"); // Turmeric
    grad.addColorStop(1, "#2E7D6F"); // Verdigris
    mainCtx.fillStyle = grad;
    mainCtx.fillRect(0, 0, W, H);

    // Decorative geometric shapes & typography
    mainCtx.fillStyle = "#1C1A17";
    mainCtx.beginPath();
    mainCtx.arc(W / 2, H / 2, 90, 0, Math.PI * 2);
    mainCtx.fill();

    mainCtx.fillStyle = "#ECE4D3";
    mainCtx.font = "bold 28px 'Fraunces', serif";
    mainCtx.textAlign = "center";
    mainCtx.fillText("PIXELLAB", W / 2, H / 2 - 10);
    mainCtx.font = "14px 'JetBrains Mono', monospace";
    mainCtx.fillText("STUDIO READY", W / 2, H / 2 + 20);

    originalImageData = mainCtx.getImageData(0, 0, W, H);
    workingImageData = mainCtx.getImageData(0, 0, W, H);

    recordHistory("Initial Test Canvas");
    renderPipeline();
  }

  // Tab Switching
  function setupNavigation() {
    document.querySelectorAll(".tab-btn").forEach(btn => {
      btn.addEventListener("click", () => {
        document.querySelectorAll(".tab-btn").forEach(b => b.classList.remove("active"));
        btn.classList.add("active");

        const targetTab = btn.dataset.tab;
        const sideFilters = document.getElementById("sidebar-filters");
        const sideInspector = document.getElementById("sidebar-inspector");
        const viewMain = document.getElementById("viewport-main");
        const viewAdv = document.getElementById("view-advanced");

        if (targetTab === "tab-studio") {
          if (sideFilters) sideFilters.style.display = "flex";
          if (sideInspector) sideInspector.style.display = "none";
          if (viewMain) viewMain.style.display = "flex";
          if (viewAdv) viewAdv.style.display = "none";
        } else if (targetTab === "tab-inspector") {
          if (sideFilters) sideFilters.style.display = "none";
          if (sideInspector) sideInspector.style.display = "flex";
          if (viewMain) viewMain.style.display = "flex";
          if (viewAdv) viewAdv.style.display = "none";
          PixelInspector.updateHistogram(mainCtx, mainCanvas.width, mainCanvas.height);
        } else if (targetTab === "tab-advanced") {
          if (sideFilters) sideFilters.style.display = "none";
          if (sideInspector) sideInspector.style.display = "none";
          if (viewMain) viewMain.style.display = "none";
          if (viewAdv) viewAdv.style.display = "flex";
        }
      });
    });

    // Category pills in sidebar
    document.querySelectorAll(".cat-pill").forEach(pill => {
      pill.addEventListener("click", () => {
        document.querySelectorAll(".cat-pill").forEach(p => p.classList.remove("active"));
        pill.classList.add("active");
        populateCategoryFilters(pill.dataset.category);
      });
    });
  }

  function populateCategoryFilters(cat) {
    const picker = document.getElementById("filter-add-select");
    if (!picker) return;
    picker.innerHTML = '<option value="">+ Add Filter from Library...</option>';

    for (const [fid, meta] of Object.entries(filterRegistry)) {
      if (cat === "all" || meta.category === cat) {
        const opt = document.createElement("option");
        opt.value = fid;
        opt.textContent = `${meta.name} — ${meta.description}`;
        picker.appendChild(opt);
      }
    }

    picker.onchange = () => {
      if (picker.value) {
        addFilterToStack(picker.value);
        picker.value = "";
      }
    };
  }

  // Setup file upload & drag/drop
  function setupDropzone() {
    const fileInput = document.getElementById("file-upload-input");
    const dropzone = document.getElementById("dropzone-overlay");

    if (fileInput) {
      fileInput.addEventListener("change", e => {
        if (e.target.files && e.target.files[0]) {
          handleFile(e.target.files[0]);
        }
      });
    }

    const dropTarget = document.getElementById("canvas-viewport-drop");
    if (dropTarget) {
      ["dragenter", "dragover"].forEach(evt => {
        dropTarget.addEventListener(evt, e => {
          e.preventDefault();
          if (dropzone) dropzone.classList.add("dragover");
        });
      });

      ["dragleave", "drop"].forEach(evt => {
        dropTarget.addEventListener(evt, e => {
          e.preventDefault();
          if (dropzone) dropzone.classList.remove("dragover");
        });
      });

      dropTarget.addEventListener("drop", e => {
        if (e.dataTransfer && e.dataTransfer.files[0]) {
          handleFile(e.dataTransfer.files[0]);
        }
      });
    }

    if (dropzone) {
      dropzone.addEventListener("click", () => {
        if (fileInput) fileInput.click();
      });
    }
  }

  function handleFile(file) {
    if (!file.type.startsWith("image/")) {
      alert("Please choose a valid image file (JPG, PNG, WEBP).");
      return;
    }

    const reader = new FileReader();
    reader.onload = e => {
      const img = new Image();
      img.onload = () => {
        loadNewImage(img);
        const dropzone = document.getElementById("dropzone-overlay");
        if (dropzone) dropzone.style.display = "none";
      };
      img.src = e.target.result;
    };
    reader.readAsDataURL(file);
  }

  function loadNewImage(img) {
    originalImage = img;
    let w = img.width;
    let h = img.height;

    // Downscale working copy if long side exceeds 1600px (keep original untouched)
    const maxSide = Math.max(w, h);
    if (maxSide > 1600) {
      const scale = 1600.0 / maxSide;
      w = Math.round(w * scale);
      h = Math.round(h * scale);
    }

    mainCanvas.width = w;
    mainCanvas.height = h;
    mainCtx.drawImage(img, 0, 0, w, h);

    originalImageData = mainCtx.getImageData(0, 0, w, h);
    workingImageData = mainCtx.getImageData(0, 0, w, h);

    zoomLevel = 1.0;
    panX = 0; panY = 0;
    updateCanvasTransform();

    recordHistory("Upload Image");
    renderPipeline();
  }

  // Setup Canvas Mouse & Keyboard Interactions
  function setupCanvasInteractions() {
    const scroller = document.getElementById("canvas-scroller");
    const wrapper = document.getElementById("canvas-wrapper");

    // Wheel Zoom
    scroller.addEventListener("wheel", e => {
      e.preventDefault();
      const zoomFactor = e.deltaY < 0 ? 1.15 : 0.87;
      setZoom(zoomLevel * zoomFactor);
    }, { passive: false });

    // Panning & Tool Dragging
    let isMouseDown = false;

    window.addEventListener("keydown", e => {
      if (e.code === "Space" && !isSpacePressed && e.target.tagName !== "INPUT") {
        isSpacePressed = true;
        scroller.classList.add("panning");
      }
    });

    window.addEventListener("keyup", e => {
      if (e.code === "Space") {
        isSpacePressed = false;
        scroller.classList.remove("panning");
      }
    });

    wrapper.addEventListener("mousedown", e => {
      if (e.button === 1 || (e.button === 0 && isSpacePressed)) {
        isPanning = true;
        panStartX = e.clientX - panX;
        panStartY = e.clientY - panY;
        return;
      }

      if (e.button === 0) {
        isMouseDown = true;
        const pt = getCanvasCoordinates(e);
        handleCanvasPointer(pt, true);
      }
    });

    window.addEventListener("mousemove", e => {
      if (isPanning) {
        panX = e.clientX - panStartX;
        panY = e.clientY - panStartY;
        updateCanvasTransform();
        return;
      }

      if (isDraggingSplit) {
        const rect = wrapper.getBoundingClientRect();
        const relX = (e.clientX - rect.left) / rect.width;
        splitRatio = Math.max(0.05, Math.min(0.95, relX));
        redrawSplitDivider();
        return;
      }

      const pt = getCanvasCoordinates(e);
      if (pt.x >= 0 && pt.x < mainCanvas.width && pt.y >= 0 && pt.y < mainCanvas.height) {
        PixelInspector.updateLoupe(mainCtx, mainCanvas.width, mainCanvas.height, pt.x, pt.y);
      }

      if (isMouseDown) {
        handleCanvasPointer(pt, false);
      }
    });

    window.addEventListener("mouseup", () => {
      if (isMouseDown) {
        isMouseDown = false;
        if (PixelInspector.getActiveTool() === "brush" || PixelInspector.getActiveTool() === "eraser") {
          recordHistory("Brush Stroke");
        }
      }
      isPanning = false;
      isDraggingSplit = false;
    });

    // Split Divider Dragging
    const divider = document.getElementById("split-divider");
    if (divider) {
      divider.addEventListener("mousedown", e => {
        e.stopPropagation();
        isDraggingSplit = true;
      });
    }
  }

  function getCanvasCoordinates(e) {
    const rect = mainCanvas.getBoundingClientRect();
    const scaleX = mainCanvas.width / rect.width;
    const scaleY = mainCanvas.height / rect.height;
    const x = Math.floor((e.clientX - rect.left) * scaleX);
    const y = Math.floor((e.clientY - rect.top) * scaleY);
    return { x, y };
  }

  function handleCanvasPointer(pt, isInitialClick) {
    const tool = PixelInspector.getActiveTool();

    if (tool === "inspect") {
      PixelInspector.inspectPixel(pt.x, pt.y, mainCtx, mainCanvas.width, mainCanvas.height);
      PixelAdvanced.setTraceCoord(pt.x, pt.y);
    } else if (tool === "brush") {
      PixelInspector.paintBrush(pt.x, pt.y, false);
    } else if (tool === "eraser") {
      PixelInspector.paintBrush(pt.x, pt.y, true);
    } else if (tool === "eyedropper" && isInitialClick) {
      const pix = mainCtx.getImageData(pt.x, pt.y, 1, 1).data;
      const hex = "#" + [pix[0], pix[1], pix[2]].map(c => c.toString(16).padStart(2, "0")).join("");
      const bColor = document.getElementById("brush-color");
      if (bColor) bColor.value = hex;
      PixelInspector.inspectPixel(pt.x, pt.y, mainCtx, mainCanvas.width, mainCanvas.height);
    }
  }

  function setZoom(val) {
    zoomLevel = Math.max(0.1, Math.min(20.0, val));
    const lbl = document.getElementById("zoom-label");
    if (lbl) lbl.textContent = `${Math.round(zoomLevel * 100)}%`;

    // Toggle pixel grid when zoomed in >= 800%
    const wrapper = document.getElementById("canvas-wrapper");
    if (wrapper) {
      if (showPixelGrid && zoomLevel >= 8.0) {
        wrapper.classList.add("pixel-grid-active");
      } else {
        wrapper.classList.remove("pixel-grid-active");
      }
    }
    updateCanvasTransform();
  }

  function updateCanvasTransform() {
    const wrapper = document.getElementById("canvas-wrapper");
    if (wrapper) {
      wrapper.style.transform = `translate(${panX}px, ${panY}px) scale(${zoomLevel})`;
    }
  }

  // Filter Stack Management
  function addFilterToStack(fid, initialParams = null) {
    const meta = filterRegistry[fid];
    if (!meta) return;

    const params = {};
    meta.params.forEach(p => {
      params[p.id] = initialParams && initialParams[p.id] !== undefined ? initialParams[p.id] : p.default;
    });

    const item = {
      id: nextFilterUid++,
      filter_id: fid,
      enabled: true,
      params
    };

    filterStack.push(item);
    recordHistory(`Add ${meta.name}`);
    renderFilterStackUI();
    renderPipeline();
  }

  function renderFilterStackUI() {
    const list = document.getElementById("filter-stack-list");
    if (!list) return;
    list.innerHTML = "";

    if (filterStack.length === 0) {
      list.innerHTML = `
        <div style="text-align: center; padding: 32px 16px; color: var(--ink-muted); font-size: 0.85rem;">
          <div style="font-family: var(--font-display); font-size: 1.1rem; margin-bottom: 4px;">Empty Filter Stack</div>
          Choose a filter from the library above or try a preset.
        </div>
      `;
      return;
    }

    filterStack.forEach((item, index) => {
      const meta = filterRegistry[item.filter_id];
      if (!meta) return;

      const card = document.createElement("div");
      card.className = `filter-card ${item.enabled ? "" : "disabled"}`;

      // Card Header
      let actionsHtml = `
        <button class="icon-btn" title="Toggle On/Off" data-act="toggle">${item.enabled ? "●" : "○"}</button>
        <button class="icon-btn" title="Move Up" data-act="up" ${index === 0 ? "disabled" : ""}>▲</button>
        <button class="icon-btn" title="Move Down" data-act="down" ${index === filterStack.length - 1 ? "disabled" : ""}>▼</button>
        <button class="icon-btn" title="Reset Default" data-act="reset">↺</button>
        <button class="icon-btn danger" title="Delete" data-act="delete">✕</button>
      `;

      // Parameters sliders
      let paramsHtml = "";
      meta.params.forEach(p => {
        const val = item.params[p.id];
        if (p.type === "range") {
          paramsHtml += `
            <div class="param-group">
              <div class="param-header">
                <span class="param-name" title="${p.tooltip || ''}">${p.name}</span>
                <span class="param-value">${val}</span>
              </div>
              <div class="slider-container">
                <input type="range" class="mixing-slider" min="${p.min}" max="${p.max}" step="${p.step}" value="${val}" data-pid="${p.id}">
              </div>
            </div>
          `;
        } else if (p.type === "select") {
          paramsHtml += `
            <div class="param-group">
              <div class="param-header">
                <span class="param-name">${p.name}</span>
              </div>
              <select class="select-input" data-pid="${p.id}" style="width: 100%;">
                ${p.options.map(opt => `<option value="${opt}" ${opt === val ? "selected" : ""}>${opt}</option>`).join("")}
              </select>
            </div>
          `;
        } else if (p.type === "seed") {
          paramsHtml += `
            <div class="param-group" style="display: flex; justify-content: space-between; align-items: center;">
              <span class="param-name">Seed: ${val}</span>
              <button class="btn btn-sm" data-act="reroll" data-pid="${p.id}">🎲 Re-roll</button>
            </div>
          `;
        }
      });

      card.innerHTML = `
        <div class="filter-card-header">
          <div class="filter-title">
            <span class="tape-label ${meta.category === 'rare' ? 'verdigris' : ''}">${meta.category}</span>
            <span>${meta.name}</span>
          </div>
          <div class="filter-card-actions">${actionsHtml}</div>
        </div>
        <div class="filter-desc">${meta.description}</div>
        <div class="filter-params">${paramsHtml}</div>
      `;

      // Event handlers for actions
      card.querySelectorAll("[data-act]").forEach(btn => {
        btn.addEventListener("click", () => {
          const act = btn.dataset.act;
          if (act === "toggle") {
            item.enabled = !item.enabled;
            renderFilterStackUI();
            renderPipeline();
          } else if (act === "up" && index > 0) {
            const tmp = filterStack[index - 1];
            filterStack[index - 1] = filterStack[index];
            filterStack[index] = tmp;
            renderFilterStackUI();
            renderPipeline();
          } else if (act === "down" && index < filterStack.length - 1) {
            const tmp = filterStack[index + 1];
            filterStack[index + 1] = filterStack[index];
            filterStack[index] = tmp;
            renderFilterStackUI();
            renderPipeline();
          } else if (act === "delete") {
            filterStack.splice(index, 1);
            renderFilterStackUI();
            renderPipeline();
          } else if (act === "reset") {
            meta.params.forEach(p => { item.params[p.id] = p.default; });
            renderFilterStackUI();
            renderPipeline();
          } else if (act === "reroll") {
            item.params["seed"] = Math.floor(Math.random() * 99999);
            renderFilterStackUI();
            renderPipeline();
          }
        });
      });

      // Event handlers for inputs
      card.querySelectorAll("input, select").forEach(inp => {
        inp.addEventListener("input", e => {
          const pid = inp.dataset.pid;
          item.params[pid] = inp.type === "range" ? parseFloat(inp.value) : inp.value;
          const valLabel = inp.closest(".param-group")?.querySelector(".param-value");
          if (valLabel) valLabel.textContent = item.params[pid];
          scheduleRenderPipeline();
        });
      });

      list.appendChild(card);
    });
  }

  // Render Pipeline: Apply filter stack sequentially using JS twins
  function scheduleRenderPipeline() {
    if (renderScheduled) return;
    renderScheduled = true;
    requestAnimationFrame(() => {
      renderPipeline();
      renderScheduled = false;
    });
  }

  function renderPipeline() {
    if (!originalImageData) return;

    let current = cloneImageData(originalImageData);

    for (const item of filterStack) {
      if (!item.enabled) continue;
      const jsFn = JS_FILTERS[item.filter_id];
      if (typeof jsFn === "function") {
        current = jsFn(current, item.params);
      }
    }

    workingImageData = current;

    if (showSplitView) {
      renderSplitView();
    } else {
      mainCtx.putImageData(current, 0, 0);
    }

    PixelInspector.updateHistogram(mainCtx, mainCanvas.width, mainCanvas.height);
  }

  // Before / After Split Slider
  function renderSplitView() {
    if (!originalImageData || !workingImageData) return;
    const w = mainCanvas.width, h = mainCanvas.height;
    const splitX = Math.floor(w * splitRatio);

    mainCtx.putImageData(workingImageData, 0, 0);

    // Draw original image on left half
    const tempCanvas = document.createElement("canvas");
    tempCanvas.width = w; tempCanvas.height = h;
    tempCanvas.getContext("2d").putImageData(originalImageData, 0, 0);

    mainCtx.save();
    mainCtx.beginPath();
    mainCtx.rect(0, 0, splitX, h);
    mainCtx.clip();
    mainCtx.drawImage(tempCanvas, 0, 0);
    mainCtx.restore();

    redrawSplitDivider();
  }

  function redrawSplitDivider() {
    const divider = document.getElementById("split-divider");
    if (!divider) return;
    if (!showSplitView) {
      divider.style.display = "none";
      return;
    }
    divider.style.display = "block";
    divider.style.left = `${splitRatio * 100}%`;
  }

  // Undo / Redo History
  function recordHistory(actionName) {
    historyStack = historyStack.slice(0, historyIndex + 1);
    historyStack.push({
      name: actionName,
      filterStack: JSON.parse(JSON.stringify(filterStack)),
      imageData: mainCtx.getImageData(0, 0, mainCanvas.width, mainCanvas.height)
    });
    if (historyStack.length > MAX_HISTORY) {
      historyStack.shift();
    }
    historyIndex = historyStack.length - 1;
    updateHistoryButtons();
  }

  function undo() {
    if (historyIndex > 0) {
      historyIndex--;
      restoreHistoryState(historyStack[historyIndex]);
    }
  }

  function redo() {
    if (historyIndex < historyStack.length - 1) {
      historyIndex++;
      restoreHistoryState(historyStack[historyIndex]);
    }
  }

  function restoreHistoryState(state) {
    filterStack = JSON.parse(JSON.stringify(state.filterStack));
    mainCtx.putImageData(state.imageData, 0, 0);
    renderFilterStackUI();
    updateHistoryButtons();
  }

  function updateHistoryButtons() {
    const btnUndo = document.getElementById("btn-undo");
    const btnRedo = document.getElementById("btn-redo");
    if (btnUndo) btnUndo.disabled = historyIndex <= 0;
    if (btnRedo) btnRedo.disabled = historyIndex >= historyStack.length - 1;
  }

  // Python Parity: Process with Python/NumPy Backend
  async function processWithPython() {
    const statusToast = document.getElementById("status-notification");
    if (statusToast) {
      statusToast.textContent = "Sending to Python/NumPy backend...";
      statusToast.style.display = "block";
    }

    try {
      const payload = {
        image: mainCanvas.toDataURL("image/png"),
        filters: filterStack.filter(f => f.enabled).map(f => ({
          filter_id: f.filter_id,
          params: f.params
        }))
      };

      const res = await fetch("/api/apply", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });
      const data = await res.json();
      if (data.status === "ok") {
        const pyImg = new Image();
        pyImg.onload = () => {
          mainCtx.drawImage(pyImg, 0, 0);
          if (statusToast) {
            statusToast.innerHTML = `✓ Processed with Python/NumPy in <strong>${data.timing_ms} ms</strong>! Output matches shape [${data.shape.join(", ")}].`;
            setTimeout(() => { statusToast.style.display = "none"; }, 4000);
          }
          recordHistory("Process with NumPy");
        };
        pyImg.src = data.image;
      } else {
        alert("Python Error: " + data.error);
        if (statusToast) statusToast.style.display = "none";
      }
    } catch (err) {
      alert("Failed to communicate with Python backend: " + err.message);
      if (statusToast) statusToast.style.display = "none";
    }
  }

  // Header & Presets
  function setupHeaderButtons() {
    // Process with Python
    const btnPy = document.getElementById("btn-process-numpy");
    if (btnPy) btnPy.addEventListener("click", processWithPython);

    // Undo / Redo
    const btnUndo = document.getElementById("btn-undo");
    if (btnUndo) btnUndo.addEventListener("click", undo);
    const btnRedo = document.getElementById("btn-redo");
    if (btnRedo) btnRedo.addEventListener("click", redo);

    // Zoom controls
    const btnZoomIn = document.getElementById("btn-zoom-in");
    if (btnZoomIn) btnZoomIn.addEventListener("click", () => setZoom(zoomLevel * 1.25));
    const btnZoomOut = document.getElementById("btn-zoom-out");
    if (btnZoomOut) btnZoomOut.addEventListener("click", () => setZoom(zoomLevel * 0.8));
    const btnZoom100 = document.getElementById("btn-zoom-100");
    if (btnZoom100) btnZoom100.addEventListener("click", () => { panX = 0; panY = 0; setZoom(1.0); });

    // Pixel grid toggle
    const btnGrid = document.getElementById("btn-toggle-grid");
    if (btnGrid) {
      btnGrid.addEventListener("click", () => {
        showPixelGrid = !showPixelGrid;
        btnGrid.classList.toggle("active", showPixelGrid);
        setZoom(zoomLevel);
      });
    }

    // Split view toggle
    const btnSplit = document.getElementById("btn-toggle-split");
    if (btnSplit) {
      btnSplit.addEventListener("click", () => {
        showSplitView = !showSplitView;
        btnSplit.classList.toggle("active", showSplitView);
        redrawSplitDivider();
        renderPipeline();
      });
    }

    // Export PNG / JPG
    const btnExportPng = document.getElementById("btn-export-png");
    if (btnExportPng) btnExportPng.addEventListener("click", () => exportImage("image/png", "pixellab-export.png"));
    const btnExportJpg = document.getElementById("btn-export-jpg");
    if (btnExportJpg) btnExportJpg.addEventListener("click", () => exportImage("image/jpeg", "pixellab-export.jpg"));

    // Theme toggle
    const btnTheme = document.getElementById("btn-toggle-theme");
    if (btnTheme) {
      btnTheme.addEventListener("click", () => {
        document.body.classList.toggle("dark-mode");
        btnTheme.textContent = document.body.classList.contains("dark-mode") ? "☀️ Light" : "🌙 Dark";
      });
    }

    // Surprise Me random filter combination
    const btnSurprise = document.getElementById("btn-surprise-me");
    if (btnSurprise) btnSurprise.addEventListener("click", applySurpriseCombination);

    // Presets dropdown
    const presetSelect = document.getElementById("preset-select");
    if (presetSelect) {
      presetSelect.addEventListener("change", e => {
        if (e.target.value) {
          applyPreset(e.target.value);
          e.target.value = "";
        }
      });
    }

    // Onboarding help modal
    const btnHelp = document.getElementById("btn-help");
    if (btnHelp) btnHelp.addEventListener("click", showOnboardingModal);
    const btnCloseModal = document.getElementById("btn-close-modal");
    if (btnCloseModal) btnCloseModal.addEventListener("click", hideOnboardingModal);
  }

  function applyPreset(name) {
    filterStack = [];
    if (name === "newspaper") {
      filterStack.push({ id: nextFilterUid++, filter_id: "grayscale", enabled: true, params: { mode: "rec709" } });
      filterStack.push({ id: nextFilterUid++, filter_id: "contrast", enabled: true, params: { factor: 1.5 } });
      filterStack.push({ id: nextFilterUid++, filter_id: "newsprint_cmyk", enabled: true, params: { dot_scale: 4.0, misregistration: 2.0 } });
    } else if (name === "thermal") {
      filterStack.push({ id: nextFilterUid++, filter_id: "thermal_vision", enabled: true, params: { contrast: 1.3 } });
    } else if (name === "vcr") {
      filterStack.push({ id: nextFilterUid++, filter_id: "vhs_tracking", enabled: true, params: { tracking_noise: 0.6, chroma_shift: 6, seed: 42 } });
    } else if (name === "blueprint") {
      filterStack.push({ id: nextFilterUid++, filter_id: "cyanotype", enabled: true, params: { blue_intensity: 1.1, contrast: 1.3, grain: 0.3 } });
    }
    renderFilterStackUI();
    renderPipeline();
    recordHistory(`Apply Preset: ${name}`);
  }

  function applySurpriseCombination() {
    filterStack = [];
    const keys = Object.keys(filterRegistry);
    const count = 2 + Math.floor(Math.random() * 2);
    for (let i = 0; i < count; i++) {
      const fid = keys[Math.floor(Math.random() * keys.length)];
      addFilterToStack(fid);
    }
  }

  function exportImage(mimeType, filename) {
    const link = document.createElement("a");
    link.download = filename;
    link.href = mainCanvas.toDataURL(mimeType, 0.95);
    link.click();
  }

  function setupShortcuts() {
    window.addEventListener("keydown", e => {
      if ((e.ctrlKey || e.metaKey) && e.key === "z") {
        if (e.shiftKey) {
          e.preventDefault();
          redo();
        } else {
          e.preventDefault();
          undo();
        }
      }
    });
  }

  function showOnboardingModal() {
    const modal = document.getElementById("onboarding-modal");
    if (modal) modal.classList.add("active");
  }

  function hideOnboardingModal() {
    const modal = document.getElementById("onboarding-modal");
    if (modal) modal.classList.remove("active");
    localStorage.setItem("pixellab_onboarded", "true");
  }

  function cloneImageData(src) {
    const copy = new ImageData(src.width, src.height);
    copy.data.set(src.data);
    return copy;
  }

  return {
    init,
    getMainCanvas: () => mainCanvas,
    getOriginalImageData: () => originalImageData,
    recordHistory,
    renderPipeline,
    redrawOverlay: () => renderPipeline(),
    getActiveFilterParams: fid => {
      const item = filterStack.find(f => f.filter_id === fid);
      return item ? item.params : null;
    }
  };
})();

// Initialize on DOM load
document.addEventListener("DOMContentLoaded", () => {
  PixelLabApp.init();
  PixelInspector.init();
  PixelAdvanced.init();
});
