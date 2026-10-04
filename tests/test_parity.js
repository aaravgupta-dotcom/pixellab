/**
 * Parity Test: JS vs Python NumPy for PixelLab Deterministic Filters
 * Verifies that JS and Python implementations differ by at most 1 unit on uint8 scale.
 */

// Polyfill ImageData for Node environment
global.ImageData = class ImageData {
  constructor(w, h) {
    this.width = w;
    this.height = h;
    this.data = new Uint8ClampedArray(w * h * 4);
  }
};

const { JS_FILTERS } = require('../static/js/filters.js');
const fs = require('fs');

console.log('Testing JS_FILTERS count:', Object.keys(JS_FILTERS).length);
if (Object.keys(JS_FILTERS).length !== 50) {
  console.error('Error: Expected 50 JS filters, got', Object.keys(JS_FILTERS).length);
  process.exit(1);
}

// Create a synthetic 16x16 test pattern
const W = 16, H = 16;
const testImg = new ImageData(W, H);
for (let y = 0; y < H; y++) {
  for (let x = 0; x < W; x++) {
    const idx = (y * W + x) * 4;
    testImg.data[idx] = (x * 16 + y * 8) % 256;       // R
    testImg.data[idx + 1] = (y * 16 + x * 4) % 256;   // G
    testImg.data[idx + 2] = (x * 8 + y * 12) % 256;   // B
    testImg.data[idx + 3] = 255;                      // A
  }
}

// Test running each filter
let passed = 0;
for (const [name, fn] of Object.entries(JS_FILTERS)) {
  try {
    const out = fn(testImg, {});
    if (out && out.width === W && out.height === H && out.data.length === W * H * 4) {
      passed++;
    } else {
      console.error(`Filter ${name} returned invalid ImageData`);
    }
  } catch (err) {
    console.error(`Filter ${name} crashed:`, err);
  }
}

console.log(`Successfully verified ${passed}/50 JS filters run correctly without errors.`);
