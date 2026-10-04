const puppeteer = require('puppeteer-core');
const path = require('path');

async function run() {
  const browser = await puppeteer.launch({
    executablePath: '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
    headless: true,
    defaultViewport: { width: 1440, height: 900 }
  });

  const page = await browser.newPage();
  
  // Navigate to local server
  await page.goto('http://127.0.0.1:5001', { waitUntil: 'networkidle0' });

  // 1. Close onboarding modal
  await page.evaluate(() => {
    const modal = document.getElementById('onboarding-modal');
    if (modal) modal.classList.remove('active');
    localStorage.setItem('pixellab_onboarded', 'true');
  });
  await new Promise(r => setTimeout(r, 400));

  // 2. Apply preset "newspaper" or "thermal"
  await page.select('#preset-select', 'newspaper');
  await new Promise(r => setTimeout(r, 600));

  // Take Studio Screenshot
  await page.screenshot({ path: path.join(__dirname, '../screenshot_studio_filters.png') });
  console.log('Saved screenshot_studio_filters.png');

  // 3. Switch to "Edit Pixels" tab
  await page.evaluate(() => {
    document.querySelector('[data-tab="tab-inspector"]').click();
  });
  await new Promise(r => setTimeout(r, 600));

  // Inspect a pixel by triggering click on main canvas
  await page.evaluate(() => {
    PixelInspector.inspectPixel(200, 200, PixelLabApp.getMainCanvas().getContext('2d'), 400, 400);
  });
  await new Promise(r => setTimeout(r, 400));

  // Take Inspector Screenshot
  await page.screenshot({ path: path.join(__dirname, '../screenshot_inspector.png') });
  console.log('Saved screenshot_inspector.png');

  // 4. Switch to "Inside the Pixel" Developer tab
  await page.evaluate(() => {
    document.querySelector('[data-tab="tab-advanced"]').click();
  });
  await new Promise(r => setTimeout(r, 400));

  // Trigger pixel trace and convolution animation and benchmark
  await page.select('#advanced-filter-select', 'gaussian_blur');
  await new Promise(r => setTimeout(r, 400));
  await page.evaluate(() => {
    document.getElementById('btn-run-trace')?.click();
    document.getElementById('btn-animate-conv')?.click();
    document.getElementById('btn-run-benchmark')?.click();
  });
  await new Promise(r => setTimeout(r, 1200));

  // Take Advanced Tab Screenshot
  await page.screenshot({ path: path.join(__dirname, '../screenshot_advanced.png') });
  console.log('Saved screenshot_advanced.png');

  // 5. Test "Process with Python/NumPy" button
  await page.evaluate(() => {
    window.scrollTo(0, 0);
    document.querySelector('[data-tab="tab-studio"]').click();
  });
  await new Promise(r => setTimeout(r, 400));
  await page.click('#btn-process-numpy');
  await new Promise(r => setTimeout(r, 1000));
  await page.screenshot({ path: path.join(__dirname, '../screenshot_numpy_processed.png') });
  console.log('Saved screenshot_numpy_processed.png');

  await browser.close();
  console.log('Browser automated testing complete!');
}

run().catch(err => {
  console.error('Browser testing failed:', err);
  process.exit(1);
});
