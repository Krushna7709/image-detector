import puppeteer from 'file:///C:/Users/Admin/.qoder/hallmark-tools/node_modules/puppeteer-core/lib/esm/puppeteer/puppeteer-core.js';
const browser = await puppeteer.launch({ executablePath: 'C:/Program Files/Google/Chrome/Application/chrome.exe', headless: 'new', args: ['--no-sandbox', '--disable-gpu'] });
const page = await browser.newPage();
await page.setViewport({ width: 1600, height: 900, deviceScaleFactor: 2 });
await page.goto('http://127.0.0.1:8010/', { waitUntil: 'networkidle0', timeout: 60000 });
await page.waitForSelector('.example-card');
await new Promise(r => setTimeout(r, 600));

await page.evaluate(() => {
  const grid = document.querySelector('.example-card').parentElement;
  grid.scrollIntoView({ block: 'center' });
  window.scrollBy(0, -60);
});
await new Promise(r => setTimeout(r, 500));
await page.screenshot({ path: 'promo/shots/06-samples.png' });

await page.evaluate(() => document.querySelector('.example-card').click());
await page.waitForFunction(() => { const t = document.getElementById('verdictTitle'); return t && t.textContent.trim().length > 2 && !/analys|reading/i.test(t.textContent); }, { timeout: 90000 });
await new Promise(r => setTimeout(r, 1500));
await page.evaluate(() => document.querySelector('.viewer-card').scrollIntoView({ block: 'start' }));
await new Promise(r => setTimeout(r, 800));
await page.screenshot({ path: 'promo/shots/07-viewer.png' });

await page.evaluate(() => document.getElementById('verdictCard').scrollIntoView({ block: 'center' }));
await new Promise(r => setTimeout(r, 600));
await page.screenshot({ path: 'promo/shots/08-verdict-full.png' });
await browser.close();
