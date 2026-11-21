import puppeteer from 'file:///C:/Users/Admin/.qoder/hallmark-tools/node_modules/puppeteer-core/lib/esm/puppeteer/puppeteer-core.js';
const browser = await puppeteer.launch({ executablePath: 'C:/Program Files/Google/Chrome/Application/chrome.exe', headless: 'new', args: ['--no-sandbox', '--disable-gpu'] });
const page = await browser.newPage();
await page.setViewport({ width: 1600, height: 900, deviceScaleFactor: 3 });
await page.goto('http://127.0.0.1:8010/', { waitUntil: 'networkidle0', timeout: 60000 });
await page.waitForSelector('.example-card');
await page.evaluate(() => document.querySelector('.example-card').click());
await page.waitForFunction(() => { const t = document.getElementById('verdictTitle'); return t && t.textContent.trim().length > 2 && !/analys|reading/i.test(t.textContent); }, { timeout: 90000 });
await new Promise(r => setTimeout(r, 2000));
await (await page.$('.readout')).screenshot({ path: 'promo/shots/09-readout3x.png' });
await (await page.$('.evidence')).screenshot({ path: 'promo/shots/10-evidence3x.png' });
const rects = await page.evaluate(() => {
  const ev = document.querySelector('.evidence').getBoundingClientRect();
  const viewer = document.querySelector('.viewer-card').getBoundingClientRect();
  const auto = document.querySelector('.attribution, .panel:last-child')?.getBoundingClientRect();
  return { evidence: { w: ev.width, h: ev.height }, viewer: { x: viewer.x - ev.x, y: viewer.y - ev.y, w: viewer.width, h: viewer.height } };
});
console.log(JSON.stringify(rects));
await browser.close();
