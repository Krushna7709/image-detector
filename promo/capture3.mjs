import puppeteer from 'file:///C:/Users/Admin/.qoder/hallmark-tools/node_modules/puppeteer-core/lib/esm/puppeteer/puppeteer-core.js';
const browser = await puppeteer.launch({ executablePath: 'C:/Program Files/Google/Chrome/Application/chrome.exe', headless: 'new', args: ['--no-sandbox', '--disable-gpu'] });
const page = await browser.newPage();
await page.setViewport({ width: 1600, height: 900, deviceScaleFactor: 2 });
await page.goto('http://127.0.0.1:8010/', { waitUntil: 'networkidle0', timeout: 60000 });
await page.waitForSelector('.example-card');
await page.evaluate(() => document.querySelector('.example-card').click());
await page.waitForFunction(() => { const t = document.getElementById('verdictTitle'); return t && t.textContent.trim().length > 2 && !/analys|reading/i.test(t.textContent); }, { timeout: 90000 });
await new Promise(r => setTimeout(r, 1800));

const rects = await page.evaluate(() => {
  const readout = document.querySelector('.readout');
  const rr = readout.getBoundingClientRect();
  const rel = (sel) => {
    const b = document.querySelector(sel).getBoundingClientRect();
    return { x: b.x - rr.x, y: b.y - rr.y, w: b.width, h: b.height };
  };
  return {
    readout: { w: rr.width, h: rr.height },
    verdict: rel('#verdictCard'),
    gauge: rel('.gauge-wrap'),
    copy: rel('.verdict-copy'),
    signals: rel('#signalList'),
    signalsPanel: rel('#signalList').y,
  };
});
console.log(JSON.stringify(rects, null, 1));

const readout = await page.$('.readout');
await readout.screenshot({ path: 'promo/shots/09-readout.png' });
const viewer = await page.$('.viewer-card');
await viewer.screenshot({ path: 'promo/shots/10-viewer-el.png' });
const surface = await page.$('.evidence');
console.log(await page.evaluate(() => {
  const e = document.querySelector('.evidence').getBoundingClientRect();
  return { w: e.width, h: e.height };
}));
await browser.close();
