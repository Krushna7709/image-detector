import puppeteer from 'file:///C:/Users/Admin/.qoder/hallmark-tools/node_modules/puppeteer-core/lib/esm/puppeteer/puppeteer-core.js';
const browser = await puppeteer.launch({ executablePath: 'C:/Program Files/Google/Chrome/Application/chrome.exe', headless: 'new', args: ['--no-sandbox', '--disable-gpu'] });
const page = await browser.newPage();
await page.setViewport({ width: 430, height: 932, deviceScaleFactor: 3 });
await page.goto('http://127.0.0.1:8010/', { waitUntil: 'networkidle0', timeout: 60000 });
await page.waitForSelector('.example-card');
await new Promise(r => setTimeout(r, 900));
await page.screenshot({ path: 'promo/shots/r01-hero.png' });

await page.evaluate(() => document.querySelector('.example-card').parentElement.scrollIntoView({ block: 'center' }));
await new Promise(r => setTimeout(r, 500));
await page.screenshot({ path: 'promo/shots/r02-samples.png' });

await page.evaluate(() => document.querySelector('.example-card').click());
await page.waitForFunction(() => { const t = document.getElementById('verdictTitle'); return t && t.textContent.trim().length > 2 && !/analys|reading/i.test(t.textContent); }, { timeout: 90000 });
await new Promise(r => setTimeout(r, 1800));

await page.evaluate(() => document.getElementById('verdictCard').scrollIntoView({ block: 'center' }));
await new Promise(r => setTimeout(r, 700));
await page.screenshot({ path: 'promo/shots/r03-verdict.png' });

await page.evaluate(() => document.getElementById('signalList').scrollIntoView({ block: 'start' }));
await new Promise(r => setTimeout(r, 500));
await page.screenshot({ path: 'promo/shots/r04-signals.png' });

await page.evaluate(() => window.scrollBy(0, 3600));
await new Promise(r => setTimeout(r, 500));
await page.screenshot({ path: 'promo/shots/r05-signals-mid.png' });

await page.evaluate(() => document.querySelector('.viewer-card').scrollIntoView({ block: 'start' }));
await new Promise(r => setTimeout(r, 600));
await page.screenshot({ path: 'promo/shots/r06-viewer.png' });

await page.evaluate(() => window.scrollBy(0, 700));
await new Promise(r => setTimeout(r, 500));
await page.screenshot({ path: 'promo/shots/r07-heatmap.png' });

// real verdict
await page.evaluate(() => {
  window.scrollTo(0, 0);
  const cards = [...document.querySelectorAll('.example-card')];
  cards.find(c => (c.querySelector('.example-tag')?.textContent || '').trim().toLowerCase() !== 'ai')?.click();
});
await page.waitForFunction(() => {
  const t = document.getElementById('verdictTitle');
  return t && /real/i.test(t.textContent);
}, { timeout: 90000 });
await new Promise(r => setTimeout(r, 1600));
await page.evaluate(() => document.getElementById('verdictCard').scrollIntoView({ block: 'center' }));
await new Promise(r => setTimeout(r, 700));
await page.screenshot({ path: 'promo/shots/r08-real.png' });

const sizes = await page.evaluate(() => ({ scrollW: document.documentElement.scrollWidth, innerW: innerWidth }));
console.log(JSON.stringify(sizes));
await browser.close();
