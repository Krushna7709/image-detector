import puppeteer from 'file:///C:/Users/Admin/.qoder/hallmark-tools/node_modules/puppeteer-core/lib/esm/puppeteer/puppeteer-core.js';

const browser = await puppeteer.launch({
  executablePath: 'C:/Program Files/Google/Chrome/Application/chrome.exe',
  headless: 'new',
  args: ['--no-sandbox', '--disable-gpu', '--force-device-scale-factor=2'],
});
const page = await browser.newPage();
await page.setViewport({ width: 1600, height: 900, deviceScaleFactor: 2 });
await page.goto('http://127.0.0.1:8010/', { waitUntil: 'networkidle0', timeout: 60000 });
await page.waitForSelector('.example-card', { timeout: 30000 });
await new Promise(r => setTimeout(r, 800));
await page.screenshot({ path: 'promo/shots/01-hero.png' });

async function analyze(kind) {
  await page.evaluate((kind) => {
    const cards = [...document.querySelectorAll('.example-card')];
    const card = cards.find(c => {
      const tag = c.querySelector('.example-tag');
      const isAi = tag && tag.textContent.trim().toLowerCase() === 'ai';
      return kind === 'ai' ? isAi : !isAi;
    });
    card?.click();
  }, kind);
  await page.waitForFunction(() => {
    const t = document.getElementById('verdictTitle');
    return !!document.getElementById('verdictCard') && t && t.textContent.trim().length > 2 && !/analys|reading/i.test(t.textContent);
  }, { timeout: 90000 });
  await new Promise(r => setTimeout(r, 1200));
}

await analyze('ai');
await page.evaluate(() => document.getElementById('results').scrollIntoView({ block: 'start' }));
await new Promise(r => setTimeout(r, 600));
await page.screenshot({ path: 'promo/shots/02-ai-verdict.png' });
const verdict = await page.evaluate(() => ({
  chip: document.getElementById('verdictChip').textContent.trim(),
  title: document.getElementById('verdictTitle').textContent.trim(),
  sub: document.getElementById('verdictSub').textContent.trim(),
}));
console.log('AI verdict:', JSON.stringify(verdict));

await page.evaluate(() => window.scrollBy(0, 620));
await new Promise(r => setTimeout(r, 500));
await page.screenshot({ path: 'promo/shots/03-signals.png' });

await page.evaluate(() => document.querySelector('.heatmap-section, .edit-section, #editHeatmap')?.scrollIntoView({ block: 'start' }));
await new Promise(r => setTimeout(r, 500));
await page.screenshot({ path: 'promo/shots/04-heatmap.png' });

await page.evaluate(() => window.scrollTo(0, 0));
await new Promise(r => setTimeout(r, 400));
await analyze('real');
await page.evaluate(() => document.getElementById('results').scrollIntoView({ block: 'start' }));
await new Promise(r => setTimeout(r, 700));
await page.screenshot({ path: 'promo/shots/05-real-verdict.png' });
const realVerdict = await page.evaluate(() => ({
  chip: document.getElementById('verdictChip').textContent.trim(),
  title: document.getElementById('verdictTitle').textContent.trim(),
}));
console.log('Real verdict:', JSON.stringify(realVerdict));

await browser.close();
