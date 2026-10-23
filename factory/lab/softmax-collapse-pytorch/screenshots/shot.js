const { chromium } = require(require('child_process').execSync('npm root -g').toString().trim() + '/playwright');
const fs = require('fs'), path = require('path');
const DIST = '/home/user/maybenull.com/dist';
const types = {'.html':'text/html','.js':'application/javascript','.css':'text/css','.json':'application/json','.svg':'image/svg+xml','.woff2':'font/woff2','.png':'image/png'};
(async () => {
  const browser = await chromium.launch({executablePath: '/opt/pw-browsers/chromium'}).catch(()=>chromium.launch());
  for (const scheme of ['light','dark']) for (const w of [1100, 390]) {
    const ctx = await browser.newContext({viewport: {width: w, height: 900}, colorScheme: scheme, deviceScaleFactor: 1});
    await ctx.route('https://maybenull.test/**', async route => {
      let p = new URL(route.request().url()).pathname; if (p.endsWith('/')) p += 'index.html';
      const f = path.join(DIST, p);
      if (fs.existsSync(f)) route.fulfill({body: fs.readFileSync(f), contentType: types[path.extname(f)] || 'application/octet-stream'});
      else route.fulfill({status: 404, body: 'nf'});
    });
    const page = await ctx.newPage();
    page.on('console', m => { if (m.type()==='error') console.log('console', m.text()); });
    await page.goto('https://maybenull.test/posts/softmax-collapse-pytorch/', {waitUntil: 'networkidle'});
    await page.waitForTimeout(1500);
    for (const id of ['sc-cliff','sc-curves']) {
      const el = await page.$('#'+id);
      await el.screenshot({path: `shots/${id}-${scheme}-${w}.png`});
    }
    if (w===1100 && scheme==='light') { await page.selectOption('#sc-seed','3'); await page.waitForTimeout(500); await (await page.$('#sc-curves')).screenshot({path:`shots/sc-curves-seed3-${scheme}-${w}.png`}); }
    await ctx.close();
  }
  await browser.close();
})();
