// The own data dialogs end to end through a running site. The steps are a job with own loops of two factors
// and contacts from a local .hic, a job with an uploaded VCF and its variants table and a job
// with contacts by link and your own cell type with listed loops and enhancers. Each step screenshots its dialog at 1440 px and prints the job's url.
//
//   SITE=http://127.0.0.1:8796 FILES=dir SHOTS=dir node web/scripts/e2e_own_data.mjs own|vcf|link|owncell
//
// FILES holds GM12878_CTCF_loops.bedpe, GM12878_RNAPII_loops.bedpe, ENCFF011KAF.hic and
// HGSVC.vcf.gz. Needs puppeteer-core and a local Chrome.
import puppeteer from 'puppeteer-core'
const SITE = process.env.SITE ?? 'http://127.0.0.1:8796'
const FILES = process.env.FILES
const SHOTS = process.env.SHOTS ?? '.'
const which = process.argv[2]
const b = await puppeteer.launch({ executablePath: '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome', headless: true })
const p = await b.newPage()
const errors = []
p.on('pageerror', (e) => errors.push(String(e)))
p.on('console', (m) => m.type() === 'error' && errors.push(m.text()))
await p.emulateMediaFeatures([{ name: 'prefers-color-scheme', value: 'light' }])
await p.setViewport({ width: 1440, height: 1040, deviceScaleFactor: 1 })
const wait = (ms) => new Promise((r) => setTimeout(r, ms))
async function click(sel, text) {
  for (const o of await p.$$(sel)) if ((await o.evaluate((n) => n.textContent)).includes(text)) return o.click()
  throw new Error(`no ${sel} with ${text}`)
}
async function region(text) {
  const f = await p.$('#region')
  await f.click({ clickCount: 3 })
  await f.type(text)
  await p.waitForSelector('.resolved', { timeout: 20000 })
}
async function structures(n) {
  const f = await p.waitForSelector('xpath/.//label[contains(., "Structures")]//input')
  await f.click({ clickCount: 3 })
  await f.type(String(n))
}
async function submit() {
  await click('button.submit', 'Build')
  await p.waitForFunction(() => /\/(jobs|p)\//.test(location.pathname), { timeout: 60000 })
  return p.url()
}

await p.goto(`${SITE}/model`, { waitUntil: 'networkidle0' })
await wait(800)
if (which === 'own') {
  await region('TNFRSF8')
  await click('button[role=radio]', 'Your own data')
  const add = await p.$('.add-file input')
  await add.uploadFile(`${FILES}/GM12878_CTCF_loops.bedpe`, `${FILES}/GM12878_RNAPII_loops.bedpe`)
  await p.waitForFunction(() => document.querySelectorAll('.factor-row .check.good').length === 2, { timeout: 60000 })
  await wait(1500)
  await p.screenshot({ path: `${SHOTS}/e2e_own_loops.png` })
  await click('button[role=tab]', 'Contacts')
  await click('.route', 'Local .hic')
  await (await p.$('.add-file input')).uploadFile(`${FILES}/ENCFF011KAF.hic`)
  await p.waitForSelector('.contact-card.good', { timeout: 120000 })
  await click('.dialog-foot button', 'Use this data')
  await structures(5)
} else if (which === 'vcf') {
  await region('chr1:109,200,000-110,200,000')
  await (await p.$('.upload input[type=file]')).uploadFile(`${FILES}/HGSVC.vcf.gz`)
  await p.waitForSelector('.sv-table', { timeout: 120000 })
  await wait(1000)
  await p.screenshot({ path: `${SHOTS}/e2e_variants.png` })
  console.log(await p.$eval('.sv-box.good', (n) => n.textContent))
  await click('.dialog-foot button', 'Apply')
  await structures(5)
} else if (which === 'owncell') {
  // Your own cell type with H1ESC CTCF loops, contacts by link to a 4DN file
  // in its bucket and GM12878 enhancers.
  await region('TNFRSF8')
  await click('button[role=radio]', 'Your own data')
  await click('button[role=tab]', 'Contacts')
  await p.type('#contact-link', process.env.LINK ?? 'https://4dn-open-data-public.s3.amazonaws.com/fourfront-webprod/wfoutput/a98ca64a-861a-4a8c-92e9-586af457b1fb/4DNFI1UEG1HD.hic')
  await click('button', 'Read')
  await p.waitForSelector('.contact-card.good, .contact-card.bad', { timeout: 180000 })
  console.log(await p.$eval('.contact-card:not(.hint-card)', (n) => n.textContent))
  await click('button[role=tab]', 'Enhancers')
  await click('.route', 'Enhancers of a listed')
  await p.select('.own-select select', 'GM12878')
  await wait(300)
  await p.screenshot({ path: `${SHOTS}/e2e_owncell_enhancers.png` })
  await click('.dialog-foot button', 'Use this data')
  await click('button[role=radio]', 'CTCF ChIA-PET')
  await p.select('.loops-of select', 'H1ESC')
  await structures(5)
} else if (which === 'link') {
  await region('TNFRSF8')
  await click('button[role=radio]', 'Your own loops')
  await click('button[role=tab]', 'Contacts')
  await p.type('#contact-link', 'https://www.encodeproject.org/files/ENCFF718AWL/@@download/ENCFF718AWL.hic')
  await click('button', 'Read')
  await p.waitForSelector('.contact-card.good, .contact-card.bad', { timeout: 180000 })
  await wait(500)
  await p.screenshot({ path: `${SHOTS}/e2e_contacts_link.png` })
  console.log(await p.$eval('.contact-card:not(.hint-card)', (n) => n.textContent))
  await click('.dialog-foot button', 'Use this data')
  await structures(5)
}
console.log('job', await submit())
console.log(errors.slice(0, 5).join('\n') || 'no errors')
await b.close()
