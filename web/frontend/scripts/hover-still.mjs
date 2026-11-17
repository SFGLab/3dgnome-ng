import puppeteer from 'puppeteer-core'
const url = process.argv[2]
const b = await puppeteer.launch({ executablePath: '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome', headless: true, args: ['--use-angle=swiftshader', '--enable-unsafe-swiftshader'] })
const p = await b.newPage()
await p.setViewport({ width: 1440, height: 980 })
await p.goto(url, { waitUntil: 'networkidle0' })
await new Promise((r) => setTimeout(r, 7000))
const box = await p.$eval('.view-box canvas', (c) => { const r = c.getBoundingClientRect(); return { x: r.left, y: r.top, w: r.width, h: r.height } })
const hov = () => p.$eval('.hover-box', (n) => n.textContent.match(/bead ([\d,]+)/)?.[1]).catch(() => null)
let tried = 0, held = 0, blocked = 0
for (let k = 0; k < 12; k++) {
  const x = box.x + box.w * (0.3 + 0.04 * k), y = box.y + box.h * (0.45 + 0.01 * k)
  await p.mouse.move(x, y)
  await new Promise((r) => setTimeout(r, 300))
  const first = await hov()
  if (!first) continue
  tried++
  const top = await p.evaluate((x, y) => document.elementFromPoint(x, y)?.tagName, x, y)
  if (top !== 'CANVAS') blocked++
  const seq = []
  for (let s = 0; s < 10; s++) { await new Promise((r) => setTimeout(r, 100)); seq.push(await hov()) }
  if (seq.every((v) => v === first)) held++
  else console.log('lost at', k, first, seq.join(' '))
}
console.log('hovered points', tried, 'covered by a label', blocked, 'held still', held)
await b.close()
