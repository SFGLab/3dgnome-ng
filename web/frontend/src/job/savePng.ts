/** Saving a figure of the page as a PNG file. */

// What a figure's style sheet sets and a picture of it must carry in its own attributes.
const STYLES = ['fill', 'stroke', 'stroke-width', 'stroke-dasharray', 'opacity', 'font-size', 'font-family', 'font-weight', 'text-anchor']
const SCALE = 3
const TITLE = 26

/**
 * A copy of an SVG figure that stands by itself: every element has the styles the page computed
 * for it, the figure has the background of its place on the page and its title is written above it.
 */
export function standalone(svg: SVGSVGElement, title: string): SVGSVGElement {
  const [, , w, h] = (svg.getAttribute('viewBox') ?? '0 0 340 270').split(/\s+/).map(Number)
  const copy = svg.cloneNode(true) as SVGSVGElement
  const from = [svg, ...svg.querySelectorAll<SVGElement>('*')]
  const to = [copy, ...copy.querySelectorAll<SVGElement>('*')]
  from.forEach((el, i) => {
    const style = getComputedStyle(el)
    for (const name of STYLES) {
      const value = style.getPropertyValue(name)
      if (value) to[i].style.setProperty(name, value)
    }
  })
  const ns = 'http://www.w3.org/2000/svg'
  const page = getComputedStyle(svg)
  const whole = document.createElementNS(ns, 'svg')
  whole.setAttribute('xmlns', ns)
  whole.setAttribute('viewBox', `0 0 ${w} ${h + TITLE}`)
  whole.setAttribute('width', String(w * SCALE))
  whole.setAttribute('height', String((h + TITLE) * SCALE))
  const back = document.createElementNS(ns, 'rect')
  back.setAttribute('width', String(w))
  back.setAttribute('height', String(h + TITLE))
  back.setAttribute('fill', page.backgroundColor && page.backgroundColor !== 'rgba(0, 0, 0, 0)' ? page.backgroundColor : '#FFFFFF')
  const head = document.createElementNS(ns, 'text')
  head.setAttribute('x', '10')
  head.setAttribute('y', '17')
  head.setAttribute('font-size', '11')
  head.setAttribute('font-weight', '600')
  head.setAttribute('font-family', page.fontFamily || 'sans-serif')
  head.setAttribute('fill', page.color || '#141413')
  head.textContent = title
  copy.setAttribute('x', '0')
  copy.setAttribute('y', String(TITLE))
  copy.setAttribute('width', String(w))
  copy.setAttribute('height', String(h))
  copy.style.removeProperty('opacity')
  whole.append(back, head, copy)
  return whole
}

/** Saves an SVG figure of the page as a PNG at three times its size, with its title above it. */
export async function savePng(svg: SVGSVGElement, title: string, file: string): Promise<void> {
  const whole = standalone(svg, title)
  const url = URL.createObjectURL(new Blob([new XMLSerializer().serializeToString(whole)], { type: 'image/svg+xml' }))
  try {
    const image = new Image()
    await new Promise<void>((done, failed) => {
      image.onload = () => done()
      image.onerror = () => failed(new Error('The figure could not be drawn as a picture.'))
      image.src = url
    })
    const canvas = document.createElement('canvas')
    canvas.width = Number(whole.getAttribute('width'))
    canvas.height = Number(whole.getAttribute('height'))
    canvas.getContext('2d')?.drawImage(image, 0, 0)
    const blob = await new Promise<Blob | null>((done) => canvas.toBlob(done, 'image/png'))
    if (!blob) return
    const link = document.createElement('a')
    link.href = URL.createObjectURL(blob)
    link.download = `${file}.png`
    link.click()
    URL.revokeObjectURL(link.href)
  } finally {
    URL.revokeObjectURL(url)
  }
}
