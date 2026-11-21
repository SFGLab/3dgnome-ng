/** The drawing of the two halves of a contact map on a canvas, one pixel for each pair of bins. */
import { colour, diverging, emptyRows, type Show } from './heatmap'

const NO_DATA: [number, number, number] = [226, 225, 221]

type Half = { m: number[][]; range: [number, number]; loops: boolean }

/** Each half on its own colour range. A loop half has no empty bins to grey: a row without a loop
 * is a row of zeros, drawn white. */
export function draw(canvas: HTMLCanvasElement, up: Half, low: Half, log: boolean, show: Show) {
  const [upper, lower] = [up.m, low.m]
  const ctx = canvas.getContext('2d')
  const n = upper.length
  if (!ctx || !n) return
  canvas.width = n
  canvas.height = n
  // Each half on its own scale: the lower half's values are brought to the upper half's range.
  const span = Math.max(Math.abs(up.range[0]), Math.abs(up.range[1]))
  const none = upper.map(() => false)
  const [emptyUp, emptyLow] = [up.loops ? none : emptyRows(upper), low.loops ? none : emptyRows(lower)]
  const img = ctx.createImageData(n, n)
  for (let i = 0; i < n; i++)
    for (let j = 0; j < n; j++) {
      const v = j >= i ? upper[i][j] : lower[i][j]
      const range = j >= i ? up.range : low.range
      // The maps hold no pair within one bin. The main diagonal is drawn as no data.
      const empty = j >= i ? emptyUp[i] || emptyUp[j] : emptyLow[i] || emptyLow[j]
      const [r, g, b] =
        i === j || empty ? NO_DATA : show === 'oe' ? diverging(v > 0 ? Math.log2(v) : -span, span) : show === 'pearson' ? diverging(v, span) : colour(v, range[0], range[1], log)
      const k = (i * n + j) * 4
      img.data.set([r, g, b, 255], k)
    }
  ctx.putImageData(img, 0, 0)
}
