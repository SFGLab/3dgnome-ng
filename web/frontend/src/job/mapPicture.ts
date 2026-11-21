/** The picture of a contact map that the page saves: the map with its region, its two halves and its axes. */

export type MapInfo = {
  chrom: string
  /** The start of each bin, in bp. */
  bins: number[]
  /** The size of a bin, in bp. */
  res: number
  upper: string
  lower: string
}

const MAP = 1200
const LEFT = 96
const TOP = 76
const RIGHT = 28
const BOTTOM = 56

/** A position as megabases with two decimals. */
const mb = (bp: number) => `${(bp / 1e6).toFixed(2)} Mb`

/**
 * A canvas with the drawn map at 1200 pixels a side or more, each bin a square of whole pixels,
 * and around it the region and the bin size, what is above and below the diagonal, and five
 * positions on each axis.
 */
export function mapPicture(map: HTMLCanvasElement, info: MapInfo): HTMLCanvasElement | null {
  const scale = Math.max(1, Math.ceil(MAP / map.width))
  const side = map.width * scale
  const out = document.createElement('canvas')
  out.width = LEFT + side + RIGHT
  out.height = TOP + side + BOTTOM
  const ctx = out.getContext('2d')
  if (!ctx) return null
  ctx.fillStyle = '#FFFFFF'
  ctx.fillRect(0, 0, out.width, out.height)
  ctx.imageSmoothingEnabled = false
  ctx.drawImage(map, LEFT, TOP, side, side)
  ctx.strokeStyle = '#141413'
  ctx.lineWidth = 1
  ctx.strokeRect(LEFT - 0.5, TOP - 0.5, side + 1, side + 1)

  const first = info.bins[0] ?? 0
  const last = (info.bins[info.bins.length - 1] ?? 0) + info.res
  ctx.fillStyle = '#141413'
  ctx.font = '600 22px sans-serif'
  ctx.textBaseline = 'alphabetic'
  ctx.fillText(`${info.chrom}:${first.toLocaleString('en-US')}-${last.toLocaleString('en-US')}, ${info.res / 1000} kb bins`, LEFT, 30)
  ctx.font = '18px sans-serif'
  ctx.fillStyle = '#5C5B57'
  ctx.fillText(`Above the diagonal ${info.upper}. Below it ${info.lower}.`, LEFT, 58)

  ctx.font = '17px sans-serif'
  ctx.fillStyle = '#141413'
  for (let k = 0; k <= 4; k++) {
    const at = (side * k) / 4
    const bp = first + ((last - first) * k) / 4
    ctx.beginPath()
    ctx.moveTo(LEFT + at, TOP + side)
    ctx.lineTo(LEFT + at, TOP + side + 8)
    ctx.moveTo(LEFT - 8, TOP + at)
    ctx.lineTo(LEFT, TOP + at)
    ctx.stroke()
    ctx.textAlign = k === 0 ? 'left' : k === 4 ? 'right' : 'center'
    ctx.textBaseline = 'top'
    ctx.fillText(mb(bp), LEFT + at, TOP + side + 14)
    ctx.textAlign = 'right'
    ctx.textBaseline = k === 0 ? 'top' : k === 4 ? 'bottom' : 'middle'
    ctx.fillText(mb(bp), LEFT - 12, TOP + at)
  }
  return out
}
