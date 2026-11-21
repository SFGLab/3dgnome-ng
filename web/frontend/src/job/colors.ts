/** The colour of each point of the chain under the three colour modes of the Structure tab. */
export type Mode = 'elements' | 'position' | 'factor'
export type Palette = { promoter: string; enhancer: string; chromatin: string; ctcf: string; rnapii: string }
export type ColorContext = {
  palette: Palette
  start: number
  end: number
  promoters: [number, number][]
  enhancers: [number, number][]
  anchors: { start: number; end: number; factors: string[] }[]
}

const RAMP: [number, number, number][] = [
  [0xd6, 0xe6, 0xf8],
  [0x1c, 0x3f, 0x7a],
]

const hex = (rgb: number[]) => `#${rgb.map((v) => Math.round(v).toString(16).padStart(2, '0')).join('')}`

const rgb = (c: string) => [1, 3, 5].map((i) => parseInt(c.slice(i, i + 2), 16))

/** The chromatin colour toned by position, lighter at the region's start and darker at its end.
 * One flat grey made strands that cross each other read as one mass. */
export function chromatinAt(base: string, t: number): string {
  const c = rgb(base)
  const light = c.map((v) => v + 0.45 * (255 - v))
  const dark = c.map((v) => v * 0.42)
  return hex(light.map((v, i) => v + t * (dark[i] - v)))
}

const inside = (bp: number, spans: [number, number][]) => spans.some(([a, b]) => bp >= a && bp <= b)

export function colorAt(mode: Mode, bp: number, ctx: ColorContext): string {
  const { palette } = ctx
  const t = Math.min(1, Math.max(0, (bp - ctx.start) / (ctx.end - ctx.start)))
  if (mode === 'position') return hex(RAMP[0].map((v, i) => v + t * (RAMP[1][i] - v)))
  if (mode === 'factor') {
    const a = ctx.anchors.find((x) => bp >= x.start && bp <= x.end)
    if (!a || a.factors.length === 0) return chromatinAt(palette.chromatin, t)
    return a.factors.includes('CTCF') ? palette.ctcf : palette.rnapii
  }
  if (inside(bp, ctx.promoters)) return palette.promoter
  if (inside(bp, ctx.enhancers)) return palette.enhancer
  return chromatinAt(palette.chromatin, t)
}

/** The palette from the page's tokens, with the design's values where a token is missing. */
export function readPalette(): Palette {
  const css = typeof window === 'undefined' ? null : getComputedStyle(document.documentElement)
  const v = (name: string, fallback: string) => css?.getPropertyValue(name).trim() || fallback
  return {
    promoter: v('--promoter', '#EB6834'),
    enhancer: v('--enhancer', '#1BAF7A'),
    chromatin: '#8A8D94',
    ctcf: v('--ctcf', '#2A78D6'),
    rnapii: v('--rnapii', '#7A3FB0'),
  }
}
