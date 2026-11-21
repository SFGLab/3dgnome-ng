import { describe, expect, it } from 'vitest'
import { chromatinAt, colorAt, type ColorContext } from './colors'

/** Returns the relative luminance of a hex colour, for the order of a ramp. */
function lightness(c: string): number {
  const [r, g, b] = [1, 3, 5].map((i) => parseInt(c.slice(i, i + 2), 16) / 255)
  return 0.2126 * r + 0.7152 * g + 0.0722 * b
}

const PALETTE = { promoter: '#EB6834', enhancer: '#1BAF7A', chromatin: '#9A9994', ctcf: '#2A78D6', rnapii: '#7A3FB0' }
const CTX: ColorContext = {
  palette: PALETTE,
  start: 1_000_000,
  end: 2_000_000,
  promoters: [[1_200_000, 1_202_000]],
  enhancers: [[1_500_000, 1_502_000]],
  anchors: [{ start: 1_700_000, end: 1_701_000, factors: ['CTCF'] }, { start: 1_800_000, end: 1_801_000, factors: ['RNAPII'] }],
}

describe('colorAt', () => {
  it('colours promoters, enhancers and the rest of the chromatin', () => {
    expect(colorAt('elements', 1_201_000, CTX)).toBe(PALETTE.promoter)
    expect(colorAt('elements', 1_501_000, CTX)).toBe(PALETTE.enhancer)
    expect(colorAt('elements', 1_600_000, CTX)).toBe(chromatinAt(PALETTE.chromatin, 0.6))
  })

  it('tones the chromatin from light to dark along the region and keeps it grey', () => {
    const tones = [0, 0.25, 0.5, 0.75, 1].map((t) => chromatinAt(PALETTE.chromatin, t))
    const ls = tones.map(lightness)
    for (let i = 1; i < ls.length; i++) expect(ls[i]).toBeLessThan(ls[i - 1])
    expect(ls[0] - ls[4]).toBeGreaterThan(0.3)
    for (const c of tones) {
      const [r, g, b] = [1, 3, 5].map((i) => parseInt(c.slice(i, i + 2), 16))
      expect(Math.max(r, g, b) - Math.min(r, g, b)).toBeLessThan(16)
    }
  })

  it('runs the position ramp from light to dark along the region', () => {
    const ls = [1_000_000, 1_250_000, 1_500_000, 1_750_000, 2_000_000].map((bp) => lightness(colorAt('position', bp, CTX)))
    for (let i = 1; i < ls.length; i++) expect(ls[i]).toBeLessThan(ls[i - 1])
  })

  it('colours anchors by their loop factor', () => {
    expect(colorAt('factor', 1_700_500, CTX)).toBe(PALETTE.ctcf)
    expect(colorAt('factor', 1_800_500, CTX)).toBe(PALETTE.rnapii)
    expect(colorAt('factor', 1_750_000, CTX)).toBe(chromatinAt(PALETTE.chromatin, 0.75))
  })
})
