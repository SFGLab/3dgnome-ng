import { describe, expect, it } from 'vitest'
import { DARKEST, colour, diverging, emptyRows, halfRange, transform } from './heatmap'

const M = [
  [4, 2, 1],
  [2, 6, 3],
  [1, 3, 8],
]

/** A map whose contacts fall off as one over the separation, with empty rows and columns. */
function decay(n: number, empty: number[] = []): number[][] {
  return Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => (empty.includes(i) || empty.includes(j) || i === j ? 0 : 1 / Math.abs(i - j))))
}

describe('transform', () => {
  it('gives each off diagonal its mean as the expectation and leaves the main diagonal empty', () => {
    expect(transform(M, 'expected')).toEqual([
      [0, 2.5, 1],
      [2.5, 0, 2.5],
      [1, 2.5, 0],
    ])
  })

  it('gives ones off the diagonal for a map that equals its expectation', () => {
    const e = transform(M, 'expected')
    const oe = transform(e, 'oe')
    expect(oe.flatMap((r, i) => r.filter((_, j) => j !== i)).every((v) => Math.abs(v - 1) < 1e-12)).toBe(true)
  })

  it('averages each separation over rows that hold contacts', () => {
    const e = transform(decay(10, [9]), 'expected')
    expect(e[0][1]).toBeCloseTo(1, 5)
    expect(e[0][4]).toBeCloseTo(0.25, 2)
  })

  it('smooths a lone stratum over its neighbours', () => {
    const m = decay(40)
    for (let i = 0; i + 16 < 40; i++) m[i][i + 16] = m[i + 16][i] = 5
    const e = transform(m, 'expected')
    expect(e[0][16] / e[0][14]).toBeLessThan(3)
  })

  it('gives a symmetric Pearson map with ones on the diagonal', () => {
    const p = transform(M, 'pearson')
    for (let i = 0; i < 3; i++) {
      expect(p[i][i]).toBeCloseTo(1)
      for (let j = 0; j < 3; j++) expect(p[i][j]).toBeCloseTo(p[j][i])
    }
  })

  it('leaves the observed map as it is', () => {
    expect(transform(M, 'observed')).toEqual(M)
  })
})

describe('colour on a log scale', () => {
  it('spreads values below one as it spreads values above, by ratio', () => {
    const small = colour(0.01, 0.001, 0.1, true)
    const big = colour(10, 1, 100, true)
    expect(small).toEqual(big)
  })
})

describe('colour', () => {
  it('runs from white at the low end to the darkest red at the high end', () => {
    expect(colour(0, 0, 10, false)).toEqual([255, 255, 255])
    expect(colour(10, 0, 10, false)).toEqual(DARKEST)
    expect(colour(20, 0, 10, true)).toEqual(DARKEST)
  })
})

describe('emptyRows', () => {
  it('marks the rows with no contact off the diagonal', () => {
    expect(emptyRows([[3, 0, 0], [0, 0, 2], [0, 2, 0]])).toEqual([true, false, false])
  })
})

describe('diverging', () => {
  it('is white at no change, blue below and red above', () => {
    expect(diverging(0, 2)).toEqual([255, 255, 255])
    const [r1, , b1] = diverging(-2, 2)
    const [r2, , b2] = diverging(2, 2)
    expect(b1).toBeGreaterThan(r1)
    expect(r2).toBeGreaterThan(b2)
  })
})

describe('halfRange', () => {
  const m = Array.from({ length: 10 }, (_, i) => Array.from({ length: 10 }, (_, j) => (i === j ? 0 : 1 + ((i * 10 + j) % 100))))
  it('runs from a low to a high percentile of the nonzero values on a log scale', () => {
    const [lo, hi] = halfRange(m, true)
    expect(lo).toBeGreaterThan(0)
    expect(hi).toBeGreaterThan(lo)
  })

  it('starts at zero on a linear scale', () => {
    expect(halfRange(m, false)[0]).toBe(0)
  })
})
