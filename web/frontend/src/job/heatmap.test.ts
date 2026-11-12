import { describe, expect, it } from 'vitest'
import { DARKEST, colour, transform } from './heatmap'

const M = [
  [4, 2, 1],
  [2, 6, 3],
  [1, 3, 8],
]

describe('transform', () => {
  it('gives each diagonal its mean as the expectation', () => {
    expect(transform(M, 'expected')).toEqual([
      [6, 2.5, 1],
      [2.5, 6, 2.5],
      [1, 2.5, 6],
    ])
  })

  it('gives ones for a map that equals its expectation', () => {
    const e = transform(M, 'expected')
    expect(transform(e, 'oe').flat().every((v) => Math.abs(v - 1) < 1e-12)).toBe(true)
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

describe('colour', () => {
  it('runs from white at the low end to the darkest red at the high end', () => {
    expect(colour(0, 0, 10, false)).toEqual([255, 255, 255])
    expect(colour(10, 0, 10, false)).toEqual(DARKEST)
    expect(colour(20, 0, 10, true)).toEqual(DARKEST)
  })
})
