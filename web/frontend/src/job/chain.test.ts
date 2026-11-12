import { describe, expect, it } from 'vitest'
import { bondRadius, fogRange, stretch } from './chain'

describe('chain geometry', () => {
  it('makes the tube a fixed share of the median bond', () => {
    const pts: [number, number, number][] = [[0, 0, 0], [1, 0, 0], [3, 0, 0], [4, 0, 0], [5, 0, 0]]
    expect(bondRadius(pts)).toBeCloseTo(0.28, 5)
  })

  it('gives a tube for a chain of one bead', () => {
    expect(bondRadius([[0, 0, 0]])).toBeGreaterThan(0)
  })

  it('brightens a stretch around the hovered bead, cut at the ends', () => {
    expect(stretch(10, 100)).toEqual([4, 16])
    expect(stretch(2, 100)).toEqual([0, 8])
    expect(stretch(98, 100)).toEqual([92, 99])
  })

  it('fogs from the middle of the structure to its far side', () => {
    expect(fogRange(10, 24)).toEqual([24, 34])
  })
})
