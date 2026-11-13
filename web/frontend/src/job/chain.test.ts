import { describe, expect, it } from 'vitest'
import { bondRadius, fogRange, pickOnChain, stretch } from './chain'

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

describe('pickOnChain', () => {
  // Screen x, y in pixels, distance from the camera, and the tube's drawn radius in pixels there.
  it('takes the tube under the cursor between two beads far apart on screen', () => {
    const chain: [number, number, number, number][] = [
      [0, 100, 10, 6],
      [100, 100, 10, 6],
    ]
    expect(pickOnChain(chain, 30, 103)).toBe(0)
    expect(pickOnChain(chain, 70, 98)).toBe(1)
  })

  it('takes the front tube where two cross', () => {
    const chain: [number, number, number, number][] = [
      [0, 100, 20, 6],
      [100, 100, 20, 6],
      [50, 0, 5, 6],
      [50, 200, 5, 6],
    ]
    expect(pickOnChain(chain, 50, 100)).toBe(2)
  })

  it('takes nothing off every tube', () => {
    const chain: [number, number, number, number][] = [
      [0, 100, 10, 6],
      [100, 100, 10, 6],
    ]
    expect(pickOnChain(chain, 50, 130)).toBe(null)
  })

  it('takes a lone bead under the cursor', () => {
    expect(pickOnChain([[10, 10, 5, 6]], 12, 12)).toBe(0)
  })
})
