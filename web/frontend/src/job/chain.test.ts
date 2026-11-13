import { describe, expect, it } from 'vitest'
import { bondRadius, fogRange, pickNearest, stretch } from './chain'

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

describe('pickNearest', () => {
  // Screen x, y in pixels and depth, smaller nearer.
  const screen: [number, number, number][] = [
    [100, 100, 0.5],
    [104, 100, 0.2],
    [300, 300, 0.1],
  ]
  it('takes the front bead among those within the radius of the cursor', () => {
    expect(pickNearest(screen, 101, 100, 8)).toBe(1)
  })

  it('takes nothing when no bead is that close', () => {
    expect(pickNearest(screen, 200, 200, 8)).toBe(null)
  })

  it('ignores a bead behind the camera', () => {
    expect(pickNearest([[100, 100, 1.5]], 100, 100, 8)).toBe(null)
  })
})
