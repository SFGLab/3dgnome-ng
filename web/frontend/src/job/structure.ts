/** The beads and the chain of a structure, with the region of a job, the colours, the bead at a position and the ranges of a focus. */
import type { Loop, Annotations, Structure } from '../api'
import { stretch } from './chain'
import { colorAt, readPalette, type Mode } from './colors'

/**
 * Returns the stretch of the genome the model keeps solid. It is the whole chain between the two
 * anchors of a selected loop and, without one, the part of the region chosen on the loop view.
 */
export function solidSpan(loop: Loop | null, part: [number, number] | null): [number, number] | null {
  return loop ? [(loop.a[0] + loop.a[1]) / 2, (loop.b[0] + loop.b[1]) / 2] : part
}

/** Returns the colour of every chain point of a structure under a colour mode. */
export function chainColors(structure: Structure, annotations: Annotations, region: [number, number], mode: Mode, palette = readPalette()): string[] {
  const ctx = {
    palette,
    start: region[0],
    end: region[1],
    promoters: annotations.genes.map((g) => [g.tss - 1000, g.tss + 1000] as [number, number]),
    enhancers: annotations.enhancers,
    anchors: structure.anchors.map((a) => ({ start: a.start, end: a.end, factors: a.loops.map((l) => l.factor) })),
  }
  return structure.chain.map((p) => colorAt(mode, p[3], ctx))
}

/** Returns the index of the bead whose span holds a genomic position, by binary search over starts. */
export function beadAt(beads: Structure['beads'], bp: number): number {
  let lo = 0
  let hi = beads.length - 1
  while (lo < hi) {
    const mid = (lo + hi + 1) >> 1
    if (beads[mid][0] <= bp) lo = mid
    else hi = mid - 1
  }
  return lo
}

/** Returns the index of the chain point nearest a genomic position, by binary search over the positions. */
export function chainAt(chain: Structure['chain'], bp: number): number {
  let lo = 0
  let hi = chain.length - 1
  while (lo < hi) {
    const mid = (lo + hi) >> 1
    if (chain[mid][3] < bp) lo = mid + 1
    else hi = mid
  }
  return lo > 0 && Math.abs(chain[lo - 1][3] - bp) <= Math.abs(chain[lo][3] - bp) ? lo - 1 : lo
}

/** Chain points either side of a hovered or a selected point that stay solid. */
export const FOCUS_REACH = 15

/**
 * Returns a short stretch of the chain at each of the beads. The whole chain between two beads far
 * apart on the genome is most of the model and would leave nothing see through.
 */
export function focusRanges(structure: Structure, beads: number[]): [number, number][] {
  const n = structure.chain.length
  return beads.filter((k) => structure.beads[k]).map((k) => stretch(chainAt(structure.chain, (structure.beads[k][0] + structure.beads[k][1]) / 2), n, FOCUS_REACH))
}
