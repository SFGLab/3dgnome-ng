/** The loops of a job as the loop views show them: the zoom, the ticks, the text of a loop and whether two are one. */
import type { Loop } from '../api'
import { bp, kb } from './format'

/** The shortest part a zoom leaves, in bp. */
const MIN_PART = 10_000

/** A part of `bounds` scaled by `factor` about the position `at`, which keeps its place in the
 * part. A part as long as the bounds is the bounds. */
export function zoomed(part: [number, number], bounds: [number, number], at: number, factor: number): [number, number] {
  const full = bounds[1] - bounds[0]
  const width = Math.min(Math.max((part[1] - part[0]) * factor, Math.min(MIN_PART, full)), full)
  const share = Math.min(1, Math.max(0, (at - part[0]) / (part[1] - part[0])))
  const start = Math.min(Math.max(at - share * width, bounds[0]), bounds[1] - width)
  return [Math.round(start), Math.round(start + width)]
}

/** One loop in words: its factor, its two anchors, its span and its PET count. */
export const describeLoop = (l: Loop) => `${l.factor} loop, ${l.chrom}:${bp(l.a[0])}-${bp(l.a[1])} to ${l.chrom}:${bp(l.b[0])}-${bp(l.b[1])}, ${kb(l.span)}, ${l.pets} PET${l.pets === 1 ? '' : 's'}`

/** Whether two loops are the same loop, by factor and anchors. */
export const sameLoop = (a: Loop, b: Loop) => a.factor === b.factor && a.a[0] === b.a[0] && a.a[1] === b.a[1] && a.b[0] === b.b[0] && a.b[1] === b.b[1]

/** About five round positions across a span, in bp. */
export function ticks(start: number, end: number): number[] {
  const rough = (end - start) / 5
  const power = 10 ** Math.floor(Math.log10(Math.max(rough, 1)))
  const step = [1, 2, 5, 10].map((m) => m * power).find((s) => s >= rough) ?? power * 10
  const out: number[] = []
  for (let v = Math.ceil(start / step) * step; v <= end; v += step) out.push(v)
  return out
}
