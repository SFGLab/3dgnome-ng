/** The window of a contact map, with its place in the region of the job and its resolution. */
import { type Locus } from '../lib/region'

export const RESOLUTIONS = [5_000, 10_000, 25_000, 50_000, 100_000, 250_000]

/** The most bins on a side the server draws, `MAX_BINS` of gnome3d_web/contacts.py. */
export const MAX_BINS = 840

/** Returns a window cut to the region of the job and at least ten bins wide, or the whole region when it does not overlap it. */
export function clamp(l: Locus, job: Locus, res: number): Locus {
  if (l.chrom !== job.chrom || l.end <= job.start || l.start >= job.end) return job
  const width = Math.min(Math.max(l.end - l.start, res * 10), job.end - job.start)
  const start = Math.min(Math.max(l.start, job.start), job.end - width)
  return { chrom: job.chrom, start: Math.round(start), end: Math.round(start + width) }
}

/**
 * Returns the bins of a window at a resolution, counted as the server counts them, from the bin
 * that holds the start, on the grid of the resolution.
 */
export const binCount = (l: Locus, r: number) => Math.ceil((l.end - (l.start - (l.start % r))) / r)

/** Returns the finest resolution not finer than `res` that keeps a window within MAX_BINS bins. */
export const fitting = (l: Locus, res: number) => RESOLUTIONS.find((r) => r >= res && binCount(l, r) <= MAX_BINS) ?? RESOLUTIONS[RESOLUTIONS.length - 1]
