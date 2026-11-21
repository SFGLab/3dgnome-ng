import { type Compare, type Summary } from '../../api'
import { FACTORS } from '../../lib/format'

const BACKGROUNDS: Record<string, string> = { hic: 'Hi-C', hicdiffusion: 'HiCDiffusion', none: 'no contact map' }

/**
 * A short name for each of two jobs, by the first thing they differ in: the cell type, the
 * sample of the variants, the loops or the contact background. Two jobs alike in all of them
 * are this job and the other job.
 */
export function sideNames(a: Summary, b: Summary): [string, string] {
  const tries: ((s: Summary) => string)[] = [
    (s) => (s.request.cell_type === 'own' ? 'own data' : s.request.cell_type),
    (s) => s.request.sample || 'no variants',
    (s) => `${FACTORS[s.request.factors] ?? s.request.factors} loops`,
    (s) => BACKGROUNDS[s.request.hicdiffusion ? 'hicdiffusion' : (s.request.background ?? 'hic')] ?? 'Hi-C',
  ]
  for (const name of tries) if (name(a) !== name(b)) return [name(a), name(b)]
  return ['this job', 'the other job']
}

/** How many times a level or a distance changed, from its log2 change, with the word for its direction. */
export const times = (log2: number, up: string, down: string) => `${(2 ** Math.abs(log2)).toFixed(1)} times ${log2 >= 0 ? up : down}`

/** The change of an expression level. A gene with next to no reads on one side has a fold change
 * in the hundreds that says no more than over 100 times. */
export const fold = (log2: number) => (2 ** Math.abs(log2) >= 100 ? `over 100 times ${log2 >= 0 ? 'higher' : 'lower'}` : times(log2, 'higher', 'lower'))

/** Whether the two jobs measure a gene to two enhancers: the separations differ by more than half a bead. */
export const twoEnhancers = (r: Compare['rows'][number]) => r.linear_bp_b !== undefined && Math.abs(r.linear_bp - r.linear_bp_b) > 500

export const pText = (p: number | null) => (p === null ? 'no test' : p < 0.001 ? p.toExponential(1) : p.toFixed(3))

export const level = (v: number) => (v >= 100 ? v.toFixed(0) : v.toFixed(1))
