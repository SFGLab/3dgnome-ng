import type { Region } from '../api'
import { regionText } from '../lib/region'
import { NO_OWN, type OwnChoice } from './upload'
import { type UploadResult } from '../api'

/** The state of the Model a region form. The cell type own means the job reads only uploads. */
export type Factors = 'CTCF' | 'RNAPII' | 'CTCF+RNAPII'
export type Background = 'hic' | 'hicdiffusion' | 'none'

export type FormState = {
  cell: string
  region: string
  factors: Factors
  svSet: string
  sample: string
  haplotype: 0 | 1
  vcfUpload: UploadResult | null
  background: Background
  /** The id of the hosted expression source of the job, own for the table of the own data, empty for none. */
  expression: string
  structures: number
  name: string
  settings: Record<string, string>
  own: OwnChoice
}

export const EMPTY_FORM: FormState = {
  cell: 'GM12878',
  region: '',
  factors: 'CTCF+RNAPII',
  svSet: '',
  sample: '',
  haplotype: 0,
  vcfUpload: null,
  background: 'hic',
  expression: '',
  structures: 100,
  name: '',
  settings: {},
  own: NO_OWN,
}

/**
 * Returns the coordinates of a typed region longer than `maxSpan` bp. The
 * server refuses such a region and the form keeps it on the loops, where it
 * can be shortened. Undefined for any other text.
 */
export function longRegion(text: string, maxSpan: number): Region | undefined {
  const m = /^(chr[0-9XYM]+):(\d+)-(\d+)$/.exec(text.replace(/,/g, '').trim())
  if (!m) return undefined
  const [start, end] = [Number(m[2]), Number(m[3])]
  if (end - start <= maxSpan) return undefined
  return { chrom: m[1], start, end, label: regionText({ chrom: m[1], start, end }) }
}

/**
 * Returns the change of the form when the own data dialog is confirmed.
 *
 * On your own cell type a job without loops of its own takes the loops of a listed cell type, the
 * first one unless another is chosen. It takes no enhancers unless a set is chosen. A table added
 * in the dialog becomes the expression of the job. A table taken away leaves the job without.
 * @param form the form before the change
 * @param own the uploads the dialog gives
 * @param opened true for a dialog opened to make your own cell type, with the step it opened on
 * @param cells the listed cell types
 */
export function withOwn(form: FormState, own: OwnChoice, opened: { cell: boolean; tab: string }, cells: string[]): Partial<FormState> {
  const asOwn = opened.cell || form.cell === 'own'
  const chosen = asOwn ? { ...own, loopsFrom: own.loops.length ? undefined : own.loopsFrom || cells[0], enhancersFrom: own.enhancersFrom ?? 'none' } : own
  const added = !!chosen.expression && (opened.tab === 'expression' || !form.own.expression)
  const expression = added ? 'own' : !chosen.expression && form.expression === 'own' ? '' : form.expression
  return opened.cell ? { own: chosen, cell: 'own', expression } : { own: chosen, expression }
}
