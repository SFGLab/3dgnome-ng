import type { Region } from '../api'
import { NO_OWN, type OwnChoice, type UploadResult } from './upload'

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
  structures: 100,
  name: '',
  settings: {},
  own: NO_OWN,
}

/**
 * The coordinates of a typed region longer than `maxSpan` bp. The server refuses such a region and
 * the form keeps it on the loops, where it can be shortened. Undefined for any other text.
 */
export function longRegion(text: string, maxSpan: number): Region | undefined {
  const m = /^(chr[0-9XYM]+):(\d+)-(\d+)$/.exec(text.replace(/,/g, '').trim())
  if (!m) return undefined
  const [start, end] = [Number(m[2]), Number(m[3])]
  if (end - start <= maxSpan) return undefined
  return { chrom: m[1], start, end, label: `${m[1]}:${start.toLocaleString('en-US')}-${end.toLocaleString('en-US')}` }
}
