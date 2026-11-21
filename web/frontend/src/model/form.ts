import type { Region, Summary } from '../api'
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
 * The coordinates of a typed region longer than `maxSpan` bp. The server refuses such a region and
 * the form keeps it on the loops, where it can be shortened. Undefined for any other text.
 */
export function longRegion(text: string, maxSpan: number): Region | undefined {
  const m = /^(chr[0-9XYM]+):(\d+)-(\d+)$/.exec(text.replace(/,/g, '').trim())
  if (!m) return undefined
  const [start, end] = [Number(m[2]), Number(m[3])]
  if (end - start <= maxSpan) return undefined
  return { chrom: m[1], start, end, label: regionText({ chrom: m[1], start, end }) }
}

/**
 * The form filled with the request of a job, to start another job from it. The files a job read
 * from uploads are not part of its request as the site shows it and are left out: the loops,
 * the contacts and the enhancers of your own data, an uploaded VCF and an uploaded expression table. A job on your own cell
 * type gets the listed cell type its loops came from, or the first one.
 */
export function fromRequest(r: Summary['request'], cells: string[]): FormState {
  const factors: Factors = r.factors === 'CTCF' || r.factors === 'RNAPII' ? r.factors : 'CTCF+RNAPII'
  const asked = r.hicdiffusion ? 'hicdiffusion' : (r.background ?? 'hic')
  const background: Background = asked === 'hicdiffusion' || asked === 'none' ? asked : 'hic'
  return {
    ...EMPTY_FORM,
    cell: r.cell_type === 'own' ? (r.own?.loops_from ?? cells[0] ?? EMPTY_FORM.cell) : r.cell_type,
    region: r.region,
    factors,
    svSet: r.sv_set,
    // A sample without a call set was read from an uploaded VCF, which the form does not have.
    sample: r.sv_set ? r.sample : '',
    haplotype: r.sv_set && r.haplotype === 1 ? 1 : 0,
    background,
    expression: r.expression === 'upload' ? '' : (r.expression ?? ''),
    structures: r.n_models,
    settings: { ...(r.overrides ?? {}) },
  }
}

/** Whether a job read files from uploads, which a form filled from its request lacks. */
export const readUploads = (r: Summary['request']) => !!r.own || (!r.sv_set && !!r.sample) || r.expression === 'upload'
