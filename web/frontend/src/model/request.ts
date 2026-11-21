/**
 * The request of a job, in the one place that states its fields: from the form to the body the
 * server takes, from an example and from the request of an earlier job to the form, and the step
 * of the form each field belongs to.
 */
import type { Summary } from '../api'
import type { Example } from './examples'
import { type Background, EMPTY_FORM, type Factors, type FormState } from './form'
import { ownField } from './upload'

/** The step of the form that has each field of a request. A refusal of the server is shown there. */
export const FIELD_STEP: Record<string, number> = {
  cell_type: 1,
  region: 1,
  own: 1,
  contacts: 1,
  factors: 2,
  loops: 2,
  sv_set: 3,
  sample: 3,
  vcf_upload: 3,
  background: 4,
  expression: 5,
  n_models: 6,
  overrides: 6,
}

/**
 * The form as the multipart body of POST /api/jobs.
 * @param fresh starts a job of its own, also when the site holds a job of the same request. The
 *   form asks so. An example does not and opens the held job, the same for every visitor.
 */
export function toFormData(form: FormState, overrides: Record<string, string>, fresh = true): FormData {
  const d = new FormData()
  d.append('cell_type', form.cell)
  d.append('region', form.region)
  d.append('factors', form.factors)
  d.append('background', form.background)
  if (form.expression) d.append('expression', form.expression)
  const table = form.own.expression
  if (form.expression === 'own' && table) {
    d.append('expression_upload', table.upload.id)
    d.append('expression_columns', JSON.stringify(table.columns))
    d.append('expression_kind', table.kind)
  }
  d.append('sv_set', form.svSet)
  d.append('sample', form.sample)
  d.append('haplotype', String(form.haplotype))
  d.append('n_models', String(form.structures))
  d.append('overrides', JSON.stringify(overrides))
  if (fresh) d.append('fresh', '1')
  if (form.vcfUpload) d.append('vcf_upload', form.vcfUpload.id)
  const own = form.own
  if (own.loops.length || own.anchors || own.contacts || own.enhancers || (form.cell === 'own' && (own.loopsFrom || own.enhancersFrom))) d.append('own', ownField(own))
  return d
}

/** The form filled with an example. */
export function fromExample(e: Example): FormState {
  return {
    ...EMPTY_FORM,
    cell: e.cell,
    region: e.region,
    factors: e.factors,
    svSet: e.svSet ?? '',
    sample: e.sample ?? '',
    expression: e.expression ?? '',
  }
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
