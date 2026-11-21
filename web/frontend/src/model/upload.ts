import { ApiError } from '../api'
import { served } from '../base'

export type UploadKind = 'loops' | 'anchors' | 'contacts' | 'enhancers' | 'vcf' | 'expression'
export type CheckError = { line: number | null; found: string; fix: string }
export type UploadResult = {
  id: string
  kind: UploadKind
  name: string
  size: number
  ok: boolean
  summary: string
  error: CheckError | null
  samples: string[]
  /** An expression table of whole numbers, which read counts are. */
  counts?: boolean
}

/** An uploaded expression table, the samples of it the job takes and what its values are. */
export type OwnExpression = { upload: UploadResult; columns: string[]; kind: 'counts' | 'values' }

/**
 * The uploads a job uses in place of the data of a listed cell type, and its expression table. On your own cell type
 * loopsFrom and enhancersFrom give a listed cell type whose loops or enhancers it takes,
 * enhancersFrom none for no enhancers.
 */
export type OwnChoice = {
  loops: { factor: string; upload: UploadResult }[]
  anchors: UploadResult | null
  contacts: UploadResult | null
  enhancers: UploadResult | null
  loopsFrom?: string
  enhancersFrom?: string
  /** An expression table with the samples the job takes from it. */
  expression?: OwnExpression | null
}

export const NO_OWN: OwnChoice = { loops: [], anchors: null, contacts: null, enhancers: null }

/** The data of the own data card, named as the data of a listed cell type is: each factor of the
 * loops, the contacts and the enhancers. A set taken from a listed cell type has its name. */
export function ownData(own: OwnChoice): string[] {
  return [
    ...own.loops.map((l) => l.factor),
    ...(own.loopsFrom ? [`${own.loopsFrom} loops`] : []),
    ...(own.contacts ? ['contacts'] : []),
    ...(own.enhancers ? ['enhancers'] : own.enhancersFrom && own.enhancersFrom !== 'none' ? [`${own.enhancersFrom} enhancers`] : []),
    ...(own.expression ? ['RNA-seq'] : []),
  ]
}

/** Sends one file to POST /api/uploads, which stores it and checks it against the region. */
export async function uploadFile(kind: UploadKind, file: File, where: { cell: string; region: string; factor?: string }): Promise<UploadResult> {
  const form = new FormData()
  form.append('kind', kind)
  form.append('file', file)
  form.append('cell', where.cell)
  form.append('region', where.region)
  if (where.factor) form.append('factor', where.factor)
  const r = await fetch(served('/api/uploads'), { method: 'POST', body: form })
  const body = await r.json().catch(() => ({}))
  if (!r.ok) throw new ApiError(body.error ?? `${r.status} from /api/uploads`, r.status, body.field)
  return body as UploadResult
}

/** The loops of a loops upload inside a region, as anchor midpoints and the count. */
export async function uploadPreview(id: string, cell: string, region: string): Promise<[number, number, number][]> {
  const r = await fetch(served(`/api/uploads/${id}/preview?${new URLSearchParams({ cell, region })}`))
  return r.ok ? r.json() : []
}

/** The uploads as the JSON field own of POST /api/jobs. */
export function ownField(own: OwnChoice): string {
  return JSON.stringify({
    loops: own.loops.map((l) => ({ factor: l.factor, upload: l.upload.id })),
    anchors: own.anchors?.id ?? null,
    contacts: own.contacts?.id ?? null,
    enhancers: own.enhancers?.id ?? null,
    loops_from: own.loops.length ? undefined : own.loopsFrom,
    enhancers_from: own.enhancers ? undefined : own.enhancersFrom,
  })
}

/** The factor a loops file most likely holds, read from its name and CTCF for the first file. */
export function guessFactor(name: string, index: number): string {
  const n = name.toUpperCase()
  if (/RNAP|POL2|POLII/.test(n)) return 'RNAPII'
  if (n.includes('CTCF')) return 'CTCF'
  return index === 0 ? 'CTCF' : `factor ${index + 1}`
}

/** A file size in MB, in kB below a tenth of a MB and in GB from one GB. */
export const megabytes = (bytes: number) =>
  bytes < 104_858 ? `${Math.max(1, Math.round(bytes / 1024))} kB` : bytes < 1_073_741_824 ? `${(bytes / 1_048_576).toFixed(1)} MB` : `${(bytes / 1_073_741_824).toFixed(1)} GB`

/** A link the server refused, with the way to fix it. */
export class LinkRefused extends Error {
  constructor(
    message: string,
    readonly fix: string,
  ) {
    super(message)
  }
}

/** Asks the server to read the region of a remote .hic or .mcool file into a contacts upload. */
export async function uploadLink(url: string, where: { cell: string; region: string }): Promise<UploadResult> {
  const form = new FormData()
  form.append('url', url)
  form.append('cell', where.cell)
  form.append('region', where.region)
  const r = await fetch(served('/api/uploads/link'), { method: 'POST', body: form })
  const body = await r.json().catch(() => ({}))
  if (!r.ok) throw new LinkRefused(body.error ?? `${r.status} from /api/uploads/link`, body.fix ?? '')
  return body as UploadResult
}
