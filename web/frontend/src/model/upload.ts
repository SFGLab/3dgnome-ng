import { ApiError } from '../api'

export type UploadKind = 'loops' | 'anchors' | 'contacts' | 'enhancers' | 'vcf'
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
}

/** The uploads a job uses in place of the data of a listed cell type. */
export type OwnChoice = {
  loops: { factor: string; upload: UploadResult }[]
  anchors: UploadResult | null
  contacts: UploadResult | null
  enhancers: UploadResult | null
}

export const NO_OWN: OwnChoice = { loops: [], anchors: null, contacts: null, enhancers: null }

/** Sends one file to POST /api/uploads, which stores it and checks it against the region. */
export async function uploadFile(kind: UploadKind, file: File, where: { cell: string; region: string; factor?: string }): Promise<UploadResult> {
  const form = new FormData()
  form.append('kind', kind)
  form.append('file', file)
  form.append('cell', where.cell)
  form.append('region', where.region)
  if (where.factor) form.append('factor', where.factor)
  const r = await fetch('/api/uploads', { method: 'POST', body: form })
  const body = await r.json().catch(() => ({}))
  if (!r.ok) throw new ApiError(body.error ?? `${r.status} from /api/uploads`, r.status, body.field)
  return body as UploadResult
}

/** The loops of a loops upload inside a region, as anchor midpoints and the count. */
export async function uploadPreview(id: string, cell: string, region: string): Promise<[number, number, number][]> {
  const r = await fetch(`/api/uploads/${id}/preview?${new URLSearchParams({ cell, region })}`)
  return r.ok ? r.json() : []
}

/** The uploads as the JSON field own of POST /api/jobs. */
export function ownField(own: OwnChoice): string {
  return JSON.stringify({
    loops: own.loops.map((l) => ({ factor: l.factor, upload: l.upload.id })),
    anchors: own.anchors?.id ?? null,
    contacts: own.contacts?.id ?? null,
    enhancers: own.enhancers?.id ?? null,
  })
}

/** The factor a loops file most likely holds, read from its name, and CTCF for the first file. */
export function guessFactor(name: string, index: number): string {
  const n = name.toUpperCase()
  if (/RNAP|POL2|POLII/.test(n)) return 'RNAPII'
  if (n.includes('CTCF')) return 'CTCF'
  return index === 0 ? 'CTCF' : `factor ${index + 1}`
}

/** A file size in MB, or in kB below a tenth of a MB. */
export const megabytes = (bytes: number) =>
  bytes < 104_858 ? `${Math.max(1, Math.round(bytes / 1024))} kB` : `${(bytes / 1_048_576).toFixed(1)} MB`
