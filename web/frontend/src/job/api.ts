import { ApiError } from '../api'

export type FileInfo = { name: string; size: number }
export type Summary = {
  state: 'queued' | 'running' | 'done' | 'failed'
  stage: string | null
  error: string | null
  log_tail: string | null
  ahead: number | null
  started: number | null
  finished: number | null
  region: string | null
  request: {
    cell_type: string
    region: string
    factors: string
    background?: string
    hicdiffusion?: boolean
    sv_set: string
    sample: string
    haplotype: number
    n_models: number
    own?: { loops: string[]; anchors: boolean; contacts: boolean; enhancers: boolean; loops_from?: string; enhancers_from?: string }
  }
  loops: Record<string, number>
  n_svs: number | null
  structures: number
  stages?: { stage: string; at: number }[]
  structures_done?: number
  structures_asked?: number | null
  log_recent?: string
  genes: number
  loop_count: number
  files: FileInfo[]
  models: FileInfo[]
}
export type Structure = {
  chain: [number, number, number, number][]
  beads: [number, number, number, number, number, 'anchor' | 'subanchor'][]
  marks: { promoters: { bead: number; genes: string[] }[]; enhancers: number[] }
  anchors: { bead: number; start: number; end: number; loops: { factor: string; other: [number, number]; pets: number }[] }[]
}
export type Annotations = {
  genes: { name: string; tss: number }[]
  enhancers: [number, number][]
  loops: Loop[]
}
export type Loop = { factor: string; chrom: string; a: [number, number]; b: [number, number]; span: number; pets: number }
export type Distance = { a: { bead: number; bp: number; offset: number }; b: { bead: number; bp: number; offset: number }; values: number[]; mean: number; sd: number; n: number }
export type GeneRow = {
  gene: string
  tss: number
  enhancer: number
  linear_bp: number
  distance: number
  promoter_bead: number
  enhancer_bead: number
  min: number
  q1: number
  median: number
  q3: number
  max: number
  sd: number
  n: number
  joined_by: string
  promoter_loops?: number
}
export type Contacts = {
  bins: number[]
  upper: number[][]
  lower: number[][]
  upper_label: string
  lower_label: string
  scc: number | null
  scc_about?: string
  loop_enrichment?: number | null
}
export type Compare = { same_region: boolean; rows: { gene: string; linear_bp: number; a: { mean: number; sd: number }; b: { mean: number; sd: number }; ratio: number }[] }

async function call<T>(path: string, text = false): Promise<T> {
  const r = await fetch(path)
  if (!r.ok) {
    const body = await r.json().catch(() => ({}))
    throw new ApiError(body.error ?? `${r.status} from ${path}`, r.status)
  }
  return (text ? await r.text() : await r.json()) as T
}

/** The endpoints of one job, `kind` jobs for a shared job and p for a private one. */
export function jobApi(kind: 'jobs' | 'p', key: string) {
  const base = `/api/${kind}/${key}`
  const q = (p: Record<string, string | number>) => new URLSearchParams(Object.entries(p).map(([k, v]) => [k, String(v)])).toString()
  return {
    url: `/${kind}/${key}`,
    base,
    summary: () => call<Summary>(`${base}/summary`),
    structure: (i: number) => call<Structure>(`${base}/structure/${i}`),
    annotations: () => call<Annotations>(`${base}/annotations`),
    picks: () => call<{ medoid: number; compact: number; random: number }>(`${base}/picks`),
    distance: (a: number, b: number) => call<Distance>(`${base}/distance?${q({ a, b })}`),
    genes: () => call<GeneRow[]>(`${base}/genes`),
    loops: () => call<Loop[]>(`${base}/loops`),
    log: () => call<string>(`${base}/log`, true),
    contacts: (p: { region: string; res: number; upper: string; lower: string; norm: string }) => call<Contacts>(`${base}/contacts?${q(p)}`),
    fileUrl: (name: string) => `/${kind}/${key}/files/${name}`,
    zipUrl: `${base}/files.zip`,
  }
}

export type JobApi = ReturnType<typeof jobApi>

export const compareJobs = (a: string, b: string) => call<Compare>(`/api/compare?${new URLSearchParams({ a, b })}`)
