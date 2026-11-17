/** The site's JSON API. */

export type CellType = { label: string; description: string; data: string[]; lymphoblastoid: boolean; note: string }
export type Catalog = {
  cell_types: Record<string, CellType>
  sv_sets: Record<string, { label: string; samples: string[] }>
  factors: string[]
  max_region_mb: number
  max_structures: number
  hicdiffusion: boolean
}
export type Region = { chrom: string; start: number; end: number; label: string; gene?: string }
export type Preview = { loops: Record<string, [number, number, number][]>; counts: Record<string, number>; genes: { name: string; tss: number }[] }
export type Setting = {
  section: string
  key: string
  type: string
  default: string
  server: string
  lo: number | null
  hi: number | null
  step: number | null
  choices: string[] | null
  meaning: string
}
export type SettingGroup = { name: string; settings: Setting[] }
export type VariantRow = { type: string; pos: number; end: number; length: number; genotype: string; change: string }
export type VariantCheck = { summary: string; applied: Record<string, number>; skipped: Record<string, number>; variants: VariantRow[] }
export type JobStatus = { state: string; stage?: string; error?: string }

/** An answer of the API that is not 2xx, with its message and the form field it concerns. */
export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
    readonly field?: string,
  ) {
    super(message)
  }
}

async function call<T>(path: string, init?: RequestInit): Promise<T> {
  const r = await fetch(path, init)
  const body = await r.json().catch(() => ({}))
  if (!r.ok) throw new ApiError(body.error ?? `${r.status} from ${path}`, r.status, body.field)
  return body as T
}

const q = (params: Record<string, string>) => new URLSearchParams(params).toString()

export const api = {
  catalog: () => call<Catalog>('/api/catalog'),
  region: (cell: string, text: string) => call<Region>(`/api/region?${q({ cell, q: text })}`),
  preview: (cell: string, region: string) => call<Preview>(`/api/preview?${q({ cell, region })}`),
  settings: (cell: string) => call<SettingGroup[]>(`/api/settings?${q({ cell })}`),
  parseIni: (file: File) => {
    const form = new FormData()
    form.append('file', file)
    return call<{ overrides: Record<string, string>; refused: string[] }>('/api/settings/ini', { method: 'POST', body: form })
  },
  submit: (form: FormData) => call<{ url: string }>('/api/jobs', { method: 'POST', body: form }),
  variants: (fields: Record<string, string>) => {
    const form = new FormData()
    for (const [k, v] of Object.entries(fields)) form.append(k, v)
    return call<VariantCheck>('/api/variants/check', { method: 'POST', body: form })
  },
  status: (url: string) => call<JobStatus>(`/api${url}`),
}
