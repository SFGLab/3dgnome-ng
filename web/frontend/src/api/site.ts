/** The routes of the form and of the lists of jobs. */
import { call, posted, query as q } from './http'
import type { Catalog, ExpressionOption, GeneHit, JobState, Preview, Region, SampleRow, SettingGroup, VariantCheck } from './types'

export const api = {
  catalog: () => call<Catalog>('/api/catalog'),
  region: (cell: string, text: string) => call<Region>(`/api/region?${q({ cell, q: text })}`),
  genes: (cell: string, text: string) => call<GeneHit[]>(`/api/genes?${q({ cell, q: text, limit: '25' })}`),
  preview: (cell: string, region: string) => call<Preview>(`/api/preview?${q({ cell, region })}`),
  sampleTable: (set: string) =>
    call<{ columns: string[]; rows: string[][] }>(`/api/sv_sets/${encodeURIComponent(set)}/samples`).then((t) =>
      t.rows.map((r) => Object.fromEntries(t.columns.map((c, i) => [c, r[i] ?? ''])) as SampleRow),
    ),
  settings: (cell: string, chain = 'anchors') => call<SettingGroup[]>(`/api/settings?${q({ cell, chain })}`),
  expression: (cell: string, svSet: string, sample: string) =>
    call<{ options: ExpressionOption[] }>(`/api/expression?${q({ cell, sv_set: svSet, sample })}`).then((r) => r.options),
  parseIni: (file: File) => call<{ overrides: Record<string, string>; refused: string[] }>('/api/settings/ini', posted({ file })),
  submit: (form: FormData) => call<{ url: string }>('/api/jobs', { method: 'POST', body: form }),
  variants: (fields: Record<string, string>) => call<VariantCheck>('/api/variants/check', posted(fields)),
  states: (urls: string[]) =>
    call<{ states: Record<string, JobState | null> }>('/api/jobs/states', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ jobs: urls }) }).then((r) => r.states),
}
