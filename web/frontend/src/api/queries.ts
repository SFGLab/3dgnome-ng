/**
 * One definition for each question the pages ask the server: its key in the cache and its call.
 * A page spreads a definition into useQuery and adds what is its own, such as when to ask again.
 */
import { queryOptions } from '@tanstack/react-query'
import type { ApiError } from './http'
import { compareExpression, compareGene, compareJobs, type JobApi } from './jobs'
import { api } from './site'
import type { Region } from './types'
import { uploadPreview } from './uploads'

/** The questions of the form, of the lists of jobs and of the comparison of two jobs. */
export const asked = {
  catalog: () => queryOptions({ queryKey: ['catalog'], queryFn: api.catalog }),
  region: (cell: string, text: string) => queryOptions<Region, ApiError>({ queryKey: ['region', cell, text], queryFn: () => api.region(cell, text) }),
  geneHits: (cell: string, text: string) => queryOptions({ queryKey: ['gene-hits', cell, text], queryFn: () => api.genes(cell, text) }),
  preview: (cell: string, span: string) => queryOptions({ queryKey: ['preview', cell, span], queryFn: () => api.preview(cell, span) }),
  settings: (cell: string) => queryOptions({ queryKey: ['settings', cell], queryFn: () => api.settings(cell) }),
  sampleTable: (set: string) => queryOptions({ queryKey: ['sample-table', set], queryFn: () => api.sampleTable(set) }),
  expression: (cell: string, svSet: string, sample: string) => queryOptions({ queryKey: ['expression', cell, svSet, sample], queryFn: () => api.expression(cell, svSet, sample) }),
  variants: (fields: Record<string, string>) => queryOptions({ queryKey: ['sv-check', fields], queryFn: () => api.variants(fields) }),
  status: (url: string) => queryOptions({ queryKey: ['status', url], queryFn: () => api.status(url) }),
  uploadPreview: (id: string, cell: string, span: string) => queryOptions({ queryKey: ['upload-preview', id, span], queryFn: () => uploadPreview(id, cell, span) }),
  compare: (a: string, b: string, shared: boolean) => queryOptions({ queryKey: ['compare', a, b, shared], queryFn: () => compareJobs(a, b, shared) }),
  compareGene: (a: string, b: string, gene: string, shared: boolean) => queryOptions({ queryKey: ['compare-gene', a, b, gene, shared], queryFn: () => compareGene(a, b, gene, shared) }),
  compareExpression: (a: string, b: string, shared: boolean) => queryOptions({ queryKey: ['compare-expression', a, b, shared], queryFn: () => compareExpression(a, b, shared) }),
}

/** The questions about one job. */
export const askedOf = (job: JobApi) => ({
  summary: () => queryOptions({ queryKey: ['summary', job.base], queryFn: job.summary }),
  structure: (i: number) => queryOptions({ queryKey: ['structure', job.base, i], queryFn: () => job.structure(i) }),
  annotations: () => queryOptions({ queryKey: ['annotations', job.base], queryFn: job.annotations }),
  genes: () => queryOptions({ queryKey: ['genes', job.base], queryFn: job.genes }),
  loops: () => queryOptions({ queryKey: ['loops', job.base], queryFn: job.loops }),
  log: () => queryOptions({ queryKey: ['log', job.base], queryFn: job.log }),
  distance: (a: number, b: number) => queryOptions({ queryKey: ['distance', job.base, a, b], queryFn: () => job.distance(a, b) }),
  contacts: (p: { region: string; res: number; upper: string; lower: string; norm: string }) =>
    queryOptions({ queryKey: ['contacts', job.base, p.region, p.res, p.upper, p.lower, p.norm], queryFn: () => job.contacts(p) }),
})
