/** The routes of a job page and of the comparison of two jobs. */
import { served } from '../base'
import { call, callText, query as q } from './http'
import type { Annotations, Compare, Contacts, Distance, ExpressionCompare, GeneDistances, GeneRow, Loop, Picks, Structure, Summary } from './types'

/**
 * Returns the whole of a symmetric map from the form the server sends, each pair of bins once. Row
 * `i` of that form holds the cells of the columns from `i` to the last.
 */
export function square(once: number[][]): number[][] {
  const n = once.length
  const m = Array.from({ length: n }, () => new Array<number>(n))
  for (let i = 0; i < n; i++)
    for (let j = i; j < n; j++) {
      const v = once[i][j - i]
      m[i][j] = v
      m[j][i] = v
    }
  return m
}

/** Returns the endpoints of one job, `kind` jobs for a shared job and p for a private one. */
export function jobApi(kind: 'jobs' | 'p', key: string) {
  const base = `/api/${kind}/${key}`
  return {
    url: `/${kind}/${key}`,
    base,
    summary: () => call<Summary>(`${base}/summary`),
    structure: (i: number) => call<Structure>(`${base}/structure/${i}`),
    annotations: () => call<Annotations>(`${base}/annotations`),
    picks: () => call<Picks>(`${base}/picks`),
    distance: (a: number, b: number) => call<Distance>(`${base}/distance?${q({ a, b })}`),
    genes: () => call<GeneRow[]>(`${base}/genes`),
    loops: () => call<Loop[]>(`${base}/loops`),
    log: () => callText(`${base}/log`),
    // The server sends each symmetric map once. The page reads whole maps.
    contacts: async (p: { region: string; res: number; upper: string; lower: string; norm: string }): Promise<Contacts> => {
      const r = await call<Contacts>(`${base}/contacts?${q(p)}`)
      return { ...r, upper: square(r.upper), lower: square(r.lower) }
    },
    fileUrl: (name: string) => served(`/${kind}/${key}/files/${name}`),
    zipUrl: served(`${base}/files.zip`),
  }
}

export type JobApi = ReturnType<typeof jobApi>

/** Returns the query of a comparison of two jobs, with the enhancers both have when `shared`. */
const pair = (a: string, b: string, shared: boolean, more: Record<string, string> = {}) => new URLSearchParams({ a, b, ...more, ...(shared ? { shared: '1' } : {}) })
export const compareJobs = (a: string, b: string, shared = false) => call<Compare>(`/api/compare?${pair(a, b, shared)}`)
/** Returns the enhancer to promoter distance of one gene in each structure of two jobs. */
export const compareGene = (a: string, b: string, gene: string, shared = false) => call<GeneDistances>(`/api/compare/gene?${pair(a, b, shared, { gene })}`)

export const compareExpression = (a: string, b: string, shared = false) => call<ExpressionCompare>(`/api/compare/expression?${pair(a, b, shared)}`)
