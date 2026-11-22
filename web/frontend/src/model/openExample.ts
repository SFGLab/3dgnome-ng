import { api } from '../api'
import type { Example } from './examples'
import type { Factors } from './form'
import { fromExample, toFormData } from './request'
import { nameOf, rememberJob } from '../myjobs/recent'

/**
 * Structures an example job asks for. Ten left the mean distance of a pair too uncertain: two runs of one
 * request differed by a third of the difference an example shows. The first run of an example takes minutes.
 * The server gives every later visitor the same job, the first job of the request.
 */
export const EXAMPLE_STRUCTURES = 100

/** Returns the request of an example for one loop set and one cell type, without the settings of the form. */
const request = (e: Example, factors: Factors, cell = e.cell) => toFormData({ ...fromExample(e), cell, factors, structures: EXAMPLE_STRUCTURES }, {}, false)

/**
 * Submits the request of an example, which the server answers with the shared job, finished or
 * running. Returns the page to open, the job on its tab or with the comparison.
 */
export async function openExample(e: Example): Promise<string> {
  // The two jobs of a compare example have the names of their loop sets,
  // not the title of the example.
  // The gene the job opens with and, for an example about one enhancer, that enhancer.
  const marked = { ...(e.gene ? { gene: e.gene } : {}), ...(e.gene && e.enhancer ? { enhancer: String(e.enhancer) } : {}) }
  const named = (f: Factors) => `${e.cellLabel} ${e.title}, ${f === 'CTCF+RNAPII' ? 'CTCF and RNAPII' : f} loops`
  if (e.compareCell) {
    // Two cell types on one region. Each job has the name of its cell type.
    const of = (cell: string) => `${cell} ${e.region}`
    const first = await api.submit(request(e, e.factors))
    rememberJob({ url: first.url, label: of(e.cell), name: nameOf(first.url), at: Date.now() })
    const second = await api.submit(request(e, e.factors, e.compareCell))
    rememberJob({ url: second.url, label: of(e.compareCell), name: nameOf(second.url), at: Date.now() })
    return `${first.url}?${new URLSearchParams({ compare: second.url, ...(e.gene ? { gene: e.gene } : {}) })}`
  }
  const { url } = await api.submit(request(e, e.factors))
  rememberJob({ url, label: e.compareFactors ? named(e.factors) : `${e.cellLabel} ${e.title}`, name: nameOf(url), at: Date.now() })
  if (e.compareFactors) {
    const other = await api.submit(request(e, e.compareFactors))
    rememberJob({ url: other.url, label: named(e.compareFactors), name: nameOf(other.url), at: Date.now() })
    return `${url}?${new URLSearchParams({ compare: other.url, ...marked })}`
  }
  const q = new URLSearchParams({ ...(e.tab ? { tab: e.tab } : {}), ...marked }).toString()
  return q ? `${url}?${q}` : url
}
