import { api } from '../api'
import type { Example } from './examples'
import type { Factors } from './form'
import { rememberJob } from './recent'

/** Structures an example job asks for. Ten keep the first run short; the server's cache serves
 * every later visitor the same job. */
export const EXAMPLE_STRUCTURES = 10

function request(e: Example, factors: Factors): FormData {
  const d = new FormData()
  d.append('cell_type', e.cell)
  d.append('region', e.region)
  d.append('factors', factors)
  d.append('background', 'hic')
  d.append('sv_set', e.svSet ?? '')
  d.append('sample', e.sample ?? '')
  d.append('haplotype', '0')
  d.append('n_models', String(EXAMPLE_STRUCTURES))
  d.append('overrides', '{}')
  return d
}

/** Submits an example's request, which the server answers with the shared job, finished or
 * running and returns the page to open: the job on its tab, or with the comparison. */
export async function openExample(e: Example): Promise<string> {
  // A compare example's two jobs are named by their loop sets, not by the example's title.
  const named = (f: Factors) => `${e.cellLabel} ${e.region}, ${f === 'CTCF+RNAPII' ? 'CTCF and RNAPII' : f} loops`
  const { url } = await api.submit(request(e, e.factors))
  rememberJob({ url, label: e.compareFactors ? named(e.factors) : `${e.cellLabel} ${e.title}`, name: '', at: Date.now() })
  if (e.compareFactors) {
    const other = await api.submit(request(e, e.compareFactors))
    rememberJob({ url: other.url, label: named(e.compareFactors), name: '', at: Date.now() })
    return `${url}?${new URLSearchParams({ compare: other.url, ...(e.gene ? { gene: e.gene } : {}) })}`
  }
  const q = new URLSearchParams({ ...(e.tab ? { tab: e.tab } : {}), ...(e.gene ? { gene: e.gene } : {}) }).toString()
  return q ? `${url}?${q}` : url
}
