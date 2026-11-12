import { vi } from 'vitest'

export const CATALOG = {
  cell_types: {
    GM12878: { label: 'GM12878, lymphoblastoid', description: 'lymphoblastoid', data: ['CTCF', 'RNAPII', 'Hi-C', 'enhancers'], lymphoblastoid: true, note: '' },
    H1ESC: { label: 'H1ESC, embryonic stem cell', description: 'embryonic stem cell', data: ['CTCF', 'RNAPII', 'Hi-C', 'Micro-C'], lymphoblastoid: false, note: '' },
  },
  sv_sets: { HGSVC: { label: 'HGSVC freeze 4', samples: ['HG00512', 'HG00733'] } },
  factors: ['CTCF', 'RNAPII', 'CTCF+RNAPII'],
  max_region_mb: 20,
  max_structures: 100,
  hicdiffusion: true,
}

export const PREVIEW = {
  loops: { CTCF: [[11600000, 11900000, 7], [12000000, 12400000, 5]], RNAPII: [[11700000, 12300000, 4]] },
  counts: { CTCF: 2, RNAPII: 1 },
  genes: [{ name: 'TNFRSF8', tss: 12063303 }],
}

export const SCHEMA = [
  {
    name: 'Distance law',
    settings: [
      { section: 'distance', key: 'polymer_exponent', type: 'float', default: '0.0', server: '0.0', lo: 0, hi: 0.6, step: 0.01, choices: null, meaning: 'Zero measures the exponent.' },
      { section: 'distance', key: 'contact_half_saturation', type: 'float', default: '1.0', server: '1.0', lo: 0.1, hi: 5, step: 0.1, choices: null, meaning: 'Loop strength at half pull.' },
    ],
  },
  {
    name: 'Chains',
    settings: [{ section: 'simulation_arcs_smooth', key: 'hard_wall', type: 'bool', default: 'no', server: 'yes', lo: null, hi: null, step: null, choices: ['yes', 'no'], meaning: 'Reject moves into overlap.' }],
  },
]

type Route = [RegExp, (url: URL, init?: RequestInit) => [number, unknown]]

/** Replaces fetch with fixed answers for the site's API. Returns the mock to inspect calls. */
export function mockApi(extra: Route[] = []) {
  const routes: Route[] = [
    ...extra,
    [/^\/api\/catalog$/, () => [200, CATALOG]],
    [/^\/api\/region$/, (u) => {
      const q = (u.searchParams.get('q') ?? '').toUpperCase()
      if (q === 'TNFRSF8') return [200, { chrom: 'chr1', start: 11500000, end: 12600000, label: 'chr1:11,500,000-12,600,000, 1.1 Mb', gene: 'TNFRSF8' }]
      if (q.startsWith('CHR8')) return [200, { chrom: 'chr8', start: 127500000, end: 129000000, label: 'chr8:127,500,000-129,000,000, 1.5 Mb' }]
      return [400, { error: `'${q}' is neither chr:start-end nor a known gene name` }]
    }],
    [/^\/api\/preview$/, () => [200, PREVIEW]],
    [/^\/api\/settings$/, () => [200, SCHEMA]],
    [/^\/api\/jobs\/[\w-]+$/, () => [200, { state: 'done', stage: 'done' }]],
  ]
  const fn = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
    const url = new URL(String(input), 'http://site')
    const hit = routes.find(([re]) => re.test(url.pathname))
    const [status, body] = hit ? hit[1](url, init) : [404, { error: 'not found' }]
    return new Response(JSON.stringify(body), { status, headers: { 'content-type': 'application/json' } })
  })
  vi.stubGlobal('fetch', fn)
  return fn
}
