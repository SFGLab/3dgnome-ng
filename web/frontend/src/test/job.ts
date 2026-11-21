/** Fixed answers of the job endpoints for tests. */
import type { Structure, Summary } from '../api'

export const KEY = 'abcdef0123456789'

export const SUMMARY_DONE: Summary = {
  state: 'done', stage: 'done', error: null, log_tail: null, ahead: null,
  started: 1000, finished: 1372, region: 'chr1:11500000-12600000',
  request: { cell_type: 'GM12878', region: 'chr1:11,500,000-12,600,000', factors: 'CTCF+RNAPII', background: 'hic', sv_set: '', sample: '', haplotype: 0, n_models: 100 },
  loops: { CTCF: 118, RNAPII: 64 }, n_svs: 0, structures: 100, genes: 312, loop_count: 182,
  files: [{ name: 'ep.tsv', size: 38000 }, { name: 'marks.json', size: 6000 }, { name: 'engine.log', size: 22000 }],
  models: Array.from({ length: 100 }, (_, i) => ({ name: `chr1_11500000_12600000_s${i + 1}.cif`, size: 412000 })),
  skipped_svs: 0, expression: null, stages: [], now: 1372, structures_done: 100, structures_asked: 100, log_recent: '',
}

export const PICKS = { medoid: 17, compact: 42, random: 5 }

export const STRUCTURE: Structure = {
  chain: [
    [0, 0, 0, 12_063_000],
    [5, 0, 0, 12_200_000],
    [9, 0, 0, 12_449_000],
  ],
  beads: [
    [12_062_500, 12_063_500, 0, 0, 0, 'anchor'],
    [12_199_500, 12_200_500, 5, 0, 0, 'subanchor'],
    [12_448_500, 12_449_500, 9, 0, 0, 'anchor'],
  ],
  marks: { promoters: [{ bead: 0, genes: ['TNFRSF8'] }], enhancers: [2] },
  anchors: [],
}

export const ANNOTATIONS = { genes: [{ name: 'TNFRSF8', tss: 12_063_302 }], enhancers: [[12_448_194, 12_450_774]], loops: [] }

export const DISTANCE = {
  a: { bead: 0, bp: 12_063_000, offset: -302 },
  b: { bead: 2, bp: 12_449_000, offset: 806 },
  values: [5.9, 6.1, 8.8, 4.1, 9.8],
  mean: 6.7,
  sd: 1.9,
  n: 5,
}

export const GENES = [
  {
    gene: 'TNFRSF8', gene_id: 'ENSG00000120949', expression: null, promoter_loops: 1, tss: 12_063_302, enhancer: 12_449_484, linear_bp: 385_000, distance: 6.7, promoter_bead: 0, enhancer_bead: 2,
    min: 4.1, q1: 5.9, median: 6.1, q3: 8.8, max: 9.8, sd: 1.9, n: 5, joined_by: 'RNAPII',
  },
  {
    gene: 'DHRS3', gene_id: 'ENSG00000162496', expression: null, promoter_loops: 1, tss: 12_540_102, enhancer: 12_568_000, linear_bp: 28_000, distance: 2.1, promoter_bead: 1, enhancer_bead: 2,
    min: 1.5, q1: 1.9, median: 2.1, q3: 2.3, max: 2.9, sd: 0.4, n: 5, joined_by: 'CTCF',
  },
]
