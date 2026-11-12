/** Fixed answers of the job endpoints for tests. */
export const KEY = 'abcdef0123456789'

export const SUMMARY_DONE = {
  state: 'done', stage: 'done', error: null, log_tail: null, ahead: null,
  started: 1000, finished: 1372, region: 'chr1:11500000-12600000',
  request: { cell_type: 'GM12878', region: 'chr1:11,500,000-12,600,000', factors: 'CTCF+RNAPII', background: 'hic', sv_set: '', sample: '', haplotype: 0, n_models: 100 },
  loops: { CTCF: 118, RNAPII: 64 }, n_svs: 0, structures: 100, genes: 312, loop_count: 182,
  files: [{ name: 'ep.tsv', size: 38000 }, { name: 'marks.json', size: 6000 }, { name: 'engine.log', size: 22000 }],
  models: Array.from({ length: 100 }, (_, i) => ({ name: `chr1_11500000_12600000_s${i + 1}.cif`, size: 412000 })),
}

export const PICKS = { medoid: 17, compact: 42, random: 5 }
