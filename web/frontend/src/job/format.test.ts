import { describe, expect, it } from 'vitest'
import { SUMMARY_DONE } from '../test/job'
import { variants } from './format'

describe('the variants line of a job', () => {
  const sampled = { ...SUMMARY_DONE, n_svs: 69, sample_note: 'PUR, AMR, female', request: { ...SUMMARY_DONE.request, sv_set: 'HGSVC', sample: 'HG00733', haplotype: 1 } }

  it('names the sample, its note and its haplotype', () => {
    expect(variants(sampled)).toBe('69 from HG00733 (PUR, AMR, female), haplotype 2')
    expect(variants({ ...sampled, sample_note: undefined, request: { ...sampled.request, sample: '' } })).toBe('69 from uploaded VCF')
  })

  it('counts the variants of a type the site does not apply', () => {
    expect(variants({ ...sampled, skipped_svs: 3 })).toBe('69 from HG00733 (PUR, AMR, female), haplotype 2, 3 of other types skipped')
    expect(variants({ ...SUMMARY_DONE, skipped_svs: 2 })).toBe('none, 2 of other types skipped')
    expect(variants({ ...SUMMARY_DONE, skipped_svs: 0 })).toBe('none')
    expect(variants(SUMMARY_DONE)).toBe('none')
  })
})
