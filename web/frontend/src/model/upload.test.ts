import { describe, expect, it } from 'vitest'
import { guessFactor, megabytes, NO_OWN, ownData, ownField } from './upload'
import { type UploadResult } from '../api'

describe('upload helpers', () => {
  it('writes a file size in MB and in kB under a tenth of a MB', () => {
    expect(megabytes(2_200_000)).toBe('2.1 MB')
    expect(megabytes(400)).toBe('1 kB')
    expect(megabytes(40_000)).toBe('39 kB')
    expect(megabytes(41_016_250_368)).toBe('38.2 GB')
  })

  it('reads the factor from a file name, CTCF for an unnamed first file', () => {
    expect(guessFactor('GM12878_RNAPII_loops.bedpe', 0)).toBe('RNAPII')
    expect(guessFactor('loops.bedpe', 0)).toBe('CTCF')
    expect(guessFactor('hichip.txt', 2)).toBe('factor 3')
  })

  it('names each upload by its id in the job field', () => {
    const u = { id: 'abc12345', kind: 'loops' as const, name: 'a', size: 1, ok: true, summary: '', error: null, samples: [], counts: false }
    expect(JSON.parse(ownField({ loops: [{ factor: 'CTCF', upload: u }], anchors: null, contacts: u, enhancers: null }))).toEqual({
      loops: [{ factor: 'CTCF', upload: 'abc12345' }],
      anchors: null,
      contacts: 'abc12345',
      enhancers: null,
    })
  })

  it('names the data of the own data card as the data of a listed cell type is named', () => {
    const file = (kind: UploadResult['kind']): UploadResult => ({ id: kind, kind, name: `${kind}.bed`, size: 1, ok: true, summary: '', error: null, samples: [], counts: false })
    expect(ownData(NO_OWN)).toEqual([])
    expect(ownData({ ...NO_OWN, loops: [{ factor: 'CTCF', upload: file('loops') }, { factor: 'RNAPII', upload: file('loops') }], contacts: file('contacts'), enhancers: file('enhancers') })).toEqual(['CTCF', 'RNAPII', 'contacts', 'enhancers'])
    // A set taken from a listed cell type has the name of that cell type. No enhancers is no chip.
    expect(ownData({ ...NO_OWN, contacts: file('contacts'), loopsFrom: 'H1ESC', enhancersFrom: 'GM12878' })).toEqual(['H1ESC loops', 'contacts', 'GM12878 enhancers'])
    expect(ownData({ ...NO_OWN, loopsFrom: 'H1ESC', enhancersFrom: 'none' })).toEqual(['H1ESC loops'])
  })
})
