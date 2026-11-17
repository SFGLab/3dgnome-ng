import { describe, expect, it } from 'vitest'
import { guessFactor, megabytes, ownField } from './upload'

describe('upload helpers', () => {
  it('writes a file size in MB, and in kB under a tenth of a MB', () => {
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
    const u = { id: 'abc12345', kind: 'loops' as const, name: 'a', size: 1, ok: true, summary: '', error: null, samples: [] }
    expect(JSON.parse(ownField({ loops: [{ factor: 'CTCF', upload: u }], anchors: null, contacts: u, enhancers: null }))).toEqual({
      loops: [{ factor: 'CTCF', upload: 'abc12345' }],
      anchors: null,
      contacts: 'abc12345',
      enhancers: null,
    })
  })
})
