import { describe, expect, it } from 'vitest'
import { DATA_SOURCES, REFERENCES } from './references'

describe('references', () => {
  it('has unique keys', () => {
    const keys = REFERENCES.map((r) => r.key)
    expect(new Set(keys).size).toBe(keys.length)
  })

  it('gives every confirmed entry a DOI or a URL', () => {
    for (const r of REFERENCES.filter((r) => !r.unconfirmed)) {
      expect(r.doi || r.url, r.key).toBeTruthy()
    }
  })

  it('writes DOIs in their plain form', () => {
    for (const r of REFERENCES.filter((r) => r.doi)) {
      expect(r.doi, r.key).toMatch(/^10\.\d{4,9}\/\S+$/)
    }
  })

  it('includes the papers the site rests on', () => {
    const keys = new Set(REFERENCES.map((r) => r.key))
    for (const k of ['gnome2016', 'gnome2', 'gnome3', 'cudammc', 'enhancer3d', 'loopsage', 'multimm', 'sadowski2019']) {
      expect(keys, k).toContain(k)
    }
  })

  it('marks the HGSVC freeze 4 reference as unconfirmed', () => {
    expect(REFERENCES.find((r) => r.key === 'hgsvc4')?.unconfirmed).toBe(true)
  })

  it('points every data row at a reference', () => {
    const keys = new Set(REFERENCES.map((r) => r.key))
    for (const d of DATA_SOURCES) expect(keys, d.data).toContain(d.ref)
  })
})
