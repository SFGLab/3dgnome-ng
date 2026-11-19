import { render, screen, within } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { describe, expect, it } from 'vitest'
import { DATA_SOURCES, REFERENCES } from '../content/references'
import { checkCopy } from '../test/copy'
import { Cite } from './Cite'

function renderCite() {
  return render(
    <MemoryRouter>
      <Cite />
    </MemoryRouter>,
  )
}

describe('Cite page', () => {
  it('lists every reference outside the data table as an entry', () => {
    renderCite()
    const listed = REFERENCES.filter((r) => r.group !== 'data')
    expect(screen.getAllByRole('article')).toHaveLength(listed.length)
  })

  it('links each DOI to doi.org', () => {
    renderCite()
    for (const r of REFERENCES.filter((r) => r.doi)) {
      const links = screen.getAllByRole('link').filter((a) => a.getAttribute('href') === `https://doi.org/${r.doi}`)
      expect(links.length, r.key).toBeGreaterThan(0)
    }
  })

  it('gives each data set its accessions, linked to their archives', () => {
    renderCite()
    const row = (name: string) => screen.getByText(name).closest('tr')!
    for (const d of DATA_SOURCES) {
      for (const a of d.accessions) {
        const link = within(row(d.data)).getByRole('link', { name: a.id })
        expect(link, a.id).toHaveAttribute('href', a.url)
      }
      if (d.pending) expect(within(row(d.data)).getByText(d.pending), d.data).toHaveClass('unconfirmed')
      expect(d.accessions.length > 0 || d.pending !== undefined || d.note !== undefined, `${d.data} says why it has no accession`).toBe(true)
    }
    expect(within(row('Hi-C, GM12878')).getByRole('link', { name: '4DNFIXP4QG5B' })).toHaveAttribute('href', 'https://data.4dnucleome.org/files-processed/4DNFIXP4QG5B/')
    expect(within(row('RNAPII ChIA-PET loops and peaks, H1ESC')).getByRole('link', { name: 'ENCSR782EKZ' })).toHaveAttribute('href', 'https://www.encodeproject.org/experiments/ENCSR782EKZ/')
    expect(within(row('RNAPII ChIA-PET loops and peaks, BJ')).getByText(/stands in for HFFc6/)).toBeInTheDocument()
  })

  it('cites a listed reference for every data set with a confirmed source', () => {
    for (const d of DATA_SOURCES) if (d.ref) expect(REFERENCES.some((r) => r.key === d.ref), d.data).toBe(true)
  })

  it('names every accession once', () => {
    const ids = DATA_SOURCES.flatMap((d) => d.accessions.map((a) => a.id))
    expect(new Set(ids).size).toBe(ids.length)
  })

  it('links the HGSVC call set to its release and its paper', () => {
    renderCite()
    expect(screen.getByRole('link', { name: 'HGSVC2 release v2.0' })).toHaveAttribute('href', expect.stringContaining('data_collections/HGSVC2/release/v2.0'))
    expect(screen.getByRole('link', { name: 'Ebert et al., Science 2021' })).toHaveAttribute('href', 'https://doi.org/10.1126/science.abf7117')
    expect(document.querySelector('.unconfirmed')).toBeNull()
  })

  it('offers the primary citation with copy buttons and the source code', () => {
    renderCite()
    expect(screen.getByRole('button', { name: /Copy BibTeX/ })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /Copy text/ })).toBeInTheDocument()
    expect(screen.getByRole('link', { name: /Source code/ })).toHaveAttribute('href', 'https://github.com/SFGLab/3dgnome-ng')
  })

  it('keeps to the writing rules', () => {
    const { container } = renderCite()
    expect(checkCopy(container)).toEqual([])
  })
})
