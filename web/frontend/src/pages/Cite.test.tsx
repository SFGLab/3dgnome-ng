import { render, screen } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { describe, expect, it } from 'vitest'
import { REFERENCES } from '../content/references'
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

  it('marks the unconfirmed reference', () => {
    renderCite()
    expect(screen.getByText('HGSVC, reference to confirm')).toHaveClass('unconfirmed')
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
