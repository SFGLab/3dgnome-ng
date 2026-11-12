import { render, screen, within } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { describe, expect, it } from 'vitest'
import { checkCopy } from '../test/copy'
import { Help } from './Help'

function renderHelp() {
  return render(
    <MemoryRouter>
      <Help />
    </MemoryRouter>,
  )
}

describe('Help page', () => {
  it('lists every input format', () => {
    renderHelp()
    const table = screen.getByRole('table', { name: 'Input formats' })
    const files = within(table).getAllByRole('row').slice(1).map((r) => within(r).getAllByRole('cell')[0].textContent)
    expect(files).toEqual([
      'Loop anchors, BED',
      'Loops, BEDPE',
      'Contacts, BEDPE',
      'Genes, GTF',
      'Enhancers, BED',
      'Segment split, BED',
      'Centromeres, BED',
      'Structural variants, VCF',
    ])
    expect(within(table).getByText(/An uploaded .hic or .mcool map is converted to this format/)).toBeInTheDocument()
  })

  it('says the 10 column loop files are not read', () => {
    renderHelp()
    expect(screen.getByText(/The 10 column loop files of 3D-GNOME 2.0 and 3.0/)).toBeInTheDocument()
  })

  it('shows the examples as code', () => {
    renderHelp()
    expect(screen.getByText(/chr1\s+12062800\s+12064300\s+R/)).toBeInTheDocument()
    expect(screen.getByText(/SVTYPE=DEL;END=12118400/)).toBeInTheDocument()
  })

  it('has no section for administrators', () => {
    renderHelp()
    expect(screen.queryByText(/administrator|admin/i)).toBeNull()
  })

  it('links the full settings reference', () => {
    renderHelp()
    expect(screen.getByRole('link', { name: /All 177 settings/ })).toHaveAttribute('href', 'https://github.com/SFGLab/3dgnome-ng/blob/main/SETTINGS.md')
  })

  it('keeps to the writing rules', () => {
    const { container } = renderHelp()
    expect(checkCopy(container)).toEqual([])
  })
})
