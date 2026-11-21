import { render, screen, within } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { describe, expect, it } from 'vitest'
import { checkCopy } from '../test/copy'
import { EXAMPLES } from '../model/examples'
import { Help } from './Help'

function renderHelp() {
  return render(
    <MemoryRouter>
      <Help />
    </MemoryRouter>,
  )
}

describe('Help page', () => {
  it('has a tutorial with numbered steps and links to sample output', () => {
    renderHelp()
    const tutorial = document.getElementById('tutorial')!
    expect(within(tutorial).getByRole('heading', { level: 2, name: 'Tutorial' })).toBeInTheDocument()
    expect(within(tutorial).getAllByRole('listitem').length).toBeGreaterThan(10)
    expect(within(tutorial).getByRole('link', { name: /The form with the example filled in/ })).toHaveAttribute('href', '/model?example=TNFRSF8')
    // Every example of the site is a link to its finished job.
    const output = within(tutorial).getByRole('list', { name: 'Sample output' })
    expect(within(output).getAllByRole('button').map((b) => b.textContent!.trim())).toEqual(EXAMPLES.map((e) => e.title))
  })

  it('offers a sample file for every kind of upload', () => {
    renderHelp()
    const files = within(screen.getByRole('list', { name: 'Sample files' })).getAllByRole('link')
    expect(files.map((f) => f.getAttribute('download'))).toEqual(['anchors.bed', 'ctcf_loops.bedpe', 'rnapii_loops.bedpe', 'contacts.bedpe', 'enhancers.bed', 'variants.vcf.gz', 'expression.tsv'])
    expect(files.every((f) => !!f.getAttribute('href'))).toBe(true)
  })

  it('guides through every part of a job page', () => {
    renderHelp()
    const guide = document.getElementById('reading')!
    expect(within(guide).getByRole('heading', { level: 2, name: 'Guide to the results' })).toBeInTheDocument()
    const heads = within(guide).getAllByRole('heading', { level: 3 }).map((h) => h.textContent)
    for (const part of ['Side panel', 'Structure tab', 'Genes tab', 'Contact map tab', 'Loops tab', 'Log tab', 'Files', 'Comparison of two jobs']) expect(heads).toContain(part)
  })

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
      'Gene expression, table',
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
    expect(screen.getByRole('link', { name: /All settings, from SETTINGS.md/ })).toHaveAttribute('href', 'https://github.com/SFGLab/3dgnome-ng/blob/main/SETTINGS.md')
  })

  it('keeps to the writing rules', () => {
    const { container } = renderHelp()
    expect(checkCopy(container)).toEqual([])
  })
})
