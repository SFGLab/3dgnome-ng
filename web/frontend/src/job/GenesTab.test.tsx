import { fireEvent, render, screen, within } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'
import { checkCopy } from '../test/copy'
import { GENES } from '../test/job'
import type { GeneRow } from '../api'
import { GenesTab } from './GenesTab'

const ROWS: GeneRow[] = [
  ...GENES,
  { ...GENES[1], gene: 'PLOD1', distance: 2.9, joined_by: '', min: 2.0, max: 3.8 },
  { ...GENES[1], gene: 'NPPA', distance: 5.0, joined_by: 'CTCF and RNAPII' },
]

function renderTab(props: Partial<Parameters<typeof GenesTab>[0]> = {}) {
  return render(<GenesTab rows={ROWS} chrom="chr1" structures={5} epUrl="/jobs/k/files/ep.tsv" filter="" onSelect={vi.fn()} {...props} />)
}

const names = () => within(screen.getByRole('table')).getAllByRole('row').slice(1).map((r) => within(r).getAllByRole('cell')[0].textContent)

describe('GenesTab', () => {
  it('has the columns of the design', () => {
    renderTab()
    expect(within(screen.getByRole('table')).getAllByRole('columnheader').map((h) => h.textContent)).toEqual([
      'Gene', 'Nearest enhancer', 'Separation', '3D distance', 'Over 5 structures', 'Loop between them',
    ])
  })

  it('offers the table as a CSV file beside the file of the engine', () => {
    renderTab()
    expect(screen.getByRole('button', { name: 'Save the table as genes.csv' })).toBeInTheDocument()
    expect(screen.getByRole('link', { name: /ep\.tsv/ })).toBeInTheDocument()
  })

  it('shows the expression of each gene when the job has it, in its unit', () => {
    renderTab()
    expect(screen.queryByRole('columnheader', { name: /Expression/ })).toBeNull()
    document.body.innerHTML = ''
    renderTab({ rows: ROWS.map((r, i) => ({ ...r, expression: i === 0 ? 312.4 : i === 1 ? 4.25 : null })), expressionUnit: 'TPM' })
    const head = screen.getByRole('columnheader', { name: /Expression, TPM/ })
    const column = screen.getAllByRole('columnheader').indexOf(head)
    const cells = () => screen.getAllByRole('row').slice(1).map((row) => within(row).getAllByRole('cell')[column].textContent)
    // A level of 100 or more has no decimals and a gene without a level an empty cell.
    expect(cells().filter(Boolean).sort()).toEqual(['312', '4.3'].sort())
    fireEvent.click(head)
    expect(cells()[cells().length - 1]).toBe('312')
  })

  it('sorts by 3D distance and by a column header', () => {
    renderTab()
    expect(names()).toEqual(['DHRS3', 'PLOD1', 'NPPA', 'TNFRSF8'])
    fireEvent.click(screen.getByRole('columnheader', { name: /Gene/ }))
    expect(names()).toEqual(['DHRS3', 'NPPA', 'PLOD1', 'TNFRSF8'])
    fireEvent.click(screen.getByRole('columnheader', { name: /Separation/ }))
    expect(names()[0]).toBe('DHRS3')
  })

  it('pages a long gene list and starts again at the first page when filtered', () => {
    const many: GeneRow[] = Array.from({ length: 75 }, (_, i) => ({ ...GENES[0], gene: `G${String(i).padStart(3, '0')}`, distance: i }))
    renderTab({ rows: many })
    expect(names()).toHaveLength(50)
    fireEvent.click(screen.getByRole('button', { name: 'Next' }))
    expect(screen.getByText('genes 51 to 75 of 75')).toBeInTheDocument()
    fireEvent.change(screen.getByPlaceholderText('Filter genes'), { target: { value: 'G0' } })
    expect(screen.getByText('genes 1 to 50 of 75')).toBeInTheDocument()
  })

  it('says when a promoter anchors loops that do not reach its enhancer and gives short separations in bp', () => {
    renderTab({ rows: [{ ...GENES[1], gene: 'CYB561D1', linear_bp: 352, joined_by: '', promoter_loops: 21 }] })
    const cells = within(screen.getAllByRole('row')[1]).getAllByRole('cell')
    expect(cells[2].textContent).toBe('352 bp')
    expect(cells[5].textContent).toBe('none, the promoter anchors 21 loops')
  })

  it('filters by name and by the loop joining the pair', () => {
    renderTab()
    fireEvent.change(screen.getByPlaceholderText('Filter genes'), { target: { value: 'pl' } })
    expect(names()).toEqual(['PLOD1'])
    fireEvent.change(screen.getByPlaceholderText('Filter genes'), { target: { value: '' } })
    fireEvent.click(screen.getByRole('button', { name: 'Joined by a loop' }))
    expect(names()).toEqual(['DHRS3', 'NPPA', 'TNFRSF8'])
    fireEvent.click(screen.getByRole('button', { name: 'Joined by RNAPII only' }))
    expect(names()).toEqual(['TNFRSF8'])
  })

  it('starts filtered when opened from a gene', () => {
    renderTab({ filter: 'TNFRSF8' })
    expect(names()).toEqual(['TNFRSF8'])
  })

  it('draws a range strip per row and labels the loop', () => {
    renderTab()
    const row = within(screen.getByRole('table')).getAllByRole('row')[1]
    expect(row.querySelector('.range .whisker')).not.toBeNull()
    expect(row.querySelector('.range .box')).not.toBeNull()
    expect(within(row).getByText('CTCF loop')).toBeInTheDocument()
  })

  it('states the medians in the histogram card', () => {
    renderTab()
    expect(screen.getByText('The median is 4.0. It is 5.0 for genes joined to their enhancer by a loop and 2.9 for the other genes.')).toBeInTheDocument()
  })

  it('selects a gene and links ep.tsv', () => {
    const onSelect = vi.fn()
    renderTab({ onSelect })
    fireEvent.click(screen.getByText('TNFRSF8'))
    expect(onSelect).toHaveBeenCalledWith('TNFRSF8')
    expect(screen.getByRole('link', { name: /ep\.tsv/ })).toHaveAttribute('href', '/jobs/k/files/ep.tsv')
  })

  it('keeps to the writing rules', () => {
    const { container } = renderTab()
    expect(checkCopy(container)).toEqual([])
  })
})
