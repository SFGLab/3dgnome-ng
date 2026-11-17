import { render, screen, within } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { describe, expect, it } from 'vitest'
import { checkCopy } from '../test/copy'
import { Results } from './Results'

function renderResults() {
  return render(
    <MemoryRouter>
      <Results />
    </MemoryRouter>,
  )
}

const values = (chart: HTMLElement) => within(chart).getAllByTestId('bar-value').map((v) => v.textContent)

describe('Results page', () => {
  it('shows the agreement with Hi-C of the original and the new models', () => {
    renderResults()
    expect(values(screen.getByRole('figure', { name: 'Agreement with Hi-C' }))).toEqual(['0.22', '0.50', '0.15', '0.36', '0.15', '0.43'])
  })

  it('shows the enhancer to promoter distances and the link to expression', () => {
    renderResults()
    expect(values(screen.getByRole('figure', { name: 'Enhancer to promoter distance' }))).toEqual(['4.3', '3.6', '4.0', '3.0', '4.6', '3.8'])
    expect(values(screen.getByRole('figure', { name: 'Link to expression' }))).toEqual(['0.19', '0.26', '0.28', '0.12', '0.18', '0.22', '0.25', '0.29', '0.31'])
  })

  it('shows the run time of the three tools with the bead count of each model', () => {
    renderResults()
    const one = screen.getByRole('figure', { name: 'One structure, 1 Mb' })
    expect(values(one)).toEqual(['6.4', '7.4', '9.0'])
    expect(within(one).getByText('cudaMMC, 350 beads')).toBeInTheDocument()
    expect(within(one).getByText('3D-GNOMEᴺᴳ, 760 beads')).toBeInTheDocument()
    expect(values(screen.getByRole('figure', { name: 'One structure, whole chromosome' }))).toEqual(['3.2', '14.2', '6.4'])
    expect(values(screen.getByRole('figure', { name: 'Ten structures, 60 Mb' }))).toEqual(['9.3', '20.1', '10.6'])
    expect(values(screen.getByRole('figure', { name: 'Three structures at equal bead counts' }))).toEqual(['10.7', '42.1', '11.2'])
    for (const name of ['One structure, 1 Mb', 'Ten structures, 60 Mb', 'One structure, whole chromosome', 'Three structures at equal bead counts']) {
      expect(values(screen.getByRole('figure', { name }))).not.toContain('placeholder')
    }
  })

  it('charts one structure by region size and by bead count, with a table of every value', () => {
    renderResults()
    for (const name of ['One structure by region size', 'One structure by bead count']) {
      const chart = screen.getByRole('figure', { name })
      expect(within(chart).getAllByRole('row')).toHaveLength(9)
      expect(within(chart).getByText('9.1 s, 662 beads')).toBeInTheDocument()
      expect(within(chart).getByText('6.4 min, 124,657 beads')).toBeInTheDocument()
      expect(within(chart).getByText('3.2 min, 50,631 beads')).toBeInTheDocument()
      expect(within(chart).getByLabelText('MultiMM, 100 Mb, 56,458 beads, 3.2 min')).toBeInTheDocument()
    }
  })

  it('compares the three tools on Hi-C and counts the runs with intervals', () => {
    renderResults()
    expect(values(screen.getByRole('figure', { name: 'Agreement with Hi-C, three tools' })).slice(0, 3)).toEqual(['0.22', '0.30', '0.51'])
    const runs = screen.getByRole('figure', { name: 'Twenty one runs with intervals' })
    expect(within(runs).getByText('21 of 21')).toBeInTheDocument()
    expect(within(runs).getByText('12 of 21')).toBeInTheDocument()
  })

  it('shows the trio counts', () => {
    renderResults()
    expect(screen.getByText('9 of 9')).toBeInTheDocument()
    expect(screen.getByText('2 of 9')).toBeInTheDocument()
  })

  it('has a key for the three kinds of model', () => {
    renderResults()
    const key = screen.getByRole('list', { name: 'Key' })
    expect(within(key).getAllByRole('listitem').map((li) => li.textContent)).toEqual([
      'original models, cudaMMC',
      '3D-GNOMEᴺᴳ, CTCF loops',
      '3D-GNOMEᴺᴳ, CTCF and RNAPII loops',
    ])
  })

  it('keeps to the writing rules', () => {
    const { container } = renderResults()
    expect(checkCopy(container)).toEqual([])
  })
})
