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
  // The cell line charts hold the whole genome readouts of the redo of 2026-11-23, design/figures/redo-2026-11-23.
  it('shows the agreement with Hi-C of the original and the new models', () => {
    renderResults()
    expect(values(screen.getByRole('figure', { name: 'Agreement with Hi-C' }))).toEqual(['0.19', '0.49', '0.52', '0.13', '0.44', '0.45', '0.15', '0.51', '0.47'])
  })

  it('shows the enhancer to promoter distances and the link to expression', () => {
    renderResults()
    expect(screen.getByText(/are from models built with every loop in every structure/)).toBeInTheDocument()
    expect(values(screen.getByRole('figure', { name: 'Enhancer to promoter distance' }))).toEqual(['4.0', '3.3', '3.9', '3.4', '4.3', '3.7'])
    expect(values(screen.getByRole('figure', { name: 'Link to expression' }))).toEqual(['0.14', '0.23', '0.31', '0.13', '0.20', '0.24', '0.14', '0.24', '0.28'])
  })

  it('shows the beads, the runtime and the runtime per 1,000 beads of each tool', () => {
    renderResults()
    const cells = (figure: HTMLElement, id: string) => within(figure).getAllByTestId(id).map((el) => el.textContent)
    const one = screen.getByRole('figure', { name: 'One structure, 1 Mb' })
    expect(screen.getByText(/timed with its default chain, the chain of loop anchors/)).toBeInTheDocument()
    expect(values(one)).toEqual(['6.4', '7.4', '9.0'])
    expect(cells(one, 'beads-value')).toEqual(['350', '760', '760'])
    expect(cells(one, 'rate-value')).toEqual(['18.3', '9.7', '11.8'])
    expect(within(one).getByRole('columnheader', { name: /^runtime, seconds/ })).toBeInTheDocument()
    expect(cells(one, 'better')).toEqual(['↑ more is finer', '↓ lower is better', '↓ lower is better'])
    expect(within(one).getByTestId('bar-best').closest('tr')).toHaveTextContent('cudaMMC')
    expect(within(one).getByTestId('rate-best').closest('tr')).toHaveTextContent('MultiMM')
    expect(values(screen.getByRole('figure', { name: 'One structure, whole chromosome' }))).toEqual(['3.2', '14.2', '6.4'])
    expect(values(screen.getByRole('figure', { name: 'Ten structures, 60 Mb' }))).toEqual(['9.3', '20.1', '10.6'])
    expect(values(screen.getByRole('figure', { name: 'Three structures at equal bead counts' }))).toEqual(['10.7', '42.1', '11.2'])
    const genome = screen.getByRole('figure', { name: 'One structure, whole genome' })
    expect(values(genome)).toEqual(['31.4', 'did not complete', '46.4'])
    expect(cells(genome, 'beads-value')).toEqual(['458,445', '1,312,597', '1,312,597'])
    expect(cells(genome, 'rate-value')).toEqual(['4.1', '', '2.1'])
    expect(within(genome).getByTestId('bar-best').closest('tr')).toHaveTextContent('cudaMMC')
    expect(within(genome).getByTestId('rate-best').closest('tr')).toHaveTextContent('3D-GNOME')
    expect(within(screen.getByRole('figure', { name: 'Agreement with Hi-C' })).getByTestId('better')).toHaveTextContent('↑ higher is better')
    expect(within(screen.getByRole('figure', { name: 'One structure by bead count' })).getByTestId('better')).toHaveTextContent('↓ lower is better')
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
    expect(values(screen.getByRole('figure', { name: 'Agreement with Hi-C, three tools' }))).toEqual(['0.22', '0.30', '0.48', '0.14', '0.16', '0.35', '0.15', '0.18', '0.47'])
    const runs = screen.getByRole('figure', { name: 'Comparison over 21 runs' })
    expect(within(runs).getByText('21 of 21')).toBeInTheDocument()
    expect(within(runs).getByText('7 of 21')).toBeInTheDocument()
  })

  it('shows the share of Micro-C loops placed, for the model sets and for the three tools', () => {
    renderResults()
    expect(values(screen.getByRole('figure', { name: 'Micro-C loops placed' }))).toEqual(['0.12', '0.26', '0.33', '0.12', '0.39', '0.45', '0.10', '0.44', '0.52'])
    expect(values(screen.getByRole('figure', { name: 'Micro-C loops with an RNAPII loop only' }))).toEqual(['0.14', '0.65', '0.19', '0.67', '0.24', '0.70'])
    expect(values(screen.getByRole('figure', { name: 'Micro-C loops placed, three tools' }))).toEqual(['0.14', '0.25', '0.29', '0.21', '0.16', '0.53', '0.16', '0.15', '0.51'])
    expect(values(screen.getByRole('figure', { name: 'Micro-C loops with beads at both ends' }))).toEqual(['0.25', '1.00', '0.98', '0.26', '1.00', '0.95', '0.22', '1.00', '0.97'])
  })

  it('shows the trio counts', () => {
    renderResults()
    // The whole genome readouts of design/figures/rnapii-2026-11-18/readouts/trio_lr/trio_genome.json.
    expect(screen.getByText('9 of 9')).toBeInTheDocument()
    // A comparison that is not significant is not shown, here the correlation with the whole map.
    expect(screen.queryByText('5 of 9')).not.toBeInTheDocument()
    expect(screen.getByText(/The rise is 44 to 67% and the median is 53%/)).toBeInTheDocument()
    expect(screen.getByText(/on chromosomes 1 to 22/)).toBeInTheDocument()
  })

  it('has a key for the kinds of model', () => {
    renderResults()
    const key = screen.getByRole('list', { name: 'Key' })
    expect(within(key).getAllByRole('listitem').map((li) => li.textContent)).toEqual([
      'original models, cudaMMC',
      '3D-GNOMEᴺᴳ, CTCF loops',
      '3D-GNOMEᴺᴳ, CTCF and RNAPII loops',
      '3D-GNOMEᴺᴳ, chain of loop anchors, CTCF loops',
    ])
  })

  it('keeps to the writing rules', () => {
    const { container } = renderResults()
    expect(checkCopy(container)).toEqual([])
  })
})
