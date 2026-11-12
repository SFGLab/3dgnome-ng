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
    expect(values(screen.getByRole('figure', { name: 'Agreement with Hi-C' }))).toEqual(['0.22', '0.37', '0.15', '0.27', '0.15', '0.33'])
  })

  it('shows the enhancer to promoter distances and the link to expression', () => {
    renderResults()
    expect(values(screen.getByRole('figure', { name: 'Enhancer to promoter distance' }))).toEqual(['4.2', '3.7', '4.0', '3.0', '4.5', '4.0'])
    expect(values(screen.getByRole('figure', { name: 'Link to expression' }))).toEqual(['0.19', '0.26', '0.29', '0.12', '0.19', '0.20', '0.25', '0.29', '0.33'])
  })

  it('marks every run time value as a placeholder', () => {
    renderResults()
    for (const name of ['One structure, 1 Mb', 'Ten structures, 60 Mb', 'One structure, whole chromosome']) {
      expect(values(screen.getByRole('figure', { name }))).toEqual(['placeholder', 'placeholder'])
    }
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
