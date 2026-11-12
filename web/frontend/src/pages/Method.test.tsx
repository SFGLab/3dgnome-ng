import { render, screen, within } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { describe, expect, it } from 'vitest'
import { checkCopy } from '../test/copy'
import { METHOD_REFERENCES, Method } from './Method'

function renderMethod() {
  return render(
    <MemoryRouter>
      <Method />
    </MemoryRouter>,
  )
}

describe('Method page', () => {
  it('tells the story in the design order', () => {
    renderMethod()
    const headings = screen.getAllByRole('heading', { level: 2 }).map((h) => h.textContent)
    expect(headings).toEqual([
      '1. Inputs',
      '2. Blocks and target distances',
      '3. Anchor phase',
      '4. Chain phase',
      '5. Ensemble and output',
      '6. Engine',
      '7. Structural variants',
      '8. Validation',
      '9. Limitations',
      'References',
    ])
  })

  it('gives every image a description', () => {
    const { container } = renderMethod()
    for (const img of container.querySelectorAll('img')) expect(img.getAttribute('alt'), img.getAttribute('src') ?? '').toBeTruthy()
  })

  it('numbers the references as the design does', () => {
    expect(METHOD_REFERENCES).toEqual([
      'szalaj2016', 'cudammc', 'lieberman2009', 'hilbert1891', 'multimm', 'byrd1995', 'zhu1997', 'kirkpatrick1983',
      'metropolis1953', 'enhancer3d', 'jax2018', 'sadowski2019', 'hicrep2017', 'kremer1990', 'rosa2008', 'tang2015', 'mirny2011',
    ])
    renderMethod()
    const list = screen.getByRole('list', { name: 'References' })
    expect(within(list).getAllByRole('listitem')).toHaveLength(17)
  })

  it('cites the sources of the shared terms', () => {
    renderMethod()
    const anchorTerms = screen.getByRole('table', { name: 'Anchor energy terms' })
    const chainTerms = screen.getByRole('table', { name: 'Chain energy terms' })
    const source = (table: HTMLElement, term: string) =>
      within(within(table).getByRole('row', { name: new RegExp(term) })).getAllByRole('cell').at(-1)?.textContent
    expect(source(anchorTerms, 'Repulsion')).toBe('[1, 5]')
    expect(source(anchorTerms, 'Excluded volume')).toBe('[5, 14]')
    expect(source(chainTerms, 'Excluded volume')).toBe('[5, 14]')
    expect(source(chainTerms, 'Confinement')).toBe('[5, 15]')
  })

  it('links a citation to its reference', () => {
    renderMethod()
    const links = screen.getAllByRole('link', { name: '17' })
    expect(links[0]).toHaveAttribute('href', '#ref-17')
  })

  it('keeps to the writing rules', () => {
    const { container } = renderMethod()
    expect(checkCopy(container)).toEqual([])
  })
})
