import { render, screen, within } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { describe, expect, it } from 'vitest'
import { checkCopy } from '../test/copy'
import { Method } from './Method'
import { METHOD_REFERENCES } from '../content/methodReferences'

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
      '5. One bead per kilobase',
      '6. Ensemble and output',
      '7. Engine',
      '8. Structural variants',
      '9. Validation',
      '10. Limitations',
      'References',
    ])
  })

  it('gives every image a description', () => {
    const { container } = renderMethod()
    for (const img of container.querySelectorAll('img')) expect(img.getAttribute('alt'), img.getAttribute('src') ?? '').toBeTruthy()
  })

  it('numbers the references as the design does', () => {
    expect(METHOD_REFERENCES).toEqual([
      'szalaj2016', 'cudammc', 'lieberman2009', 'shi2023', 'multimm', 'byrd1995', 'zhu1997', 'kirkpatrick1983',
      'metropolis1953', 'enhancer3d', 'jax2018', 'sadowski2019', 'hicrep2017', 'kremer1990', 'rosa2008', 'tang2015', 'mirny2011',
    ])
    renderMethod()
    const list = screen.getByRole('list', { name: 'References' })
    expect(within(list).getAllByRole('listitem')).toHaveLength(17)
  })

  it('explains the target distances with a figure per step', () => {
    renderMethod()
    const section = document.getElementById('targets')!
    expect(within(section).getAllByRole('heading', { level: 3 }).map((h) => h.textContent)).toEqual([
      'Blocks',
      'Distance of a pair without a loop',
      'The default exponent',
      'Distance of a loop',
      'Distance from the Hi-C map',
    ])
    const figures = [...section.querySelectorAll('figure')]
    expect(figures.map((f) => f.querySelector('figcaption b')?.textContent)).toEqual(['Figure 1', 'Figure 2', 'Figure 3', 'Figure 4', 'Figure 5'])
    // The default is the mean of the three exponents the page gives.
    expect(section).toHaveTextContent('The three values, 0.309, 0.269 and 0.277, are close to each other')
    expect(((0.309 + 0.269 + 0.277) / 3).toFixed(3)).toBe('0.285')
    expect(section).toHaveTextContent('uses ν = 0.285')
  })

  it('tells the anchor phase and the chain phase by their terms and the additions of this work', () => {
    renderMethod()
    const h3 = (id: string) => within(document.getElementById(id)!).getAllByRole('heading', { level: 3 }).map((h) => h.textContent)
    expect(h3('anchors')).toEqual(['The four terms', 'The start and the fit'])
    expect(h3('chain')).toEqual(['The five terms', 'The start and the wall', 'Annealing', 'The last pass'])
    const anchors = document.getElementById('anchors')!
    // The three kinds of pair of the figure add up to its pairs.
    expect(46 + 55 + 1277).toBe((53 * 52) / 2)
    expect(anchors).toHaveTextContent('53 anchors and 1,378 pairs of anchors')
    expect(anchors.querySelector('figure figcaption b')?.textContent).toBe('Figure 6')
    expect(anchors).toHaveTextContent('two are new in this work, the background spring and the window term')
  })

  it('cites the sources of the shared terms', () => {
    renderMethod()
    const anchorTerms = screen.getByRole('table', { name: 'Anchor energy terms' })
    const chainTerms = screen.getByRole('table', { name: 'Chain energy terms' })
    const source = (table: HTMLElement, term: string) =>
      within(within(table).getByRole('row', { name: new RegExp(term) })).getAllByRole('cell').at(-1)?.textContent
    expect(source(anchorTerms, 'Window size')).toBe('this work')
    expect(source(anchorTerms, 'Excluded volume')).toBe('[5, 14]')
    expect(source(chainTerms, 'Excluded volume')).toBe('[5, 14]')
    expect(source(chainTerms, 'Confinement')).toBe('[5, 15]')
  })

  it('draws every term on a small chain with a sentence on what the term does', () => {
    renderMethod()
    const anchorTerms = screen.getByRole('table', { name: 'Anchor energy terms' })
    const chainTerms = screen.getByRole('table', { name: 'Chain energy terms' })
    expect(anchorTerms.querySelectorAll('svg.term-sketch')).toHaveLength(4)
    expect(chainTerms.querySelectorAll('svg.term-sketch')).toHaveLength(5)
    expect(within(within(chainTerms).getByRole('row', { name: /Bending/ })).getByRole('img', { name: 'Straightens sharp turns of the chain.' })).toBeInTheDocument()
    expect(within(within(anchorTerms).getByRole('row', { name: /Loop spring/ })).getByRole('img', { name: /two anchors of a loop/ })).toBeInTheDocument()
    // The orientation term is least for motifs that point the same way, as the engine scores it.
    const orientation = within(chainTerms).getByRole('row', { name: /CTCF orientation/ })
    expect(within(orientation).getByRole('img', { name: /point the same way in space.*face each other along the genome give a hairpin/ })).toBeInTheDocument()
    expect(within(orientation).getByAltText(/1 - \\mathbf\{o\}_i \\cdot \\mathbf\{o\}_j/)).toBeInTheDocument()
  })

  it('gives the rule of each kind of variant, tandem duplications among them', () => {
    renderMethod()
    const section = document.getElementById('variants')!
    expect(within(section).getAllByRole('heading', { level: 3 }).map((h) => h.textContent)).toEqual(['Deletion', 'Insertion', 'Inversion', 'Duplication'])
    expect(section).toHaveTextContent('a loop to an anchor after it moves to the second')
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
