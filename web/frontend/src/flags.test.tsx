import { render, screen } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { afterEach, describe, expect, it } from 'vitest'
import { Layout } from './components/Layout'
import { hidden } from './flags'
import { Cite } from './pages/Cite'
import { Help } from './pages/Help'
import { Results } from './pages/Results'

/** The tag the server writes into the page for the features its `config.json` turns off. */
function hide(parts: string) {
  const meta = document.createElement('meta')
  meta.name = 'site-hide'
  meta.content = parts
  document.head.appendChild(meta)
}

afterEach(() => document.querySelectorAll('meta[name="site-hide"]').forEach((m) => m.remove()))

const links = () => screen.getAllByRole('link').map((a) => a.textContent)

describe('the parts the server hides', () => {
  it('hides nothing on a page without the tag', () => {
    expect(hidden('results')).toBe(false)
    expect(hidden('method')).toBe(false)
    expect(hidden('trios')).toBe(false)
  })

  it('leaves the hidden pages out of the top bar', () => {
    hide('results,method')
    render(
      <MemoryRouter>
        <Layout>
          <p>page</p>
        </Layout>
      </MemoryRouter>,
    )
    expect(links()).not.toContain('Results')
    expect(links()).not.toContain('Method')
    expect(links()).toContain('Help')
  })

  it('leaves the nine people out of the Results page', () => {
    hide('trios')
    const { container } = render(
      <MemoryRouter>
        <Results />
      </MemoryRouter>,
    )
    expect(screen.queryByRole('figure', { name: 'Nine people, three trios' })).not.toBeInTheDocument()
    expect(container.textContent).not.toMatch(/nine people|trios/i)
    expect(screen.getByRole('figure', { name: 'Link to expression' })).toBeInTheDocument()
  })

  it('leaves the RNA-seq of the nine people out of the Cite page and the Help page', () => {
    const cite = render(
      <MemoryRouter>
        <Cite />
      </MemoryRouter>,
    )
    expect(cite.container.textContent).toMatch(/PRJEB11271/)
    cite.unmount()
    const help = render(<Help />)
    expect(help.container.textContent).toMatch(/The server holds four sets\..*nine people of three 1000 Genomes trios/)
    help.unmount()
    hide('trios')
    const hiddenCite = render(
      <MemoryRouter>
        <Cite />
      </MemoryRouter>,
    )
    expect(hiddenCite.container.textContent).not.toMatch(/PRJEB11271|nine people/i)
    expect(hiddenCite.container.textContent).toMatch(/MAGE, 731 people/)
    hiddenCite.unmount()
    const hiddenHelp = render(<Help />)
    expect(hiddenHelp.container.textContent).toMatch(/The server holds three sets\./)
    expect(hiddenHelp.container.textContent).not.toMatch(/trios/i)
  })

  it('does not link to a hidden Method page or name a hidden page', () => {
    hide('method,results')
    const results = render(
      <MemoryRouter>
        <Results />
      </MemoryRouter>,
    )
    expect(links()).not.toContain('Limitations')
    results.unmount()
    const { container } = render(
      <MemoryRouter>
        <Cite />
      </MemoryRouter>,
    )
    expect(container.textContent).not.toMatch(/Method page|Results page/)
  })
})
