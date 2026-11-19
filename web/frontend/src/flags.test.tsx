import { render, screen } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { afterEach, describe, expect, it } from 'vitest'
import { Layout } from './components/Layout'
import { hidden } from './flags'
import { Cite } from './pages/Cite'
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
