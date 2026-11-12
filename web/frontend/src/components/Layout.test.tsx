import { render, screen, within } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { describe, expect, it } from 'vitest'
import { ROUTES } from '../routes'
import { Layout } from './Layout'

function renderLayout() {
  return render(
    <MemoryRouter>
      <Layout>
        <p>body</p>
      </Layout>
    </MemoryRouter>,
  )
}

describe('Layout', () => {
  it('shows the wordmark with a superscript NG', () => {
    renderLayout()
    const mark = screen.getByTestId('wordmark')
    expect(mark).toHaveTextContent('3D-GNOMENG')
    expect(within(mark).getByText('NG').tagName).toBe('SUP')
  })

  it('links the navigation in the design order', () => {
    renderLayout()
    const nav = screen.getByRole('navigation')
    const links = within(nav).getAllByRole('link')
    expect(links.map((a) => a.textContent)).toEqual(['Model a region', 'My jobs', 'Method', 'Results', 'Help', 'Cite'])
    expect(links.map((a) => a.getAttribute('href'))).toEqual([ROUTES.model, ROUTES.jobs, ROUTES.method, ROUTES.results, ROUTES.help, ROUTES.cite])
  })

  it('has the cite strip with a link to the cite page', () => {
    renderLayout()
    expect(screen.getByText(/If you use these models, please cite 3D-GNOMEᴺᴳ/)).toBeInTheDocument()
    expect(screen.getByRole('link', { name: /All citations/ })).toHaveAttribute('href', ROUTES.cite)
  })

  it('renders its children', () => {
    renderLayout()
    expect(screen.getByText('body')).toBeInTheDocument()
  })
})
