import { fireEvent, render, screen, within } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { checkCopy } from '../test/copy'
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

  describe('theme switch', () => {
    afterEach(() => {
      vi.restoreAllMocks()
      localStorage.clear()
      delete document.documentElement.dataset.theme
    })

    it('offers System, Light and Dark and sets the page theme', () => {
      const { container } = renderLayout()
      const group = screen.getByRole('radiogroup', { name: 'Theme' })
      expect(within(group).getAllByRole('radio').map((r) => r.getAttribute('aria-label'))).toEqual(['System', 'Light', 'Dark'])
      expect(within(group).getByRole('radio', { name: 'System' })).toBeChecked()
      fireEvent.click(within(group).getByRole('radio', { name: 'Dark' }))
      expect(document.documentElement.dataset.theme).toBe('dark')
      expect(localStorage.getItem('gnome3d-theme')).toBe('dark')
      fireEvent.click(within(group).getByRole('radio', { name: 'System' }))
      expect(document.documentElement.dataset.theme).toBeUndefined()
      expect(checkCopy(container)).toEqual([])
    })

    it('applies a stored choice', () => {
      localStorage.setItem('gnome3d-theme', 'light')
      renderLayout()
      expect(document.documentElement.dataset.theme).toBe('light')
      expect(screen.getByRole('radio', { name: 'Light' })).toBeChecked()
    })

    it('stays on the system theme when storage throws', () => {
      vi.spyOn(Storage.prototype, 'getItem').mockImplementation(() => {
        throw new Error('blocked')
      })
      vi.spyOn(Storage.prototype, 'setItem').mockImplementation(() => {
        throw new Error('blocked')
      })
      renderLayout()
      expect(screen.getByRole('radio', { name: 'System' })).toBeChecked()
      fireEvent.click(screen.getByRole('radio', { name: 'Dark' }))
      expect(document.documentElement.dataset.theme).toBe('dark')
    })
  })
})
