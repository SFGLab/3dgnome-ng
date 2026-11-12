import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { MemoryRouter, Route, Routes, useLocation } from 'react-router-dom'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { EXAMPLES } from '../model/examples'
import { mockApi } from '../test/api'
import { checkCopy } from '../test/copy'
import { Examples } from './Examples'

function Where() {
  const l = useLocation()
  return <p data-testid="where">{l.pathname + l.search}</p>
}

function renderPage() {
  return render(
    <MemoryRouter initialEntries={['/examples']}>
      <Routes>
        <Route path="/examples" element={<Examples />} />
        <Route path="*" element={<Where />} />
      </Routes>
    </MemoryRouter>,
  )
}

describe('Examples page', () => {
  afterEach(() => {
    vi.unstubAllGlobals()
    localStorage.clear()
  })

  it('lists every example', () => {
    const { container } = renderPage()
    expect(screen.getAllByRole('button', { name: /Open/ })).toHaveLength(EXAMPLES.length)
    expect(checkCopy(container)).toEqual([])
  })

  it('opens an example as its job and remembers it', async () => {
    mockApi([[/^\/api\/jobs$/, () => [201, { url: '/jobs/aaaa1111bbbb2222' }]]])
    renderPage()
    fireEvent.click(screen.getByRole('button', { name: /Open MYC/ }))
    await waitFor(() => expect(screen.getByTestId('where')).toHaveTextContent('/jobs/aaaa1111bbbb2222'))
    expect(JSON.parse(localStorage.getItem('gnome3d.recentJobs') ?? '[]')[0].url).toBe('/jobs/aaaa1111bbbb2222')
  })

  it('shows the refusal of the server', async () => {
    mockApi([[/^\/api\/jobs$/, () => [503, { error: 'The queue is full. Try again later.' }]]])
    renderPage()
    fireEvent.click(screen.getByRole('button', { name: /Open MYC/ }))
    expect(await screen.findByRole('alert')).toHaveTextContent('The queue is full')
  })
})
