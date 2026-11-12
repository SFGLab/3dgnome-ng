import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { render, screen, within } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { rememberJob } from '../model/recent'
import { mockApi } from '../test/api'
import { checkCopy } from '../test/copy'
import { MyJobs } from './MyJobs'

function renderPage() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  return render(
    <QueryClientProvider client={client}>
      <MemoryRouter>
        <MyJobs />
      </MemoryRouter>
    </QueryClientProvider>,
  )
}

describe('My jobs', () => {
  beforeEach(() => {
    mockApi([[/^\/api\/(jobs|p)\/[\w-]+$/, (u) => [200, u.pathname.endsWith('aaaa1111bbbb2222') ? { state: 'done', stage: 'done' } : { state: 'running', stage: 'modelling' }]]])
  })
  afterEach(() => {
    vi.unstubAllGlobals()
    vi.restoreAllMocks()
    localStorage.clear()
  })

  it('says so when this browser has no jobs', () => {
    const { container } = renderPage()
    expect(screen.getByText('You have no jobs in this browser yet.')).toBeInTheDocument()
    expect(screen.getByRole('link', { name: /Model a region/ })).toHaveAttribute('href', '/model')
    expect(checkCopy(container)).toEqual([])
  })

  it('lists the jobs of this browser with their state', async () => {
    rememberJob({ url: '/jobs/aaaa1111bbbb2222', label: 'GM12878 chr1:11.5-12.6 Mb', name: '', at: Date.UTC(2026, 10, 12, 9, 30) })
    rememberJob({ url: '/p/privatetoken123', label: 'H1ESC chr7:27.0-27.4 Mb', name: 'HOXA run', at: Date.UTC(2026, 10, 12, 10, 0) })
    const { container } = renderPage()
    const rows = screen.getAllByRole('row').slice(1)
    expect(rows).toHaveLength(2)
    expect(within(rows[0]).getByRole('link', { name: 'HOXA run' })).toHaveAttribute('href', '/p/privatetoken123')
    expect(await within(rows[0]).findByText('running, modelling stage')).toBeInTheDocument()
    expect(await within(rows[1]).findByText('done')).toBeInTheDocument()
    expect(within(rows[1]).getByText('GM12878 chr1:11.5-12.6 Mb')).toBeInTheDocument()
    expect(checkCopy(container)).toEqual([])
  })

  it('shows the empty state when storage throws', () => {
    vi.spyOn(Storage.prototype, 'getItem').mockImplementation(() => {
      throw new Error('blocked')
    })
    renderPage()
    expect(screen.getByText('You have no jobs in this browser yet.')).toBeInTheDocument()
  })
})
