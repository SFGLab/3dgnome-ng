import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { fireEvent, render, screen, within } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { recentJobs, rememberJob } from '../model/recent'
import { resetSync, syncState } from '../sync/sync'
import { mockApi } from '../test/api'
import { checkCopy } from '../test/copy'
import { mockLists } from '../test/lists'
import { MyJobs } from './MyJobs'

function renderPage(path = '/jobs') {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  return render(
    <QueryClientProvider client={client}>
      <MemoryRouter initialEntries={[path]}>
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
    resetSync()
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

  it('offers a sync code and a place to enter one, with no jobs too', () => {
    const { container } = renderPage()
    expect(screen.getByRole('heading', { name: 'Your jobs on every computer' })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Get a sync code' })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Enter a code' })).toBeInTheDocument()
    expect(checkCopy(container)).toEqual([])
  })

  it('makes a code, private jobs left out unless the box is ticked', async () => {
    const { lists } = mockLists()
    rememberJob({ url: '/jobs/aaaa1111bbbb2222', label: 'GM12878 chr1', name: '', at: 1 })
    rememberJob({ url: '/p/privatetoken123', label: 'H1ESC chr7', name: '', at: 2 })
    const { container } = renderPage()
    fireEvent.click(screen.getByRole('button', { name: 'Get a sync code' }))
    expect(screen.getByRole('checkbox', { name: /Include private jobs/ })).not.toBeChecked()
    expect(checkCopy(container)).toEqual([])
    fireEvent.click(screen.getByRole('button', { name: 'Create the code' }))
    expect(await screen.findByLabelText('Sync code')).toHaveTextContent('AAAA-BBBB-CCCC-DDDD')
    expect(screen.getByRole('heading', { name: 'This list is synced' })).toBeInTheDocument()
    expect(screen.getByText('Private jobs stay on this computer. They are not on the shared list.')).toBeInTheDocument()
    expect(screen.getByText('private, on this computer only')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Copy code' })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Copy link' })).toBeInTheDocument()
    expect(lists['AAAA-BBBB-CCCC-DDDD'].jobs.map((j) => j.url)).toEqual(['/jobs/aaaa1111bbbb2222'])
    expect(checkCopy(container)).toEqual([])
  })

  it('shows what a typed code holds, then joins it', async () => {
    mockLists({ 'K7M2-9QXD-4THP-W3RZ': { private: false, jobs: [{ url: '/jobs/far0000000000000', label: 'HFFc6 chr2', name: 'From the lab', at: 5 }] } })
    rememberJob({ url: '/jobs/aaaa1111bbbb2222', label: 'GM12878 chr1', name: '', at: 1 })
    const { container } = renderPage()
    fireEvent.click(screen.getByRole('button', { name: 'Enter a code' }))
    const input = screen.getByRole('textbox', { name: 'Sync code' })
    expect(screen.getByRole('button', { name: 'Join this list' })).toBeDisabled()
    fireEvent.change(input, { target: { value: 'k7m29qxd4thpw3rz' } })
    expect(input).toHaveValue('K7M2-9QXD-4THP-W3RZ')
    expect(await screen.findByText('This list has 1 job. Private jobs are not on it. 1 job of this computer will be added to it.')).toBeInTheDocument()
    expect(checkCopy(container)).toEqual([])
    fireEvent.click(screen.getByRole('button', { name: 'Join this list' }))
    expect(await screen.findByRole('link', { name: 'From the lab' })).toHaveAttribute('href', '/jobs/far0000000000000')
    expect(syncState()?.code).toBe('K7M2-9QXD-4THP-W3RZ')
    expect(screen.getAllByRole('row').slice(1)).toHaveLength(2)
  })

  it('says so for a code no list has', async () => {
    mockLists()
    renderPage()
    fireEvent.click(screen.getByRole('button', { name: 'Enter a code' }))
    fireEvent.change(screen.getByRole('textbox', { name: 'Sync code' }), { target: { value: '0000-0000-0000-0000' } })
    expect(await screen.findByRole('alert')).toHaveTextContent('No list has this code')
    expect(screen.getByRole('button', { name: 'Join this list' })).toBeDisabled()
  })

  it('opens the form with the code of a link', async () => {
    mockLists({ 'K7M2-9QXD-4THP-W3RZ': { private: true, jobs: [] } })
    renderPage('/jobs?sync=K7M2-9QXD-4THP-W3RZ')
    expect(screen.getByRole('textbox', { name: 'Sync code' })).toHaveValue('K7M2-9QXD-4THP-W3RZ')
    expect(await screen.findByText('This list has 0 jobs. It includes private jobs.')).toBeInTheDocument()
  })

  it('stops syncing after a second step and keeps the jobs', async () => {
    mockLists({ 'K7M2-9QXD-4THP-W3RZ': { private: false, jobs: [{ url: '/jobs/far0000000000000', label: 'HFFc6 chr2', name: '', at: 5 }] } })
    localStorage.setItem('gnome3d.sync', JSON.stringify({ code: 'K7M2-9QXD-4THP-W3RZ', private: false }))
    const { container } = renderPage()
    expect(await screen.findByRole('link', { name: 'HFFc6 chr2' })).toBeInTheDocument()
    // The button is beside the code and its two copy buttons.
    const stop = screen.getByRole('button', { name: 'Stop syncing on this computer' })
    expect(stop.parentElement).toBe(screen.getByRole('button', { name: 'Copy link' }).parentElement)
    fireEvent.click(stop)
    const dialog = screen.getByRole('dialog', { name: 'Stop syncing on this computer' })
    expect(dialog).toHaveTextContent('The jobs stay on this computer and the code keeps working on the others.')
    expect(checkCopy(container)).toEqual([])
    expect(checkCopy(dialog)).toEqual([])
    // Keep syncing is the emphasised button. It closes the dialog and changes nothing.
    expect(within(dialog).getByRole('button', { name: 'Keep syncing' })).toHaveClass('primary')
    expect(within(dialog).getByRole('button', { name: 'Stop syncing' })).toHaveClass('secondary')
    fireEvent.click(within(dialog).getByRole('button', { name: 'Keep syncing' }))
    expect(screen.queryByRole('dialog')).toBeNull()
    expect(syncState()).not.toBeNull()
    fireEvent.click(screen.getByRole('button', { name: 'Stop syncing on this computer' }))
    fireEvent.click(within(screen.getByRole('dialog')).getByRole('button', { name: 'Stop syncing' }))
    expect(screen.queryByRole('dialog')).toBeNull()
    expect(screen.getByRole('button', { name: 'Get a sync code' })).toBeInTheDocument()
    expect(syncState()).toBeNull()
    expect(recentJobs()).toHaveLength(1)
  })

  it('says so when the server cannot be reached and keeps the list', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => Promise.reject(new TypeError('offline'))))
    localStorage.setItem('gnome3d.sync', JSON.stringify({ code: 'K7M2-9QXD-4THP-W3RZ', private: false }))
    rememberJob({ url: '/jobs/aaaa1111bbbb2222', label: 'GM12878 chr1', name: '', at: 1 })
    const { container } = renderPage()
    expect(await screen.findByRole('heading', { name: 'The last sync failed' })).toBeInTheDocument()
    expect(screen.getByRole('alert')).toHaveTextContent('The server did not answer. The jobs on this computer are unchanged.')
    expect(screen.getByRole('button', { name: /Try again/ })).toBeInTheDocument()
    expect(screen.getAllByRole('row').slice(1)).toHaveLength(1)
    expect(checkCopy(container)).toEqual([])
  })
})
