import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import { MemoryRouter, Route, Routes, useNavigate } from 'react-router-dom'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { mockApi } from '../test/api'
import { checkCopy } from '../test/copy'
import { GENES, KEY, PICKS, SUMMARY_DONE } from '../test/job'
import { recentJobs } from '../model/recent'
import { Job } from './Job'

function Back() {
  const navigate = useNavigate()
  return (
    <button type="button" onClick={() => navigate(-1)}>
      history back
    </button>
  )
}

function renderJob(path = `/jobs/${KEY}`, back = false) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  return render(
    <QueryClientProvider client={client}>
      <MemoryRouter initialEntries={[path]}>
        <Routes>
          <Route path="/jobs/:key" element={<Job kind="jobs" pollMs={50} />} />
          <Route path="/p/:key" element={<Job kind="p" pollMs={50} />} />
        </Routes>
        {back && <Back />}
      </MemoryRouter>
    </QueryClientProvider>,
  )
}

describe('Job explorer', () => {
  afterEach(() => {
    vi.unstubAllGlobals()
    localStorage.clear()
  })

  it('shows the tabs of a done job with their counts', async () => {
    mockApi([[/summary$/, () => [200, SUMMARY_DONE]]])
    renderJob()
    const tabs = await screen.findByRole('tablist')
    expect(within(tabs).getAllByRole('tab').map((t) => t.textContent)).toEqual(['Structure', 'Genes312', 'Contact map', 'Loops182', 'Log'])
  })

  it('shows the facts and the files in the side panel', async () => {
    mockApi([[/summary$/, () => [200, SUMMARY_DONE]]])
    renderJob()
    expect(await screen.findByText('Done in 6 min 12 s')).toBeInTheDocument()
    expect(screen.getByText('118 CTCF, 64 RNAPII')).toBeInTheDocument()
    expect(screen.getByText('CTCF and RNAPII')).toBeInTheDocument()
    expect(screen.getByRole('link', { name: /ep\.tsv/ })).toHaveAttribute('href', `/jobs/${KEY}/files/ep.tsv`)
    expect(screen.getByText('99 more')).toBeInTheDocument()
    expect(screen.getByRole('link', { name: /all files, zip/ })).toHaveAttribute('href', `/api/jobs/${KEY}/files.zip`)
  })

  it('names own loops and an uploaded contact map in the side panel', async () => {
    const own = { ...SUMMARY_DONE, request: { ...SUMMARY_DONE.request, cell_type: 'own', own: { loops: ['CTCF', 'H3K27ac'], anchors: false, contacts: true, enhancers: false } } }
    mockApi([[/summary$/, () => [200, own]]])
    renderJob()
    expect(await screen.findByText('CTCF and H3K27ac, your files')).toBeInTheDocument()
    expect(screen.getByText('your map')).toBeInTheDocument()
  })

  it('names the listed cell type your own cell type took its loops from', async () => {
    const own = { ...SUMMARY_DONE, request: { ...SUMMARY_DONE.request, cell_type: 'own', factors: 'CTCF', own: { loops: [], anchors: false, contacts: true, enhancers: false, loops_from: 'H1ESC' } } }
    mockApi([[/summary$/, () => [200, own]]])
    renderJob()
    expect(await screen.findByText('CTCF of H1ESC')).toBeInTheDocument()
  })

  it('goes back to the previous tab with the browser', async () => {
    mockApi([[/summary$/, () => [200, SUMMARY_DONE]]])
    renderJob(`/jobs/${KEY}`, true)
    fireEvent.click(await screen.findByRole('tab', { name: /Genes/ }))
    fireEvent.click(screen.getByRole('tab', { name: /Loops/ }))
    fireEvent.click(screen.getByRole('button', { name: 'history back' }))
    await waitFor(() => expect(screen.getByRole('tab', { name: /Genes/ })).toHaveAttribute('aria-selected', 'true'))
  })

  it('opens with the gene named in the link selected', async () => {
    mockApi([[/summary$/, () => [200, SUMMARY_DONE]], [/genes$/, () => [200, GENES]]])
    renderJob(`/jobs/${KEY}?tab=genes&gene=DHRS3`)
    const row = await screen.findByRole('row', { name: /DHRS3/ })
    expect(row).toHaveClass('selected')
  })

  it('opens the comparison named in the link', async () => {
    mockApi([[/summary$/, () => [200, SUMMARY_DONE]]])
    renderJob(`/jobs/${KEY}?compare=${encodeURIComponent('/jobs/cccc3333dddd4444')}`)
    expect(await screen.findByText('Link rotation and zoom')).toBeInTheDocument()
  })

  it('moves between structures and to the picks', async () => {
    mockApi([[/summary$/, () => [200, SUMMARY_DONE]], [/picks$/, () => [200, PICKS]]])
    renderJob()
    const select = await screen.findByLabelText('Structure')
    expect(select).toHaveValue('1')
    fireEvent.click(screen.getByRole('button', { name: 'Next structure' }))
    expect(select).toHaveValue('2')
    fireEvent.click(screen.getByRole('button', { name: 'Previous structure' }))
    expect(select).toHaveValue('1')
    fireEvent.click(screen.getByRole('button', { name: 'Medoid' }))
    await waitFor(() => expect(select).toHaveValue('17'))
  })

  it('shows a queued job and its tabs once it is done, without a reload', async () => {
    let calls = 0
    mockApi([[/summary$/, () => [200, ++calls < 3 ? { ...SUMMARY_DONE, state: 'queued', ahead: 2, models: [], structures: 0 } : SUMMARY_DONE]]])
    renderJob()
    expect(await screen.findByText('Queued, 2 jobs ahead')).toBeInTheDocument()
    expect(await screen.findByRole('tablist', {}, { timeout: 3000 })).toBeInTheDocument()
  })

  it('shows the progress of a running job and its status log', async () => {
    const running = {
      ...SUMMARY_DONE,
      state: 'running',
      stage: 'modelling',
      models: [],
      structures: 0,
      structures_done: 2,
      structures_asked: 5,
      stages: [
        { stage: 'queued', at: 1000 },
        { stage: 'inputs', at: 1012 },
        { stage: 'modelling', at: 1020 },
      ],
      log_recent: 'structure 1 written\nstructure 2 written',
    }
    mockApi([[/summary$/, () => [200, running]]])
    const { container } = render(
      <QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>
        <MemoryRouter initialEntries={[`/jobs/${KEY}`]}>
          <Routes>
            <Route path="/jobs/:key" element={<Job kind="jobs" pollMs={100000} />} />
          </Routes>
        </MemoryRouter>
      </QueryClientProvider>,
    )
    const bar = await screen.findByRole('progressbar', { name: 'Progress of the job' })
    // inputs 5, then modelling's 85 by two fifths: 5 + 34 = 39
    expect(bar).toHaveAttribute('aria-valuenow', '39')
    expect(screen.getByText('Modelling the structures, 2 of 5 written')).toBeInTheDocument()
    const log = screen.getByRole('list', { name: 'Status log' })
    expect(within(log).getAllByRole('listitem').map((li) => li.textContent)).toEqual(['Queued0 s', 'Building the inputs12 s', 'Modelling the structures20 s'])
    expect(screen.getByText(/structure 2 written/)).toBeInTheDocument()
    expect(checkCopy(container)).toEqual([])
  })

  it('shows a failed job with its error and log', async () => {
    mockApi([[/summary$/, () => [200, { ...SUMMARY_DONE, state: 'failed', error: 'RuntimeError: engine exploded', log_tail: 'line 1\nengine exploded', models: [] }]]])
    renderJob()
    expect(await screen.findByText('RuntimeError: engine exploded')).toBeInTheDocument()
    expect(document.querySelector('pre.log-tail')?.textContent).toBe('line 1\nengine exploded')
  })

  it('renames the job in this browser', async () => {
    mockApi([[/summary$/, () => [200, SUMMARY_DONE]]])
    renderJob()
    fireEvent.click(await screen.findByRole('button', { name: /Rename job/ }))
    fireEvent.change(screen.getByLabelText('Job name'), { target: { value: 'TNFRSF8 with RNAPII loops' } })
    fireEvent.keyDown(screen.getByLabelText('Job name'), { key: 'Enter' })
    expect(await screen.findByRole('heading', { level: 1, name: 'TNFRSF8 with RNAPII loops' })).toBeInTheDocument()
    expect(recentJobs()[0]).toMatchObject({ url: `/jobs/${KEY}`, name: 'TNFRSF8 with RNAPII loops' })
  })

  it('reads a private job under /api/p', async () => {
    const f = mockApi([[/summary$/, () => [200, SUMMARY_DONE]]])
    renderJob(`/p/${KEY}`)
    await screen.findByRole('tablist')
    expect(f.mock.calls.some(([u]) => String(u) === `/api/p/${KEY}/summary`)).toBe(true)
  })

  it('keeps to the writing rules', async () => {
    mockApi([[/summary$/, () => [200, SUMMARY_DONE]]])
    const { container } = renderJob()
    await screen.findByRole('tablist')
    expect(checkCopy(container)).toEqual([])
  })
})
