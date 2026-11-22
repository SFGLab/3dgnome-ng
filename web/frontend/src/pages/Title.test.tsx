import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import { MemoryRouter, Route, Routes, useLocation } from 'react-router-dom'
import { describe, expect, it, vi } from 'vitest'
import { CATALOG, mockApi } from '../test/api'
import { Layout } from '../components/Layout'
import { ROUTES } from '../routes'
import { checkCopy } from '../test/copy'
import { Title } from './Title'

function Where() {
  const l = useLocation()
  return <p data-testid="where">{l.pathname + l.search}</p>
}

function renderTitle() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  return render(
    <QueryClientProvider client={client}>
      <MemoryRouter>
        <Layout>
          <Routes>
            <Route path="/" element={<Title />} />
            <Route path="*" element={<Where />} />
          </Routes>
        </Layout>
      </MemoryRouter>
    </QueryClientProvider>,
  )
}

describe('Title page', () => {
  it('has the name as its main heading and the lead', () => {
    renderTitle()
    expect(screen.getByRole('heading', { level: 1 })).toHaveTextContent('3D-GNOMENG')
    expect(screen.getByText(/computes 3D structures of genomic regions up to 20 Mb/)).toBeInTheDocument()
  })

  it('states the terms of use on the landing page: free, no login and the licence', () => {
    mockApi()
    renderTitle()
    expect(screen.getByText(/The server is free to use for all, your data stays private\./)).toBeInTheDocument()
    expect(within(screen.getByRole('main')).getByRole('link', { name: 'GNU General Public License, version 3' })).toHaveAttribute('href', 'https://www.gnu.org/licenses/gpl-3.0.html')
  })

  it('states the longest region the server allows', async () => {
    mockApi([[/^\/api\/catalog$/, () => [200, { ...CATALOG, max_region_mb: 30 }]]])
    renderTitle()
    expect(await screen.findByText(/computes 3D structures of genomic regions up to 30 Mb/)).toBeInTheDocument()
    vi.unstubAllGlobals()
  })

  it('links to the form and the results', () => {
    renderTitle()
    const model = screen.getAllByRole('link', { name: /Model a region/ }).map((a) => a.getAttribute('href'))
    expect(model).toContain(ROUTES.model)
    expect(screen.queryByRole('link', { name: /Comparison with Hi-C and expression data/ })).toBeNull()
  })

  it('links to the previous services', () => {
    renderTitle()
    expect(screen.getByRole('link', { name: /Open 3D-GNOME 3.0/ })).toHaveAttribute('href', 'https://3dgnome.mini.pw.edu.pl/old')
    expect(screen.getByRole('link', { name: /Open enhancer3D/ })).toHaveAttribute('href', 'https://3dgnome.mini.pw.edu.pl/old/enhancer3D/')
  })

  it('shows the ECI2 comparison and the contact map as examples', () => {
    renderTitle()
    for (const t of ['ECI2 with and without its RNAPII loop', 'Contacts against Hi-C', 'Deletion at GSTM1', 'CD52 in GM12878 and H1ESC, distance beside expression']) {
      expect(screen.getByText(t)).toBeInTheDocument()
    }
  })

  it('draws the locus with and without the RNAPII loop in one card, each with its distance', () => {
    renderTitle()
    const card = screen.getByRole('button', { name: /ECI2 with and without its RNAPII loop/ })
    expect(within(card).getByRole('img', { name: /CTCF and RNAPII loops.*3\.4 ± 1\.3 bead units apart/ })).toBeInTheDocument()
    expect(within(card).getByRole('img', { name: /CTCF loops alone.*4\.3 ± 1\.6 bead units apart/ })).toBeInTheDocument()
    expect(card).toHaveTextContent('3.4 ± 1.3 bead units with the loop and 4.3 ± 1.6 with CTCF loops alone')
  })

  it('marks A, B and the distance on the picture at the top', () => {
    renderTitle()
    const art = screen.getByRole('img', { name: /A 3D structure of the ECI2 locus/ })
    expect(art).toHaveTextContent('3.4 ± 1.3')
    expect(screen.getByText('A to B 3.4 ± 1.3 bead units')).toBeInTheDocument()
    expect(screen.queryByText(/stay private/)).toBeNull()
  })

  it('opens an example as its finished job, on its tab', async () => {
    const fetchMock = mockApi([[/^\/api\/jobs$/, () => [201, { url: '/jobs/aaaa1111bbbb2222' }]]])
    renderTitle()
    fireEvent.click(screen.getByRole('button', { name: /Contacts against Hi-C/ }))
    await waitFor(() => expect(screen.getByTestId('where')).toHaveTextContent('/jobs/aaaa1111bbbb2222?tab=contacts'))
    const body = fetchMock.mock.calls.find(([u]) => String(u) === '/api/jobs')![1]!.body as FormData
    expect(body.get('region')).toBe('chr1:20,000,000-25,000,000')
    expect(body.get('n_models')).toBe('100')
    vi.unstubAllGlobals()
  })

  it('opens the ECI2 example compared with CTCF loops alone', async () => {
    let n = 0
    const fetchMock = mockApi([[/^\/api\/jobs$/, () => [201, { url: n++ === 0 ? '/jobs/aaaa1111bbbb2222' : '/jobs/cccc3333dddd4444' }]]])
    renderTitle()
    fireEvent.click(screen.getByRole('button', { name: /ECI2 with and without its RNAPII loop/ }))
    await waitFor(() => expect(screen.getByTestId('where')).toHaveTextContent('/jobs/aaaa1111bbbb2222?compare=%2Fjobs%2Fcccc3333dddd4444&gene=ECI2&enhancer=4022561'))
    const posted = fetchMock.mock.calls.filter(([u]) => String(u) === '/api/jobs').map(([, i]) => (i!.body as FormData).get('factors'))
    expect(posted).toEqual(['CTCF+RNAPII', 'CTCF'])
    expect(JSON.parse(localStorage.getItem('gnome3d.recentJobs') ?? '[]').map((j: { label: string }) => j.label)).toEqual([
      'GM12878 ECI2, CTCF loops',
      'GM12878 ECI2, CTCF and RNAPII loops',
    ])
    localStorage.clear()
    vi.unstubAllGlobals()
  })

  it('has two rows of examples: a wide card with a narrow one, then a narrow card with a wide one', () => {
    renderTitle()
    const rows = [...document.querySelectorAll('.example-row')].map((row) => [...row.querySelectorAll(':scope > .example')].map((card) => card.classList.contains('story')))
    expect(rows).toEqual([
      [true, false],
      [false, true],
    ])
  })

  it('draws CD52 in the two cell types, each with its own nearest enhancer, beside its expression', () => {
    renderTitle()
    const card = screen.getByRole('button', { name: /CD52 in GM12878 and H1ESC/ })
    expect(within(card).getByRole('img', { name: /CD52 locus in GM12878.*2\.3 ± 0\.9 bead units apart/ })).toBeInTheDocument()
    expect(within(card).getByRole('img', { name: /CD52 locus in H1ESC.*2\.6 ± 0\.8 bead units apart/ })).toBeInTheDocument()
    expect(card).toHaveTextContent('The promoter A and its nearest enhancer B are 2.3 ± 0.9 bead units apart in GM12878 and 2.6 ± 0.8 bead units apart in H1ESC, expression about 250 times higher in GM12878')
    // The fold of the card is the DESeq2 log2 fold change of the site, 7.95.
    expect(Math.round(2 ** 7.95 / 10) * 10).toBe(250)
    // The two drawings have one size of view, each about the middle of its own structure.
    const views = [...card.querySelectorAll('svg.structure-preview')].map((s) => s.getAttribute('viewBox')!.split(' ').map(Number))
    expect(views[0].slice(2)).toEqual(views[1].slice(2))
    expect(views[0].slice(0, 2)).not.toEqual(views[1].slice(0, 2))
    // The card of the deletion has one line of text.
    expect(screen.getByRole('button', { name: /Deletion at GSTM1/ })).toHaveTextContent('GM12878, the 18 kb deletion of GSTM1 in HG00512')
  })

  it('opens the expression example as two jobs of two cell types with their ENCODE expression', async () => {
    let n = 0
    const fetchMock = mockApi([[/^\/api\/jobs$/, () => [201, { url: n++ === 0 ? '/jobs/aaaa1111bbbb2222' : '/jobs/cccc3333dddd4444' }]]])
    renderTitle()
    fireEvent.click(screen.getByRole('button', { name: /CD52 in GM12878 and H1ESC/ }))
    await waitFor(() => expect(screen.getByTestId('where')).toHaveTextContent('/jobs/aaaa1111bbbb2222?compare=%2Fjobs%2Fcccc3333dddd4444&gene=CD52'))
    const posted = fetchMock.mock.calls.filter(([u]) => String(u) === '/api/jobs').map(([, i]) => i!.body as FormData)
    expect(posted.map((b) => [b.get('cell_type'), b.get('expression'), b.get('region'), b.get('fresh')])).toEqual([
      ['GM12878', 'ENCODE', 'chr1:25,900,000-26,900,000', null],
      ['H1ESC', 'ENCODE', 'chr1:25,900,000-26,900,000', null],
    ])
    localStorage.clear()
    vi.unstubAllGlobals()
  })

  it('opens the deletion example with its call set and its sample', async () => {
    const fetchMock = mockApi([[/^\/api\/jobs$/, () => [201, { url: '/jobs/aaaa1111bbbb2222' }]]])
    renderTitle()
    fireEvent.click(screen.getByRole('button', { name: /Deletion at GSTM1/ }))
    await waitFor(() => expect(screen.getByTestId('where')).toHaveTextContent('/jobs/aaaa1111bbbb2222'))
    const body = fetchMock.mock.calls.find(([u]) => String(u) === '/api/jobs')![1]!.body as FormData
    expect([body.get('sv_set'), body.get('sample')]).toEqual(['HGSVC', 'HG00512'])
    localStorage.clear()
    vi.unstubAllGlobals()
  })

  it('links All examples to the examples page', () => {
    renderTitle()
    expect(screen.getByRole('link', { name: 'All examples' })).toHaveAttribute('href', ROUTES.examples)
  })

  it('keeps to the writing rules', () => {
    const { container } = renderTitle()
    expect(checkCopy(container)).toEqual([])
  })
})
