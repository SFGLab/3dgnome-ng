import { fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import { MemoryRouter, Route, Routes, useLocation } from 'react-router-dom'
import { describe, expect, it, vi } from 'vitest'
import { mockApi } from '../test/api'
import { Layout } from '../components/Layout'
import { ROUTES } from '../routes'
import { checkCopy } from '../test/copy'
import { Title } from './Title'

function Where() {
  const l = useLocation()
  return <p data-testid="where">{l.pathname + l.search}</p>
}

function renderTitle() {
  return render(
    <MemoryRouter>
      <Layout>
        <Routes>
          <Route path="/" element={<Title />} />
          <Route path="*" element={<Where />} />
        </Routes>
      </Layout>
    </MemoryRouter>,
  )
}

describe('Title page', () => {
  it('has the name as its main heading and the lead', () => {
    renderTitle()
    expect(screen.getByRole('heading', { level: 1 })).toHaveTextContent('3D-GNOMENG')
    expect(screen.getByText(/computes 3D structures of genomic regions up to 20 Mb/)).toBeInTheDocument()
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

  it('shows the TNFRSF8 comparison and the contact map as examples', () => {
    renderTitle()
    for (const t of ['TNFRSF8 with and without its RNAPII loop', 'Contacts against Hi-C']) {
      expect(screen.getByText(t)).toBeInTheDocument()
    }
  })

  it('draws the locus with and without the RNAPII loop in one card, each with its distance', () => {
    renderTitle()
    const card = screen.getByRole('button', { name: /TNFRSF8 with and without its RNAPII loop/ })
    expect(within(card).getByRole('img', { name: /CTCF and RNAPII loops.*6\.0 ± 1\.6 bead units apart/ })).toBeInTheDocument()
    expect(within(card).getByRole('img', { name: /CTCF loops alone.*9\.9 ± 2\.9 bead units apart/ })).toBeInTheDocument()
    expect(card).toHaveTextContent('6.0 ± 1.6 bead units with the loop and 9.9 ± 2.9 with CTCF loops alone')
  })

  it('marks A, B and the distance on the picture at the top', () => {
    renderTitle()
    const art = screen.getByRole('img', { name: /A 3D structure of the TNFRSF8 locus/ })
    expect(art).toHaveTextContent('6.0 ± 1.6')
    expect(screen.getByText('A to B 6.0 ± 1.6 bead units')).toBeInTheDocument()
    expect(screen.queryByText(/stay private/)).toBeNull()
  })

  it('opens an example as its finished job, on its tab', async () => {
    const fetchMock = mockApi([[/^\/api\/jobs$/, () => [201, { url: '/jobs/aaaa1111bbbb2222' }]]])
    renderTitle()
    fireEvent.click(screen.getByRole('button', { name: /Contacts against Hi-C/ }))
    await waitFor(() => expect(screen.getByTestId('where')).toHaveTextContent('/jobs/aaaa1111bbbb2222?tab=contacts'))
    const body = fetchMock.mock.calls.find(([u]) => String(u) === '/api/jobs')![1]!.body as FormData
    expect(body.get('region')).toBe('chr1:20,000,000-25,000,000')
    expect(body.get('n_models')).toBe('10')
    vi.unstubAllGlobals()
  })

  it('opens the TNFRSF8 example compared with CTCF loops alone', async () => {
    let n = 0
    const fetchMock = mockApi([[/^\/api\/jobs$/, () => [201, { url: n++ === 0 ? '/jobs/aaaa1111bbbb2222' : '/jobs/cccc3333dddd4444' }]]])
    renderTitle()
    fireEvent.click(screen.getByRole('button', { name: /TNFRSF8 with and without its RNAPII loop/ }))
    await waitFor(() => expect(screen.getByTestId('where')).toHaveTextContent('/jobs/aaaa1111bbbb2222?compare=%2Fjobs%2Fcccc3333dddd4444&gene=TNFRSF8'))
    const posted = fetchMock.mock.calls.filter(([u]) => String(u) === '/api/jobs').map(([, i]) => (i!.body as FormData).get('factors'))
    expect(posted).toEqual(['CTCF+RNAPII', 'CTCF'])
    expect(JSON.parse(localStorage.getItem('gnome3d.recentJobs') ?? '[]').map((j: { label: string }) => j.label)).toEqual([
      'GM12878 TNFRSF8, CTCF loops',
      'GM12878 TNFRSF8, CTCF and RNAPII loops',
    ])
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
