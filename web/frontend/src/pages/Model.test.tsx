import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { rememberJob } from '../model/recent'
import { mockApi } from '../test/api'
import { recentJobs } from '../model/recent'
import { checkCopy } from '../test/copy'
import { Model } from './Model'

function renderModel(path = '/model', onSubmitted = vi.fn()) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  return render(
    <QueryClientProvider client={client}>
      <MemoryRouter initialEntries={[path]}>
        <Model onSubmitted={onSubmitted} />
      </MemoryRouter>
    </QueryClientProvider>,
  )
}

describe('Model a region page', () => {
  beforeEach(() => {
    mockApi()
  })
  afterEach(() => {
    vi.unstubAllGlobals()
    localStorage.clear()
  })

  it('has the five steps in the design order', async () => {
    renderModel()
    const steps = await screen.findAllByRole('heading', { level: 2 })
    expect(steps.map((h) => h.textContent)).toEqual([
      '1Cell type and region',
      '2Loops',
      '3Structural variantsoptional',
      '4Contact background',
      '5Ensemble and simulation settings',
    ])
  })

  it('lists the cell types from the catalog with their data', async () => {
    renderModel()
    const card = await screen.findByRole('radio', { name: /GM12878/ })
    expect(within(card).getByText('lymphoblastoid')).toBeInTheDocument()
    expect(within(card).getByText('enhancers')).toBeInTheDocument()
    expect(screen.getByRole('radio', { name: /Your own data/ })).toBeEnabled()
  })

  it('resolves a gene name to its region', async () => {
    renderModel()
    fireEvent.change(await screen.findByLabelText('Region, chr:start-end or a gene name'), { target: { value: 'TNFRSF8' } })
    expect(await screen.findByText('chr1:11,500,000-12,600,000, 1.1 Mb')).toBeInTheDocument()
  })

  it('shows the error of a region it cannot resolve', async () => {
    renderModel()
    fireEvent.change(await screen.findByLabelText('Region, chr:start-end or a gene name'), { target: { value: 'nowhere' } })
    expect(await screen.findByText(/is neither chr:start-end nor a known gene name/)).toBeInTheDocument()
  })

  it('fills the form from an example', async () => {
    renderModel()
    fireEvent.click(await screen.findByRole('button', { name: /MYC/ }))
    expect(screen.getByLabelText('Region, chr:start-end or a gene name')).toHaveValue('chr8:127,500,000-129,000,000')
    expect(screen.getByRole('radio', { name: /GM12878/ })).toBeChecked()
  })

  it('opens filled from an example link', async () => {
    renderModel('/model?example=TNFRSF8')
    await waitFor(() => expect(screen.getByLabelText('Region, chr:start-end or a gene name')).toHaveValue('TNFRSF8'))
  })

  it('lists the recent jobs of this browser', async () => {
    rememberJob({ url: '/jobs/abc12345', label: 'GM12878 chr1:11.5-12.6 Mb', name: 'TNFRSF8 run', at: 1 })
    renderModel()
    const link = await screen.findByRole('link', { name: /TNFRSF8 run/ })
    expect(link).toHaveAttribute('href', '/jobs/abc12345')
    expect(await within(link).findByText('done')).toBeInTheDocument()
  })

  it('names the structures on the submit button', async () => {
    renderModel()
    expect(await screen.findByRole('button', { name: /Build 100 structures/ })).toBeInTheDocument()
  })

  it('posts the job, remembers it and moves to its page', async () => {
    const fetchMock = mockApi([[/^\/api\/jobs$/, () => [201, { url: '/jobs/abcdef0123456789' }]]])
    const onSubmitted = vi.fn()
    renderModel('/model?example=TNFRSF8', onSubmitted)
    fireEvent.change(await screen.findByLabelText('Job name, optional'), { target: { value: 'my run' } })
    fireEvent.change(screen.getByLabelText('polymer_exponent'), { target: { value: '0.3' } })
    fireEvent.click(screen.getByRole('button', { name: /Build 100 structures/ }))
    await waitFor(() => expect(onSubmitted).toHaveBeenCalledWith('/jobs/abcdef0123456789'))
    const call = fetchMock.mock.calls.find(([u]) => String(u) === '/api/jobs')!
    const body = call[1]!.body as FormData
    expect(body.get('cell_type')).toBe('GM12878')
    expect(body.get('region')).toBe('TNFRSF8')
    expect(body.get('factors')).toBe('CTCF+RNAPII')
    expect(body.get('background')).toBe('hic')
    expect(body.get('n_models')).toBe('100')
    expect(JSON.parse(String(body.get('overrides')))).toEqual({ 'distance.polymer_exponent': '0.3' })
    expect(recentJobs()[0]).toMatchObject({ url: '/jobs/abcdef0123456789', name: 'my run' })
  })

  it('shows a refusal at the step of its field', async () => {
    mockApi([[/^\/api\/jobs$/, () => [400, { error: 'distance.polymer_exponent takes a number from 0 to 0.6, not 9', field: 'overrides' }]]])
    renderModel('/model?example=TNFRSF8')
    fireEvent.click(await screen.findByRole('button', { name: /Build 100 structures/ }))
    const alert = await screen.findByRole('alert')
    expect(alert).toHaveTextContent('takes a number from 0 to 0.6')
    expect(alert.closest('section')?.querySelector('h2')?.textContent).toBe('5Ensemble and simulation settings')
  })

  it('opens the own data dialog from the card and from the loops option', async () => {
    renderModel()
    fireEvent.click(await screen.findByRole('radio', { name: /Your own data/ }))
    expect(screen.getByRole('dialog', { name: 'Your own data' })).toBeInTheDocument()
    fireEvent.click(screen.getByRole('button', { name: 'Cancel' }))
    expect(screen.queryByRole('dialog')).not.toBeInTheDocument()
    fireEvent.click(screen.getByRole('radio', { name: /Your own loops/ }))
    expect(screen.getByRole('dialog', { name: 'Your own data' })).toBeInTheDocument()
  })

  it('posts the uploads of the dialog as own', async () => {
    const fetchMock = mockApi([
      [/^\/api\/uploads$/, () => [201, { id: 'loopsupload1', kind: 'loops', name: 'my_loops.bedpe', size: 1000, ok: true, summary: '12 loops read.', error: null, samples: [] }]],
      [/^\/api\/uploads\/[\w-]+\/preview$/, () => [200, []]],
      [/^\/api\/jobs$/, () => [201, { url: '/p/token' }]],
    ])
    renderModel('/model?example=TNFRSF8')
    fireEvent.click(await screen.findByRole('radio', { name: /Your own loops/ }))
    fireEvent.change(screen.getByLabelText(/Add a factor, or drop a BEDPE file here/), { target: { files: [new File(['x'], 'my_loops.bedpe')] } })
    await screen.findByText('12 loops read.')
    fireEvent.click(screen.getByRole('button', { name: 'Use these loops' }))
    expect(screen.queryByRole('dialog')).not.toBeInTheDocument()
    expect(screen.getByRole('radio', { name: /Your own loops/ })).toHaveAttribute('aria-checked', 'true')
    expect(screen.getByText(/CTCF from my_loops.bedpe/)).toBeInTheDocument()
    fireEvent.click(screen.getByRole('button', { name: /Build 100 structures/ }))
    await waitFor(() => expect(fetchMock.mock.calls.some(([u]) => String(u) === '/api/jobs')).toBe(true))
    const body = fetchMock.mock.calls.find(([u]) => String(u) === '/api/jobs')![1]!.body as FormData
    expect(JSON.parse(String(body.get('own')))).toEqual({ loops: [{ factor: 'CTCF', upload: 'loopsupload1' }], anchors: null, contacts: null, enhancers: null })
    expect(body.get('cell_type')).toBe('GM12878')
  })

  it('applies a dropped VCF through the variants dialog', async () => {
    const fetchMock = mockApi([
      [/^\/api\/uploads$/, () => [201, { id: 'vcfupload1', kind: 'vcf', name: 'mine.vcf.gz', size: 1000, ok: true, summary: 'VCF with 1 sample.', error: null, samples: ['HG00733'] }]],
      [/^\/api\/variants\/check$/, () => [200, { summary: '1 variant of HG00733 on the first haplotype lies inside chr1:11,500,000-12,600,000.', applied: { DEL: 1 }, skipped: {}, variants: [{ type: 'DEL', pos: 11_812_400, end: 11_829_600, length: 17_200, genotype: '1|0', change: 'no anchor inside' }] }]],
      [/^\/api\/jobs$/, () => [201, { url: '/p/token' }]],
    ])
    renderModel('/model?example=TNFRSF8')
    await screen.findByText('chr1:11,500,000-12,600,000, 1.1 Mb')
    fireEvent.change(screen.getByLabelText(/Or drop your own VCF here/), { target: { files: [new File(['x'], 'mine.vcf.gz')] } })
    expect(screen.getByRole('dialog', { name: 'Structural variants' })).toBeInTheDocument()
    fireEvent.click(await screen.findByRole('button', { name: 'Apply 1 variant' }))
    expect(screen.queryByRole('dialog')).not.toBeInTheDocument()
    fireEvent.click(screen.getByRole('button', { name: /Build 100 structures/ }))
    await waitFor(() => expect(fetchMock.mock.calls.some(([u]) => String(u) === '/api/jobs')).toBe(true))
    const body = fetchMock.mock.calls.find(([u]) => String(u) === '/api/jobs')![1]!.body as FormData
    expect(body.get('vcf_upload')).toBe('vcfupload1')
    expect(body.get('vcf')).toBeNull()
    expect(body.get('sample')).toBe('HG00733')
    expect(body.get('sv_set')).toBe('')
  })

  it('posts your own cell type with a listed loop set and its contacts', async () => {
    const fetchMock = mockApi([
      [/^\/api\/uploads$/, () => [201, { id: 'contactsup1', kind: 'contacts', name: 'region.bedpe', size: 1000, ok: true, summary: '12 contact pairs read.', error: null, samples: [] }]],
      [/^\/api\/jobs$/, () => [201, { url: '/p/token' }]],
    ])
    renderModel('/model?example=TNFRSF8')
    fireEvent.click(await screen.findByRole('radio', { name: /Your own data/ }))
    fireEvent.click(screen.getByRole('tab', { name: /Contacts/ }))
    fireEvent.click(screen.getByRole('radio', { name: /BEDPE of the region/ }))
    fireEvent.change(screen.getByLabelText(/Choose a BEDPE file/), { target: { files: [new File(['x'], 'region.bedpe')] } })
    await screen.findByText('12 contact pairs read.')
    fireEvent.click(screen.getByRole('button', { name: 'Use this map' }))
    fireEvent.click(screen.getByRole('radio', { name: /^CTCF CTCF ChIA/ }))
    fireEvent.change(screen.getByLabelText('Loops of'), { target: { value: 'H1ESC' } })
    fireEvent.click(screen.getByRole('button', { name: /Build 100 structures/ }))
    await waitFor(() => expect(fetchMock.mock.calls.some(([u]) => String(u) === '/api/jobs')).toBe(true))
    const body = fetchMock.mock.calls.find(([u]) => String(u) === '/api/jobs')![1]!.body as FormData
    expect(body.get('cell_type')).toBe('own')
    expect(body.get('factors')).toBe('CTCF')
    expect(JSON.parse(String(body.get('own')))).toMatchObject({ loops: [], contacts: 'contactsup1', loops_from: 'H1ESC', enhancers_from: 'none' })
  })

  it('keeps to the writing rules', async () => {
    const { container } = renderModel()
    await screen.findByRole('radio', { name: /GM12878/ })
    expect(checkCopy(container)).toEqual([])
  })
})
