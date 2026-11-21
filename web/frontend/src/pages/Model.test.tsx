import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { rememberJob } from '../model/recent'
import { mockApi } from '../test/api'
import { SUMMARY_DONE } from '../test/job'
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

  it('has the six steps in the design order', async () => {
    renderModel()
    const steps = await screen.findAllByRole('heading', { level: 2 })
    expect(steps.map((h) => h.textContent)).toEqual([
      '1Cell type and region',
      '2Loops',
      '3Structural variantsoptional',
      '4Contact background',
      '5Gene expressionoptional',
      '6Ensemble and simulation settings',
    ])
  })

  it('lists the cell types from the catalog with their data', async () => {
    renderModel()
    const card = await screen.findByRole('radio', { name: /GM12878/ })
    expect(within(card).getByText('lymphoblastoid')).toBeInTheDocument()
    expect(within(card).getByText('enhancers')).toBeInTheDocument()
    // The own data card has a chip for each kind of data given, and none before.
    const own = screen.getByRole('radio', { name: /Your own data/ })
    expect(own).toBeEnabled()
    expect(own.querySelector('.chip')).toBeNull()
  })

  it('resolves a gene name to its region', async () => {
    renderModel()
    // A gene name reaches the form once the field is left.
    const field = await screen.findByLabelText('Region, chr:start-end or a gene name')
    fireEvent.change(field, { target: { value: 'TNFRSF8' } })
    fireEvent.blur(field)
    expect(await screen.findByText('chr1:11,500,000-12,600,000, 1.1 Mb')).toBeInTheDocument()
  })

  it('shows the error of a region it cannot resolve', async () => {
    renderModel()
    const field = await screen.findByLabelText('Region, chr:start-end or a gene name')
    fireEvent.change(field, { target: { value: 'nowhere' } })
    expect(screen.queryByText(/is neither chr:start-end nor a known gene name/)).toBeNull()
    fireEvent.keyDown(field, { key: 'Enter' })
    expect(await screen.findByText(/is neither chr:start-end nor a known gene name/)).toBeInTheDocument()
  })

  it('keeps the loops under a region longer than the limit and takes them away for any other refusal', async () => {
    vi.unstubAllGlobals()
    mockApi([[/^\/api\/region$/, (u) => {
      const q = u.searchParams.get('q') ?? ''
      if (q.startsWith('chr11')) return [400, { error: `'${q}' spans 34,307,000 bp, over the 20,000,000 bp limit` }]
      return [400, { error: `'${q}' is neither chr:start-end nor a known gene name` }]
    }]])
    const { container } = renderModel()
    const field = await screen.findByLabelText('Region, chr:start-end or a gene name')
    fireEvent.change(field, { target: { value: 'chr11:37,615,000-71,922,000' } })
    expect(await screen.findByRole('status')).toHaveTextContent('This region is 34.3 Mb long and a job takes 20 Mb at most. Drag an edge inwards to shorten it.')
    // The line under the loops is the one message. The field is marked and does not repeat it.
    expect(screen.queryByText(/over the 20,000,000 bp limit/)).toBeNull()
    expect(container.querySelector('.region-input')).toHaveClass('error')
    expect(container.querySelector('.loop-preview')).toHaveClass('too-long')
    expect(screen.getByRole('slider', { name: 'End of the region' })).toHaveAttribute('aria-valuenow', '71922000')
    // The button of the line brings the region inside the limit and the field follows.
    fireEvent.click(screen.getByRole('button', { name: 'Keep the first 20 Mb' }))
    expect(field).toHaveValue('chr11:37,615,000-57,615,000')
    fireEvent.change(field, { target: { value: 'nowhere' } })
    fireEvent.keyDown(field, { key: 'Enter' })
    expect(await screen.findByText(/is neither chr:start-end nor a known gene name/)).toBeInTheDocument()
    expect(container.querySelector('.loop-preview')).toBeNull()
  })

  it('fills the form from an example', async () => {
    renderModel()
    fireEvent.click(await screen.findByRole('button', { name: /MYC/ }))
    expect(screen.getByLabelText('Region, chr:start-end or a gene name')).toHaveValue('chr8:127,000,000-128,500,000')
    expect(screen.getByRole('radio', { name: /GM12878/ })).toBeChecked()
  })

  it('opens filled from an example link', async () => {
    renderModel('/model?example=TNFRSF8')
    await waitFor(() => expect(screen.getByLabelText('Region, chr:start-end or a gene name')).toHaveValue('TNFRSF8'))
  })

  it('fills the form with the request of a job it is opened from', async () => {
    const request = { cell_type: 'H1ESC', region: 'chr8:127,500,000-129,000,000', factors: 'CTCF', background: 'none', sv_set: 'HGSVC', sample: 'HG00733', haplotype: 1, n_models: 7, overrides: { 'springs.gyration_weight': '0.3' } }
    const fetchMock = mockApi([
      [/^\/api\/jobs\/abcdef0123456789\/summary$/, () => [200, { ...SUMMARY_DONE, request }]],
      [/^\/api\/jobs$/, () => [201, { url: '/jobs/new0000000000000' }]],
    ])
    renderModel(`/model?from=${encodeURIComponent('/jobs/abcdef0123456789')}`)
    expect(await screen.findByRole('status')).toHaveTextContent('The form is filled with the request of job abcdef01. Change what you need and start a new job.')
    await waitFor(() => expect(screen.getByLabelText('Region, chr:start-end or a gene name')).toHaveValue('chr8:127,500,000-129,000,000'))
    expect(screen.getByRole('radio', { name: /H1ESC/ })).toHaveAttribute('aria-checked', 'true')
    expect(within(screen.getByRole('radiogroup', { name: 'Contact background' })).getByRole('radio', { name: /^None/ })).toHaveAttribute('aria-checked', 'true')
    // The new job is posted with the request of the old one.
    fireEvent.click(await screen.findByRole('button', { name: /Build 7 structures/ }))
    await waitFor(() => expect(fetchMock.mock.calls.some(([u, o]) => String(u).endsWith('/api/jobs') && (o as RequestInit)?.method === 'POST')).toBe(true))
    const body = fetchMock.mock.calls.find(([u, o]) => String(u).endsWith('/api/jobs') && (o as RequestInit)?.method === 'POST')![1]!.body as FormData
    expect(Object.fromEntries(['cell_type', 'region', 'factors', 'background', 'sv_set', 'sample', 'haplotype', 'n_models'].map((k) => [k, body.get(k)]))).toEqual({
      cell_type: 'H1ESC', region: 'chr8:127,500,000-129,000,000', factors: 'CTCF', background: 'none', sv_set: 'HGSVC', sample: 'HG00733', haplotype: '1', n_models: '7',
    })
  })

  it('says which files a form filled from a job lacks, and when the job is not found', async () => {
    const request = { cell_type: 'GM12878', region: 'TNFRSF8', factors: 'CTCF+RNAPII', background: 'hic', sv_set: '', sample: 'HG00733', haplotype: 1, n_models: 10 }
    mockApi([
      [/^\/api\/p\/tokentokentoken1\/summary$/, () => [200, { ...SUMMARY_DONE, request }]],
      [/^\/api\/jobs\/gone000000000000\/summary$/, () => [404, { error: 'no such job' }]],
    ])
    const first = renderModel(`/model?from=${encodeURIComponent('/p/tokentokentoken1')}`)
    expect(await screen.findByRole('status')).toHaveTextContent('That job read uploaded files. They are not filled in. Add them again in their steps.')
    await waitFor(() => expect(screen.getByLabelText('Region, chr:start-end or a gene name')).toHaveValue('TNFRSF8'))
    first.unmount()
    renderModel(`/model?from=${encodeURIComponent('/jobs/gone000000000000')}`)
    expect(await screen.findByText('The job of this link was not found and the form is not filled.')).toBeInTheDocument()
    // A link that is not a job of this site asks for nothing.
    expect(screen.getByLabelText('Region, chr:start-end or a gene name')).toHaveValue('')
  })

  it('offers the hosted expression of the cell type as an opt in and posts the chosen source', async () => {
    const options = [
      { id: 'ENCODE', label: 'ENCODE RNA-seq of GM12878, 2 replicates', kind: 'counts', unit: '', columns: ['a', 'b'], other_cell_type: false },
      { id: 'MAGE', label: 'MAGE of HG00512, 1 library', kind: 'values', unit: 'TMM', columns: ['HG00512'], other_cell_type: true },
    ]
    const fetchMock = mockApi([
      [/^\/api\/expression$/, () => [200, { options }]],
      [/^\/api\/jobs$/, () => [201, { url: '/jobs/new0000000000000' }]],
    ])
    const { container } = renderModel('/model?example=TNFRSF8')
    const group = await screen.findByRole('radiogroup', { name: 'Gene expression' })
    // None is chosen until the reader picks a source.
    expect(within(group).getByRole('radio', { name: /^None/ })).toHaveAttribute('aria-checked', 'true')
    expect(await within(group).findByRole('radio', { name: /ENCODE RNA-seq of GM12878, 2 replicates/ })).toHaveTextContent('compared with DESeq2')
    expect(screen.queryByText(/is of a lymphoblastoid cell line/)).toBeNull()
    // A person's lymphoblastoid line beside another cell type gets a warning when it is chosen.
    fireEvent.click(within(group).getByRole('radio', { name: /MAGE of HG00512/ }))
    expect(screen.getByText('This RNA-seq is of a lymphoblastoid cell line. GM12878 is another cell type and its expression differs.')).toBeInTheDocument()
    fireEvent.click(within(group).getByRole('radio', { name: /ENCODE RNA-seq/ }))
    expect(screen.queryByText(/is of a lymphoblastoid cell line/)).toBeNull()
    expect(checkCopy(container)).toEqual([])
    fireEvent.click(screen.getByRole('button', { name: /Build \d+ structures/ }))
    await waitFor(() => expect(fetchMock.mock.calls.some(([u, o]) => String(u).endsWith('/api/jobs') && (o as RequestInit)?.method === 'POST')).toBe(true))
    const body = fetchMock.mock.calls.find(([u, o]) => String(u).endsWith('/api/jobs') && (o as RequestInit)?.method === 'POST')![1]!.body as FormData
    expect(body.get('expression')).toBe('ENCODE')
  })

  it('takes an uploaded expression table through its dialog, with the chosen samples and what the values are', async () => {
    const upload = { id: 'upl0expression00', kind: 'expression', name: 'counts.tsv', size: 20480, ok: true, summary: '400 genes by name and 3 samples. Its values are whole numbers, as read counts are.', error: null, samples: ['s1', 's2', 's3'], counts: true }
    const fetchMock = mockApi([
      [/^\/api\/uploads$/, () => [201, upload]],
      [/^\/api\/jobs$/, () => [201, { url: '/p/private0000000000000000' }]],
    ])
    const { container } = renderModel('/model?example=TNFRSF8')
    const group = await screen.findByRole('radiogroup', { name: 'Gene expression' })
    fireEvent.click(within(group).getByRole('radio', { name: /^Your own table/ }))
    const dialog = screen.getByRole('dialog', { name: 'Your own expression table' })
    // The card is not chosen and the table cannot be used before a file passes its check.
    expect(within(group).getByRole('radio', { name: /^Your own table/ })).toHaveAttribute('aria-checked', 'false')
    expect(within(dialog).getByRole('button', { name: 'Use this table' })).toBeDisabled()
    fireEvent.change(dialog.querySelector('input[type=file]') as HTMLInputElement, { target: { files: [new File(['gene\ts1\n'], 'counts.tsv')] } })
    expect(await within(dialog).findByText(upload.summary)).toBeInTheDocument()
    const samples = within(dialog).getByRole('group', { name: 'Samples of this job' })
    expect(within(samples).getAllByRole('checkbox').every((c) => (c as HTMLInputElement).checked)).toBe(true)
    fireEvent.click(within(samples).getByRole('checkbox', { name: 's2' }))
    // A table of whole numbers starts as read counts.
    expect(within(dialog).getByRole('combobox', { name: 'The values are' })).toHaveValue('counts')
    expect(checkCopy(container)).toEqual([])
    fireEvent.click(within(dialog).getByRole('button', { name: 'Use this table' }))
    expect(screen.queryByRole('dialog', { name: 'Your own expression table' })).toBeNull()
    expect(within(group).getByRole('radio', { name: /^Your own table/ })).toHaveAttribute('aria-checked', 'true')
    expect(screen.getByText('counts.tsv, 2 samples, read counts')).toBeInTheDocument()
    fireEvent.click(screen.getByRole('button', { name: /Build \d+ structures/ }))
    await waitFor(() => expect(fetchMock.mock.calls.some(([u, o]) => String(u).endsWith('/api/jobs') && (o as RequestInit)?.method === 'POST')).toBe(true))
    const body = fetchMock.mock.calls.find(([u, o]) => String(u).endsWith('/api/jobs') && (o as RequestInit)?.method === 'POST')![1]!.body as FormData
    expect([body.get('expression'), body.get('expression_upload'), body.get('expression_columns'), body.get('expression_kind')]).toEqual(['own', 'upl0expression00', '["s1","s3"]', 'counts'])
    // Change opens the dialog on the same table and the cross takes the table away.
    fireEvent.click(screen.getByRole('button', { name: 'Change' }))
    expect(within(screen.getByRole('dialog', { name: 'Your own expression table' })).getByRole('checkbox', { name: 's2' })).not.toBeChecked()
    fireEvent.click(within(screen.getByRole('dialog', { name: 'Your own expression table' })).getByRole('button', { name: 'Cancel' }))
    fireEvent.click(screen.getByRole('button', { name: 'Remove the table' }))
    expect(within(group).getByRole('radio', { name: /^None/ })).toHaveAttribute('aria-checked', 'true')
  })

  it('lists the recent jobs of this browser', async () => {
    rememberJob({ url: '/jobs/abc12345', label: 'GM12878 chr1:11.5-12.6 Mb', name: 'TNFRSF8 run', at: 1 })
    renderModel()
    const link = await screen.findByRole('link', { name: /TNFRSF8 run/ })
    expect(link).toHaveAttribute('href', '/jobs/abc12345')
    expect(await within(link).findByText('done')).toBeInTheDocument()
  })

  it('warns when My jobs already shows a job under the name typed', async () => {
    rememberJob({ url: '/jobs/abc12345', label: 'GM12878 chr1:11.5-12.6 Mb', name: 'TNFRSF8 run', at: 1 })
    renderModel('/model?example=TNFRSF8')
    const name = await screen.findByLabelText('Job name (optional)')
    const warning = 'My jobs already lists a job as TNFRSF8 run. Give this job a name of its own to tell the two apart.'
    expect(screen.queryByText(warning)).toBeNull()
    fireEvent.change(name, { target: { value: ' TNFRSF8 run ' } })
    expect(screen.getByText(warning)).toBeInTheDocument()
    fireEvent.change(name, { target: { value: 'TNFRSF8 run, again' } })
    expect(screen.queryByText(warning)).toBeNull()
  })

  it('warns when a job without a name would read as one My jobs has', async () => {
    rememberJob({ url: '/jobs/abc12345', label: 'GM12878 TNFRSF8', name: '', at: 1 })
    renderModel('/model?example=TNFRSF8')
    await screen.findByLabelText('Job name (optional)')
    expect(await screen.findByText(/My jobs already lists a job as GM12878 /)).toBeInTheDocument()
  })

  it('names the structures on the submit button', async () => {
    renderModel()
    expect(await screen.findByRole('button', { name: /Build 100 structures/ })).toBeInTheDocument()
  })

  it('posts the job, remembers it and moves to its page', async () => {
    const fetchMock = mockApi([[/^\/api\/jobs$/, () => [201, { url: '/jobs/abcdef0123456789' }]]])
    const onSubmitted = vi.fn()
    renderModel('/model?example=TNFRSF8', onSubmitted)
    fireEvent.change(await screen.findByLabelText('Job name (optional)'), { target: { value: 'my run' } })
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
    // The form asks for a job of its own. The same request twice gives two jobs.
    expect(body.get('fresh')).toBe('1')
    expect(JSON.parse(String(body.get('overrides')))).toEqual({ 'distance.polymer_exponent': '0.3' })
    expect(recentJobs()[0]).toMatchObject({ url: '/jobs/abcdef0123456789', name: 'my run' })
  })

  it('shows a refusal at the step of its field', async () => {
    mockApi([[/^\/api\/jobs$/, () => [400, { error: 'distance.polymer_exponent takes a number from 0 to 0.6, not 9', field: 'overrides' }]]])
    renderModel('/model?example=TNFRSF8')
    fireEvent.click(await screen.findByRole('button', { name: /Build 100 structures/ }))
    const alert = await screen.findByRole('alert')
    expect(alert).toHaveTextContent('takes a number from 0 to 0.6')
    expect(alert.closest('section')?.querySelector('h2')?.textContent).toBe('6Ensemble and simulation settings')
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
    fireEvent.click(screen.getByRole('button', { name: 'Use this data' }))
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
    fireEvent.click(screen.getByRole('button', { name: 'Use this data' }))
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
