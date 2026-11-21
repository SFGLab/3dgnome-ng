import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { fireEvent, render, screen, within } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { rememberJob } from '../model/recent'
import { mockApi } from '../test/api'
import { checkCopy } from '../test/copy'
import { ANNOTATIONS, GENES, KEY, STRUCTURE, SUMMARY_DONE } from '../test/job'
import { jobApi } from './api'
import { Compare, shownRows } from './Compare'

vi.mock('./Scene', () => ({ Scene: (p: { markers?: { label: string }[] }) => <div data-testid="scene">{(p.markers ?? []).map((m) => m.label).join('')}</div> }))

const OTHER = 'fedcba9876543210'
const CTCF_SUMMARY = { ...SUMMARY_DONE, request: { ...SUMMARY_DONE.request, factors: 'CTCF' } }
const COMPARE = {
  same_region: true,
  rows: [{ gene: 'TNFRSF8', linear_bp: 385000, a: { mean: 6.7, sd: 1.9 }, b: { mean: 15.8, sd: 3.4 }, ratio: 15.8 / 6.7 }],
}

function renderCompare(gene: string | null = 'TNFRSF8') {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  return render(
    <QueryClientProvider client={client}>
      <Compare api={jobApi('jobs', KEY)} summary={SUMMARY_DONE} selected={1} gene={gene} onClose={vi.fn()} />
    </QueryClientProvider>,
  )
}

describe('Compare', () => {
  beforeEach(() => {
    mockApi([
      [new RegExp(`${OTHER}/summary$`), () => [200, CTCF_SUMMARY]],
      [/summary$/, () => [200, SUMMARY_DONE]],
      [/structure\/\d+$/, () => [200, STRUCTURE]],
      [/annotations$/, () => [200, ANNOTATIONS]],
      [new RegExp(`${OTHER}/genes$`), () => [200, [{ ...GENES[0], distance: 15.8, sd: 3.4 }]]],
      [/genes$/, () => [200, GENES]],
      [/\/api\/compare$/, () => [200, COMPARE]],
    ])
  })
  afterEach(() => {
    vi.unstubAllGlobals()
    localStorage.clear()
  })

  it('offers the recent jobs of this browser and a pasted link', async () => {
    rememberJob({ url: `/jobs/${OTHER}`, label: 'GM12878 TNFRSF8', name: 'TNFRSF8 CTCF', at: 2 })
    rememberJob({ url: `/jobs/${KEY}`, label: 'GM12878 TNFRSF8', name: 'this one', at: 3 })
    renderCompare()
    // Both sides are there from the start and the empty side has its list of jobs open.
    expect(await screen.findByText('CTCF and RNAPII loops')).toBeInTheDocument()
    expect(screen.getByText('Choose a job to compare')).toBeInTheDocument()
    const choose = screen.getByRole('group', { name: 'Jobs to compare with' })
    expect(within(choose).getByRole('button', { name: /TNFRSF8 CTCF/ })).toBeInTheDocument()
    expect(within(choose).queryByRole('button', { name: /this one/ })).toBeNull()
    fireEvent.change(within(choose).getByPlaceholderText('Paste a job link'), { target: { value: `https://site/jobs/${OTHER}` } })
    fireEvent.click(within(choose).getByRole('button', { name: 'Compare' }))
    expect(await screen.findByText('CTCF loops')).toBeInTheDocument()
    expect(screen.queryByRole('group', { name: 'Jobs to compare with' })).toBeNull()
  })

  it('opens the jobs under the head of the other side to change it', async () => {
    rememberJob({ url: `/jobs/${OTHER}`, label: 'GM12878 TNFRSF8', name: 'TNFRSF8 CTCF', at: 2 })
    rememberJob({ url: '/jobs/cccc3333dddd4444', label: 'GM12878 MYC', name: '', at: 1 })
    renderCompare()
    fireEvent.click(screen.getByRole('button', { name: /TNFRSF8 CTCF/ }))
    const head = (await screen.findByText('CTCF loops')).closest('button')!
    expect(head).toHaveAttribute('aria-expanded', 'false')
    fireEvent.click(head)
    const menu = screen.getByRole('group', { name: 'Jobs to compare with' })
    expect(within(menu).getByRole('button', { name: 'GM12878 MYC' })).toBeInTheDocument()
    expect(within(menu).queryByRole('button', { name: /TNFRSF8 CTCF/ })).toBeNull()
    expect(screen.getAllByTestId('scene')).toHaveLength(2)
    fireEvent.keyDown(head, { key: 'Escape' })
    expect(screen.queryByRole('group', { name: 'Jobs to compare with' })).toBeNull()
  })

  it('shows two panels, the linked view and the table of changes', async () => {
    rememberJob({ url: `/jobs/${OTHER}`, label: 'GM12878 TNFRSF8', name: 'TNFRSF8 CTCF', at: 2 })
    renderCompare()
    fireEvent.click(screen.getByRole('button', { name: /TNFRSF8 CTCF/ }))
    expect(await screen.findByText('CTCF and RNAPII loops')).toBeInTheDocument()
    expect(await screen.findByText('CTCF loops')).toBeInTheDocument()
    expect(screen.getAllByTestId('scene')).toHaveLength(2)
    expect(screen.getByRole('checkbox', { name: 'Link rotation and zoom' })).toBeChecked()
    expect(await screen.findByRole('checkbox', { name: 'Same region' })).toBeChecked()
    await screen.findAllByText('15.8 ± 3.4')
    expect([...document.querySelectorAll('.ab-label')].map((e) => e.textContent)).toEqual(['A to B 6.7 ± 1.9', 'A to B 15.8 ± 3.4'])
    expect(screen.getAllByTestId('scene').map((s) => s.textContent)).toEqual(['AB', 'AB'])
    const table = await screen.findByRole('table', { name: 'Largest changes in enhancer to promoter distance' })
    expect(within(table).getByText('TNFRSF8, enhancer 385 kb away')).toBeInTheDocument()
    expect(within(table).getByText('2.4 times farther')).toBeInTheDocument()
    expect(within(table).getAllByRole('columnheader').at(-1)!.textContent).toBe('CTCF against CTCF and RNAPII')
  })

  it('selects the gene of a clicked row of the table', async () => {
    rememberJob({ url: `/jobs/${OTHER}`, label: 'GM12878 TNFRSF8', name: 'TNFRSF8 CTCF', at: 2 })
    const onGene = vi.fn()
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
    render(
      <QueryClientProvider client={client}>
        <Compare api={jobApi('jobs', KEY)} summary={SUMMARY_DONE} selected={1} gene={null} onGene={onGene} onClose={vi.fn()} />
      </QueryClientProvider>,
    )
    fireEvent.click(screen.getByRole('button', { name: /TNFRSF8 CTCF/ }))
    const table = await screen.findByRole('table', { name: 'Largest changes in enhancer to promoter distance' })
    fireEvent.click(within(table).getByText('TNFRSF8, enhancer 385 kb away'))
    expect(onGene).toHaveBeenCalledWith('TNFRSF8')
    expect(screen.queryByRole('button', { name: /Clear the selection/ })).toBeNull()
  })

  it('clears the selection from the selected row and from the toolbar', async () => {
    rememberJob({ url: `/jobs/${OTHER}`, label: 'GM12878 TNFRSF8', name: 'TNFRSF8 CTCF', at: 2 })
    const onGene = vi.fn()
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
    render(
      <QueryClientProvider client={client}>
        <Compare api={jobApi('jobs', KEY)} summary={SUMMARY_DONE} selected={1} gene="TNFRSF8" onGene={onGene} onClose={vi.fn()} />
      </QueryClientProvider>,
    )
    fireEvent.click(screen.getByRole('button', { name: /TNFRSF8 CTCF/ }))
    const table = await screen.findByRole('table', { name: 'Largest changes in enhancer to promoter distance' })
    fireEvent.click(within(table).getByText('TNFRSF8, enhancer 385 kb away'))
    expect(onGene).toHaveBeenLastCalledWith(null)
    fireEvent.click(screen.getByRole('button', { name: 'Clear the selection of TNFRSF8' }))
    expect(onGene).toHaveBeenCalledTimes(2)
    // Both are drawn as the buttons of the job page, not as links.
    for (const name of ['Clear the selection of TNFRSF8', 'Close the comparison']) expect(screen.getByRole('button', { name })).toHaveClass('action')
  })

  it('compares the expression of two jobs that both have it, beside the change of distance', async () => {
    const withData = { ...SUMMARY_DONE, expression: { label: 'ENCODE RNA-seq of GM12878, 2 replicates', kind: 'counts' as const, unit: 'TPM', samples: 2, source: 'ENCODE' } }
    const change = {
      mode: 'deseq2', reason: '', label_a: 'ENCODE RNA-seq of GM12878, 2 replicates', label_b: 'ENCODE RNA-seq of H1ESC, 2 replicates', rho: -0.41,
      rows: [
        { gene: 'TNFRSF8', distance: 1.24, log2fc: -3, padj: 0.0000002, a: 80, b: 10 },
        { gene: 'MIIP', distance: -0.2, log2fc: 1, padj: 0.2, a: 5, b: 10 },
        { gene: 'DRAXIN', distance: -0.85, log2fc: 8.41, padj: 1.4e-71, a: 0.2, b: 68 },
      ],
    }
    const f = mockApi([
      [/summary$/, () => [200, withData]],
      [/structure\/\d+$/, () => [200, STRUCTURE]],
      [/annotations$/, () => [200, ANNOTATIONS]],
      [/genes$/, () => [200, GENES]],
      [/\/api\/compare\/expression$/, () => [200, change]],
      [/\/api\/compare$/, () => [200, COMPARE]],
    ])
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
    const { container } = render(
      <QueryClientProvider client={client}>
        <Compare api={jobApi('jobs', KEY)} summary={withData} selected={1} gene="TNFRSF8" initial={`/jobs/${OTHER}`} onClose={vi.fn()} />
      </QueryClientProvider>,
    )
    const block = await screen.findByRole('region', { name: 'Expression against 3D distance' })
    expect(await within(block).findByText(/The fold change and its adjusted p are those of DESeq2\./)).toHaveTextContent('Over 3 genes the change of expression and the change of distance have a rank correlation of -0.41.')
    const table = within(block).getByRole('table', { name: 'Largest changes in expression' })
    const row = within(table).getByRole('row', { name: /TNFRSF8/ })
    expect(within(row).getAllByRole('cell').map((c) => c.textContent)).toEqual(['TNFRSF8', '8.0 times lower', '2.0e-7', '2.4 times farther'])
    expect(within(within(table).getByRole('row', { name: /MIIP/ })).getAllByRole('cell')[2]).toHaveTextContent('0.200')
    // A fold change in the hundreds comes from next to no reads on one side and is capped.
    expect(within(within(table).getByRole('row', { name: /DRAXIN/ })).getAllByRole('cell')[1]).toHaveTextContent('over 100 times higher')
    expect(within(block).getByRole('img', { name: 'Change of expression against change of distance for 3 genes' }).querySelectorAll('circle')).toHaveLength(3)
    expect(String(f.mock.calls.find(([u]) => String(u).includes('/api/compare/expression'))![0])).toContain(`b=%2Fjobs%2F${OTHER}`)
    expect(checkCopy(container)).toEqual([])
  })

  it('does not ask for a comparison of expression when a job lacks it, and says what a ratio is', async () => {
    const f = mockApi([
      [new RegExp(`${OTHER}/summary$`), () => [200, CTCF_SUMMARY]],
      [/summary$/, () => [200, SUMMARY_DONE]],
      [/structure\/\d+$/, () => [200, STRUCTURE]],
      [/annotations$/, () => [200, ANNOTATIONS]],
      [/genes$/, () => [200, GENES]],
      [/\/api\/compare$/, () => [200, COMPARE]],
    ])
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
    render(
      <QueryClientProvider client={client}>
        <Compare api={jobApi('jobs', KEY)} summary={SUMMARY_DONE} selected={1} gene={null} initial={`/jobs/${OTHER}`} onClose={vi.fn()} />
      </QueryClientProvider>,
    )
    await screen.findByRole('table', { name: 'Largest changes in enhancer to promoter distance' })
    expect(screen.queryByRole('region', { name: 'Expression against 3D distance' })).toBeNull()
    expect(f.mock.calls.some(([u]) => String(u).includes('/api/compare/expression'))).toBe(false)
  })

  it('keeps the ten largest changes in their order and puts a selected gene outside them first', () => {
    const rows = Array.from({ length: 14 }, (_, i) => ({ gene: `G${i}` }))
    expect(shownRows(rows, 'G3').map((r) => r.gene)).toEqual(rows.slice(0, 10).map((r) => r.gene))
    expect(shownRows(rows, 'G12').map((r) => r.gene)).toEqual(['G12', ...rows.slice(0, 10).map((r) => r.gene)])
    expect(shownRows(rows, null)).toHaveLength(10)
  })

  it('keeps to the writing rules', async () => {
    rememberJob({ url: `/jobs/${OTHER}`, label: 'GM12878 TNFRSF8', name: 'TNFRSF8 CTCF', at: 2 })
    const { container } = renderCompare()
    fireEvent.click(screen.getByRole('button', { name: /TNFRSF8 CTCF/ }))
    await screen.findByRole('table', { name: 'Largest changes in enhancer to promoter distance' })
    expect(checkCopy(container)).toEqual([])
  })
})
