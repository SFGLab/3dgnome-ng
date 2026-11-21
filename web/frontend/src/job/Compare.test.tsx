import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { rememberJob } from '../model/recent'
import { mockApi } from '../test/api'
import { checkCopy } from '../test/copy'
import { ANNOTATIONS, GENES, KEY, STRUCTURE, SUMMARY_DONE } from '../test/job'
import { jobApi } from '../api'
import { Compare } from './Compare'
import { sideNames } from './CompareResults'
import { csvText } from './saveCsv'
import { standalone } from './savePng'

vi.mock('./Scene', () => ({ Scene: (p: { markers?: { label: string }[] }) => <div data-testid="scene">{(p.markers ?? []).map((m) => m.label).join('')}</div> }))

const OTHER = 'fedcba9876543210'
const CTCF_SUMMARY = { ...SUMMARY_DONE, request: { ...SUMMARY_DONE.request, factors: 'CTCF' } }
const COMPARE = {
  same_region: true,
  rows: [{ gene: 'TNFRSF8', linear_bp: 385000, a: { mean: 6.7, sd: 1.9 }, b: { mean: 15.8, sd: 3.4 }, ratio: 15.8 / 6.7, p: 0.004 }],
}
const ONE_GENE = { gene: 'TNFRSF8', a: [5, 6, 7, 9], b: [12, 15, 16, 20], p: 0.004 }

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
      [/\/api\/compare\/gene$/, () => [200, ONE_GENE]],
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
    expect((await screen.findAllByText('CTCF loops')).length).toBeGreaterThan(0)
    expect(screen.queryByRole('group', { name: 'Jobs to compare with' })).toBeNull()
  })

  it('opens the jobs under the head of the other side to change it', async () => {
    rememberJob({ url: `/jobs/${OTHER}`, label: 'GM12878 TNFRSF8', name: 'TNFRSF8 CTCF', at: 2 })
    rememberJob({ url: '/jobs/cccc3333dddd4444', label: 'GM12878 MYC', name: '', at: 1 })
    renderCompare()
    fireEvent.click(screen.getByRole('button', { name: /TNFRSF8 CTCF/ }))
    const head = (await screen.findAllByText('CTCF loops')).map((e) => e.closest('button')).find(Boolean)!
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
    expect((await screen.findAllByText('CTCF and RNAPII loops')).length).toBeGreaterThan(0)
    expect((await screen.findAllByText('CTCF loops')).length).toBeGreaterThan(0)
    expect(screen.getAllByTestId('scene')).toHaveLength(2)
    expect(screen.getByRole('checkbox', { name: 'Link rotation and zoom' })).toBeChecked()
    expect(await screen.findByRole('checkbox', { name: 'Same region' })).toBeChecked()
    await screen.findAllByText('15.8 ± 3.4')
    expect([...document.querySelectorAll('.ab-label')].map((e) => e.textContent)).toEqual(['A to B 6.7 ± 1.9', 'A to B 15.8 ± 3.4'])
    expect(screen.getAllByTestId('scene').map((s) => s.textContent)).toEqual(['AB', 'AB'])
    const table = await screen.findByRole('table', { name: 'Changes in enhancer to promoter distance' })
    expect(within(table).getByText('TNFRSF8, enhancer 385 kb away')).toBeInTheDocument()
    expect(within(table).getByText('2.4 times farther')).toBeInTheDocument()
    // The two jobs are named by the loops they differ in, and every change says in which job it is.
    expect(within(table).getAllByRole('columnheader').map((h) => h.textContent)).toEqual(['Gene', 'CTCF and RNAPII loops', 'CTCF loops', 'CTCF loops against CTCF and RNAPII loops', 'Mann-Whitney p'])
    expect(within(table).getByText('0.004')).toBeInTheDocument()
    // The selected gene has its values in a card and a box plot of its distance in the structures of both jobs.
    const card = screen.getByRole('region', { name: 'Change of TNFRSF8' })
    expect(card).toHaveTextContent('2.4 times farther in CTCF loops')
    expect(card).toHaveTextContent('6.7 ± 1.9 bead units in CTCF and RNAPII loops, 15.8 ± 3.4 in CTCF loops. Mann-Whitney p 0.004.')
    const box = await screen.findByRole('img', { name: 'Enhancer to promoter distance of TNFRSF8 in the structures of CTCF and RNAPII loops and of CTCF loops' })
    expect(box.querySelectorAll('rect.quartiles')).toHaveLength(2)
    expect(box.querySelectorAll('.box circle.structure')).toHaveLength(8)
    expect([...box.querySelectorAll('.axis-title')].map((x) => x.textContent)).toEqual(['3D distance, bead units'])
    expect(within(box).getAllByText('4 structures')).toHaveLength(2)
    expect(screen.getByRole('button', { name: 'Save the figure Enhancer to promoter distance of TNFRSF8 as a PNG' })).toBeInTheDocument()
    // The picture of a figure stands by itself: its title is written above it and it has a background.
    const picture = standalone(box as unknown as SVGSVGElement, 'Enhancer to promoter distance of TNFRSF8')
    expect(picture.querySelector('text')!.textContent).toBe('Enhancer to promoter distance of TNFRSF8')
    expect(picture.querySelector('rect')!.getAttribute('fill')).toBeTruthy()
    expect(picture.getAttribute('width')).toBe('1020')
    // The picture of each of the two structures can be saved.
    expect(screen.getByRole('button', { name: 'Save the structure of this job as a PNG' })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Save the structure of TNFRSF8 CTCF as a PNG' })).toBeInTheDocument()
    // The table is saved whole as a CSV file.
    expect(screen.getByRole('button', { name: 'Save the table as distance_changes.csv' })).toBeInTheDocument()
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
    const table = await screen.findByRole('table', { name: 'Changes in enhancer to promoter distance' })
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
    const table = await screen.findByRole('table', { name: 'Changes in enhancer to promoter distance' })
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
      volcano: { fold: 12, p: 50, points: [[-3, 6.5], [0, 0], [0.2, 0.5], [1, 0.5], [8.4, 50], [-0.4, 1]] as [number, number][] },
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
      [/\/api\/compare\/gene$/, () => [200, ONE_GENE]],
      [/\/api\/compare$/, () => [200, COMPARE]],
    ])
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
    const { container } = render(
      <QueryClientProvider client={client}>
        <Compare api={jobApi('jobs', KEY)} summary={withData} selected={1} gene="TNFRSF8" initial={`/jobs/${OTHER}`} onClose={vi.fn()} />
      </QueryClientProvider>,
    )
    // The main value is the rank correlation, as a card, and the text beside it is short.
    const card = await screen.findByRole('region', { name: 'Expression against 3D distance' })
    expect(within(card).getByText('-0.41')).toHaveClass('big-value')
    expect(card).toHaveTextContent('Of the change of expression with the change of distance over 3 genes, the other job against this job. Tested with DESeq2.')
    expect(screen.getByRole('region', { name: 'Change of TNFRSF8' })).toHaveTextContent('8.0 times lower expression in the other job')
    // The tables are two tabs. The 3D distance is shown first.
    const tabs = screen.getByRole('tablist', { name: 'Tables of the comparison' })
    expect(within(tabs).getAllByRole('tab').map((x) => x.textContent)).toEqual(['3D distance 1', 'Expression 3'])
    expect(screen.queryByRole('table', { name: 'Changes in expression' })).toBeNull()
    fireEvent.click(within(tabs).getByRole('tab', { name: 'Expression 3' }))
    expect(screen.queryByRole('table', { name: 'Changes in enhancer to promoter distance' })).toBeNull()
    const table = screen.getByRole('table', { name: 'Changes in expression' })
    expect(within(table).getAllByRole('columnheader').map((h) => h.textContent)).toEqual(['Gene', 'this job', 'the other job', 'Expression, the other job against this job', 'Adjusted p', '3D distance, the other job against this job'])
    const row = within(table).getByRole('row', { name: /TNFRSF8/ })
    expect(within(row).getAllByRole('cell').map((c) => c.textContent)).toEqual(['TNFRSF8', '80.0', '10.0', '8.0 times lower', '2.0e-7', '2.4 times farther'])
    expect(within(within(table).getByRole('row', { name: /MIIP/ })).getAllByRole('cell')[4]).toHaveTextContent('0.200')
    // A change that is not significant is grey.
    expect(within(table).getByRole('row', { name: /MIIP/ })).toHaveClass('insignificant')
    expect(row).not.toHaveClass('insignificant')
    // A fold change in the hundreds comes from next to no reads on one side and is capped.
    expect(within(within(table).getByRole('row', { name: /DRAXIN/ })).getAllByRole('cell')[3]).toHaveTextContent('over 100 times higher')
    expect(screen.getByRole('button', { name: 'Save the table as expression_changes.csv' })).toBeInTheDocument()
    // The figures are an accordion. With expression it starts on the plot of the two changes.
    const block = screen.getByRole('region', { name: 'Changes between the two jobs' })
    const heads = ['Enhancer to promoter distance of TNFRSF8', 'Change of expression against change of distance', 'Volcano plot of the change of expression']
    expect(heads.map((name) => within(block).getByRole('button', { name }).getAttribute('aria-expanded'))).toEqual(['false', 'true', 'false'])
    const scatter = within(block).getByRole('img', { name: 'Change of expression against change of distance for 3 genes' })
    expect(scatter.querySelectorAll(':scope > circle')).toHaveLength(3)
    // Each plot names its two axes, says which job is over which, has ticks on both and a key of its marks.
    expect([...scatter.querySelectorAll('.axis-title')].map((x) => x.textContent)).toEqual(['3D distance, log2 of the other job over this job', 'Expression, log2 of the other job over this job'])
    expect(scatter.querySelectorAll('.tick').length).toBeGreaterThan(5)
    expect(within(scatter).getByText('adjusted p under 0.05')).toBeInTheDocument()
    // An open figure can be saved as a PNG. Several figures can be open at once.
    expect(within(block).getAllByRole('button', { name: /^Save the figure .* as a PNG$/ })).toHaveLength(1)
    fireEvent.click(within(block).getByRole('button', { name: heads[2] }))
    expect(within(block).getByRole('img', { name: /^Change of expression against/ })).toBeInTheDocument()
    expect(within(block).getAllByRole('button', { name: /^Save the figure .* as a PNG$/ })).toHaveLength(2)
    const volcano = within(block).getByRole('img', { name: 'Volcano plot of 3 genes of the region among the genes of the test' })
    expect([...volcano.querySelectorAll('.axis-title')].map((x) => x.textContent)).toEqual(['Expression, log2 of the other job over this job', '-log10 of the adjusted p'])
    // The genes of the test are the background and the genes of the region are drawn over them.
    expect(volcano.querySelectorAll(':scope > circle.back')).toHaveLength(6)
    expect(volcano.querySelectorAll(':scope > circle.point')).toHaveLength(3)
    expect(volcano.querySelectorAll(':scope > circle.point.sure')).toHaveLength(1)
    expect(within(volcano).getByText('adjusted p 0.05')).toBeInTheDocument()
    expect(within(volcano).getByText('genes outside the region')).toBeInTheDocument()
    // The last tick of a capped axis stands for every value above it.
    expect(within(volcano).getByText('50+')).toBeInTheDocument()
    expect(within(block).getByRole('button', { name: 'Save the figure Volcano plot of the change of expression as a PNG' })).toBeInTheDocument()
    // A second click on the head of an open figure closes that figure alone.
    fireEvent.click(within(block).getByRole('button', { name: heads[2] }))
    expect(within(block).getAllByRole('img').map((x) => x.getAttribute('aria-label'))).toEqual(['Change of expression against change of distance for 3 genes'])
    // The main values are in the column of the figures, under them.
    expect(block.querySelector('.compare-side .compare-figures + .compare-cards')).not.toBeNull()
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
      [/\/api\/compare\/gene$/, () => [200, ONE_GENE]],
      [/\/api\/compare$/, () => [200, COMPARE]],
    ])
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
    render(
      <QueryClientProvider client={client}>
        <Compare api={jobApi('jobs', KEY)} summary={SUMMARY_DONE} selected={1} gene={null} initial={`/jobs/${OTHER}`} onClose={vi.fn()} />
      </QueryClientProvider>,
    )
    await screen.findByRole('table', { name: 'Changes in enhancer to promoter distance' })
    expect(screen.queryByRole('region', { name: 'Expression against 3D distance' })).toBeNull()
    expect(f.mock.calls.some(([u]) => String(u).includes('/api/compare/expression'))).toBe(false)
  })

  it('lists every gene of the comparison and shows the page of the selected one', async () => {
    const rows = Array.from({ length: 30 }, (_, k) => ({ gene: `G${k}`, linear_bp: 1000 * (k + 1), a: { mean: 2, sd: 0.5 }, b: { mean: 2 + k, sd: 0.5 }, ratio: (2 + k) / 2, p: 0.5 }))
    mockApi([
      [new RegExp(`${OTHER}/summary$`), () => [200, CTCF_SUMMARY]],
      [/summary$/, () => [200, SUMMARY_DONE]],
      [/structure\/\d+$/, () => [200, STRUCTURE]],
      [/annotations$/, () => [200, ANNOTATIONS]],
      [/genes$/, () => [200, GENES]],
      [/\/api\/compare\/gene$/, () => [200, { gene: 'G27', a: [1, 2], b: [3, 4], p: 0.33 }]],
      [/\/api\/compare$/, () => [200, { same_region: true, rows }]],
    ])
    localStorage.setItem('gnome3d.rows', '25')
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
    const view = (gene: string | null) => (
      <QueryClientProvider client={client}>
        <Compare api={jobApi('jobs', KEY)} summary={SUMMARY_DONE} selected={1} gene={gene} initial={`/jobs/${OTHER}`} onClose={vi.fn()} />
      </QueryClientProvider>
    )
    const { rerender } = render(view(null))
    const table = await screen.findByRole('table', { name: 'Changes in enhancer to promoter distance' })
    // No gene is left out of the table. Thirty genes are two pages of 25.
    expect(screen.getByRole('tab', { name: '3D distance 30' })).toHaveAttribute('aria-selected', 'true')
    expect(within(table).getAllByRole('row')).toHaveLength(26)
    expect(within(table).queryByText(/^G27,/)).toBeNull()
    expect(screen.getByText(/Select a gene in a table or a plot/)).toBeInTheDocument()
    // A gene selected in a plot or on the other tab is on a later page. The table goes to that page.
    rerender(view('G27'))
    expect(await within(table).findByText(/^G27,/)).toBeInTheDocument()
    expect(within(table).getByRole('row', { name: /^G27,/ })).toHaveAttribute('aria-selected', 'true')
  })

  it('writes a table as CSV, with quotes around a cell that needs them', () => {
    expect(csvText(['gene', 'note'], [['MYC', 'a, b'], ['TP53', 'say "x"'], ['X', null], ['Y', 1.5]])).toBe('gene,note\r\nMYC,"a, b"\r\nTP53,"say ""x"""\r\nX,\r\nY,1.5\r\n')
  })

  it('states the two enhancers of a gene of two cell types and offers the enhancers both have', async () => {
    const h1 = { ...SUMMARY_DONE, request: { ...SUMMARY_DONE.request, cell_type: 'H1ESC' } }
    const own = { same_region: true, differ: 1, shared: false, shared_enhancers: null, rows: [{ gene: 'TNFRSF8', linear_bp: 20000, linear_bp_b: 849, enhancer_bead_a: 3, enhancer_bead_b: 4, a: { mean: 2.5, sd: 0.7 }, b: { mean: 1.0, sd: 0.1 }, ratio: 0.4, p: 0.0002 }] }
    const both = { same_region: true, differ: 0, shared: true, shared_enhancers: 12, rows: [{ gene: 'TNFRSF8', linear_bp: 20000, linear_bp_b: 20000, enhancer_bead_a: 3, enhancer_bead_b: 3, a: { mean: 2.5, sd: 0.7 }, b: { mean: 2.6, sd: 1.2 }, ratio: 1.04, p: 0.8 }] }
    const f = mockApi([
      [new RegExp(`${OTHER}/summary$`), () => [200, h1]],
      [/summary$/, () => [200, SUMMARY_DONE]],
      [/structure\/\d+$/, () => [200, STRUCTURE]],
      [/annotations$/, () => [200, ANNOTATIONS]],
      [/genes$/, () => [200, GENES]],
      [/\/api\/compare\/gene$/, () => [200, ONE_GENE]],
      [/\/api\/compare$/, (u) => [200, u.searchParams.get('shared') === '1' ? both : own]],
    ])
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
    const { container } = render(
      <QueryClientProvider client={client}>
        <Compare api={jobApi('jobs', KEY)} summary={SUMMARY_DONE} selected={1} gene="TNFRSF8" initial={`/jobs/${OTHER}`} onClose={vi.fn()} />
      </QueryClientProvider>,
    )
    // By default each job has its own nearest enhancer, and the table says so where the two differ.
    const table = await screen.findByRole('table', { name: 'Changes in enhancer to promoter distance' })
    expect(within(table).getByText('TNFRSF8, enhancer 20 kb away in GM12878, 849 bp in H1ESC')).toBeInTheDocument()
    expect(screen.getByText('Each job measures a gene to its own nearest enhancer. 1 of 1 genes have another enhancer in H1ESC than in GM12878.')).toBeInTheDocument()
    // The distance under each structure is the one of the comparison.
    expect([...document.querySelectorAll('.ab-label')].map((e) => e.textContent)).toEqual(['A to B 2.5 ± 0.7', 'A to B 1.0 ± 0.1'])
    expect(checkCopy(container)).toEqual([])
    // The option is off at first. On, both jobs measure the nearest enhancer they share.
    const option = screen.getByRole('checkbox', { name: 'Enhancers of both jobs only' })
    expect(option).not.toBeChecked()
    fireEvent.click(option)
    expect(await screen.findByText('Each gene is measured to its nearest enhancer among the 12 that both jobs have.')).toBeInTheDocument()
    expect(await within(screen.getByRole('table', { name: 'Changes in enhancer to promoter distance' })).findByText('TNFRSF8, enhancer 20 kb away')).toBeInTheDocument()
    expect(f.mock.calls.some(([u]) => String(u).includes('/api/compare/gene') && String(u).includes('shared=1'))).toBe(true)
    expect(checkCopy(container)).toEqual([])
  })

  it('does not mark a gene that has no enhancer both jobs have', async () => {
    const h1 = { ...SUMMARY_DONE, request: { ...SUMMARY_DONE.request, cell_type: 'H1ESC' } }
    const row = { gene: 'MIIP', linear_bp: 1000, linear_bp_b: 1000, enhancer_bead_a: 3, enhancer_bead_b: 3, a: { mean: 2, sd: 0.5 }, b: { mean: 3, sd: 0.5 }, ratio: 1.5, p: 0.2 }
    mockApi([
      [new RegExp(`${OTHER}/summary$`), () => [200, h1]],
      [/summary$/, () => [200, SUMMARY_DONE]],
      [/structure\/\d+$/, () => [200, STRUCTURE]],
      [/annotations$/, () => [200, ANNOTATIONS]],
      [/genes$/, () => [200, GENES]],
      [/\/api\/compare\/gene$/, () => [404, { error: 'TNFRSF8 is not a gene of both jobs' }]],
      [/\/api\/compare$/, (u) => [200, { same_region: true, differ: 0, shared: u.searchParams.get('shared') === '1', shared_enhancers: 5, rows: [row] }]],
    ])
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
    const { container } = render(
      <QueryClientProvider client={client}>
        <Compare api={jobApi('jobs', KEY)} summary={SUMMARY_DONE} selected={1} gene="TNFRSF8" initial={`/jobs/${OTHER}`} onClose={vi.fn()} />
      </QueryClientProvider>,
    )
    // With each job's own enhancer the gene is marked from the gene table of each job.
    await screen.findByRole('table', { name: 'Changes in enhancer to promoter distance' })
    await waitFor(() => expect(screen.getAllByTestId('scene').map((s) => s.textContent)).toEqual(['AB', 'AB']))
    fireEvent.click(screen.getByRole('checkbox', { name: 'Enhancers of both jobs only' }))
    expect(await screen.findByText('TNFRSF8 has no enhancer that both jobs have. It is not in this comparison and is not marked in the structures.')).toBeInTheDocument()
    expect(screen.getAllByTestId('scene').map((s) => s.textContent)).toEqual(['', ''])
    expect(document.querySelectorAll('.ab-label')).toHaveLength(0)
    expect(checkCopy(container)).toEqual([])
  })

  it('does not offer the shared enhancers to two jobs of one cell type', async () => {
    rememberJob({ url: `/jobs/${OTHER}`, label: 'GM12878 TNFRSF8', name: 'TNFRSF8 CTCF', at: 2 })
    renderCompare()
    fireEvent.click(screen.getByRole('button', { name: /TNFRSF8 CTCF/ }))
    await screen.findByRole('table', { name: 'Changes in enhancer to promoter distance' })
    expect(screen.queryByRole('checkbox', { name: 'Enhancers of both jobs only' })).toBeNull()
    expect(screen.queryByText(/its own nearest enhancer/)).toBeNull()
  })

  it('names two jobs by the first thing they differ in', () => {
    const job = (request: Partial<typeof SUMMARY_DONE.request>) => ({ ...SUMMARY_DONE, request: { ...SUMMARY_DONE.request, ...request } })
    expect(sideNames(job({ cell_type: 'GM12878' }), job({ cell_type: 'H1ESC', factors: 'CTCF' }))).toEqual(['GM12878', 'H1ESC'])
    expect(sideNames(job({ sample: '' }), job({ sample: 'HG00512' }))).toEqual(['no variants', 'HG00512'])
    expect(sideNames(job({ factors: 'CTCF' }), job({ factors: 'CTCF+RNAPII' }))).toEqual(['CTCF loops', 'CTCF and RNAPII loops'])
    expect(sideNames(job({ background: 'hic' }), job({ background: 'none' }))).toEqual(['Hi-C', 'no contact map'])
    expect(sideNames(job({}), job({}))).toEqual(['this job', 'the other job'])
  })

  it('keeps to the writing rules', async () => {
    rememberJob({ url: `/jobs/${OTHER}`, label: 'GM12878 TNFRSF8', name: 'TNFRSF8 CTCF', at: 2 })
    const { container } = renderCompare()
    fireEvent.click(screen.getByRole('button', { name: /TNFRSF8 CTCF/ }))
    await screen.findByRole('table', { name: 'Changes in enhancer to promoter distance' })
    expect(checkCopy(container)).toEqual([])
  })
})
