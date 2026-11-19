import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { fireEvent, render, screen, within } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { mockApi } from '../test/api'
import { checkCopy } from '../test/copy'
import { ANNOTATIONS, KEY, SUMMARY_DONE } from '../test/job'
import { jobApi } from './api'
import { ContactMapTab } from './ContactMapTab'

const MAP = {
  bins: [11_500_000, 11_525_000, 11_550_000],
  upper: [[5, 2, 1], [2, 6, 3], [1, 3, 7]],
  lower: [[9, 4, 1], [4, 8, 2], [1, 2, 9]],
  upper_label: 'Hi-C of GM12878',
  lower_label: 'This job, 1/(d+1)³ between bins, mean of 100 structures',
  scc: 0.37,
  rank: 0.21,
  rank_reach: 2000000,
  rank_about: 'Hi-C of GM12878 against This job, each distance up to 2 Mb ranked on its own, Hi-C balanced, over the bins Hi-C maps',
  scc_about: 'Hi-C of GM12878, 4DN 4DNFIXP4QG5B against this job\'s contacts within the median bond, raw counts, diagonals up to 2 Mb, as the validation battery computes it',
}

function renderTab() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  return render(
    <QueryClientProvider client={client}>
      <ContactMapTab api={jobApi('jobs', KEY)} summary={{ ...SUMMARY_DONE, region: 'chr1:11500000-11575000' }} />
    </QueryClientProvider>,
  )
}

describe('ContactMapTab', () => {
  afterEach(() => vi.unstubAllGlobals())

  it('asks for the job region at 25 kb with Hi-C above and the model below', async () => {
    const f = mockApi([[/contacts$/, () => [200, MAP]], [/annotations$/, () => [200, ANNOTATIONS]]])
    renderTab()
    await screen.findByText('0.21')
    const url = String(f.mock.calls.find(([u]) => String(u).includes('/contacts'))![0])
    expect(url).toContain('region=chr1%3A11500000-11575000')
    expect(url).toContain('res=25000')
    expect(url).toContain('upper=hic')
    expect(url).toContain('lower=model')
  })

  it('shows the agreement with Hi-C and how it was computed', async () => {
    mockApi([[/contacts$/, () => [200, MAP]], [/annotations$/, () => [200, ANNOTATIONS]]])
    renderTab()
    const card = await screen.findByRole('region', { name: 'Agreement of the two halves' })
    expect(within(card).getByText('0.21')).toBeInTheDocument()
    expect(within(card).getByText('Spearman correlation')).toBeInTheDocument()
    expect(within(card).getByText('Pairs of bins the same distance apart are compared, up to 2 Mb. The fall of contacts with distance adds nothing to this number.')).toBeInTheDocument()
    expect(within(card).getByText('SCC 0.37, the same comparison on the values and not their ranks.')).toHaveAttribute('title', expect.stringContaining('within the median bond'))
    expect(within(card).getByText('0.21').parentElement).toHaveAttribute('title', expect.stringContaining('ranked on its own'))
    expect(card.textContent!.length).toBeLessThan(260)
    expect(document.querySelector('.map-labels')).toBeNull()
  })

  it('keeps zooming out inside the job region', async () => {
    const f = mockApi([[/contacts$/, () => [200, MAP]], [/annotations$/, () => [200, ANNOTATIONS]]])
    renderTab()
    await screen.findByText('0.21')
    const canvas = document.querySelector('canvas')!
    for (let i = 0; i < 6; i++) fireEvent.wheel(canvas, { deltaY: 100 })
    await new Promise((r) => setTimeout(r, 400))
    const asked = f.mock.calls.filter(([u]) => String(u).includes('/contacts')).map(([u]) => new URL(String(u), 'http://x').searchParams.get('region')!)
    for (const r of asked) {
      const [, a, b] = r.match(/:(\d+)-(\d+)/)!.map(Number)
      expect(a).toBeGreaterThanOrEqual(11_500_000)
      expect(b).toBeLessThanOrEqual(11_575_000)
    }
  })

  it('marks the part of the map a zoom in will show and no part on a zoom out', async () => {
    const wide = { ...MAP, bins: [10_000_000, 15_000_000] }
    mockApi([[/contacts$/, () => [200, wide]], [/annotations$/, () => [200, ANNOTATIONS]]])
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
    render(
      <QueryClientProvider client={client}>
        <ContactMapTab api={jobApi('jobs', KEY)} summary={{ ...SUMMARY_DONE, region: 'chr1:10000000-20000000' }} />
      </QueryClientProvider>,
    )
    await screen.findByRole('region', { name: 'Agreement of the two halves' })
    expect(document.querySelector('.zoom-box')).toBeNull()
    const canvas = document.querySelector('canvas')!
    fireEvent.wheel(canvas, { deltaY: -100 })
    const box = document.querySelector<HTMLElement>('.zoom-box')!
    expect([box.style.left, box.style.top, box.style.width, box.style.height].map(parseFloat).map(Math.round)).toEqual([10, 10, 80, 80])
    fireEvent.wheel(canvas, { deltaY: 100 })
    fireEvent.wheel(canvas, { deltaY: 100 })
    expect(document.querySelector('.zoom-box')).toBeNull()
  })

  it('asks once for a burst of wheel steps', async () => {
    const f = mockApi([[/contacts$/, () => [200, MAP]], [/annotations$/, () => [200, ANNOTATIONS]]])
    render(
      <QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>
        <ContactMapTab api={jobApi('jobs', KEY)} summary={{ ...SUMMARY_DONE, region: 'chr1:10000000-14000000' }} />
      </QueryClientProvider>,
    )
    await screen.findByText('0.21')
    const before = f.mock.calls.filter(([u]) => String(u).includes('/contacts')).length
    const canvas = document.querySelector('canvas')!
    for (let i = 0; i < 5; i++) fireEvent.wheel(canvas, { deltaY: -100 })
    await new Promise((r) => setTimeout(r, 500))
    expect(f.mock.calls.filter(([u]) => String(u).includes('/contacts')).length - before).toBe(1)
  })

  it('keeps a typed locus inside the job region and says so', async () => {
    mockApi([[/contacts$/, () => [200, MAP]], [/annotations$/, () => [200, ANNOTATIONS]]])
    renderTab()
    await screen.findByText('0.21')
    fireEvent.change(screen.getByLabelText('Locus'), { target: { value: 'chr1:13,781,651-14,281,651' } })
    fireEvent.click(screen.getByRole('button', { name: 'Go' }))
    expect(screen.getByText('The job covers chr1:11,500,000-11,575,000. The map shows only that region.')).toBeInTheDocument()
  })

  it('offers the model as the MultiMM map or as counted contacts', async () => {
    const f = mockApi([[/contacts$/, () => [200, MAP]], [/annotations$/, () => [200, ANNOTATIONS]]])
    renderTab()
    await screen.findByText('0.21')
    const lower = screen.getByLabelText('Lower half')
    expect(within(lower).getAllByRole('option').map((o) => (o as HTMLOptionElement).value)).toEqual(['model', 'model_count', 'hic', 'loops_ctcf', 'loops_rnapii'])
    fireEvent.change(lower, { target: { value: 'model_count' } })
    await new Promise((r) => setTimeout(r, 50))
    expect(f.mock.calls.some(([u]) => String(u).includes('lower=model_count'))).toBe(true)
  })

  it('starts the log colour range at a low percentile, not at zero', async () => {
    mockApi([[/contacts$/, () => [200, MAP]], [/annotations$/, () => [200, ANNOTATIONS]]])
    renderTab()
    await screen.findByText('0.21')
    expect(Number((screen.getByLabelText(/min/) as HTMLInputElement).value)).toBeGreaterThan(0)
  })

  it('offers Observed, O/E and Pearson', async () => {
    mockApi([[/contacts$/, () => [200, MAP]], [/annotations$/, () => [200, ANNOTATIONS]]])
    renderTab()
    await screen.findByText('0.21')
    expect(within(screen.getByRole('group', { name: 'Show' })).getAllByRole('button').map((b) => b.textContent)).toEqual(['Observed', 'O/E', 'Pearson'])
  })

  it('offers the loops of each factor in either half and shows their enrichment', async () => {
    const f = mockApi([[/contacts$/, (u) => [200, u.searchParams.get('upper') === 'loops_ctcf' ? { ...MAP, upper_label: 'CTCF loops, PETs at the anchors', loop_enrichment: 2.4 } : MAP]], [/annotations$/, () => [200, ANNOTATIONS]]])
    renderTab()
    await screen.findByText('0.21')
    const upper = screen.getByLabelText('Upper half')
    expect(within(upper).getAllByRole('option').map((o) => (o as HTMLOptionElement).value)).toEqual(['hic', 'model', 'model_count', 'loops_ctcf', 'loops_rnapii'])
    fireEvent.change(upper, { target: { value: 'loops_ctcf' } })
    expect(await screen.findByText('At the loops the other half is 2.4 times its expected contact at the same distance.')).toBeInTheDocument()
    expect(f.mock.calls.some(([u]) => String(u).includes('upper=loops_ctcf'))).toBe(true)
  })

  it('disables a resolution too fine for the window and says why', async () => {
    mockApi([[/contacts$/, () => [200, MAP]], [/annotations$/, () => [200, ANNOTATIONS]]])
    render(
      <QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>
        <ContactMapTab api={jobApi('jobs', KEY)} summary={{ ...SUMMARY_DONE, region: 'chr1:10000000-15000000' }} />
      </QueryClientProvider>,
    )
    await screen.findByText('0.21')
    const five = within(screen.getByLabelText('Resolution')).getByRole('option', { name: /^5 kb/ }) as HTMLOptionElement
    expect(five.disabled).toBe(true)
    expect(five.textContent).toBe('5 kb bins, zoom in to use')
  })

  it('says it is reading while a new resolution loads', async () => {
    let release = () => {}
    mockApi([
      [/contacts$/, (u) => (u.searchParams.get('res') === '50000' ? new Promise<[number, unknown]>((r) => (release = () => r([200, MAP]))) : [200, MAP])],
      [/annotations$/, () => [200, ANNOTATIONS]],
    ])
    renderTab()
    await screen.findByText('0.21')
    fireEvent.change(screen.getByLabelText('Resolution'), { target: { value: '50000' } })
    expect(await screen.findByText(/Reading the map at 50 kb\./)).toBeInTheDocument()
    release()
  })

  it('shows the server message for a resolution it refuses', async () => {
    mockApi([
      [/contacts$/, (u) => (u.searchParams.get('res') === '5000' ? [400, { error: 'chr1:1-20000000 at 5 kb is more than 800 bins on a side. Use 25 kb or a shorter window.' }] : [200, MAP])],
      [/annotations$/, () => [200, ANNOTATIONS]],
    ])
    renderTab()
    await screen.findByText('0.21')
    fireEvent.change(screen.getByLabelText('Resolution'), { target: { value: '5000' } })
    expect(await screen.findByText(/more than 800 bins on a side/)).toBeInTheDocument()
  })

  it('puts no label over the controls of the toolbar and still names each of them', async () => {
    mockApi([[/contacts$/, () => [200, MAP]], [/annotations$/, () => [200, ANNOTATIONS]]])
    const { container } = renderTab()
    await screen.findByLabelText('Locus')
    const bar = container.querySelector('.map-toolbar')!
    for (const word of ['Locus', 'Show', 'Normalization', 'Resolution']) expect(bar.textContent).not.toContain(word)
    for (const name of ['Locus', 'Normalization', 'Resolution']) expect(screen.getByLabelText(name)).toHaveAttribute('title', name)
    expect(within(screen.getByLabelText('Resolution')).getByRole('option', { name: '25 kb bins' })).toBeInTheDocument()
  })

  it('keeps to the writing rules', async () => {
    mockApi([[/contacts$/, () => [200, MAP]], [/annotations$/, () => [200, ANNOTATIONS]]])
    const { container } = renderTab()
    await screen.findByText('0.21')
    expect(checkCopy(container)).toEqual([])
  })
})
