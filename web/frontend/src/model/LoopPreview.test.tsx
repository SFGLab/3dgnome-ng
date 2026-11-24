import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { mockApi } from '../test/api'
import { LoopPreview } from './LoopPreview'
import { around } from './loopWindow'

function renderPreview(factors: 'CTCF' | 'RNAPII' | 'CTCF+RNAPII' = 'CTCF+RNAPII') {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  return render(
    <QueryClientProvider client={client}>
      <LoopPreview cell="GM12878" region={{ chrom: 'chr1', start: 11500000, end: 12600000, label: '', gene: 'TNFRSF8' }} factors={factors} />
    </QueryClientProvider>,
  )
}

describe('LoopPreview', () => {
  beforeEach(() => {
    mockApi()
  })
  afterEach(() => vi.unstubAllGlobals())

  it('draws CTCF arcs above the axis and RNAPII arcs below', async () => {
    const { container } = renderPreview()
    await screen.findByText('2 CTCF loops')
    expect(container.querySelectorAll('path.arc.ctcf')).toHaveLength(2)
    expect(container.querySelectorAll('path.arc.rnapii')).toHaveLength(1)
    for (const p of container.querySelectorAll('path.arc.ctcf')) expect(p.getAttribute('d')).toMatch(/ 0 0 1 /)
    for (const p of container.querySelectorAll('path.arc.rnapii')) expect(p.getAttribute('d')).toMatch(/ 0 0 0 /)
  })

  it('keys the loop counts', async () => {
    renderPreview()
    expect(await screen.findByText('2 CTCF loops')).toBeInTheDocument()
    expect(screen.getByText('1 RNAPII loop')).toBeInTheDocument()
  })

  it('gives every arc its ends and count', async () => {
    const { container } = renderPreview()
    await screen.findByText('2 CTCF loops')
    expect(container.querySelector('path.arc.rnapii title')?.textContent).toBe('RNAPII loop, 11,700,000 to 12,300,000, 4 PETs')
  })

  it('draws a factor that is not chosen faint', async () => {
    const { container } = renderPreview('CTCF')
    await screen.findByText('2 CTCF loops')
    expect(container.querySelector('path.arc.rnapii')).toHaveClass('faint')
    expect(container.querySelector('path.arc.ctcf')).not.toHaveClass('faint')
  })

  it('draws the region with its own length on either side', () => {
    expect(around(11_500_000, 12_600_000)).toEqual([10_400_000, 13_700_000])
    // A region at the start of a chromosome keeps a view three times its length.
    expect(around(200_000, 1_200_000)).toEqual([0, 3_000_000])
  })

  it('counts the loops inside the region and draws the ones outside it faint', async () => {
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
    const { container } = render(
      <QueryClientProvider client={client}>
        <LoopPreview cell="GM12878" region={{ chrom: 'chr1', start: 11500000, end: 11950000, label: '' }} factors="CTCF+RNAPII" />
      </QueryClientProvider>,
    )
    expect(await screen.findByText('1 CTCF loop')).toBeInTheDocument()
    expect(screen.getByText('0 RNAPII loops')).toBeInTheDocument()
    expect(container.querySelectorAll('path.arc.ctcf.faint')).toHaveLength(1)
    expect(screen.getByText('chr1:11,500,000-11,950,000, 0.45 Mb')).toBeInTheDocument()
  })

  it('moves an edge of the region with a drag or the arrow keys and reports the coordinates', async () => {
    const onRegion = vi.fn()
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
    const { container } = render(
      <QueryClientProvider client={client}>
        <LoopPreview cell="GM12878" region={{ chrom: 'chr1', start: 11500000, end: 12600000, label: '' }} factors="CTCF+RNAPII" onRegion={onRegion} />
      </QueryClientProvider>,
    )
    await screen.findByText('2 CTCF loops')
    const svg = container.querySelector('svg')!
    // The view is 3.3 Mb over 780 units and 78 px are 330 kb.
    svg.getBoundingClientRect = () => ({ left: 0, top: 0, width: 780, height: 150, right: 780, bottom: 150, x: 0, y: 0, toJSON: () => ({}) })
    const end = screen.getByRole('slider', { name: 'End of the region' })
    expect(end).toHaveAttribute('aria-valuenow', '12600000')
    // The pointer events of jsdom have no position. Mouse events of the same names stand in.
    const at = (el: Element, type: string, clientX: number) => fireEvent(el, new MouseEvent(type, { clientX, bubbles: true }))
    at(end, 'pointerdown', 520)
    at(end, 'pointermove', 442)
    // While the edge moves the count follows it. The second CTCF loop ends at 12.4 Mb.
    expect(screen.getByText('1 CTCF loop')).toBeInTheDocument()
    expect(onRegion).not.toHaveBeenCalled()
    at(end, 'pointerup', 442)
    expect(onRegion).toHaveBeenLastCalledWith('chr1:11,500,000-12,270,000')
    fireEvent.keyDown(screen.getByRole('slider', { name: 'Start of the region' }), { key: 'ArrowRight' })
    expect(onRegion).toHaveBeenLastCalledWith('chr1:11,533,000-12,600,000')
    // The box itself moves the region and keeps its length.
    const box = container.querySelector('.region-box')!
    at(box, 'pointerdown', 400)
    at(box, 'pointermove', 322)
    at(box, 'pointerup', 322)
    expect(onRegion).toHaveBeenLastCalledWith('chr1:11,170,000-12,270,000')
  })

  it('moves the view to a region that a drag brought to its edge', async () => {
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
    const preview = (start: number, end: number) => (
      <QueryClientProvider client={client}>
        <LoopPreview cell="GM12878" region={{ chrom: 'chr1', start, end, label: '' }} factors="CTCF+RNAPII" onRegion={vi.fn()} />
      </QueryClientProvider>
    )
    const { rerender } = render(preview(11_500_000, 12_600_000))
    await screen.findByText('2 CTCF loops')
    const edge = () => screen.getByRole('slider', { name: 'End of the region' })
    // The view is the region with its length on either side, 10.4 to 13.7 Mb.
    expect(edge()).toHaveAttribute('aria-valuemax', '13700000')
    // A region inside the view leaves the view where it is.
    rerender(preview(11_500_000, 13_000_000))
    expect(edge()).toHaveAttribute('aria-valuemax', '13700000')
    // The end of the region at the end of the view: the view moves and has room again.
    rerender(preview(11_500_000, 13_700_000))
    await waitFor(() => expect(Number(edge().getAttribute('aria-valuemax'))).toBeGreaterThan(14_000_000))
  })

  it('does not let a drag make the region longer than a job takes', async () => {
    const onRegion = vi.fn()
    const { container } = render(
      <QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>
        <LoopPreview cell="GM12878" region={{ chrom: 'chr1', start: 11500000, end: 12600000, label: '' }} factors="CTCF+RNAPII" maxSpan={1_500_000} onRegion={onRegion} />
      </QueryClientProvider>,
    )
    await screen.findByText('2 CTCF loops')
    const svg = container.querySelector('svg')!
    svg.getBoundingClientRect = () => ({ left: 0, top: 0, width: 780, height: 150, right: 780, bottom: 150, x: 0, y: 0, toJSON: () => ({}) })
    const at = (el: Element, type: string, clientX: number) => fireEvent(el, new MouseEvent(type, { clientX, bubbles: true }))
    // The end is dragged to the edge of the view, 1.1 Mb further. It stops 1.5 Mb after the start.
    const end = screen.getByRole('slider', { name: 'End of the region' })
    at(end, 'pointerdown', 520)
    at(end, 'pointermove', 780)
    expect(screen.getByText('chr1:11,500,000-13,000,000, 1.50 Mb, the limit')).toBeInTheDocument()
    at(end, 'pointerup', 780)
    expect(onRegion).toHaveBeenLastCalledWith('chr1:11,500,000-13,000,000')
    // The start stops 1.5 Mb before the end.
    const start = screen.getByRole('slider', { name: 'Start of the region' })
    at(start, 'pointerdown', 260)
    at(start, 'pointermove', 0)
    at(start, 'pointerup', 0)
    expect(onRegion).toHaveBeenLastCalledWith('chr1:11,100,000-12,600,000')
    expect(container.querySelector('.loop-preview')).not.toHaveClass('too-long')
  })

  it('keeps a region longer than a job takes on the loops and lets a drag shorten it', async () => {
    const onRegion = vi.fn()
    const { container } = render(
      <QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>
        <LoopPreview cell="GM12878" region={{ chrom: 'chr1', start: 11_000_000, end: 13_000_000, label: '' }} factors="CTCF+RNAPII" maxSpan={1_500_000} onRegion={onRegion} />
      </QueryClientProvider>,
    )
    await screen.findByText('2 CTCF loops')
    expect(container.querySelector('.loop-preview')).toHaveClass('too-long')
    expect(screen.getByRole('status')).toHaveTextContent('This region is 2.00 Mb long and a job takes 1.5 Mb at most. Drag an edge inwards to shorten it.')
    // The view is three times the longest region, with the region in its middle.
    expect(around(11_000_000, 13_000_000, 4_500_000)).toEqual([9_750_000, 14_250_000])
    const svg = container.querySelector('svg')!
    svg.getBoundingClientRect = () => ({ left: 0, top: 0, width: 780, height: 150, right: 780, bottom: 150, x: 0, y: 0, toJSON: () => ({}) })
    const at = (el: Element, type: string, clientX: number) => fireEvent(el, new MouseEvent(type, { clientX, bubbles: true }))
    // An edge of the region moves inwards alone. A drag outwards leaves the region as it is.
    const end = screen.getByRole('slider', { name: 'End of the region' })
    at(end, 'pointerdown', 500)
    at(end, 'pointermove', 700)
    at(end, 'pointerup', 700)
    expect(onRegion).not.toHaveBeenCalled()
    // 780 units are 4.5 Mb and 130 units inwards are 750 kb.
    at(end, 'pointerdown', 500)
    at(end, 'pointermove', 370)
    at(end, 'pointerup', 370)
    expect(onRegion).toHaveBeenLastCalledWith('chr1:11,000,000-12,250,000')
    fireEvent.click(screen.getByRole('button', { name: 'Keep the first 1.5 Mb' }))
    expect(onRegion).toHaveBeenLastCalledWith('chr1:11,000,000-12,500,000')
  })

  it('draws a region of the whole view or more from a little before its start', () => {
    expect(around(50_000_000, 250_000_000, 60_000_000)).toEqual([47_000_000, 107_000_000])
    expect(around(1_000_000, 250_000_000, 60_000_000)).toEqual([0, 60_000_000])
  })

  it('has no sliders where the region cannot be changed', async () => {
    renderPreview()
    await screen.findByText('2 CTCF loops')
    expect(screen.queryByRole('slider')).not.toBeInTheDocument()
  })

  it('names the gene of the region', async () => {
    renderPreview()
    expect(await screen.findByText('TNFRSF8')).toBeInTheDocument()
  })
})
