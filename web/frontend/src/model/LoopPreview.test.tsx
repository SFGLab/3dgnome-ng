import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { fireEvent, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { mockApi } from '../test/api'
import { around, LoopPreview } from './LoopPreview'

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
    // The view is 3.3 Mb over 780 units, so 78 px are 330 kb.
    svg.getBoundingClientRect = () => ({ left: 0, top: 0, width: 780, height: 150, right: 780, bottom: 150, x: 0, y: 0, toJSON: () => ({}) })
    const end = screen.getByRole('slider', { name: 'End of the region' })
    expect(end).toHaveAttribute('aria-valuenow', '12600000')
    // jsdom's pointer events carry no position, so mouse events of the same names stand in.
    const at = (el: Element, type: string, clientX: number) => fireEvent(el, new MouseEvent(type, { clientX, bubbles: true }))
    at(end, 'pointerdown', 520)
    at(end, 'pointermove', 442)
    // While the edge moves the count follows it: the second CTCF loop ends at 12.4 Mb.
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
