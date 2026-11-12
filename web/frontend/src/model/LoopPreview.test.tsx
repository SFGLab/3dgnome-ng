import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { render, screen } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { mockApi } from '../test/api'
import { LoopPreview } from './LoopPreview'

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

  it('names the gene of the region', async () => {
    renderPreview()
    expect(await screen.findByText('TNFRSF8')).toBeInTheDocument()
  })
})
