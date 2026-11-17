import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { fireEvent, render, screen, within } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { mockApi } from '../test/api'
import { checkCopy } from '../test/copy'
import { ANNOTATIONS, DISTANCE, GENES, KEY, STRUCTURE, SUMMARY_DONE } from '../test/job'
import { jobApi } from './api'
import { StructureTab } from './StructureTab'

vi.mock('./Scene', () => ({
  Scene: (p: { onPick?: (i: number) => void; markers?: { label: string }[] }) => (
    <div data-testid="scene">
      {[0, 1, 2].map((i) => (
        <button key={i} type="button" onClick={() => p.onPick?.(i)}>{`point ${i}`}</button>
      ))}
      <span data-testid="markers">{(p.markers ?? []).map((m) => m.label).join('')}</span>
    </div>
  ),
}))

function renderTab(onOpenGenes = vi.fn()) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  const summary = { ...SUMMARY_DONE, structures: 5, region: 'chr1:11500000-12600000' }
  return render(
    <QueryClientProvider client={client}>
      <StructureTab api={jobApi('jobs', KEY)} summary={summary} selected={2} onOpenGenes={onOpenGenes} />
    </QueryClientProvider>,
  )
}

describe('Measure, find a gene and the gene card', () => {
  let fetchMock: ReturnType<typeof mockApi>
  beforeEach(() => {
    fetchMock = mockApi([
      [/structure\/\d+$/, () => [200, STRUCTURE]],
      [/annotations$/, () => [200, ANNOTATIONS]],
      [/distance$/, () => [200, DISTANCE]],
      [/genes$/, () => [200, GENES]],
    ])
  })
  afterEach(() => vi.unstubAllGlobals())

  it('measures between two picked beads over the ensemble', async () => {
    renderTab()
    fireEvent.click(await screen.findByRole('button', { name: 'Measure' }))
    fireEvent.click(screen.getByRole('button', { name: 'point 0' }))
    fireEvent.click(screen.getByRole('button', { name: 'point 2' }))
    const card = await screen.findByRole('region', { name: 'Distance A to B' })
    expect(within(card).getByText('6.7 ± 1.9')).toBeInTheDocument()
    expect(within(card).getByText('Mean and standard deviation over 5 structures. The value in this structure is 6.1.')).toBeInTheDocument()
    expect(card.querySelectorAll('.strip .dot')).toHaveLength(5)
    expect(card.querySelector('.strip .dot.current')).not.toBeNull()
    expect(within(card).getByText(/TNFRSF8 TSS, chr1:12,063,000/)).toBeInTheDocument()
    expect(within(card).getByText(/enhancer, chr1:12,449,000/)).toBeInTheDocument()
    expect(screen.getByTestId('markers')).toHaveTextContent('AB')
    const call = fetchMock.mock.calls.find(([u]) => String(u).includes('/distance'))!
    expect(String(call[0])).toContain('a=12063000&b=12449000')
  })

  it('starts over on a third pick', async () => {
    renderTab()
    fireEvent.click(await screen.findByRole('button', { name: 'Measure' }))
    for (const i of [0, 2, 1]) fireEvent.click(screen.getByRole('button', { name: `point ${i}` }))
    expect(screen.getByTestId('markers')).toHaveTextContent(/^A$/)
  })

  it('finds a gene and shows its card', async () => {
    renderTab()
    const find = await screen.findByPlaceholderText('Find a gene')
    fireEvent.change(find, { target: { value: 'tnfrsf8' } })
    fireEvent.keyDown(find, { key: 'Enter' })
    const card = await screen.findByRole('region', { name: 'Selected gene' })
    expect(within(card).getByText('TNFRSF8')).toBeInTheDocument()
    expect(within(card).getByText('TSS chr1:12,063,302')).toBeInTheDocument()
    expect(within(card).getByText('6.7')).toBeInTheDocument()
    expect(within(card).getByText('chr1:12,449,484')).toBeInTheDocument()
    expect(within(card).getByText('385 kb')).toBeInTheDocument()
    expect(within(card).getByText('4.1 to 9.8')).toBeInTheDocument()
    expect(within(card).getByText('RNAPII loop')).toBeInTheDocument()
    expect(screen.getByTestId('markers')).toHaveTextContent('AB')
  })

  it('labels the enhancer of a selected gene as the enhancer even next to a TSS', async () => {
    fetchMock = mockApi([
      [/structure\/\d+$/, () => [200, STRUCTURE]],
      [/annotations$/, () => [200, { ...ANNOTATIONS, genes: [...ANNOTATIONS.genes, { name: 'NEARBY', tss: 12_450_000 }] }]],
      [/distance$/, () => [200, DISTANCE]],
      [/genes$/, () => [200, GENES]],
    ])
    renderTab()
    const find = await screen.findByPlaceholderText('Find a gene')
    fireEvent.change(find, { target: { value: 'TNFRSF8' } })
    fireEvent.keyDown(find, { key: 'Enter' })
    const card = await screen.findByRole('region', { name: 'Distance A to B' })
    expect(within(card).getByText(/^\s*enhancer, chr1:12,449,000/)).toBeInTheDocument()
  })

  it('opens the gene in the Genes tab', async () => {
    const onOpenGenes = vi.fn()
    renderTab(onOpenGenes)
    const find = await screen.findByPlaceholderText('Find a gene')
    fireEvent.change(find, { target: { value: 'TNFRSF8' } })
    fireEvent.keyDown(find, { key: 'Enter' })
    fireEvent.click(await screen.findByRole('button', { name: /All enhancers of this gene/ }))
    expect(onOpenGenes).toHaveBeenCalledWith('TNFRSF8')
  })

  it('keeps to the writing rules', async () => {
    const { container } = renderTab()
    fireEvent.click(await screen.findByRole('button', { name: 'Measure' }))
    fireEvent.click(screen.getByRole('button', { name: 'point 0' }))
    fireEvent.click(screen.getByRole('button', { name: 'point 2' }))
    await screen.findByRole('region', { name: 'Distance A to B' })
    expect(checkCopy(container)).toEqual([])
  })
})
