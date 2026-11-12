import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { render, screen, within } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { mockApi } from '../test/api'
import { checkCopy } from '../test/copy'
import { KEY } from '../test/job'
import { jobApi } from './api'
import { LogTab } from './LogTab'
import { LoopsTab } from './LoopsTab'

function wrap(node: React.ReactNode) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  return render(<QueryClientProvider client={client}>{node}</QueryClientProvider>)
}

describe('Loops and Log tabs', () => {
  afterEach(() => vi.unstubAllGlobals())

  it('lists the loops with factor, ends, span and PETs', async () => {
    mockApi([[/loops$/, () => [200, [{ factor: 'RNAPII', chrom: 'chr1', a: [12062800, 12064300], b: [12447900, 12451100], span: 383600, pets: 7 }]]]])
    const { container } = wrap(<LoopsTab api={jobApi('jobs', KEY)} />)
    const row = (await screen.findAllByRole('row'))[1]
    expect(within(row).getAllByRole('cell').map((c) => c.textContent)).toEqual(['RNAPII', 'chr1:12,062,800-12,064,300', 'chr1:12,447,900-12,451,100', '384 kb', '7'])
    expect(checkCopy(container)).toEqual([])
  })

  it('shows the engine log', async () => {
    mockApi([[/log$/, () => [200, 'reconstruct 3 structure(s)']]])
    wrap(<LogTab api={jobApi('jobs', KEY)} />)
    expect(await screen.findByText(/reconstruct 3 structure/)).toBeInTheDocument()
  })
})
