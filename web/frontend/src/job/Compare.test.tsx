import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { fireEvent, render, screen, within } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { rememberJob } from '../model/recent'
import { mockApi } from '../test/api'
import { checkCopy } from '../test/copy'
import { ANNOTATIONS, GENES, KEY, STRUCTURE, SUMMARY_DONE } from '../test/job'
import { jobApi } from './api'
import { Compare } from './Compare'

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
    const choose = screen.getByRole('region', { name: 'Compare with another job' })
    expect(within(choose).getByRole('button', { name: /TNFRSF8 CTCF/ })).toBeInTheDocument()
    expect(within(choose).queryByRole('button', { name: /this one/ })).toBeNull()
    fireEvent.change(within(choose).getByPlaceholderText('Paste a job link'), { target: { value: `https://site/jobs/${OTHER}` } })
    fireEvent.click(within(choose).getByRole('button', { name: 'Compare' }))
    expect(await screen.findByText('CTCF loops')).toBeInTheDocument()
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

  it('keeps to the writing rules', async () => {
    rememberJob({ url: `/jobs/${OTHER}`, label: 'GM12878 TNFRSF8', name: 'TNFRSF8 CTCF', at: 2 })
    const { container } = renderCompare()
    fireEvent.click(screen.getByRole('button', { name: /TNFRSF8 CTCF/ }))
    await screen.findByRole('table', { name: 'Largest changes in enhancer to promoter distance' })
    expect(checkCopy(container)).toEqual([])
  })
})
