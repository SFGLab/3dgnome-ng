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
  lower_label: 'This job, sum of 100 structures',
  scc: 0.37,
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
    await screen.findByText('0.37')
    const url = String(f.mock.calls.find(([u]) => String(u).includes('/contacts'))![0])
    expect(url).toContain('region=chr1%3A11500000-11575000')
    expect(url).toContain('res=25000')
    expect(url).toContain('upper=hic')
    expect(url).toContain('lower=model')
  })

  it('shows the agreement with Hi-C and how it was computed', async () => {
    mockApi([[/contacts$/, () => [200, MAP]], [/annotations$/, () => [200, ANNOTATIONS]]])
    renderTab()
    const card = await screen.findByRole('region', { name: 'Agreement with Hi-C' })
    expect(within(card).getByText('0.37')).toBeInTheDocument()
    expect(within(card).getByText('Computed on 25 kb bins up to 2 Mb apart in this view.')).toBeInTheDocument()
    expect(document.querySelector('.map-labels')?.textContent).toBe('This job, sum of 100 structuresHi-C of GM12878')
  })

  it('keeps zooming out inside the job region', async () => {
    const f = mockApi([[/contacts$/, () => [200, MAP]], [/annotations$/, () => [200, ANNOTATIONS]]])
    renderTab()
    await screen.findByText('0.37')
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

  it('asks once for a burst of wheel steps', async () => {
    const f = mockApi([[/contacts$/, () => [200, MAP]], [/annotations$/, () => [200, ANNOTATIONS]]])
    render(
      <QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>
        <ContactMapTab api={jobApi('jobs', KEY)} summary={{ ...SUMMARY_DONE, region: 'chr1:10000000-14000000' }} />
      </QueryClientProvider>,
    )
    await screen.findByText('0.37')
    const before = f.mock.calls.filter(([u]) => String(u).includes('/contacts')).length
    const canvas = document.querySelector('canvas')!
    for (let i = 0; i < 5; i++) fireEvent.wheel(canvas, { deltaY: -100 })
    await new Promise((r) => setTimeout(r, 500))
    expect(f.mock.calls.filter(([u]) => String(u).includes('/contacts')).length - before).toBe(1)
  })

  it('keeps a typed locus inside the job region and says so', async () => {
    mockApi([[/contacts$/, () => [200, MAP]], [/annotations$/, () => [200, ANNOTATIONS]]])
    renderTab()
    await screen.findByText('0.37')
    fireEvent.change(screen.getByLabelText('Locus'), { target: { value: 'chr1:13,781,651-14,281,651' } })
    fireEvent.click(screen.getByRole('button', { name: 'Go' }))
    expect(screen.getByText('The job covers chr1:11,500,000-11,575,000. The map shows only that region.')).toBeInTheDocument()
  })

  it('shows the server message for a resolution it refuses', async () => {
    mockApi([
      [/contacts$/, (u) => (u.searchParams.get('res') === '5000' ? [400, { error: 'chr1:1-20000000 at 5 kb is more than 800 bins on a side. Use 25 kb or a shorter window.' }] : [200, MAP])],
      [/annotations$/, () => [200, ANNOTATIONS]],
    ])
    renderTab()
    await screen.findByText('0.37')
    fireEvent.change(screen.getByLabelText('Resolution'), { target: { value: '5000' } })
    expect(await screen.findByText(/more than 800 bins on a side/)).toBeInTheDocument()
  })

  it('keeps to the writing rules', async () => {
    mockApi([[/contacts$/, () => [200, MAP]], [/annotations$/, () => [200, ANNOTATIONS]]])
    const { container } = renderTab()
    await screen.findByText('0.37')
    expect(checkCopy(container)).toEqual([])
  })
})
