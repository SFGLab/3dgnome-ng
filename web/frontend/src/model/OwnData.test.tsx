import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { mockApi } from '../test/api'
import { checkCopy } from '../test/copy'
import { OwnData } from './OwnData'

const RESULTS: Record<string, object> = {
  'GM12878_CTCF_loops.bedpe': { ok: true, summary: '12,433 loops read. 118 are in the region.', error: null },
  'GM12878_RNAPII_loops.bedpe': { ok: true, summary: '31,902 loops read. 64 are in the region.', error: null },
  'hichip_loops.txt': {
    ok: false, summary: '',
    error: { line: 1, found: 'Line 1 has 10 columns, and column 7 is the letter R where a count is expected.', fix: 'This is the loop format of 3D-GNOME 3.0. Move the orientations to the anchors file and keep one file per factor with seven columns.' },
  },
  'anchors.bed': { ok: true, summary: '1,204 anchors read, with orientations.', error: null },
  'enhancers.bed': { ok: true, summary: '71 enhancers read.', error: null },
}

let n = 0
function mock() {
  return mockApi([
    [/^\/api\/uploads$/, (_u, init) => {
      const file = (init!.body as FormData).get('file') as File
      return [201, { id: `upload${++n}abcd`, kind: (init!.body as FormData).get('kind'), name: file.name, size: 2_100_000, samples: [], ...RESULTS[file.name] }]
    }],
    [/^\/api\/uploads\/[\w-]+\/preview$/, () => [200, [[11_600_000, 11_900_000, 7]]]],
  ])
}

function renderDialog(onUse = vi.fn()) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  return render(
    <QueryClientProvider client={client}>
      <OwnData cell="GM12878" region={{ chrom: 'chr1', start: 11_500_000, end: 12_600_000, label: 'chr1:11,500,000-12,600,000, 1.1 Mb' }} initial={null} onClose={vi.fn()} onUse={onUse} />
    </QueryClientProvider>,
  )
}

const addLoops = (...names: string[]) =>
  fireEvent.change(screen.getByLabelText(/Add a factor, or drop a BEDPE file here/), { target: { files: names.map((x) => new File(['x'], x)) } })

describe('Your own data dialog', () => {
  beforeEach(() => {
    n = 0
    mock()
  })
  afterEach(() => vi.unstubAllGlobals())

  it('checks each loops file and names its factor', async () => {
    renderDialog()
    addLoops('GM12878_CTCF_loops.bedpe', 'GM12878_RNAPII_loops.bedpe')
    expect(await screen.findByText('12,433 loops read. 118 are in the region.')).toBeInTheDocument()
    expect(await screen.findByText('31,902 loops read. 64 are in the region.')).toBeInTheDocument()
    expect(screen.getAllByLabelText('Factor').map((i) => (i as HTMLInputElement).value)).toEqual(['CTCF', 'RNAPII'])
    expect(screen.getByRole('tab', { name: /Loops/ })).toHaveTextContent('2 files')
  })

  it('stops on a file that fails its check and goes on once it is removed', async () => {
    renderDialog()
    addLoops('GM12878_CTCF_loops.bedpe')
    await screen.findByText(/12,433 loops read/)
    addLoops('hichip_loops.txt')
    expect(await screen.findByText(/Line 1 has 10 columns/)).toBeInTheDocument()
    const row = screen.getAllByRole('group', { name: /Loops of factor/ })[1]
    fireEvent.change(within(row).getByLabelText('Factor'), { target: { value: 'H3K27ac' } })
    expect(screen.getByText('Fix or remove the H3K27ac file to continue.')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Use these loops' })).toBeDisabled()
    fireEvent.click(within(row).getByRole('button', { name: /Remove/ }))
    expect(screen.getByRole('button', { name: 'Use these loops' })).toBeEnabled()
  })

  it('takes one anchors file and one enhancers file', async () => {
    renderDialog()
    fireEvent.click(screen.getByRole('tab', { name: /Anchors/ }))
    fireEvent.change(screen.getByLabelText(/anchors file/), { target: { files: [new File(['x'], 'anchors.bed')] } })
    expect(await screen.findByText('1,204 anchors read, with orientations.')).toBeInTheDocument()
    fireEvent.click(screen.getByRole('tab', { name: /Enhancers/ }))
    fireEvent.change(screen.getByLabelText(/enhancers file/), { target: { files: [new File(['x'], 'enhancers.bed')] } })
    expect(await screen.findByText('71 enhancers read.')).toBeInTheDocument()
  })

  it('hands the uploads to the form', async () => {
    const onUse = vi.fn()
    renderDialog(onUse)
    addLoops('GM12878_CTCF_loops.bedpe')
    await screen.findByText(/12,433 loops read/)
    fireEvent.click(screen.getByRole('button', { name: 'Use these loops' }))
    await waitFor(() => expect(onUse).toHaveBeenCalled())
    expect(onUse.mock.calls[0][0].loops[0]).toMatchObject({ factor: 'CTCF', upload: { name: 'GM12878_CTCF_loops.bedpe', ok: true } })
  })

  it('keeps to the writing rules', async () => {
    const { container } = renderDialog()
    addLoops('GM12878_CTCF_loops.bedpe', 'hichip_loops.txt')
    await screen.findByText(/Line 1 has 10 columns/)
    expect(checkCopy(container)).toEqual([])
  })
})
