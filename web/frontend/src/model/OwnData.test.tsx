import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { mockApi } from '../test/api'
import { checkCopy } from '../test/copy'
import { readHicWindow } from './hicLocal'
import { OwnData } from './OwnData'

vi.mock('./hicLocal', () => ({ readHicWindow: vi.fn(async () => 'chr1\t11600000\t11625000\tchr1\t11900000\t11925000\t12\n') }))

const RESULTS: Record<string, object> = {
  'GM12878_CTCF_loops.bedpe': { ok: true, summary: '12,433 loops read. 118 are in the region.', error: null },
  'GM12878_RNAPII_loops.bedpe': { ok: true, summary: '31,902 loops read. 64 are in the region.', error: null },
  'hichip_loops.txt': {
    ok: false, summary: '',
    error: { line: 1, found: 'Line 1 has 10 columns, and column 7 is the letter R where a count is expected.', fix: 'This is the loop format of 3D-GNOME 3.0. Move the orientations to the anchors file and keep one file per factor with seven columns.' },
  },
  'anchors.bed': { ok: true, summary: '1,204 anchors read, with orientations.', error: null },
  'enhancers.bed': { ok: true, summary: '71 enhancers read.', error: null },
  'region.bedpe': { ok: true, summary: '5,500 contacts read.', error: null },
  'sample.chr1_11500000_12600000.bedpe': { ok: true, summary: '1 contact read.', error: null },
}

const LINK_OK = {
  id: 'linkupload1', kind: 'contacts', name: '4DNFI1UEG1HD.hic', size: 41_016_250_368, samples: [], ok: true, error: null,
  summary: 'Resolutions from 5 kb to 2.5 Mb, 25 kb is used. 1,284,512 contacts in chr1:11,500,000-12,600,000 are read and 5,500 are kept.',
}

let n = 0
function mock() {
  return mockApi([
    [/^\/api\/uploads$/, (_u, init) => {
      const file = (init!.body as FormData).get('file') as File
      return [201, { id: `upload${++n}abcd`, kind: (init!.body as FormData).get('kind'), name: file.name, size: 2_100_000, samples: [], ...RESULTS[file.name] }]
    }],
    [/^\/api\/uploads\/link$/, (_u, init) => {
      const url = String((init!.body as FormData).get('url'))
      return url.includes('example.org')
        ? [400, { error: 'example.org does not answer range requests', fix: 'The server would have to download the whole file. Host the file where range requests work, or choose Local .hic file and the page reads your region in the browser.' }]
        : [201, LINK_OK]
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
    expect(screen.getByRole('button', { name: 'Use this data' })).toBeDisabled()
    fireEvent.click(within(row).getByRole('button', { name: /Remove/ }))
    expect(screen.getByRole('button', { name: 'Use this data' })).toBeEnabled()
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

  it('offers an upload, a listed cell type or none for enhancers on your own cell type', async () => {
    const onUse = vi.fn()
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
    render(
      <QueryClientProvider client={client}>
        <OwnData cell="GM12878" region={null} initial={null} tab="enhancers" ownCell cells={['GM12878', 'H1ESC']} onClose={vi.fn()} onUse={onUse} />
      </QueryClientProvider>,
    )
    expect(screen.getByRole('radio', { name: /None/ })).toBeChecked()
    fireEvent.click(screen.getByRole('radio', { name: /Enhancers of a listed cell type/ }))
    fireEvent.change(screen.getByLabelText('Enhancers of'), { target: { value: 'H1ESC' } })
    expect(screen.getByRole('tab', { name: /Enhancers/ }).querySelector('.step-uses')?.textContent).toBe('H1ESC')
    fireEvent.click(screen.getByRole('tab', { name: /Contacts/ }))
    fireEvent.click(screen.getByRole('radio', { name: /BEDPE of the region/ }))
    fireEvent.change(screen.getByLabelText(/Choose a BEDPE file/), { target: { files: [new File(['x'], 'region.bedpe')] } })
    await screen.findByText('5,500 contacts read.')
    fireEvent.click(screen.getByRole('button', { name: 'Use this data' }))
    expect(onUse.mock.calls[0][0]).toMatchObject({ enhancersFrom: 'H1ESC', enhancers: null })
    expect(checkCopy(document.body)).toEqual([])
  })

  it('lists four steps in a side panel, each naming what the job will use', () => {
    renderDialog()
    const steps = screen.getAllByRole('tab')
    expect(steps.map((t) => t.querySelector('.step-title')?.textContent)).toEqual(['Loops', 'Anchors', 'Contacts', 'Enhancers'])
    expect(steps.map((t) => t.querySelector('.step-uses')?.textContent)).toEqual(['the listed sets', 'the loop ends', "the cell type's Hi-C", "the cell type's"])
    expect(screen.getByRole('tablist')).toHaveAttribute('aria-orientation', 'vertical')
  })

  it('says contacts are needed on your own cell type', () => {
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
    render(
      <QueryClientProvider client={client}>
        <OwnData cell="GM12878" region={null} initial={null} ownCell cells={['GM12878']} onClose={vi.fn()} onUse={vi.fn()} />
      </QueryClientProvider>,
    )
    expect(screen.getByRole('tab', { name: /Contacts/ }).querySelector('.step-uses')?.textContent).toBe('needed')
    expect(screen.getByRole('tab', { name: /Enhancers/ }).querySelector('.step-uses')?.textContent).toBe('none')
  })

  it('moves between steps with Next and Back and keeps one action', () => {
    renderDialog()
    expect(screen.getByRole('button', { name: 'Back' })).toBeDisabled()
    fireEvent.click(screen.getByRole('button', { name: 'Next' }))
    expect(screen.getByRole('tab', { name: /Anchors/ })).toHaveAttribute('aria-selected', 'true')
    expect(screen.getByRole('button', { name: 'Use this data' })).toBeInTheDocument()
    fireEvent.click(screen.getByRole('button', { name: 'Next' }))
    fireEvent.click(screen.getByRole('button', { name: 'Next' }))
    expect(screen.getByRole('tab', { name: /Enhancers/ })).toHaveAttribute('aria-selected', 'true')
    expect(screen.getByRole('button', { name: 'Next' })).toBeDisabled()
    fireEvent.click(screen.getByRole('button', { name: 'Back' }))
    expect(screen.getByRole('tab', { name: /Contacts/ })).toHaveAttribute('aria-selected', 'true')
  })

  it('names an uploaded file as what the job will use', async () => {
    renderDialog()
    addLoops('GM12878_CTCF_loops.bedpe', 'GM12878_RNAPII_loops.bedpe')
    await screen.findByText(/31,902 loops read/)
    expect(screen.getByRole('tab', { name: /Loops/ }).querySelector('.step-uses')?.textContent).toBe('2 files')
    fireEvent.click(screen.getByRole('tab', { name: /Anchors/ }))
    fireEvent.change(screen.getByLabelText(/anchors file/), { target: { files: [new File(['x'], 'anchors.bed')] } })
    await screen.findByText(/1,204 anchors read/)
    expect(screen.getByRole('tab', { name: /Anchors/ }).querySelector('.step-uses')?.textContent).toBe('anchors.bed')
  })

  it('hands the uploads to the form', async () => {
    const onUse = vi.fn()
    renderDialog(onUse)
    addLoops('GM12878_CTCF_loops.bedpe')
    await screen.findByText(/12,433 loops read/)
    fireEvent.click(screen.getByRole('button', { name: 'Use this data' }))
    await waitFor(() => expect(onUse).toHaveBeenCalled())
    expect(onUse.mock.calls[0][0].loops[0]).toMatchObject({ factor: 'CTCF', upload: { name: 'GM12878_CTCF_loops.bedpe', ok: true } })
  })

  it('offers three routes to a contact map and the .mcool hint', () => {
    renderDialog()
    fireEvent.click(screen.getByRole('tab', { name: /Contacts/ }))
    expect(screen.getAllByRole('radio').map((r) => r.querySelector('.route-title')?.textContent)).toEqual(['Link to a file', 'Local .hic file', 'BEDPE of the region'])
    expect(screen.getByText(/The server must answer range requests/)).toBeInTheDocument()
    expect(screen.getByText(/gnome3d-ng extract --mcool sample.mcool --region chr1:11500000-12600000 --out region.bedpe/)).toBeInTheDocument()
    expect(screen.getByText("The map replaces the cell type's map for this job.")).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Use this data' })).toBeInTheDocument()
  })

  it('reads a map by link and shows its summary', async () => {
    renderDialog()
    fireEvent.click(screen.getByRole('tab', { name: /Contacts/ }))
    fireEvent.change(screen.getByLabelText('Link to a .hic or .mcool file'), { target: { value: 'https://4dn-open-data-public.s3.amazonaws.com/x/4DNFI1UEG1HD.hic' } })
    fireEvent.click(screen.getByRole('button', { name: 'Read' }))
    expect(await screen.findByText('4DNFI1UEG1HD.hic, 38.2 GB')).toBeInTheDocument()
    expect(screen.getByText(/1,284,512 contacts/)).toBeInTheDocument()
    expect(screen.getByRole('tab', { name: /Contacts/ }).querySelector('.step-uses')?.textContent).toBe('4DNFI1UEG1HD.hic')
  })

  it('shows the refusal of a link with its fix', async () => {
    renderDialog()
    fireEvent.click(screen.getByRole('tab', { name: /Contacts/ }))
    fireEvent.change(screen.getByLabelText('Link to a .hic or .mcool file'), { target: { value: 'https://example.org/map.hic' } })
    fireEvent.click(screen.getByRole('button', { name: 'Read' }))
    expect(await screen.findByText('example.org does not answer range requests')).toBeInTheDocument()
    expect(screen.getByText(/Host the file where range requests work/)).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Use this data' })).toBeDisabled()
  })

  it('reads the region of a local .hic file in the browser and uploads it', async () => {
    renderDialog()
    fireEvent.click(screen.getByRole('tab', { name: /Contacts/ }))
    fireEvent.click(screen.getByRole('radio', { name: /Local .hic file/ }))
    fireEvent.change(screen.getByLabelText(/Choose a .hic file/), { target: { files: [new File(['x'], 'sample.hic')] } })
    expect(await screen.findByText('1 contact read.')).toBeInTheDocument()
    expect(vi.mocked(readHicWindow)).toHaveBeenCalledWith(expect.any(File), { chrom: 'chr1', start: 11_500_000, end: 12_600_000 }, 25_000)
  })

  it('takes a BEDPE of the region', async () => {
    renderDialog()
    fireEvent.click(screen.getByRole('tab', { name: /Contacts/ }))
    fireEvent.click(screen.getByRole('radio', { name: /BEDPE of the region/ }))
    fireEvent.change(screen.getByLabelText(/Choose a BEDPE file/), { target: { files: [new File(['x'], 'region.bedpe')] } })
    expect(await screen.findByText('5,500 contacts read.')).toBeInTheDocument()
  })

  it('keeps to the writing rules on the contacts tab', async () => {
    const { container } = renderDialog()
    fireEvent.click(screen.getByRole('tab', { name: /Contacts/ }))
    fireEvent.change(screen.getByLabelText('Link to a .hic or .mcool file'), { target: { value: 'https://example.org/map.hic' } })
    fireEvent.click(screen.getByRole('button', { name: 'Read' }))
    await screen.findByText('example.org does not answer range requests')
    expect(checkCopy(container)).toEqual([])
  })

  it('keeps to the writing rules', async () => {
    const { container } = renderDialog()
    addLoops('GM12878_CTCF_loops.bedpe', 'hichip_loops.txt')
    await screen.findByText(/Line 1 has 10 columns/)
    expect(checkCopy(container)).toEqual([])
  })
})
