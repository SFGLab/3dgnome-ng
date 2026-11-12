import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { CATALOG, mockApi } from '../test/api'
import { checkCopy } from '../test/copy'
import { VariantsDialog } from './VariantsDialog'

const REGION = { chrom: 'chr1', start: 11_500_000, end: 12_600_000, label: 'chr1:11,500,000-12,600,000, 1.1 Mb' }
const CHECK = {
  summary: '6 variants of HG00733 on the first haplotype lie inside chr1:11,500,000-12,600,000. 3 deletions, 2 insertions and 1 inversion are applied.',
  applied: { DEL: 3, INS: 2, INV: 1 },
  skipped: {},
  variants: [
    { type: 'DEL', pos: 11_812_400, end: 11_829_600, length: 17_200, genotype: '1|0', change: 'removes 1 CTCF anchor and its 3 loops' },
    { type: 'INS', pos: 11_941_880, end: 11_941_880, length: 310, genotype: '1|1', change: 'adds 1 CTCF anchor, motif score 14.2, 1 new loop' },
    { type: 'DEL', pos: 12_102_600, end: 12_106_700, length: 4_100, genotype: '1|1', change: 'shortens 2 loops' },
    { type: 'INV', pos: 12_205_300, end: 12_217_600, length: 12_300, genotype: '1|0', change: 'flips 2 anchors, replaces 4 loops' },
    { type: 'INS', pos: 12_330_050, end: 12_330_050, length: 1_100, genotype: '1|0', change: 'no CTCF motif above the cutoff' },
    { type: 'DEL', pos: 12_388_900, end: 12_391_500, length: 2_600, genotype: '1|1', change: 'no anchor inside' },
  ],
}

let checks: FormData[] = []
function mock() {
  checks = []
  return mockApi([
    [/^\/api\/uploads$/, () => [201, { id: 'vcfupload1', kind: 'vcf', name: 'HG00733.sv.vcf.gz', size: 48_000_000, ok: true, summary: 'VCF with 2 samples.', error: null, samples: ['HG00733', 'HG00512'] }]],
    [/^\/api\/uploads\/[\w-]+\/preview$/, () => [200, [[11_600_000, 11_900_000, 7]]]],
    [/^\/api\/variants\/check$/, (_u, init) => {
      checks.push(init!.body as FormData)
      return [200, CHECK]
    }],
  ])
}

function renderDialog(props: Partial<Parameters<typeof VariantsDialog>[0]> = {}) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  const onApply = vi.fn()
  const view = render(
    <QueryClientProvider client={client}>
      <VariantsDialog catalog={CATALOG} cell="GM12878" region={REGION} factors="CTCF" mode="upload" file={new File(['x'], 'HG00733.sv.vcf.gz')} svSet="" sample="" haplotype={0} onClose={vi.fn()} onApply={onApply} {...props} />
    </QueryClientProvider>,
  )
  return { ...view, onApply }
}

describe('Structural variants dialog', () => {
  beforeEach(() => mock())
  afterEach(() => vi.unstubAllGlobals())

  it('reads the samples of an uploaded VCF and checks the first one', async () => {
    renderDialog()
    expect(screen.getByRole('tab', { name: 'Upload a VCF' })).toHaveAttribute('aria-selected', 'true')
    expect(await screen.findByText(/6 variants of HG00733 on the first haplotype/)).toBeInTheDocument()
    expect(screen.getByText('HG00733.sv.vcf.gz')).toBeInTheDocument()
    expect(within(screen.getByLabelText('Sample, from the VCF header')).getAllByRole('option').map((o) => o.textContent)).toEqual(['HG00733', 'HG00512'])
    expect(checks[0].get('upload')).toBe('vcfupload1')
    expect(checks[0].get('sample')).toBe('HG00733')
    expect(checks[0].get('haplotype')).toBe('0')
  })

  it('lists each variant with its change to the loops', async () => {
    renderDialog()
    const table = await screen.findByRole('table')
    expect(within(table).getAllByRole('columnheader').map((h) => h.textContent)).toEqual(['Type', 'Position', 'Length', 'Genotype', 'Change to the loops'])
    const first = within(table).getAllByRole('row')[1]
    expect(within(first).getAllByRole('cell').map((c) => c.textContent)).toEqual(['Deletion', 'chr1:11,812,400', '17.2 kb', '1|0', 'removes 1 CTCF anchor and its 3 loops'])
    expect(within(table).getAllByRole('row')[2]).toHaveTextContent('310 bp')
    expect(screen.getByRole('img', { name: /Variants in chr1/ }).querySelectorAll('.sv-mark')).toHaveLength(6)
  })

  it('reuses an upload it was given rather than uploading again', async () => {
    const fetchMock = mock()
    const upload = { id: 'vcfupload9', kind: 'vcf' as const, name: 'kept.vcf.gz', size: 1000, ok: true, summary: '', error: null, samples: ['HG00733'] }
    renderDialog({ file: null, upload, sample: 'HG00733' })
    await screen.findByRole('table')
    expect(fetchMock.mock.calls.some(([u]) => String(u) === '/api/uploads')).toBe(false)
    expect(checks[0].get('upload')).toBe('vcfupload9')
    expect(screen.getByText('kept.vcf.gz')).toBeInTheDocument()
  })

  it('checks again for the second haplotype', async () => {
    renderDialog()
    await screen.findByRole('table')
    fireEvent.change(screen.getByLabelText('Haplotype'), { target: { value: '1' } })
    await waitFor(() => expect(checks.at(-1)!.get('haplotype')).toBe('1'))
  })

  it('applies the VCF, the sample and the haplotype', async () => {
    const { onApply } = renderDialog()
    await screen.findByRole('table')
    fireEvent.click(screen.getByRole('button', { name: 'Apply 6 variants' }))
    expect(onApply).toHaveBeenCalledWith(expect.objectContaining({ svSet: '', sample: 'HG00733', haplotype: 0 }))
    expect(onApply.mock.calls[0][0].vcfUpload.id).toBe('vcfupload1')
  })

  it('checks a sample of a hosted call set', async () => {
    const { onApply } = renderDialog({ mode: 'hosted', file: null, svSet: 'HGSVC', sample: 'HG00733' })
    expect(screen.getByRole('tab', { name: 'Hosted call set' })).toHaveAttribute('aria-selected', 'true')
    await screen.findByRole('table')
    expect(checks[0].get('sv_set')).toBe('HGSVC')
    fireEvent.click(screen.getByRole('button', { name: 'Apply 6 variants' }))
    expect(onApply).toHaveBeenCalledWith({ svSet: 'HGSVC', sample: 'HG00733', haplotype: 0, vcfUpload: null })
  })

  it('checks the variants against own loops and draws them', async () => {
    const mine = { id: 'loopsupload1', kind: 'loops' as const, name: 'mine.bedpe', size: 10, ok: true, summary: '', error: null, samples: [] }
    renderDialog({ own: { loops: [{ factor: 'H3K27ac', upload: mine }], anchors: null, contacts: null, enhancers: null } })
    await screen.findByRole('table')
    expect(JSON.parse(String(checks[0].get('own'))).loops).toEqual([{ factor: 'H3K27ac', upload: 'loopsupload1' }])
    await waitFor(() => expect(screen.getByRole('img', { name: /Variants in chr1/ }).querySelectorAll('.arc.own')).toHaveLength(1))
  })

  it('keeps to the writing rules', async () => {
    const { container } = renderDialog()
    await screen.findByRole('table')
    expect(screen.getByText('The haplotype of a phased genotype decides which variants apply. An unphased variant goes to the first haplotype.')).toBeInTheDocument()
    expect(checkCopy(container)).toEqual([])
  })
})
