import { fireEvent, render, screen } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'
import { CATALOG } from '../test/api'
import { checkCopy } from '../test/copy'
import { Background } from './Background'
import { LoopOptions } from './LoopOptions'
import { Variants } from './Variants'

const REGION_1MB = { chrom: 'chr1', start: 11500000, end: 12600000, label: '' }
const REGION_3MB = { chrom: 'chr1', start: 10000000, end: 13000000, label: '' }

describe('LoopOptions', () => {
  it('offers the four loop sets with own loops disabled', () => {
    const onChange = vi.fn()
    render(<LoopOptions value="CTCF+RNAPII" onChange={onChange} />)
    expect(screen.getAllByRole('radio').map((r) => r.querySelector('.option-title')?.textContent)).toEqual(['CTCF', 'RNAPII', 'CTCF and RNAPII', 'Your own loops'])
    expect(screen.getByRole('radio', { name: /CTCF and RNAPII/ })).toBeChecked()
    expect(screen.getByRole('radio', { name: /Your own loops/ })).toBeDisabled()
    fireEvent.click(screen.getByRole('radio', { name: /^RNAPII/ }))
    expect(onChange).toHaveBeenCalledWith('RNAPII')
  })
})

describe('Variants', () => {
  it('lists the samples of the chosen call set and the haplotypes', () => {
    const onChange = vi.fn()
    render(<Variants catalog={CATALOG} svSet="HGSVC" sample="" haplotype={0} vcf={null} onChange={onChange} />)
    expect(screen.getByLabelText('Sample')).toHaveTextContent('HG00512')
    fireEvent.change(screen.getByLabelText('Sample'), { target: { value: 'HG00733' } })
    expect(onChange).toHaveBeenCalledWith({ sample: 'HG00733' })
    fireEvent.change(screen.getByLabelText('Haplotype'), { target: { value: '1' } })
    expect(onChange).toHaveBeenCalledWith({ haplotype: 1 })
  })

  it('takes a VCF file in place of the call set', () => {
    const onChange = vi.fn()
    render(<Variants catalog={CATALOG} svSet="" sample="" haplotype={0} vcf={null} onChange={onChange} />)
    const file = new File(['x'], 'mine.vcf.gz')
    fireEvent.change(screen.getByLabelText(/Or drop your own VCF here/), { target: { files: [file] } })
    expect(onChange).toHaveBeenCalledWith({ vcf: file, svSet: '', sample: '' })
  })
})

describe('Background', () => {
  it('starts on the Hi-C of the cell type', () => {
    render(<Background value="hic" region={REGION_1MB} available lymphoblastoid onChange={() => {}} />)
    expect(screen.getByRole('radio', { name: /Hi-C of this cell type/ })).toBeChecked()
  })

  it('disables HiCDiffusion for a region over 2 Mb and says why', () => {
    render(<Background value="hic" region={REGION_3MB} available lymphoblastoid onChange={() => {}} />)
    expect(screen.getByRole('radio', { name: /Predicted by HiCDiffusion/ })).toBeDisabled()
    expect(screen.getByText('HiCDiffusion predicts regions up to 2 Mb.')).toBeInTheDocument()
  })

  it('disables HiCDiffusion when the server has no HiCDiffusion', () => {
    render(<Background value="hic" region={REGION_1MB} available={false} lymphoblastoid onChange={() => {}} />)
    expect(screen.getByRole('radio', { name: /Predicted by HiCDiffusion/ })).toBeDisabled()
    expect(screen.getByText('HiCDiffusion is not set up on this server.')).toBeInTheDocument()
  })

  it('warns for a cell type that is not lymphoblastoid', () => {
    const { rerender } = render(<Background value="hic" region={REGION_1MB} available lymphoblastoid onChange={() => {}} />)
    expect(screen.queryByText(/HiCDiffusion was trained on GM12878/)).toBeNull()
    rerender(<Background value="hic" region={REGION_1MB} available lymphoblastoid={false} onChange={() => {}} />)
    expect(screen.getByText(/HiCDiffusion was trained on GM12878/)).toBeInTheDocument()
  })

  it('keeps to the writing rules', () => {
    const { container } = render(
      <>
        <LoopOptions value="CTCF" onChange={() => {}} />
        <Variants catalog={CATALOG} svSet="" sample="" haplotype={0} vcf={null} onChange={() => {}} />
        <Background value="hic" region={REGION_3MB} available={false} lymphoblastoid={false} onChange={() => {}} />
      </>,
    )
    expect(checkCopy(container)).toEqual([])
  })
})
