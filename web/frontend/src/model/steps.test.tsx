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
  it('offers the four loop sets and opens the uploads for own loops', () => {
    const onChange = vi.fn()
    const onOwn = vi.fn()
    render(<LoopOptions value="CTCF+RNAPII" ownCell={false} cells={[]} loopsFrom="" onLoopsFrom={() => {}} onChange={onChange} onOwn={onOwn} />)
    expect(screen.getAllByRole('radio').map((r) => r.querySelector('.option-title')?.textContent)).toEqual(['CTCF', 'RNAPII', 'CTCF and RNAPII', 'Your own loops'])
    expect(screen.getByRole('radio', { name: /CTCF and RNAPII/ })).toBeChecked()
    fireEvent.click(screen.getByRole('radio', { name: /Your own loops/ }))
    expect(onOwn).toHaveBeenCalled()
    fireEvent.click(screen.getByRole('radio', { name: /^RNAPII/ }))
    expect(onChange).toHaveBeenCalledWith('RNAPII')
  })

  it('offers the listed loop sets of a chosen cell type on your own cell type', () => {
    const onLoopsFrom = vi.fn()
    const onChange = vi.fn()
    render(<LoopOptions value="CTCF" ownCell cells={['GM12878', 'H1ESC']} loopsFrom="GM12878" onLoopsFrom={onLoopsFrom} onChange={onChange} onOwn={() => {}} />)
    expect(screen.getAllByRole('radio').every((r) => !(r as HTMLButtonElement).disabled)).toBe(true)
    fireEvent.change(screen.getByLabelText('Loops of'), { target: { value: 'H1ESC' } })
    expect(onLoopsFrom).toHaveBeenCalledWith('H1ESC')
    fireEvent.click(screen.getByRole('radio', { name: /^RNAPII/ }))
    expect(onChange).toHaveBeenCalledWith('RNAPII')
  })

  it('hides the cell type select when own loops are chosen or the cell type is listed', () => {
    const { rerender } = render(<LoopOptions value="own" ownCell cells={['GM12878']} loopsFrom="GM12878" onLoopsFrom={() => {}} onChange={() => {}} onOwn={() => {}} />)
    expect(screen.queryByLabelText('Loops of')).toBeNull()
    rerender(<LoopOptions value="CTCF" ownCell={false} cells={['GM12878']} loopsFrom="" onLoopsFrom={() => {}} onChange={() => {}} onOwn={() => {}} />)
    expect(screen.queryByLabelText('Loops of')).toBeNull()
  })
})

describe('Variants', () => {
  it('lists the samples of the chosen call set and the haplotypes', () => {
    const onChange = vi.fn()
    render(<Variants catalog={CATALOG} svSet="HGSVC" sample="" haplotype={0} vcfUpload={null} onChange={onChange} onOpen={vi.fn()} />)
    expect(screen.getByLabelText('Sample')).toHaveTextContent('HG00512')
    fireEvent.change(screen.getByLabelText('Sample'), { target: { value: 'HG00733' } })
    expect(onChange).toHaveBeenCalledWith({ sample: 'HG00733' })
    fireEvent.change(screen.getByLabelText('Haplotype'), { target: { value: '1' } })
    expect(onChange).toHaveBeenCalledWith({ haplotype: 1 })
  })

  it('opens the dialog with a VCF file', () => {
    const onOpen = vi.fn()
    render(<Variants catalog={CATALOG} svSet="" sample="" haplotype={0} vcfUpload={null} onChange={vi.fn()} onOpen={onOpen} />)
    const file = new File(['x'], 'mine.vcf.gz')
    fireEvent.change(screen.getByLabelText(/Or drop your own VCF here/), { target: { files: [file] } })
    expect(onOpen).toHaveBeenCalledWith('upload', file)
  })

  it('opens the dialog for a chosen sample of a call set', () => {
    const onOpen = vi.fn()
    render(<Variants catalog={CATALOG} svSet="HGSVC" sample="HG00733" haplotype={0} vcfUpload={null} onChange={vi.fn()} onOpen={onOpen} />)
    fireEvent.click(screen.getByRole('button', { name: 'Show the variants in the region' }))
    expect(onOpen).toHaveBeenCalledWith('hosted', null)
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
        <LoopOptions value="CTCF" ownCell={false} cells={[]} loopsFrom="" onLoopsFrom={() => {}} onChange={() => {}} onOwn={() => {}} />
        <Variants catalog={CATALOG} svSet="" sample="" haplotype={0} vcfUpload={null} onChange={() => {}} onOpen={() => {}} />
        <Background value="hic" region={REGION_3MB} available={false} lymphoblastoid={false} onChange={() => {}} />
      </>,
    )
    expect(checkCopy(container)).toEqual([])
  })
})
