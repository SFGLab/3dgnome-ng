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

  it('takes a typed sample of a large call set once it names one', () => {
    const onChange = vi.fn()
    const samples = Array.from({ length: 3202 }, (_, i) => `HG${String(i).padStart(5, '0')}`)
    const big = { ...CATALOG, sv_sets: { ...CATALOG.sv_sets, KGP: { label: '1000 Genomes', samples } } }
    render(<Variants catalog={big} svSet="KGP" sample="" haplotype={0} vcfUpload={null} onChange={onChange} onOpen={vi.fn()} />)
    const field = screen.getByLabelText('Sample')
    expect(field.tagName).toBe('INPUT')
    expect(field).toHaveAttribute('placeholder', 'type one of 3,202, e.g. HG00000')
    fireEvent.change(field, { target: { value: 'HG0009' } })
    expect(onChange).not.toHaveBeenCalled()
    expect(field).toHaveAttribute('aria-invalid', 'true')
    fireEvent.change(field, { target: { value: 'HG00096' } })
    expect(onChange).toHaveBeenCalledWith({ sample: 'HG00096' })
  })

  it('drops the sample of a large call set when the text no longer names one', () => {
    const onChange = vi.fn()
    const samples = Array.from({ length: 100 }, (_, i) => `NA${String(i).padStart(5, '0')}`)
    const big = { ...CATALOG, sv_sets: { KGP: { label: '1000 Genomes', samples } } }
    render(<Variants catalog={big} svSet="KGP" sample="NA00007" haplotype={0} vcfUpload={null} onChange={onChange} onOpen={vi.fn()} />)
    expect(screen.getByLabelText('Sample')).toHaveValue('NA00007')
    fireEvent.change(screen.getByLabelText('Sample'), { target: { value: 'NA0000' } })
    expect(onChange).toHaveBeenCalledWith({ sample: '' })
  })

  const FAMILY = [
    { sample: 'HG00731', sex: 'male', population: 'PUR', superpopulation: 'AMR', father: '', mother: '' },
    { sample: 'HG00732', sex: 'female', population: 'PUR', superpopulation: 'AMR', father: '', mother: '' },
    { sample: 'HG00733', sex: 'female', population: 'PUR', superpopulation: 'AMR', father: 'HG00731', mother: 'HG00732' },
    { sample: 'NA19240', sex: 'female', population: 'YRI', superpopulation: 'AFR', father: '', mother: '' },
  ]
  const TRIO = { ...CATALOG, sv_sets: { KGP: { label: '1000 Genomes', samples: FAMILY.map((r) => r.sample), sample_table: true } } }

  it('states the population, sex and parents of a chosen sample and chooses a parent on a click', () => {
    const onChange = vi.fn()
    render(<Variants catalog={TRIO} svSet="KGP" sample="HG00733" haplotype={0} vcfUpload={null} sampleTable={FAMILY} onChange={onChange} onOpen={vi.fn()} />)
    const facts = document.querySelector('.sample-facts')!
    expect(facts.textContent).toBe('HG00733: PUR, AMR, female. Father HG00731. Mother HG00732.')
    fireEvent.click(screen.getByRole('button', { name: 'HG00732' }))
    expect(onChange).toHaveBeenCalledWith({ sample: 'HG00732' })
  })

  it('lists the children of a chosen parent', () => {
    render(<Variants catalog={TRIO} svSet="KGP" sample="HG00731" haplotype={0} vcfUpload={null} sampleTable={FAMILY} onChange={vi.fn()} onOpen={vi.fn()} />)
    expect(document.querySelector('.sample-facts')!.textContent).toBe('HG00731: PUR, AMR, male. Child HG00733.')
  })

  it('lists the samples of a population typed in place of a sample', () => {
    const onChange = vi.fn()
    const samples = Array.from({ length: 100 }, (_, i) => `NA${String(i).padStart(5, '0')}`)
    const table = samples.map((sample, i) => ({ sample, sex: 'female', population: i < 30 ? 'YRI' : 'GBR', superpopulation: i < 30 ? 'AFR' : 'EUR', father: '', mother: '' }))
    const big = { ...CATALOG, sv_sets: { KGP: { label: '1000 Genomes', samples, sample_table: true } } }
    render(<Variants catalog={big} svSet="KGP" sample="" haplotype={0} vcfUpload={null} sampleTable={table} onChange={onChange} onOpen={vi.fn()} />)
    expect(document.querySelector('.sample-facts')).toBeNull()
    expect(document.querySelector('#sample-names option')).toHaveAttribute('label', 'YRI, AFR, female')
    fireEvent.change(screen.getByLabelText('Sample'), { target: { value: 'yri' } })
    const facts = document.querySelector('.sample-facts')!
    expect(facts.textContent).toMatch(/^YRI, 30 samples: NA00000, NA00001, .* and 20 more\.$/)
    fireEvent.click(screen.getByRole('button', { name: 'NA00003' }))
    expect(onChange).toHaveBeenCalledWith({ sample: 'NA00003' })
  })

  it('shows no sample facts for a call set without a table', () => {
    render(<Variants catalog={CATALOG} svSet="HGSVC" sample="HG00733" haplotype={0} vcfUpload={null} onChange={vi.fn()} onOpen={vi.fn()} />)
    expect(document.querySelector('.sample-facts')).toBeNull()
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
  it('offers the variants once step 1 has a region and says so before', () => {
    const { rerender } = render(<Variants catalog={CATALOG} svSet="HGSVC" sample="HG00733" haplotype={0} vcfUpload={null} hasRegion={false} onChange={vi.fn()} onOpen={vi.fn()} />)
    expect(screen.queryByRole('button', { name: /Show the variants/ })).toBeNull()
    expect(screen.getByText('Enter the region in step 1 to see the variants inside it.')).toBeInTheDocument()
    rerender(<Variants catalog={CATALOG} svSet="HGSVC" sample="HG00733" haplotype={0} vcfUpload={null} hasRegion onChange={vi.fn()} onOpen={vi.fn()} />)
    expect(screen.getByRole('button', { name: /Show the variants/ })).toBeInTheDocument()
  })

  it('starts on the Hi-C of the cell type', () => {
    render(<Background value="hic" region={REGION_1MB} available lymphoblastoid onChange={() => {}} />)
    expect(screen.getByRole('radio', { name: /Hi-C of this cell type/ })).toBeChecked()
  })

  it('disables HiCDiffusion for a region over 2 Mb and says why', () => {
    render(<Background value="hic" region={REGION_3MB} available lymphoblastoid onChange={() => {}} />)
    expect(screen.getByRole('radio', { name: /Predicted by HiCDiffusion/ })).toBeDisabled()
    expect(screen.getByText('HiCDiffusion predicts regions up to 2 Mb.')).toBeInTheDocument()
  })

  it('takes the HiCDiffusion limit of the server', () => {
    render(<Background value="hic" region={REGION_1MB} available limitMb={0.5} lymphoblastoid onChange={() => {}} />)
    expect(screen.getByRole('radio', { name: /Predicted by HiCDiffusion/ })).toBeDisabled()
    expect(screen.getByText('HiCDiffusion predicts regions up to 0.5 Mb.')).toBeInTheDocument()
    expect(screen.getByText(/for regions up to 0\.5 Mb/)).toBeInTheDocument()
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
