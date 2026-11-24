import { describe, expect, it } from 'vitest'
import { labelRows, packGenes, type TrackGene } from './packGenes'

const gene = (name: string, start: number, end: number, strand = '+'): TrackGene => ({ name, tss: strand === '+' ? start : end - 1, start, end, strand })

describe('the rows of the genes track', () => {
  it('keeps two genes that do not meet in one row and puts a gene over another in the next', () => {
    const { placed, left } = packGenes([gene('A', 0, 100_000), gene('B', 500_000, 600_000), gene('C', 50_000, 150_000)], [0, 1_000_000], 1000, 4)
    expect(left).toBe(0)
    expect(Object.fromEntries(placed.map((g) => [g.name, g.row]))).toEqual({ A: 0, B: 0, C: 1 })
    expect(placed.every((g) => g.named)).toBe(true)
  })

  it('draws a gene without its name where the name has no room, and leaves out a gene with no room for its bar', () => {
    const crowd = Array.from({ length: 12 }, (_, i) => gene(`GENE${i}`, 100_000 + i * 1_000, 400_000))
    const { placed, left } = packGenes(crowd, [0, 1_000_000], 600, 3)
    expect(placed).toHaveLength(3)
    expect(left).toBe(9)
  })

  it('places the gene the view is about first and the genes without a name of their own last, without a name', () => {
    const genes = [gene('ENSG00000228793', 0, 300_000), gene('ECI2', 100_000, 200_000, '-'), gene('PRPF4B', 0, 250_000)]
    const { placed } = packGenes(genes, [0, 1_000_000], 1000, 1, 'ECI2')
    expect(placed.map((g) => g.name)).toEqual(['ECI2'])
    const three = packGenes(genes, [0, 1_000_000], 1000, 3, 'ECI2').placed
    expect(three.find((g) => g.name === 'ECI2')?.row).toBe(0)
    expect(three.find((g) => g.name === 'ENSG00000228793')?.named).toBe(false)
  })

  it('leaves out a gene outside the view and keeps one that reaches into it', () => {
    const { placed } = packGenes([gene('OUT', 2_000_000, 2_100_000), gene('EDGE', 900_000, 1_200_000)], [0, 1_000_000], 1000, 2)
    expect(placed.map((g) => g.name)).toEqual(['EDGE'])
  })
})

describe('the rows of the labels of marks', () => {
  it('puts the labels of two close marks on two rows and leaves out the label of a third', () => {
    expect(labelRows([100, 108, 114, 400], [20, 20, 20, 20], 2)).toEqual([0, 1, null, 0])
  })
})

