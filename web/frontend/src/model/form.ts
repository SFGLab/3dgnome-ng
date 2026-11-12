/** The state of the Model a region form. */
export type Factors = 'CTCF' | 'RNAPII' | 'CTCF+RNAPII'
export type Background = 'hic' | 'hicdiffusion' | 'none'

export type FormState = {
  cell: string
  region: string
  factors: Factors
  svSet: string
  sample: string
  haplotype: 0 | 1
  vcf: File | null
  background: Background
  structures: number
  name: string
  settings: Record<string, string>
}

export const EMPTY_FORM: FormState = {
  cell: 'GM12878',
  region: '',
  factors: 'CTCF+RNAPII',
  svSet: '',
  sample: '',
  haplotype: 0,
  vcf: null,
  background: 'hic',
  structures: 100,
  name: '',
  settings: {},
}
