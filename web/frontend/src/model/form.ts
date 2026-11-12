import { NO_OWN, type OwnChoice } from './upload'

/** The state of the Model a region form. The cell type own means the job reads only uploads. */
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
  own: OwnChoice
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
  own: NO_OWN,
}
