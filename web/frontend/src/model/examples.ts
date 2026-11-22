import type { Factors } from './form'

/**
 * The examples. Each is a request that fills the form, with the tab or the
 * second loop set of its finished job. Each description is read from the
 * hosted data and must be checked again when the data changes.
 */
export type Example = {
  id: string
  title: string
  cell: string
  cellLabel: string
  description: string
  region: string
  factors: Factors
  svSet?: string
  sample?: string
  tab?: 'structure' | 'genes' | 'contacts' | 'loops'
  compareFactors?: Factors
  /** The hosted expression of the job, by the id of its source. */
  expression?: string
  /** A second cell type the job is compared with, on the same region and loops. */
  compareCell?: string
  /** A gene the job opens with selected, its promoter and enhancer marked in each structure. */
  gene?: string
  /**
   * The enhancer of the gene that the example is about, by a position inside it. The job marks it in place of the
   * enhancer nearest to the gene, which can be a bead beside the promoter.
   */
  enhancer?: number
}

export const EXAMPLES: Example[] = [
  {
    id: 'ECI2',
    title: 'ECI2',
    cell: 'GM12878',
    cellLabel: 'GM12878',
    description: 'An enhancer 113 kb away joined by an RNAPII loop, with no other enhancer nearer to the promoter, compared with CTCF loops alone',
    region: 'chr6:3,600,000-4,600,000',
    factors: 'CTCF+RNAPII',
    compareFactors: 'CTCF',
    gene: 'ECI2',
    enhancer: 4022561,
  },
  {
    id: 'MYC',
    title: 'MYC',
    cell: 'GM12878',
    cellLabel: 'GM12878',
    description: 'MYC and the enhancers 430 to 560 kb upstream that CTCF and RNAPII loops join to its promoter',
    region: 'chr8:127,000,000-128,500,000',
    factors: 'CTCF+RNAPII',
  },
  { id: 'HOXA', title: 'HOXA cluster', cell: 'H1ESC', cellLabel: 'H1ESC', description: 'The HOXA gene cluster, 400 kb with 8 CTCF loops', region: 'chr7:27,000,000-27,400,000', factors: 'CTCF' },
  {
    id: 'GSTM1',
    title: 'Deletion at GSTM1',
    cell: 'GM12878',
    cellLabel: 'GM12878, HGSVC',
    description: 'A common 18 kb deletion that removes 4 CTCF anchors and their loops',
    region: 'chr1:109,200,000-110,200,000',
    factors: 'CTCF+RNAPII',
    svSet: 'HGSVC',
    sample: 'HG00512',
  },
  {
    id: 'ECI2-CTCF',
    title: 'ECI2, CTCF loops alone',
    cell: 'GM12878',
    cellLabel: 'GM12878',
    description: 'The same locus without the RNAPII loop that joins the enhancer',
    region: 'chr6:3,600,000-4,600,000',
    factors: 'CTCF',
  },
  {
    id: 'CD52',
    title: 'CD52 in GM12878 and H1ESC',
    cell: 'GM12878',
    cellLabel: 'GM12878, H1ESC',
    description: 'The distance of a gene to its nearest enhancer beside its expression, two cell types compared with ENCODE RNA-seq',
    region: 'chr1:25,900,000-26,900,000',
    factors: 'CTCF+RNAPII',
    expression: 'ENCODE',
    compareCell: 'H1ESC',
    gene: 'CD52',
  },
  {
    id: 'contacts',
    title: 'Contacts against Hi-C',
    cell: 'GM12878',
    cellLabel: 'GM12878',
    description: 'The model contact map beside the 4DN Hi-C map',
    region: 'chr1:20,000,000-25,000,000',
    factors: 'CTCF+RNAPII',
    tab: 'contacts',
  },
  {
    id: 'TNFRSF8',
    title: 'TNFRSF8',
    cell: 'GM12878',
    cellLabel: 'GM12878',
    description: 'The example of the tutorial on the Help page, an enhancer 385 kb away joined by an RNAPII loop, compared with CTCF loops alone',
    region: 'TNFRSF8',
    factors: 'CTCF+RNAPII',
    compareFactors: 'CTCF',
    gene: 'TNFRSF8',
  },
]
