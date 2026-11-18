import type { Factors } from './form'

/** The examples: each a request the form can be filled with and the tab or second loop set its
 * finished job opens on. Each description is read from the hosted data and a change of the data
 * needs them read again. TNFRSF8: one RNAPII loop of 3 PETs from 3 kb inside the gene to the
 * enhancer at chr1:12,448,194-12,450,774 and no CTCF loop between the two. MYC: loops from the
 * promoter to anchors with enhancers 428 to 557 kb upstream. HOXA: the loop count of the preview.
 * GSTM1: the variants check of HG00512 and 48 of 69 alleles of the call set carry the deletion. */
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
  /** A gene the job opens with selected, its promoter and enhancer marked in each structure. */
  gene?: string
}

export const EXAMPLES: Example[] = [
  {
    id: 'TNFRSF8',
    title: 'TNFRSF8',
    cell: 'GM12878',
    cellLabel: 'GM12878',
    description: 'An enhancer 385 kb away joined by an RNAPII loop, compared with CTCF loops alone',
    region: 'TNFRSF8',
    factors: 'CTCF+RNAPII',
    compareFactors: 'CTCF',
    gene: 'TNFRSF8',
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
    id: 'TNFRSF8-CTCF',
    title: 'TNFRSF8, CTCF loops alone',
    cell: 'GM12878',
    cellLabel: 'GM12878',
    description: 'The same locus without the RNAPII loop that joins the enhancer',
    region: 'TNFRSF8',
    factors: 'CTCF',
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
]
