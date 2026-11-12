import type { Factors } from './form'

/** The examples: each a request the form can be filled with, and the tab or second loop set its
 * finished job opens on. */
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
}

export const EXAMPLES: Example[] = [
  {
    id: 'TNFRSF8',
    title: 'TNFRSF8',
    cell: 'GM12878',
    cellLabel: 'GM12878',
    description: 'An enhancer 385 kb away joined by an RNAPII loop',
    region: 'TNFRSF8',
    factors: 'CTCF+RNAPII',
  },
  {
    id: 'MYC',
    title: 'MYC',
    cell: 'GM12878',
    cellLabel: 'GM12878',
    description: 'The MYC locus and its distal enhancers',
    region: 'chr8:127,500,000-129,000,000',
    factors: 'CTCF+RNAPII',
  },
  { id: 'HOXA', title: 'HOXA cluster', cell: 'H1ESC', cellLabel: 'H1ESC', description: 'A dense cluster of CTCF loops', region: 'chr7:27,000,000-27,400,000', factors: 'CTCF' },
  {
    id: 'GSTM1',
    title: 'Deletion at GSTM1',
    cell: 'GM12878',
    cellLabel: 'GM12878, HGSVC',
    description: 'A common deletion that removes a looped CTCF anchor',
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
    description: 'The two loop sets side by side in the compare view',
    region: 'TNFRSF8',
    factors: 'CTCF+RNAPII',
    compareFactors: 'CTCF',
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
