/** Every work the site cites, in the order of the Cite page. Copy from web/design/site.pen. */
export type Reference = {
  key: string
  group: 'gnome' | 'algorithms' | 'related' | 'data'
  authors: string
  title: string
  venue: string
  year: number
  doi?: string
  url?: string
  unconfirmed?: boolean
}

export const REFERENCES: Reference[] = [
  {
    key: 'gnome3', group: 'gnome', year: 2023, doi: '10.1093/nar/gkad354',
    title: '3D-GNOME 3.0: a three-dimensional genome modelling engine for analysing changes of promoter-enhancer contacts in the human genome',
    authors: 'Wlasnowolski M, Kadlof M, Sengupta K, Plewczynski D', venue: 'Nucleic Acids Research 51, W5-W10',
  },
  {
    key: 'gnome2', group: 'gnome', year: 2020, doi: '10.1093/nar/gkaa388',
    title: '3D-GNOME 2.0: a three-dimensional genome modeling engine for predicting structural variation-driven alterations of chromatin spatial structure in the human genome',
    authors: 'Wlasnowolski M, Sadowski M, Czarnota T, Jodkowska K, Szalaj P, Tang Z, Ruan Y, Plewczynski D', venue: 'Nucleic Acids Research 48, W170-W176',
  },
  {
    key: 'gnome2016', group: 'gnome', year: 2016, doi: '10.1093/nar/gkw437',
    title: '3D-GNOME: an integrated web service for structural modeling of the 3D genome',
    authors: 'Szalaj P, Michalski PJ, Wróblewski P, Tang Z, Kadlof M, Mazzocco G, Ruan Y, Plewczynski D', venue: 'Nucleic Acids Research 44, W288-W293',
  },
  {
    key: 'cudammc', group: 'gnome', year: 2023, doi: '10.1093/bioinformatics/btad588',
    title: 'cudaMMC: GPU-enhanced multiscale Monte Carlo chromatin 3D modelling',
    authors: 'Wlasnowolski M, Grabowski P, Roszczyk D, Kaczmarski K, Plewczynski D', venue: 'Bioinformatics 39, btad588',
  },
  {
    key: 'sadowski2019', group: 'gnome', year: 2019, doi: '10.1186/s13059-019-1728-x',
    title: 'Spatial chromatin architecture alteration by structural variations in human genomes at the population scale',
    authors: 'Sadowski M, Kraft A, Szalaj P, Wlasnowolski M, Tang Z, Ruan Y, Plewczynski D', venue: 'Genome Biology 20, 148',
  },
  {
    key: 'enhancer3d', group: 'gnome', year: 2026, doi: '10.1093/nar/gkaf1256',
    title: 'enhancer3D: 3D chromatin structures and enhancer-promoter distance profiles for archaic and modern human genomes',
    authors: 'Wlasnowolski M, Kozlov N, Wojcik M, Jacobs G, Plewczynski D', venue: 'Nucleic Acids Research 54, D1046-D1052',
  },
  {
    key: 'szalaj2016', group: 'algorithms', year: 2016, doi: '10.1101/gr.205062.116',
    title: 'An integrated 3-dimensional genome modeling engine for data-driven simulation of spatial genome organization',
    authors: 'Szałaj P, Tang Z, Michalski P, et al.', venue: 'Genome Research 26, 1697-1709',
  },
  {
    key: 'lieberman2009', group: 'algorithms', year: 2009, doi: '10.1126/science.1181369',
    title: 'Comprehensive mapping of long-range interactions reveals folding principles of the human genome',
    authors: 'Lieberman-Aiden E, van Berkum NL, Williams L, et al.', venue: 'Science 326, 289-293',
  },
  {
    key: 'byrd1995', group: 'algorithms', year: 1995, doi: '10.1137/0916069',
    title: 'A limited memory algorithm for bound constrained optimization',
    authors: 'Byrd RH, Lu P, Nocedal J, Zhu C', venue: 'SIAM Journal on Scientific Computing 16, 1190-1208',
  },
  {
    key: 'zhu1997', group: 'algorithms', year: 1997, doi: '10.1145/279232.279236',
    title: 'Algorithm 778: L-BFGS-B',
    authors: 'Zhu C, Byrd RH, Lu P, Nocedal J', venue: 'ACM Transactions on Mathematical Software 23, 550-560',
  },
  {
    key: 'kirkpatrick1983', group: 'algorithms', year: 1983, doi: '10.1126/science.220.4598.671',
    title: 'Optimization by simulated annealing',
    authors: 'Kirkpatrick S, Gelatt CD, Vecchi MP', venue: 'Science 220, 671-680',
  },
  {
    key: 'metropolis1953', group: 'algorithms', year: 1953, doi: '10.1063/1.1699114',
    title: 'Equation of state calculations by fast computing machines',
    authors: 'Metropolis N, Rosenbluth AW, Rosenbluth MN, Teller AH, Teller E', venue: 'Journal of Chemical Physics 21, 1087-1092',
  },
  {
    key: 'hilbert1891', group: 'algorithms', year: 1891, doi: '10.1007/BF01199431',
    title: 'Über die stetige Abbildung einer Linie auf ein Flächenstück',
    authors: 'Hilbert D', venue: 'Mathematische Annalen 38, 459-460',
  },
  {
    key: 'hicrep2017', group: 'algorithms', year: 2017, doi: '10.1101/gr.220640.117',
    title: 'HiCRep: assessing the reproducibility of Hi-C data using a stratum-adjusted correlation coefficient',
    authors: 'Yang T, Zhang F, Yardımcı GG, et al.', venue: 'Genome Research 27, 1939-1949',
  },
  {
    key: 'jax2018', group: 'algorithms', year: 2018, url: 'https://github.com/jax-ml/jax',
    title: 'JAX: composable transformations of Python and NumPy programs',
    authors: 'Bradbury J, Frostig R, Hawkins P, et al.', venue: 'Software',
  },
  {
    key: 'kremer1990', group: 'algorithms', year: 1990, doi: '10.1063/1.458541',
    title: 'Dynamics of entangled linear polymer melts: a molecular-dynamics simulation',
    authors: 'Kremer K, Grest GS', venue: 'Journal of Chemical Physics 92, 5057-5086',
  },
  {
    key: 'rosa2008', group: 'algorithms', year: 2008, doi: '10.1371/journal.pcbi.1000153',
    title: 'Structure and dynamics of interphase chromosomes',
    authors: 'Rosa A, Everaers R', venue: 'PLoS Computational Biology 4, e1000153',
  },
  {
    key: 'mirny2011', group: 'algorithms', year: 2011, doi: '10.1007/s10577-010-9177-0',
    title: 'The fractal globule as a model of chromatin architecture in the cell',
    authors: 'Mirny LA', venue: 'Chromosome Research 19, 37-51',
  },
  {
    key: 'multimm', group: 'related', year: 2024, doi: '10.1016/j.csbj.2024.09.025',
    title: 'Multiscale molecular modeling of chromatin with MultiMM: from nucleosomes to the whole genome',
    authors: 'Korsak S, Banecki K, Plewczynski D', venue: 'Computational and Structural Biotechnology Journal 23, 3537-3548',
  },
  {
    key: 'loopsage', group: 'related', year: 2024, doi: '10.1016/j.ymeth.2024.01.015',
    title: 'LoopSage: an energy-based Monte Carlo approach for the loop extrusion modeling of chromatin',
    authors: 'Korsak S, Plewczynski D', venue: 'Methods 223, 106-117',
  },
  {
    key: 'tang2015', group: 'data', year: 2015, doi: '10.1016/j.cell.2015.11.024',
    title: 'CTCF-mediated human 3D genome architecture reveals chromatin topology for transcription',
    authors: 'Tang Z, Luo OJ, Li X, et al.', venue: 'Cell 163, 1611-1627',
  },
  {
    key: '4dn2017', group: 'data', year: 2017, doi: '10.1038/nature23884',
    title: 'The 4D nucleome project',
    authors: 'Dekker J, Belmont AS, Guttman M, et al.', venue: 'Nature 549, 219-226',
  },
  {
    key: 'enhanceratlas2020', group: 'data', year: 2020, doi: '10.1093/nar/gkz980',
    title: 'EnhancerAtlas 2.0: an updated resource with enhancer annotation in 586 tissue/cell types across nine species',
    authors: 'Gao T, Qian J', venue: 'Nucleic Acids Research 48, D58-D64',
  },
  {
    key: 'hgsvc4', group: 'data', year: 2025, unconfirmed: true,
    title: 'Human Genome Structural Variation Consortium, freeze 4',
    authors: 'HGSVC', venue: 'reference to confirm',
  },
  {
    key: 'hicdiffusion2024', group: 'data', year: 2024, doi: '10.1186/s12864-024-10885-z',
    title: 'HiCDiffusion - diffusion-enhanced, transformer-based prediction of chromatin interactions from DNA sequences',
    authors: 'Chiliński M, Plewczynski D', venue: 'BMC Genomics 25, 964',
  },
]

/** The rows of the Cite page's data table. */
export type DataSource = { data: string; usedFor: string; label: string; ref: string }

export const DATA_SOURCES: DataSource[] = [
  { data: 'CTCF and RNAPII ChIA-PET, GM12878', usedFor: 'loops', label: 'Tang et al., Cell 2015', ref: 'tang2015' },
  { data: 'Hi-C, 4DN data portal', usedFor: 'contact background, agreement', label: '4DN Network, Nature 2017', ref: '4dn2017' },
  { data: 'Enhancers, EnhancerAtlas 2.0', usedFor: 'enhancer marks and distances', label: 'Gao and Qian, NAR 2020', ref: 'enhanceratlas2020' },
  { data: 'Structural variants, HGSVC freeze 4', usedFor: 'variant structures', label: 'HGSVC, reference to confirm', ref: 'hgsvc4' },
  { data: 'HiCDiffusion', usedFor: 'contact background from sequence', label: 'Chiliński and Plewczynski, BMC Genomics 2024', ref: 'hicdiffusion2024' },
]

/** The link of a reference, its DOI or its URL. */
export function referenceLink(r: Reference): string | undefined {
  return r.doi ? `https://doi.org/${r.doi}` : r.url
}

/** A reference as one line of plain text. */
export function referenceText(r: Reference): string {
  const link = r.doi ? ` doi:${r.doi}` : r.url ? ` ${r.url}` : ''
  return `${r.authors}. ${r.title}. ${r.venue} (${r.year}).${link}`
}
