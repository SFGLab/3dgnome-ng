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
    key: 'shi2023', group: 'algorithms', year: 2023, doi: '10.1038/s41467-023-36412-4',
    title: 'A maximum-entropy model to predict 3D structural ensembles of chromatin from pairwise distances with applications to interphase chromosomes and structural variants',
    authors: 'Shi G, Thirumalai D', venue: 'Nature Communications 14, 1150',
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
    key: 'tang2015', group: 'algorithms', year: 2015, doi: '10.1016/j.cell.2015.11.024',
    title: 'CTCF-mediated human 3D genome architecture reveals chromatin topology for transcription',
    authors: 'Tang Z, Luo OJ, Li X, et al.', venue: 'Cell 163, 1611-1627',
  },
  {
    key: '4dn2017', group: 'data', year: 2017, doi: '10.1038/nature23884',
    title: 'The 4D nucleome project',
    authors: 'Dekker J, Belmont AS, Guttman M, et al.', venue: 'Nature 549, 219-226',
  },
  {
    key: 'encode2020', group: 'data', year: 2020, doi: '10.1038/s41586-020-2493-4',
    title: 'Expanded encyclopaedias of DNA elements in the human and mouse genomes',
    authors: 'The ENCODE Project Consortium', venue: 'Nature 583, 699-710',
  },
  {
    key: 'hong2025', group: 'data', year: 2025, doi: '10.5281/zenodo.15303879',
    title: 'Comprehensive mapping of genome structure with ultra-high-resolution 3D genomics and deep learning (experimental data)',
    authors: 'Hong CKY', venue: 'Zenodo',
  },
  {
    key: 'gencode2021', group: 'data', year: 2021, doi: '10.1093/nar/gkaa1087',
    title: 'GENCODE 2021',
    authors: 'Frankish A, Diekhans M, Jungreis I, et al.', venue: 'Nucleic Acids Research 49, D916-D923',
  },
  {
    key: 'grch38', group: 'data', year: 2017, doi: '10.1101/gr.213611.116',
    title: 'Evaluation of GRCh38 and de novo haploid genome assemblies demonstrates the enduring quality of the reference assembly',
    authors: 'Schneider VA, Graves-Lindsay T, Howe K, et al.', venue: 'Genome Research 27, 849-864',
  },
  {
    key: 'enhanceratlas2020', group: 'data', year: 2020, doi: '10.1093/nar/gkz980',
    title: 'EnhancerAtlas 2.0: an updated resource with enhancer annotation in 586 tissue/cell types across nine species',
    authors: 'Gao T, Qian J', venue: 'Nucleic Acids Research 48, D58-D64',
  },
  {
    key: 'kgp30x', group: 'data', year: 2022, doi: '10.1016/j.cell.2022.08.004',
    title: 'High-coverage whole-genome sequencing of the expanded 1000 Genomes Project cohort including 602 trios',
    authors: 'Byrska-Bishop M, Evani US, Zhao X, et al.', venue: 'Cell 185, 3426-3440',
  },
  {
    key: 'tbooo', group: 'data', year: 2026, url: 'https://github.com/SFGLab/tbooo',
    title: 'TBOOO, The Biobank Of Our Own: the data layout of a biobank rebuilt from the 1000 Genomes Project and the Simons Genome Diversity Project',
    authors: 'SFGLab', venue: 'GitHub repository',
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

/** An accession with the page of its archive. */
export type Accession = { id: string; url: string }

/**
 * A row of the Cite page's data table.
 * `accessions` are the records the data came from, the experiment or set first and its files
 * after it. `pending` says what is still to be recorded and stands in the accession column.
 * A row without `ref` has no confirmed source to cite yet.
 */
export type DataSource = { data: string; usedFor: string; accessions: Accession[]; note?: string; pending?: string; label: string; ref?: string }

const encode = (id: string): Accession => ({ id, url: `https://www.encodeproject.org/${id.startsWith('ENCSR') ? 'experiments' : 'files'}/${id}/` })
const fourDn = (id: string): Accession => ({ id, url: `https://data.4dnucleome.org/${id.startsWith('4DNES') ? 'experiment-set-replicates' : 'files-processed'}/${id}/` })
const geo = (id: string): Accession => ({ id, url: `https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=${id}` })

export const DATA_SOURCES: DataSource[] = [
  { data: 'CTCF in situ ChIA-PET loops, GM12878', usedFor: 'loops', accessions: [fourDn('4DNES7IB5LY9'), encode('ENCSR184YZV')], label: '4DN Network, Nature 2017', ref: '4dn2017' },
  { data: 'CTCF in situ ChIA-PET loops, H1ESC', usedFor: 'loops', accessions: [fourDn('4DNESR9S8R38'), encode('ENCSR095SXC'), geo('GSE242609')], label: '4DN Network, Nature 2017', ref: '4dn2017' },
  { data: 'CTCF in situ ChIA-PET loops, HFFc6', usedFor: 'loops', accessions: [fourDn('4DNESCQ7ZD21')], label: '4DN Network, Nature 2017', ref: '4dn2017' },
  {
    data: 'RNAPII ChIA-PET loops and peaks, GM12878', usedFor: 'loops',
    accessions: ['ENCSR905HWW', 'ENCFF040KUS', 'ENCFF913VWM', 'ENCFF109HCZ', 'ENCFF578JKO'].map(encode),
    label: 'ENCODE, Nature 2020', ref: 'encode2020',
  },
  {
    data: 'RNAPII ChIA-PET loops and peaks, H1ESC', usedFor: 'loops',
    accessions: ['ENCSR782EKZ', 'ENCFF324UIT', 'ENCFF753NSM', 'ENCFF665IXY', 'ENCFF116RLS'].map(encode),
    label: 'ENCODE, Nature 2020', ref: 'encode2020',
  },
  {
    data: 'RNAPII ChIA-PET loops and peaks, BJ', usedFor: 'loops of HFFc6', note: 'BJ is another foreskin fibroblast line. It stands in for HFFc6.',
    accessions: ['ENCSR956IXE', 'ENCFF601NEC', 'ENCFF808QOQ', 'ENCFF775SCY', 'ENCFF428VXI', 'ENCFF084CXY', 'ENCFF953MBK'].map(encode),
    label: 'ENCODE, Nature 2020', ref: 'encode2020',
  },
  { data: 'CTCF ChIP-seq peaks, GM12878', usedFor: 'telling RNAPII loops from CTCF loops', accessions: ['ENCSR000DZN', 'ENCFF796WRU'].map(encode), label: 'ENCODE, Nature 2020', ref: 'encode2020' },
  { data: 'CTCF ChIP-seq peaks, H1ESC', usedFor: 'telling RNAPII loops from CTCF loops', accessions: ['ENCSR000BNH', 'ENCFF497ZGV'].map(encode), label: 'ENCODE, Nature 2020', ref: 'encode2020' },
  { data: 'CTCF ChIP-seq peaks, HFFc6', usedFor: 'telling RNAPII loops from CTCF loops', accessions: ['ENCSR163ULN', 'ENCFF414XTM'].map(encode), label: 'ENCODE, Nature 2020', ref: 'encode2020' },
  { data: 'Hi-C, GM12878', usedFor: 'contact background, agreement', accessions: ['4DNES3JX38V5', '4DNFIXP4QG5B'].map(fourDn), label: '4DN Network, Nature 2017', ref: '4dn2017' },
  { data: 'Hi-C, H1ESC', usedFor: 'contact background, agreement', accessions: ['4DNES2M5JIGV', '4DNFI6HDY7WZ'].map(fourDn), label: '4DN Network, Nature 2017', ref: '4dn2017' },
  { data: 'Hi-C, HFFc6', usedFor: 'contact background, agreement', accessions: ['4DNES2R6PUEK', '4DNFIB59T7NN'].map(fourDn), label: '4DN Network, Nature 2017', ref: '4dn2017' },
  {
    data: 'Micro-C, GM12878', usedFor: 'loops the models are checked against',
    accessions: [{ id: '10.5281/zenodo.15303879', url: 'https://doi.org/10.5281/zenodo.15303879' }],
    label: 'Hong, Zenodo 2025', ref: 'hong2025',
  },
  { data: 'Micro-C, H1ESC', usedFor: 'loops the models are checked against', accessions: [fourDn('4DNFI2TK7L2F')], label: '4DN Network, Nature 2017', ref: '4dn2017' },
  { data: 'Micro-C, HFFc6', usedFor: 'loops the models are checked against', accessions: [fourDn('4DNFI18Q799K')], label: '4DN Network, Nature 2017', ref: '4dn2017' },
  { data: 'Gene expression of the three cell lines', usedFor: 'link to expression', accessions: [], pending: 'accession to confirm', label: 'source to confirm' },
  { data: 'Enhancers, EnhancerAtlas 2.0', usedFor: 'enhancer marks and distances', accessions: [], note: 'One file per cell line. The atlas gives no accession.', label: 'Gao and Qian, NAR 2020', ref: 'enhanceratlas2020' },
  { data: 'Genes, GENCODE release 40, basic annotation', usedFor: 'promoter marks and gene names', accessions: [{ id: 'GENCODE v40', url: 'https://www.gencodegenes.org/human/release_40.html' }], label: 'Frankish et al., NAR 2021', ref: 'gencode2021' },
  {
    data: 'Structural variants, 1000 Genomes 30x, 3,202 samples', usedFor: 'variant structures',
    accessions: [
      { id: 'PRJEB31736', url: 'https://www.ebi.ac.uk/ena/browser/view/PRJEB31736' },
      { id: 'PRJEB36890', url: 'https://www.ebi.ac.uk/ena/browser/view/PRJEB36890' },
    ],
    note: 'The phased panel of the 2,504 unrelated samples and the 698 that complete the trios. Deletions, insertions, inversions and duplications on the autosomes are applied.',
    label: 'Byrska-Bishop et al., Cell 2022', ref: 'kgp30x',
  },
  { data: 'Structural variants, HGSVC freeze 4', usedFor: 'variant structures', accessions: [], pending: 'accession to confirm', label: 'HGSVC, reference to confirm', ref: 'hgsvc4' },
  {
    data: 'Reference genome, hg38', usedFor: 'sequence for HiCDiffusion',
    accessions: [{ id: 'GCA_000001405.15', url: 'https://www.ncbi.nlm.nih.gov/datasets/genome/GCA_000001405.15/' }],
    note: 'GRCh38, the UCSC hg38 FASTA.', label: 'Schneider et al., Genome Research 2017', ref: 'grch38',
  },
  { data: 'HiCDiffusion model', usedFor: 'contact background from sequence', accessions: [], note: 'The checkpoints of the authors. They have no accession.', label: 'Chiliński and Plewczynski, BMC Genomics 2024', ref: 'hicdiffusion2024' },
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
