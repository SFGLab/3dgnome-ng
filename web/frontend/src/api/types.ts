/**
 * The shapes of the server's answers, by the names the pages use. They come from schema.d.ts,
 * which `npm run api` generates from the description of the server.
 */
import type { components } from './schema'

type Shapes = components['schemas']

export type Catalog = Shapes['CatalogInfo']
export type CellType = Shapes['CellTypeInfo']
/** A gene the region field suggests. */
export type GeneHit = Shapes['GeneHit']
export type Region = Shapes['RegionInfo']
/** A loop of a preview is its two anchor middles, its PET count and the outer edges of its anchors. */
export type Preview = Shapes['Preview']
export type Setting = Shapes['Setting']
export type SettingGroup = Shapes['SettingGroup']
/** A hosted expression table a job can take: the id of its source, what the form calls it, counts
 * or values, the unit of values and whether it is a lymphoblastoid line beside another cell type. */
export type ExpressionOption = Shapes['ExpressionOption']
export type VariantRow = Shapes['VariantRow']
export type VariantCheck = Shapes['VariantCheck']
export type JobStatus = Shapes['Status']
/** A sample of a call set. An unknown field is an empty string and so is a parent the set does not hold. */
export type SampleRow = { sample: string; sex: string; population: string; superpopulation: string; father: string; mother: string }

export type FileInfo = Shapes['FileSize']
export type Summary = Shapes['Summary']
export type Structure = Shapes['Structure']
export type Annotations = Shapes['Annotations']
export type Loop = Shapes['Loop']
export type Distance = Shapes['Distance']
export type GeneRow = Shapes['GeneRow']
export type Picks = Shapes['Picks']
/** The two maps of a contact map view. The server sends each symmetric map once and `jobApi` gives whole maps. */
export type Contacts = Shapes['ContactMaps']
/**
 * The genes two jobs share. `ratio` is the distance of b over a and `p` that of a Mann-Whitney U
 * test of the distances in their structures, null with one structure on a side. A job measures a
 * gene to its own nearest enhancer: `linear_bp` is the separation in a and `linear_bp_b` in b,
 * and `differ` counts the genes with two enhancers. With `shared` both jobs measure the nearest of
 * the `shared_enhancers` that the two have.
 */
export type Compare = Shapes['DistanceChanges']
/** The enhancer to promoter distance of one gene in each structure of two jobs. */
export type GeneDistances = Shapes['GeneDistances']
/**
 * The expression of job b against job a for the genes both have. `distance` is log2 of the
 * enhancer to promoter distance of b over a, `log2fc` the log2 fold change of the expression and
 * `padj` its adjusted p, null where no test was run. `mode` is deseq2, ratio or none and `reason`
 * says why it is not deseq2. `volcano` has every gene of the test as [log2 fold change, -log10
 * adjusted p], kept within `fold` and `p`, null without a test.
 */
export type ExpressionCompare = Shapes['ExpressionChanges']

export type UploadKind = Shapes['UploadResult']['kind']
export type CheckError = Shapes['ProblemFound']
export type UploadResult = Shapes['UploadResult']
/** A loop of an upload inside a region: its two anchor middles and its count. */
export type LoopDrawn = [number, number, number]

export type ListedJob = Shapes['ListedJob']
export type SharedList = Shapes['SharedList']
