/** The rows of a track of genes. A gene keeps a row to itself and its name while there is room. */

export type TrackGene = { name: string; tss: number; start: number; end: number; strand: string }

/** A gene with its row and whether its name is drawn beside it. */
export type PlacedGene = TrackGene & { row: number; named: boolean }

/** The width of a letter of a name and the gap between two genes of a row, in pixels. */
const LETTER = 7.2
const GAP = 10
/** A gene is drawn at least this wide. */
export const MIN_BAR = 3

/** Returns true for a gene without a name of its own, which the annotation lists by its Ensembl id. */
export const unnamed = (g: TrackGene) => /^ENSG\d/.test(g.name)

/**
 * Returns the genes of a view in rows, as a genome browser packs them. A gene goes to the first row
 * that is free from its left end to the right end of its name. A gene that finds no such row is
 * drawn without its name. A gene that finds no room for its bar is left out. The genes with a
 * name of their own are placed before the genes the annotation lists by an Ensembl id, which are
 * drawn without a name.
 * @param genes the genes, in any order
 * @param view the start and the end of the view in bp
 * @param width the width of the track in pixels
 * @param rows the most rows
 * @param first the name of a gene placed before the others, the one the view is about
 */
export function packGenes(genes: TrackGene[], view: [number, number], width: number, rows: number, first?: string): { placed: PlacedGene[]; left: number } {
  const span = Math.max(view[1] - view[0], 1)
  const px = (p: number) => ((p - view[0]) / span) * width
  const inView = genes.filter((g) => g.end > view[0] && g.start < view[1])
  // The first gene of the view before every other, then the named genes, then the rest. Longer
  // genes first among genes of one start.
  const order = [...inView].sort((a, b) => Number(b.name === first) - Number(a.name === first) || Number(unnamed(a)) - Number(unnamed(b)) || a.start - b.start || b.end - a.end)
  const taken: [number, number][][] = Array.from({ length: rows }, () => [])
  const free = (row: number, a: number, b: number) => taken[row].every(([s, e]) => b <= s || a >= e)
  const placed: PlacedGene[] = []
  let left = 0
  for (const g of order) {
    const a = Math.max(0, px(g.start))
    const b = Math.min(width, Math.max(px(g.end), a + MIN_BAR))
    const withName = Math.min(width, b + 8 + g.name.length * LETTER)
    let row = unnamed(g) ? -1 : taken.findIndex((_, r) => free(r, a - GAP, withName + GAP))
    const named = row >= 0
    // A bar without a name keeps the room of its arrowhead on either side.
    if (row < 0) row = taken.findIndex((_, r) => free(r, a - 6, b + 6))
    if (row < 0) {
      left += 1
      continue
    }
    taken[row].push([a, named ? withName : b])
    placed.push({ ...g, row, named })
  }
  return { placed, left }
}

/**
 * Returns the row of the label of each mark, or null for a label that has no room. The labels are
 * centred on their marks and a label goes to the first row where it does not meet another.
 * @param centres the middle of each mark along the track, in its units
 * @param widths the width of each label, in the same units
 * @param rows the most rows of labels
 */
export function labelRows(centres: number[], widths: number[], rows: number): (number | null)[] {
  const taken: [number, number][][] = Array.from({ length: rows }, () => [])
  const out: (number | null)[] = centres.map(() => null)
  for (const i of centres.map((_, k) => k).sort((a, b) => centres[a] - centres[b])) {
    const [a, b] = [centres[i] - widths[i] / 2 - 2, centres[i] + widths[i] / 2 + 2]
    const row = taken.findIndex((r) => r.every(([s, e]) => b <= s || a >= e))
    if (row < 0) continue
    taken[row].push([a, b])
    out[i] = row
  }
  return out
}

