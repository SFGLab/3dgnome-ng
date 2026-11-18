type Window = { chrom: string; start: number; end: number }
type Meta = { chromosomes: { name: string; size: number }[]; resolutions: number[] }
type ContactRecord = { bin1: number; bin2: number; counts: number }

const length = (bp: number) => (bp >= 1_000_000 ? `${bp / 1_000_000} Mb` : `${bp / 1000} kb`)
const list = (xs: string[]) => (xs.length < 2 ? xs.join('') : `${xs.slice(0, -1).join(', ')} and ${xs.at(-1)}`)

/** Reads the bin pairs of a region from a local .hic file in the browser, as BEDPE text. Only
 * the blocks of the region are read from the file, as Juicebox web reads a local file.
 *
 * @param file the .hic file the user chose
 * @param region the window to read
 * @param res the bin size in bp
 * @returns one line per off diagonal bin pair, chr1 start1 end1 chr2 start2 end2 count, upper triangle
 */
export async function readHicWindow(file: File, region: Window, res: number): Promise<string> {
  // The reader loads with the dialog's first .hic file, not with the page.
  const { default: Straw } = await import('hic-straw')
  const straw = new Straw({ blob: file })
  const meta = (await straw.getMetaData()) as Meta
  const have = [...meta.resolutions].sort((a, b) => a - b)
  if (!have.includes(res)) throw new Error(`${file.name} has no ${length(res)} resolution. It has ${list(have.map(length))}.`)
  const bare = region.chrom.replace(/^chr/, '')
  const name = meta.chromosomes.map((c) => c.name).find((n) => n === region.chrom || n === bare || n === `chr${bare}`)
  if (!name) throw new Error(`${file.name} has no chromosome ${region.chrom}.`)
  const span = { chr: name, start: region.start, end: region.end }
  const records = (await straw.getContactRecords('NONE', span, span, 'BP', res)) as ContactRecord[]
  const c = region.chrom
  return records
    .map((r) => [Math.min(r.bin1, r.bin2) * res, Math.max(r.bin1, r.bin2) * res, Math.round(r.counts)] as const)
    // The hosted maps hold no pair within one bin and the engine reads every map alike.
    .filter(([a, b, n]) => n > 0 && a !== b)
    .map(([a, b, n]) => `${c}\t${a}\t${a + res}\t${c}\t${b}\t${b + res}\t${n}\n`)
    .join('')
}
