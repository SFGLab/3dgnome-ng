/** Regions as text, chr:start-end. */
import { bp } from './format'

export type Locus = { chrom: string; start: number; end: number }

/** A region the site stated itself, with plain numbers. Null for any other text. */
export function parseRegion(region: string | null | undefined): Locus | null {
  const m = region?.match(/^(\w+):(\d+)-(\d+)$/)
  return m ? { chrom: m[1], start: Number(m[2]), end: Number(m[3]) } : null
}

/** A region a visitor typed, commas allowed, its end after its start. Null for any other text. */
export const readRegion = (t: string): Locus | null => {
  const m = t.replace(/,/g, '').match(/^(\w+):(\d+)-(\d+)$/)
  return m && Number(m[3]) > Number(m[2]) ? { chrom: m[1], start: Number(m[2]), end: Number(m[3]) } : null
}

/** A region with thousands separators, chr1:11,500,000-12,600,000. */
export const regionText = (l: Locus) => `${l.chrom}:${bp(l.start)}-${bp(l.end)}`
