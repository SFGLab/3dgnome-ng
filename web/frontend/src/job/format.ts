/** Formats shared by the explorer. */
import type { Summary } from '../api'

/** The variants a job applied and from whom, with the count of those of a type the site does not apply. */
export function variants(s: Summary): string {
  const skipped = s.skipped_svs ? `, ${s.skipped_svs} of other types skipped` : ''
  if (!s.n_svs) return `none${skipped}`
  const sample = s.sample_note ? `${s.request.sample} (${s.sample_note})` : s.request.sample
  const who = s.request.sample ? `${sample}, haplotype ${s.request.haplotype + 1}` : 'uploaded VCF'
  return `${s.n_svs} from ${who}${skipped}`
}
