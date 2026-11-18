/** Formats shared by the explorer. */
export const bp = (n: number) => n.toLocaleString('en-US')

/** A region in megabases with no more digits than its positions need, chr8:127-128.5 Mb. */
export function shortRegion(chrom: string, start: number, end: number): string {
  const mb = (n: number) => String(Number((n / 1_000_000).toFixed(3)))
  return `${chrom}:${mb(start)}-${mb(end)} Mb`
}

export function kb(n: number): string {
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(2)} Mb`
  if (n < 1000) return `${n} bp`
  return `${Math.round(n / 1000)} kb`
}

export function size(bytes: number): string {
  if (bytes >= 1_000_000) return `${(bytes / 1_000_000).toFixed(1)} MB`
  return `${Math.max(1, Math.round(bytes / 1000))} KB`
}

export function duration(seconds: number): string {
  const m = Math.floor(seconds / 60)
  const s = Math.round(seconds % 60)
  return m > 0 ? `${m} min ${s} s` : `${s} s`
}

export const FACTORS: Record<string, string> = { CTCF: 'CTCF', RNAPII: 'RNAPII', 'CTCF+RNAPII': 'CTCF and RNAPII' }
