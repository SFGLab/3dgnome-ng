/** Formats shared by the explorer. */
export const bp = (n: number) => n.toLocaleString('en-US')

export function kb(n: number): string {
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(2)} Mb`
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
