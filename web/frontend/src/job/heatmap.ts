/** Views of a contact map and its colour ramp. */
export type Show = 'observed' | 'expected' | 'oe' | 'pearson'

export const DARKEST: [number, number, number] = [110, 18, 8]
const STOPS: [number, number, number][] = [[255, 255, 255], [241, 120, 60], DARKEST]

function expected(m: number[][]): number[][] {
  const n = m.length
  const mean = Array.from({ length: n }, (_, k) => {
    let s = 0
    for (let i = 0; i + k < n; i++) s += m[i][i + k]
    return s / (n - k)
  })
  return m.map((row, i) => row.map((_, j) => mean[Math.abs(i - j)]))
}

function oe(m: number[][]): number[][] {
  const e = expected(m)
  return m.map((row, i) => row.map((v, j) => (e[i][j] > 0 ? v / e[i][j] : 0)))
}

function pearson(m: number[][]): number[][] {
  const rows = oe(m)
  const n = rows.length
  const centred = rows.map((r) => {
    const mu = r.reduce((a, b) => a + b, 0) / n
    const c = r.map((v) => v - mu)
    const norm = Math.sqrt(c.reduce((a, b) => a + b * b, 0)) || 1
    return c.map((v) => v / norm)
  })
  return centred.map((a) => centred.map((b) => a.reduce((s, v, k) => s + v * b[k], 0)))
}

export function transform(m: number[][], show: Show): number[][] {
  if (show === 'expected') return expected(m)
  if (show === 'oe') return oe(m)
  if (show === 'pearson') return pearson(m)
  return m
}

/** The ramp colour of a value between lo and hi, on a log scale when asked. */
export function colour(v: number, lo: number, hi: number, log: boolean): [number, number, number] {
  const f = (x: number) => (log ? Math.log1p(Math.max(0, x)) : x)
  const span = f(hi) - f(lo)
  const t = Math.min(1, Math.max(0, span > 0 ? (f(v) - f(lo)) / span : 0))
  const s = t * (STOPS.length - 1)
  const i = Math.min(STOPS.length - 2, Math.floor(s))
  const u = s - i
  return STOPS[i].map((c, k) => Math.round(c + u * (STOPS[i + 1][k] - c))) as [number, number, number]
}

/** A high percentile of the nonzero values, the default top of the colour range. */
export function percentile(m: number[][], p: number): number {
  const xs = m.flat().filter((v) => v > 0).sort((a, b) => a - b)
  return xs.length ? xs[Math.min(xs.length - 1, Math.floor(p * xs.length))] : 1
}
