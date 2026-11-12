/** Views of a contact map and its colour ramp. */
export type Show = 'observed' | 'expected' | 'oe' | 'pearson'

export const DARKEST: [number, number, number] = [110, 18, 8]
const STOPS: [number, number, number][] = [[255, 255, 255], [241, 120, 60], DARKEST]

/** The expected contact at each separation: the mean over pairs whose rows both hold contacts,
 * and from three bins apart the median of the neighbouring separations, so one sparse stratum
 * does not stand out as a stripe. The main diagonal is left empty; the maps hold no pair within
 * one bin. */
function expected(m: number[][]): number[][] {
  const n = m.length
  const live = m.map((row, i) => row.some((v, j) => j !== i && v > 0))
  const mean = Array.from({ length: n }, (_, k) => {
    if (k === 0) return 0
    let s = 0
    let c = 0
    for (let i = 0; i + k < n; i++)
      if (live[i] && live[i + k]) {
        s += m[i][i + k]
        c++
      }
    return c ? s / c : 0
  })
  const smooth = mean.map((v, k) => {
    if (k < 3) return v
    const w = Math.max(1, Math.round(0.15 * k))
    const near = mean.slice(Math.max(1, k - w), Math.min(n, k + w + 1)).sort((a, b) => a - b)
    return near[Math.floor(near.length / 2)]
  })
  return m.map((row, i) => row.map((_, j) => smooth[Math.abs(i - j)]))
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

/** The rows of a map with no contact off the diagonal, a bin with no bead or no read. */
export function emptyRows(m: number[][]): boolean[] {
  return m.map((row, i) => !row.some((v, j) => j !== i && v > 0))
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
