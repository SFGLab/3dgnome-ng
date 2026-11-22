/** Views of a contact map and its colour ramp. */
export type Show = 'observed' | 'expected' | 'oe' | 'pearson'

export const DARKEST: [number, number, number] = [110, 18, 8]
const STOPS: [number, number, number][] = [[255, 255, 255], [241, 120, 60], DARKEST]

/**
 * Returns the expected contact at each separation, the mean over pairs whose rows both hold
 * contacts. From three bins apart it is the median of the neighbouring separations. The main
 * diagonal is left empty.
 */
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

/** Returns the rows of a map with no contact off the diagonal, a bin with no bead or no read. */
export function emptyRows(m: number[][]): boolean[] {
  return m.map((row, i) => !row.some((v, j) => j !== i && v > 0))
}

export function transform(m: number[][], show: Show): number[][] {
  if (show === 'expected') return expected(m)
  if (show === 'oe') return oe(m)
  if (show === 'pearson') return pearson(m)
  return m
}

const BLUE: [number, number, number] = [42, 110, 200]
const RED: [number, number, number] = [192, 38, 27]

/** Returns a two sided ramp for a change, white at 0, blue below and red above, full at +-span. */
export function diverging(v: number, span: number): [number, number, number] {
  const t = Math.max(-1, Math.min(1, span > 0 ? v / span : 0))
  const end = t < 0 ? BLUE : RED
  const u = Math.abs(t)
  return [0, 1, 2].map((k) => Math.round(255 + u * (end[k] - 255))) as [number, number, number]
}

/** Returns the ramp colour of a value between lo and hi, on a log scale when asked. */
export function colour(v: number, lo: number, hi: number, log: boolean): [number, number, number] {
  // A true log spreads a map of values below one as it spreads a map of counts. A lower end of
  // zero takes a floor of a thousandth of the upper end.
  const floor = lo > 0 ? lo : hi / 1000
  const f = (x: number) => (log ? Math.log(Math.max(x, floor)) : x)
  const span = f(hi) - f(lo)
  const t = Math.min(1, Math.max(0, span > 0 ? (f(v) - f(lo)) / span : 0))
  const s = t * (STOPS.length - 1)
  const i = Math.min(STOPS.length - 2, Math.floor(s))
  const u = s - i
  return STOPS[i].map((c, k) => Math.round(c + u * (STOPS[i + 1][k] - c))) as [number, number, number]
}

/** Returns a high percentile of the nonzero values, the default top of the colour range. */
export function percentile(m: number[][], p: number): number {
  const xs = m
    .flat()
    .filter((v) => v > 0)
    .sort((a, b) => a - b)
  return xs.length ? xs[Math.min(xs.length - 1, Math.floor(p * xs.length))] : 1
}

/**
 * Returns the default colour range of one half, from the 5th to the 99th percentile of its nonzero
 * values on a log scale and from zero on a linear one.
 */
export function halfRange(m: number[][], log: boolean): [number, number] {
  const hi = percentile(m, 0.99)
  const lo = log ? Math.min(percentile(m, 0.05), hi / 10) : 0
  return [lo, hi]
}

const cents = (v: number) => Math.round(v * 100) / 100

/**
 * Returns the two halves of a map as a view draws them and the colour range of each. Each half has
 * a range of its own. A range typed in is the range of the upper half. The lower half keeps the
 * same share of its own spread.
 */
export function shownMaps(up: number[][], low: number[][], show: Show, log: boolean, range: [number, number] | null) {
  const upper = transform(up, show)
  const lower = transform(low, show)
  const oeSpan =
    show === 'oe'
      ? Math.max(
          0.5,
          cents(
            percentile(
              upper.map((r) => r.map((v) => (v > 0 ? Math.abs(Math.log2(v)) : 0))),
              0.95,
            ),
          ),
        )
      : 1
  const auto = (m: number[][]): [number, number] => (show === 'oe' ? [-oeSpan, oeSpan] : show === 'pearson' ? [-1, 1] : halfRange(m, log))
  const [autoUp, autoLow] = [auto(upper), auto(lower)]
  const upperRange: [number, number] = range ?? [cents(autoUp[0]), cents(autoUp[1])]
  const scale = show === 'observed' && autoUp[1] > 0 ? autoLow[1] / autoUp[1] : 1
  const lowerRange: [number, number] = range ? [range[0] * scale, range[1] * scale] : autoLow
  return { upper, lower, upperRange, lowerRange }
}
