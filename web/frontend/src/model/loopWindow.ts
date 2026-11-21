/**
 * The stretch drawn around a region: the region with its own length on either side, from 0. The
 * stretch is `cap` bp at most. A region that leaves less room gets less on either side, and a
 * region of `cap` or more is drawn from a little before its start.
 */
export function around(start: number, end: number, cap = Infinity): [number, number] {
  const span = end - start
  if (span >= cap) {
    const from = Math.max(0, start - Math.floor(cap / 20))
    return [from, from + cap]
  }
  const pad = Math.min(span, Math.floor((cap - span) / 2))
  const from = Math.max(0, start - pad)
  return [from, from === 0 ? Math.max(end + pad, span + 2 * pad) : end + pad]
}
