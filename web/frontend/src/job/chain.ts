/** Geometry of the chain as the 3D view draws it. */
type P = [number, number, number]

/** The tube radius, a fixed share of the median bond. A chain then looks the same at any scale. */
export function bondRadius(points: P[]): number {
  const bonds: number[] = []
  for (let i = 1; i < points.length; i++) {
    const [a, b] = [points[i - 1], points[i]]
    bonds.push(Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]))
  }
  if (!bonds.length) return 0.3
  bonds.sort((x, y) => x - y)
  return 0.28 * bonds[Math.floor(bonds.length / 2)]
}

/** The beads either side of a hovered bead that are brightened with it, cut at the chain's ends. */
export function stretch(i: number, n: number, k = 6): [number, number] {
  return [Math.max(0, i - k), Math.min(n - 1, i + k)]
}

/**
 * Fog that starts at the middle of the structure and would be full at 2.5 radii behind it. The far
 * side fades by two fifths.
 */
export function fogRange(radius: number, cameraDistance: number): [number, number] {
  return [cameraDistance, cameraDistance + 2.5 * radius]
}

/** The points of the chain inside any of the ranges, as a mask. Null for no ranges, which marks
 * no part of the chain over the rest. */
export function focusMask(n: number, ranges: [number, number][]): Uint8Array | null {
  if (!ranges.length) return null
  const mask = new Uint8Array(n)
  for (const [a, b] of ranges) mask.fill(1, Math.max(0, Math.min(a, b)), Math.min(n, Math.max(a, b) + 1))
  return mask
}

/**
 * The bead whose tube is drawn under the cursor, or null. The cursor is on a bond when it is within
 * the drawn radius of the bond plus two pixels of its line on the screen. The bond nearest the
 * camera wins and of its two beads the one nearer the cursor.
 *
 * @param chain the screen x and y of each bead in pixels, its distance from the camera and the
 * drawn radius of the tube in pixels at that bead
 */
export function pickOnChain(chain: [number, number, number, number][], x: number, y: number): number | null {
  if (chain.length === 1) {
    const [sx, sy, , r] = chain[0]
    return Math.hypot(sx - x, sy - y) <= r + 2 ? 0 : null
  }
  let best: number | null = null
  let depth = Infinity
  for (let i = 0; i + 1 < chain.length; i++) {
    const [ax, ay, ad, ar] = chain[i]
    const [bx, by, bd, br] = chain[i + 1]
    if (ad <= 0 || bd <= 0) continue
    const dx = bx - ax
    const dy = by - ay
    const len2 = dx * dx + dy * dy
    const t = len2 > 0 ? Math.max(0, Math.min(1, ((x - ax) * dx + (y - ay) * dy) / len2)) : 0
    const off = Math.hypot(ax + t * dx - x, ay + t * dy - y)
    const radius = ar + t * (br - ar)
    const d = ad + t * (bd - ad)
    if (off <= radius + 2 && d < depth) {
      depth = d
      best = t <= 0.5 ? i : i + 1
    }
  }
  return best
}
