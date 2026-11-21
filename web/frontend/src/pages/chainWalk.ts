/** The two chains of the page of an address the site lacks: the number 404 folded from one chain of
 * beads, and a chain of 404 steps folded at random. */

/** The size of the drawing, in its own units. */
export const WIDTH = 900
export const HEIGHT = 430

/** Depth steps of the drawing, the farthest first, as the drawings of the structures have. */
export const LAYERS = 6

/** A bead of the chain: its place, its depth step and whether it is part of a digit. */
export type Bead = { x: number; y: number; layer: number; digit: boolean }

type Point = [number, number]

const SPACING = 9
const DIGIT = { w: 170, h: 250, top: 78 }
const LEFTS = [96, 365, 634]

/** Points every `SPACING` units along the straight line from a to b, a itself left out. */
function along(a: Point, b: Point): Point[] {
  const n = Math.max(1, Math.round(Math.hypot(b[0] - a[0], b[1] - a[1]) / SPACING))
  return Array.from({ length: n }, (_, k) => [a[0] + ((b[0] - a[0]) * (k + 1)) / n, a[1] + ((b[1] - a[1]) * (k + 1)) / n] as Point)
}

/** A strand from a to b that hangs below the straight line between them, as a loose stretch of a chain does. */
function hanging(a: Point, b: Point, drop: number): Point[] {
  const n = Math.max(2, Math.round(Math.hypot(b[0] - a[0], b[1] - a[1]) / SPACING))
  return Array.from({ length: n }, (_, k) => {
    const t = (k + 1) / n
    return [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t + Math.sin(Math.PI * t) * drop] as Point
  })
}

/** A loose end of the chain: a walk of `steps` beads from a point, kept inside the drawing. */
function loose(from: Point, steps: number, heading: number, random: () => number): Point[] {
  const out: Point[] = []
  let [x, y] = from
  let turn = heading
  for (let i = 0; i < steps; i++) {
    turn += (random() - 0.5) * 1.3
    let nx = x + Math.cos(turn) * SPACING
    let ny = y + Math.sin(turn) * SPACING
    // A step that would leave the drawing turns back.
    if (nx < 24 || nx > WIDTH - 24) {
      turn = Math.PI - turn
      nx = x + Math.cos(turn) * SPACING
    }
    if (ny < 24 || ny > HEIGHT - 24) {
      turn = -turn
      ny = y + Math.sin(turn) * SPACING
    }
    ;[x, y] = [nx, ny]
    out.push([x, y])
  }
  return out
}

/**
 * One chain that spells 404: a loose end, a 4 drawn from its bar to the foot of its stem, a
 * strand to a 0 that goes once round, a strand to a second 4 drawn from the foot of its stem to
 * its bar, and a loose end. Every bead is moved a little from its place, so no two folds are the
 * same. The beads of the digits are in the nearer depth steps and the rest in the farther ones.
 *
 * @param random the source of random numbers, from 0 to 1
 */
export function fold404(random: () => number = Math.random): Bead[] {
  const { w, h, top } = DIGIT
  const foot = top + h
  const bar = top + 0.64 * h
  const parts: { points: Point[]; digit: boolean }[] = []
  const [a, o, b] = LEFTS
  // The first 4: its bar from the right, the slope up to the top of the stem, the stem down.
  const start: Point = [a + w, bar]
  parts.push({ points: loose(start, 26, Math.PI * 0.9, random).reverse(), digit: false })
  parts.push({ points: [start, ...along(start, [a, bar]), ...along([a, bar], [a + 0.7 * w, top]), ...along([a + 0.7 * w, top], [a + 0.7 * w, foot])], digit: true })
  // The 0: once round from its lowest point.
  const [cx, cy, rx, ry] = [o + w / 2, top + h / 2, 0.44 * w, h / 2]
  const round = Math.round((2 * Math.PI * Math.sqrt((rx * rx + ry * ry) / 2)) / SPACING)
  const ring = Array.from({ length: round + 3 }, (_, k) => {
    const t = Math.PI / 2 + (2 * Math.PI * k) / round
    return [cx + rx * Math.cos(t), cy + ry * Math.sin(t)] as Point
  })
  parts.push({ points: hanging([a + 0.7 * w, foot], ring[0], 34).slice(0, -1), digit: false })
  parts.push({ points: ring, digit: true })
  // The second 4, drawn the other way: the stem up, the slope down to the bar, the bar to the right.
  const stem: Point = [b + 0.7 * w, foot]
  parts.push({ points: hanging(ring[ring.length - 1], stem, 34).slice(0, -1), digit: false })
  const end: Point = [b + w, bar]
  parts.push({ points: [stem, ...along(stem, [b + 0.7 * w, top]), ...along([b + 0.7 * w, top], [b, bar]), ...along([b, bar], end)], digit: true })
  parts.push({ points: loose(end, 26, -Math.PI * 0.1, random), digit: false })

  const beads: Bead[] = []
  let [dx, dy, depth] = [0, 0, 0]
  for (const part of parts) {
    for (const [x, y] of part.points) {
      // The shift of a bead follows the shift of the bead before it, as the bends of a chain do.
      const reach = part.digit ? 2.6 : 5
      dx = 0.55 * dx + (random() - 0.5) * reach
      dy = 0.55 * dy + (random() - 0.5) * reach
      depth = 0.8 * depth + (random() - 0.5)
      const layer = part.digit ? Math.max(3, Math.min(LAYERS - 1, Math.round(4 + depth * 1.6))) : Math.max(0, Math.min(2, Math.round(1 + depth * 1.6)))
      beads.push({ x: Math.round((x + dx) * 10) / 10, y: Math.round((y + dy) * 10) / 10, layer, digit: part.digit })
    }
  }
  return beads
}

const r1 = (v: number) => Math.round(v * 10) / 10

/** The chain from bead i to the next as a smooth piece: a Catmull-Rom segment written as a cubic. */
export function piece(beads: Bead[], i: number): string {
  const p0 = beads[Math.max(0, i - 1)]
  const p1 = beads[i]
  const p2 = beads[i + 1]
  const p3 = beads[Math.min(beads.length - 1, i + 2)]
  return `M${p1.x} ${p1.y}C${r1(p1.x + (p2.x - p0.x) / 6)} ${r1(p1.y + (p2.y - p0.y) / 6)} ${r1(p2.x - (p3.x - p1.x) / 6)} ${r1(p2.y - (p3.y - p1.y) / 6)} ${p2.x} ${p2.y}`
}

/** The size of the drawing of the random chain, in its own units. */
export const SIZE = 420

/** A chain of `steps` steps of one length, each in a direction of its own, scaled to fit its drawing. */
export function walk(steps: number, random: () => number = Math.random): [number, number][] {
  const points: [number, number][] = [[0, 0]]
  let turn = random() * Math.PI * 2
  for (let i = 0; i < steps; i++) {
    // A step turns a little from the one before it. The chain then bends as a polymer does.
    turn += (random() - 0.5) * 1.6
    const [x, y] = points[i]
    points.push([x + Math.cos(turn), y + Math.sin(turn)])
  }
  const xs = points.map((p) => p[0])
  const ys = points.map((p) => p[1])
  const [x0, y0] = [Math.min(...xs), Math.min(...ys)]
  const scale = (SIZE - 40) / Math.max(Math.max(...xs) - x0, Math.max(...ys) - y0, 1)
  const [w, h] = [(Math.max(...xs) - x0) * scale, (Math.max(...ys) - y0) * scale]
  return points.map(([x, y]) => [Math.round(((SIZE - w) / 2 + (x - x0) * scale) * 10) / 10, Math.round(((SIZE - h) / 2 + (y - y0) * scale) * 10) / 10])
}
