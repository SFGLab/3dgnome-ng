/** The chain of the page of an address the site lacks, 404 steps folded at random. */

/** The size of the drawing of the chain, in its own units. */
export const SIZE = 420

/** Returns a chain of `steps` steps of one length, each in a direction of its own, scaled to fit its drawing. */
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
