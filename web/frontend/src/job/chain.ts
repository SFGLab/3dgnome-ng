/** Geometry of the chain as the 3D view draws it. */
type P = [number, number, number]

/** The tube radius, a fixed share of the median bond, so a chain reads the same at any scale. */
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

/** Fog from the structure's middle to its far side, seen from the camera's distance. */
export function fogRange(radius: number, cameraDistance: number): [number, number] {
  return [cameraDistance, cameraDistance + radius]
}
