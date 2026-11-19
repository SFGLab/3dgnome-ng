import './structure-preview.css'

/**
 * A structure drawn flat: its beads in drawing units from A, each with a depth step, the two marked
 * beads and the distance between them over the ensemble. Written by web/scripts/make_previews.py.
 */
export type PreviewData = { points: number[][]; a: number; b: number; mean: number; sd: number; n: number; structure: number; here: number }

/** Depth steps of a drawing, the farthest first. */
const LAYERS = 6

/** Beads either side of a marked bead drawn in its colour. */
const REACH = 5

const r1 = (v: number) => Math.round(v * 10) / 10

/** The chain from bead i to the next as a smooth piece: a Catmull-Rom segment written as a cubic. */
function piece(pts: number[][], i: number): string {
  const p0 = pts[Math.max(0, i - 1)]
  const p1 = pts[i]
  const p2 = pts[i + 1]
  const p3 = pts[Math.min(pts.length - 1, i + 2)]
  return `M${p1[0]} ${p1[1]}C${r1(p1[0] + (p2[0] - p0[0]) / 6)} ${r1(p1[1] + (p2[1] - p0[1]) / 6)} ${r1(p2[0] - (p3[0] - p1[0]) / 6)} ${r1(p2[1] - (p3[1] - p1[1]) / 6)} ${p2[0]} ${p2[1]}`
}

const pieces = (pts: number[][], from: number, to: number) =>
  Array.from({ length: Math.max(0, Math.min(to, pts.length - 1) - Math.max(0, from)) }, (_, k) => piece(pts, Math.max(0, from) + k)).join('')

/**
 * A flat drawing of one structure with A and B marked, the line between them and the distance. A
 * nearer strand is drawn over a farther one.
 *
 * @param view the part drawn as x, y, width and height in drawing units. Drawings to be compared
 * take one view.
 */
export function StructurePreview({ data, label, view }: { data: PreviewData; label: string; view: [number, number, number, number] }) {
  const { points, a, b } = data
  const [lo, hi] = a < b ? [a, b] : [b, a]
  const rest: string[] = Array.from({ length: LAYERS }, () => '')
  const between: string[] = Array.from({ length: LAYERS }, () => '')
  for (let i = 0; i + 1 < points.length; i++) (i >= lo && i < hi ? between : rest)[Math.max(points[i][2], points[i + 1][2])] += piece(points, i)
  const B = points[b]
  const text = `${data.mean.toFixed(1)} ± ${data.sd.toFixed(1)}`
  // The distance is on the side of the line with fewer beads under it.
  const under = (y0: number) => points.filter((p) => p[0] > B[0] / 2 - 110 && p[0] < B[0] / 2 + 110 && p[1] > y0 && p[1] < y0 + 50).length
  const ly = under(-84) <= under(34) ? -84 : 34
  return (
    <svg className="structure-preview" viewBox={view.join(' ')} role="img" aria-label={label}>
      {rest.map((d, k) => (
        <g key={k}>
          <path className="edge" d={d + between[k]} />
          <path className={`chain step${k}`} d={d} />
          <path className={`chain between step${k}`} d={between[k]} />
        </g>
      ))}
      <path className="edge mark" d={pieces(points, a - REACH, a + REACH) + pieces(points, b - REACH, b + REACH)} />
      <path className="mark promoter" d={pieces(points, a - REACH, a + REACH)} />
      <path className="mark enhancer" d={pieces(points, b - REACH, b + REACH)} />
      <line className="ab" x1={0} y1={0} x2={B[0]} y2={B[1]} />
      <rect className="ab-box" x={B[0] / 2 - 100} y={ly} width={200} height={50} rx={10} />
      <text className="ab-text" x={B[0] / 2} y={ly + 36}>
        {text}
      </text>
      {(
        [
          ['A', [0, 0]],
          ['B', B],
        ] as const
      ).map(([name, p]) => (
        <g key={name}>
          <circle className="end" cx={p[0]} cy={p[1]} r={28} />
          <text className="end-text" x={p[0]} y={p[1] + 12}>
            {name}
          </text>
        </g>
      ))}
    </svg>
  )
}
