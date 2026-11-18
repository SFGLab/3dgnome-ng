import './structure-preview.css'

/** A structure drawn flat: its beads in a box of 1000 by 1000 with a depth step each, the two
 * marked beads and the distance between them over the ensemble. Written by
 * web/scripts/make_previews.py from a finished job. */
export type PreviewData = { points: number[][]; a: number; b: number; mean: number; sd: number; n: number; structure: number; here: number }

/** Beads either side of a marked bead drawn in its colour. */
const REACH = 5

const path = (pts: number[][]) => pts.map((p, i) => `${i ? 'L' : 'M'}${p[0]} ${p[1]}`).join('')

/** The chain as one path per depth step, each run of beads of one step joined to the next bead. */
function byDepth(points: number[][]): string[] {
  const d = ['', '', '']
  for (let i = 0; i + 1 < points.length; i++) d[points[i][2]] += `M${points[i][0]} ${points[i][1]}L${points[i + 1][0]} ${points[i + 1][1]}`
  return d
}

/** A flat drawing of one structure with A and B marked, the line between them and the distance.
 * Every drawing has the scale of the others, so a longer line is a longer distance.
 *
 * @param view the width and height of the part drawn, centred between A and B, in the units of
 *   the box. Without it the whole structure is drawn. Drawings to be compared take one view.
 */
export function StructurePreview({ data, label, view }: { data: PreviewData; label: string; view?: [number, number] }) {
  const { points, a, b } = data
  const [far, mid, near] = byDepth(points)
  const A = points[a]
  const B = points[b]
  const around = (i: number) => path(points.slice(Math.max(0, i - REACH), i + REACH + 1))
  const text = `${data.mean.toFixed(1)} ± ${data.sd.toFixed(1)}`
  // The label is over the middle of the line, on the side with more room.
  const lx = (A[0] + B[0]) / 2
  const ly = (A[1] + B[1]) / 2 + (A[1] + B[1] > 1000 ? -34 : 62)
  const box = view ? `${(A[0] + B[0]) / 2 - view[0] / 2} ${(A[1] + B[1]) / 2 - view[1] / 2} ${view[0]} ${view[1]}` : '0 0 1000 1000'
  return (
    <svg className="structure-preview" viewBox={box} role="img" aria-label={label}>
      <path className="chain far" d={far} />
      <path className="chain mid" d={mid} />
      <path className="chain near" d={near} />
      <path className="mark promoter" d={around(a)} />
      <path className="mark enhancer" d={around(b)} />
      <line className="ab" x1={A[0]} y1={A[1]} x2={B[0]} y2={B[1]} />
      <rect className="ab-box" x={lx - 110} y={ly - 36} width={220} height={50} rx={10} />
      <text className="ab-text" x={lx} y={ly}>
        {text}
      </text>
      {(
        [
          ['A', A],
          ['B', B],
        ] as const
      ).map(([name, p]) => (
        <g key={name}>
          <circle className="end" cx={p[0]} cy={p[1]} r={30} />
          <text className="end-text" x={p[0]} y={p[1] + 12}>
            {name}
          </text>
        </g>
      ))}
    </svg>
  )
}
