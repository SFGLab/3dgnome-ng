import type { ReactNode } from 'react'

/** One energy term on a small polymer: the chain before the term acts on it and after. Dark
 * beads are anchors. The blue marks are the pair, the length or the radius the term reads. */
type Pt = [number, number]

/** A state of the small polymer. `hot` beads are the ones the term acts on. */
function Chain({ pts, anchors, hot = [], open = false }: { pts: Pt[]; anchors: number[] | 'all'; hot?: number[]; open?: boolean }) {
  return (
    <g>
      <polyline className={`sk-chain${open ? ' open' : ''}`} points={pts.map((p) => p.join(',')).join(' ')} />
      {pts.map((p, i) => {
        const anchor = anchors === 'all' || anchors.includes(i)
        return <circle key={i} className={`sk-bead${anchor ? ' anchor' : ''}${hot.includes(i) ? ' hot' : ''}`} cx={p[0]} cy={p[1]} r={anchor ? 4.6 : 3.6} />
      })}
    </g>
  )
}

const Mark = ({ a, b }: { a: Pt; b: Pt }) => <line className="sk-mark" x1={a[0]} y1={a[1]} x2={b[0]} y2={b[1]} />
const Label = ({ at, children }: { at: Pt; children: string }) => (
  <text className="sk-label" x={at[0]} y={at[1]}>
    {children}
  </text>
)
/** A short arrow from a point along a direction, for the direction of a CTCF motif. */
const Motif = ({ from, dx }: { from: Pt; dx: number }) => (
  <g className="sk-motif">
    <line x1={from[0]} y1={from[1]} x2={from[0] + dx} y2={from[1]} />
    <path d={`M${from[0] + dx} ${from[1]}l${dx > 0 ? -5 : 5} -3.5v7z`} />
  </g>
)

const RING: Pt[] = [[26, 56], [14, 38], [26, 20], [45, 13], [64, 20], [76, 38], [64, 56]]

/** Each sketch as the state before and the state after, both drawn in a box of 90 by 84. */
const SKETCHES: Record<string, { text: string; before: ReactNode; after: ReactNode }> = {
  'anchor:Loop spring': {
    text: 'Brings the two anchors of a loop to the target distance of the loop.',
    before: (
      <>
        <Mark a={[8, 62]} b={[82, 62]} />
        <Chain open anchors="all" hot={[0, 4]} pts={[[8, 62], [24, 34], [45, 22], [66, 34], [82, 62]]} />
      </>
    ),
    after: (
      <>
        <Mark a={[33, 64]} b={[57, 64]} />
        <Label at={[45, 77]}>t</Label>
        <Chain open anchors="all" hot={[0, 4]} pts={[[33, 64], [17, 40], [45, 16], [73, 40], [57, 64]]} />
      </>
    ),
  },
  'anchor:Background spring': {
    text: 'Holds two nearby anchors without a loop at the distance the law gives for their separation.',
    before: <Chain open anchors="all" hot={[1, 2]} pts={[[8, 56], [39, 40], [49, 46], [82, 28]]} />,
    after: (
      <>
        <Mark a={[31, 50]} b={[59, 34]} />
        <Label at={[50, 58]}>b</Label>
        <Chain open anchors="all" hot={[1, 2]} pts={[[8, 62], [31, 50], [59, 34], [82, 22]]} />
      </>
    ),
  },
  'anchor:Window size': {
    text: 'Keeps the anchors of a genomic window as spread out as the law gives for its length.',
    before: (
      <>
        <circle className="sk-mark" cx={45} cy={42} r={25} />
        <Chain open anchors="all" pts={[[5, 18], [20, 70], [38, 28], [56, 74], [70, 16], [86, 58]]} />
      </>
    ),
    after: (
      <>
        <circle className="sk-mark" cx={45} cy={42} r={25} />
        <Label at={[45, 80]}>R</Label>
        <Chain open anchors="all" pts={[[24, 34], [32, 58], [42, 30], [52, 58], [58, 30], [66, 50]]} />
      </>
    ),
  },
  'anchor:Excluded volume': {
    text: 'Pushes apart two anchors closer than the radius r₀.',
    before: <Chain open anchors="all" hot={[1, 4]} pts={[[8, 24], [42, 36], [78, 30], [70, 62], [49, 42], [14, 62]]} />,
    after: (
      <>
        <circle className="sk-mark" cx={44} cy={32} r={11} />
        <Chain open anchors="all" hot={[1, 4]} pts={[[8, 20], [44, 32], [78, 28], [70, 62], [46, 56], [14, 66]]} />
      </>
    ),
  },
  'chain:Bond': {
    text: 'Keeps each bond at its target length.',
    before: (
      <>
        <Mark a={[20, 46]} b={[62, 30]} />
        <Chain anchors={[0, 4]} pts={[[8, 52], [20, 46], [62, 30], [72, 40], [84, 52]]} />
      </>
    ),
    after: (
      <>
        <Mark a={[27, 38]} b={[45, 32]} />
        <Label at={[35, 28]}>t</Label>
        <Chain anchors={[0, 4]} pts={[[8, 52], [27, 38], [45, 32], [63, 38], [84, 52]]} />
      </>
    ),
  },
  'chain:Bending': {
    text: 'Straightens sharp turns of the chain.',
    before: <Chain anchors={[0, 5]} hot={[2, 3]} pts={[[6, 58], [22, 24], [38, 60], [54, 24], [70, 60], [86, 30]]} />,
    after: <Chain anchors={[0, 5]} hot={[2, 3]} pts={[[6, 58], [20, 42], [38, 32], [56, 32], [72, 42], [86, 58]]} />,
  },
  'chain:CTCF orientation': {
    text: 'Turns the two anchors of a CTCF loop until their motifs point in opposite directions.',
    before: (
      <>
        <Chain anchors={[0, 6]} pts={RING} />
        <Motif from={[26, 70]} dx={15} />
        <Motif from={[64, 70]} dx={15} />
      </>
    ),
    after: (
      <>
        <Chain anchors={[0, 6]} pts={RING} />
        <Motif from={[26, 70]} dx={15} />
        <Motif from={[64, 70]} dx={-15} />
      </>
    ),
  },
  'chain:Excluded volume': {
    text: 'Pushes apart two beads closer than the radius r₀.',
    before: <Chain anchors={[0, 5]} hot={[1, 4]} pts={[[8, 24], [42, 36], [78, 30], [70, 62], [49, 42], [14, 62]]} />,
    after: (
      <>
        <circle className="sk-mark" cx={44} cy={32} r={11} />
        <Chain anchors={[0, 5]} hot={[1, 4]} pts={[[8, 20], [44, 32], [78, 28], [70, 62], [46, 56], [14, 66]]} />
      </>
    ),
  },
  'chain:Confinement': {
    text: 'Brings beads outside the sphere of radius R back toward it.',
    before: (
      <>
        <circle className="sk-mark" cx={40} cy={46} r={27} />
        <Chain anchors={[0]} hot={[4, 5]} pts={[[24, 46], [38, 34], [52, 44], [66, 30], [80, 18], [86, 36]]} />
      </>
    ),
    after: (
      <>
        <circle className="sk-mark" cx={40} cy={46} r={27} />
        <Label at={[78, 62]}>R</Label>
        <Chain anchors={[0]} hot={[4, 5]} pts={[[24, 46], [38, 34], [52, 44], [58, 58], [44, 64], [30, 60]]} />
      </>
    ),
  },
}

/** The sketch of one term: the small polymer before the term acts, an arrow and the polymer after.
 * Its sentence is the picture's description.
 *
 * @param phase anchor for the anchor energy, chain for the chain energy
 * @param name the term's name in the phase's table
 */
export function TermSketch({ phase, name }: { phase: 'anchor' | 'chain'; name: string }) {
  const s = SKETCHES[`${phase}:${name}`]
  return (
    <svg className="term-sketch" viewBox="0 0 220 84" role="img" aria-label={s.text}>
      <title>{s.text}</title>
      {s.before}
      <g className="sk-then">
        <line x1={99} y1={42} x2={117} y2={42} />
        <path d="M121 42l-6 -4v8z" />
      </g>
      <g transform="translate(130 0)">{s.after}</g>
    </svg>
  )
}
