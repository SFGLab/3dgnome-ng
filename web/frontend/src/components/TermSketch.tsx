import type { ReactNode } from 'react'

/**
 * One energy term on a small polymer, the chain before the term acts on it and after. Dark beads
 * are anchors. The blue marks are the pair, the length or the radius the term reads.
 */
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
/** The direction of the CTCF motif at an anchor, an arrow beside the anchor. */
function Motif({ from, to }: { from: Pt; to: Pt }) {
  const turn = (Math.atan2(to[1] - from[1], to[0] - from[0]) * 180) / Math.PI
  return (
    <g className="sk-motif">
      <line x1={from[0]} y1={from[1]} x2={to[0]} y2={to[1]} />
      <path d="M2 0l-6.5 -3.8v7.6z" transform={`translate(${to[0]} ${to[1]}) rotate(${turn})`} />
    </g>
  )
}

/**
 * A loop along the genome. It has the line, its two anchors, the loop over them and the direction
 * of each motif under its anchor. The first motif points right.
 */
function Genome({ second }: { second: 'left' | 'right' }) {
  const chevron = (x: number, right: boolean) => <path className="sk-chevron" d={right ? `M${x - 3} 17l5 4l-5 4` : `M${x + 3} 17l-5 4l5 4`} />
  return (
    <g>
      <line className="sk-genome" x1={12} y1={11} x2={88} y2={11} />
      <path className="sk-mark" d="M30 11Q50 -7 70 11" />
      <circle className="sk-bead anchor" cx={30} cy={11} r={3.4} />
      <circle className="sk-bead anchor" cx={70} cy={11} r={3.4} />
      {chevron(30, true)}
      {chevron(70, second === 'right')}
    </g>
  )
}

/** Each sketch as the state before and the state after, both drawn in a box of 90 by 84, or as
 * one whole drawing of 220 by 108. */
const SKETCHES: Record<string, { text: string; before?: ReactNode; after?: ReactNode; whole?: ReactNode }> = {
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
  // The drawing follows the figure of the 3D-GNOME paper. A motif on the reverse strand points
  // against the chain. Motifs that face each other along the genome are side by side in a hairpin
  // and motifs of one direction in a loop of a full turn.
  'chain:CTCF orientation': {
    text: 'Turns the chain at the two anchors of a CTCF loop until their motifs point the same way in space. Motifs that face each other along the genome give a hairpin and motifs of one direction give a loop of a full turn.',
    whole: (
      <>
        <Genome second="left" />
        <path className="sk-chain" d="M34 104C36 94 41 91 42 84C44 66 22 62 26 46C30 28 70 28 74 46C78 62 56 66 58 84C59 91 64 94 66 104" />
        <Motif from={[42, 84]} to={[43, 66]} />
        <Motif from={[58, 84]} to={[57, 66]} />
        <circle className="sk-bead anchor" cx={42} cy={84} r={4.6} />
        <circle className="sk-bead anchor" cx={58} cy={84} r={4.6} />
        <line className="sk-split" x1={110} y1={6} x2={110} y2={100} />
        <g transform="translate(122 0)">
          <Genome second="right" />
          <path className="sk-chain" d="M36 104C37 96 38 90 38 84C38 70 40 56 52 48C66 38 86 48 86 66C86 84 70 96 58 94C52 93 50 90 50 84C50 70 48 52 46 34" />
          <Motif from={[38, 84]} to={[38, 66]} />
          <Motif from={[50, 84]} to={[50, 66]} />
          <circle className="sk-bead anchor" cx={38} cy={84} r={4.6} />
          <circle className="sk-bead anchor" cx={50} cy={84} r={4.6} />
        </g>
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

/**
 * The sketch of one term. It has the small polymer before the term acts, an arrow and the polymer
 * after. Its sentence is the description of the picture.
 *
 * @param phase anchor for the anchor energy, chain for the chain energy
 * @param name the name of the term in the table of the phase
 */
export function TermSketch({ phase, name }: { phase: 'anchor' | 'chain'; name: string }) {
  const s = SKETCHES[`${phase}:${name}`]
  return (
    <svg className="term-sketch" viewBox={s.whole ? '0 0 220 108' : '0 0 220 84'} role="img" aria-label={s.text}>
      <title>{s.text}</title>
      {s.whole ?? (
        <>
          {s.before}
          <g className="sk-then">
            <line x1={99} y1={42} x2={117} y2={42} />
            <path d="M121 42l-6 -4v8z" />
          </g>
          <g transform="translate(130 0)">{s.after}</g>
        </>
      )}
    </svg>
  )
}
