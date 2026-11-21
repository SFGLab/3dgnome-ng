import { ArrowRight, RefreshCw } from 'lucide-react'
import { useMemo, useState } from 'react'
import { Link } from 'react-router-dom'
import { ROUTES } from '../routes'
import { fold404, HEIGHT, LAYERS, piece, SIZE, walk, WIDTH } from './chainWalk'
import './notfound.css'

const STEPS = 404

/** Beads either side of an end of the digits drawn in its colour, as a promoter and an enhancer are. */
const REACH = 5

/** The sign of the page: the number 404 folded from one chain, the digits as the nearer strands. */
function Sign({ fold }: { fold: number }) {
  // eslint-disable-next-line react-hooks/exhaustive-deps
  const beads = useMemo(() => fold404(), [fold])
  const far: string[] = Array.from({ length: LAYERS }, () => '')
  const near: string[] = Array.from({ length: LAYERS }, () => '')
  for (let i = 0; i + 1 < beads.length; i++) (beads[i].digit && beads[i + 1].digit ? near : far)[Math.max(beads[i].layer, beads[i + 1].layer)] += piece(beads, i)
  const first = beads.findIndex((b) => b.digit)
  const last = beads.length - 1 - [...beads].reverse().findIndex((b) => b.digit)
  const mark = (at: number) =>
    Array.from({ length: 2 * REACH }, (_, k) => at - REACH + k)
      .filter((i) => i >= 0 && i + 1 < beads.length)
      .map((i) => piece(beads, i))
      .join('')
  return (
    <svg key={fold} className="not-found-sign" viewBox={`0 0 ${WIDTH} ${HEIGHT}`} role="img" aria-label="404, folded from one chain of beads">
      {far.map((d, k) => (
        <g key={k}>
          <path className="edge" d={d + near[k]} />
          <path className={`strand far step${k}`} d={d} />
          <path className={`strand near step${k}`} d={near[k]} />
        </g>
      ))}
      <path className="edge" d={mark(first) + mark(last)} />
      <path className="strand end promoter" d={mark(first)} />
      <path className="strand end enhancer" d={mark(last)} />
    </svg>
  )
}

/** The page of an address the site does not have: the sign 404 folded from one chain, a chain of
 * 404 steps folded at random, and the ways back. */
export function NotFound() {
  const [fold, setFold] = useState(1)
  // eslint-disable-next-line react-hooks/exhaustive-deps
  const chain = useMemo(() => walk(STEPS), [fold])
  const [a, b] = [chain[0], chain[chain.length - 1]]
  return (
    <div className="not-found">
      <div className="not-found-text">
        <Sign fold={fold} />
        <h1>No structure at this address</h1>
        <p className="lead">
          The link may be mistyped, or it was the link for a private job, that was deleted after 14 days. In its place the page has folded a chain of 404 steps for you using random data. Enjoy.
        </p>
        <p className="signed">- Nik</p>
        <div className="buttons">
          <Link className="button primary" to={ROUTES.model}>
            Model a real region <ArrowRight size={16} aria-hidden />
          </Link>
          <Link className="button secondary" to={ROUTES.examples}>
            Open an example
          </Link>
          <Link className="button secondary" to={ROUTES.jobs}>
            My jobs
          </Link>
        </div>
      </div>
      <figure className="not-found-figure">
        <svg viewBox={`0 0 ${SIZE} ${SIZE}`} role="img" aria-label={`A chain of ${STEPS} steps folded at random, structure ${fold}`}>
          <polyline key={fold} className="chain" points={chain.map((p) => p.join(',')).join(' ')} />
          <circle className="end promoter" cx={a[0]} cy={a[1]} r="7" />
          <circle className="end enhancer" cx={b[0]} cy={b[1]} r="7" />
        </svg>
        <figcaption>
          <span>{`Structure ${fold} of an ensemble that fits no data`}</span>
          <button type="button" className="refold" onClick={() => setFold(fold + 1)}>
            <RefreshCw size={13} aria-hidden /> Fold it again
          </button>
        </figcaption>
      </figure>
    </div>
  )
}
