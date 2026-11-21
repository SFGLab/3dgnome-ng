import { ArrowRight, RefreshCw } from 'lucide-react'
import { useMemo, useState } from 'react'
import { Link } from 'react-router-dom'
import { ROUTES } from '../routes'
import { SIZE, walk } from './chainWalk'
import './notfound.css'

const STEPS = 404

/** The page of an address the site does not have: a chain of 404 steps folded at random, and the ways back. */
export function NotFound() {
  const [fold, setFold] = useState(1)
  // eslint-disable-next-line react-hooks/exhaustive-deps
  const chain = useMemo(() => walk(STEPS), [fold])
  const [a, b] = [chain[0], chain[chain.length - 1]]
  return (
    <div className="not-found">
      <div className="not-found-text">
        <p className="not-found-code">404</p>
        <h1>No structure at this address</h1>
        <p className="lead">
          The site has no page here. The link may be mistyped, or it was the link of a private job, which is deleted after 14 days. In its place the page has folded a chain of 404 steps for you, with no loops and no Hi-C
          to guide it.
        </p>
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
