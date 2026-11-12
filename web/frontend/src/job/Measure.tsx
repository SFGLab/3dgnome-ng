import type { Distance } from './api'

/** The Distance A to B card: the mean and SD over the ensemble, the value in this structure,
 * a dot per structure and the two positions. */
export function DistanceCard({ d, current, labels }: { d: Distance; current: number; labels: [string, string] }) {
  const lo = Math.min(...d.values)
  const hi = Math.max(...d.values)
  const x = (v: number) => (hi > lo ? ((v - lo) / (hi - lo)) * 100 : 50)
  const here = d.values[current - 1]
  return (
    <section className="side-card dark-border" aria-label="Distance A to B">
      <p className="card-label">Distance A to B</p>
      <p className="big">
        <span className="big-value">{`${d.mean.toFixed(1)} ± ${d.sd.toFixed(1)}`}</span> <span className="unit">bead units</span>
      </p>
      <p className="card-text">
        {`Mean and standard deviation over ${d.n} structure${d.n === 1 ? '' : 's'}.`}
        {here !== undefined && ` The value in this structure is ${here.toFixed(1)}.`}
      </p>
      <div className="strip" aria-hidden>
        <span className="axis" />
        {d.values.map((v, i) => (
          <span key={i} className={i === current - 1 ? 'dot current' : 'dot'} style={{ left: `${x(v)}%` }} />
        ))}
      </div>
      <p className="point">
        <span className="marker">A</span> {labels[0]}
      </p>
      <p className="point">
        <span className="marker">B</span> {labels[1]}
      </p>
    </section>
  )
}
