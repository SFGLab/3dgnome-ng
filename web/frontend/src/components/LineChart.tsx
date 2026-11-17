import { useState } from 'react'
import { Better, ChartKey, type Direction, type Series } from './BarChart'
import './barchart.css'
import './linechart.css'

export type Point = { x: number; seconds: number; region: string; beads: number }
export type Line = { series: Series; name: string; points: Point[] }

const W = 460
const H = 250
const PAD = { left: 46, right: 92, top: 12, bottom: 34 }
const TIME_TICKS: [number, string][] = [
  [10, '10 s'],
  [60, '1 min'],
  [600, '10 min'],
  [3600, '1 h'],
]

/** Seconds as the reader says them, seconds under a minute and minutes above. */
function duration(seconds: number): string {
  return seconds < 60 ? `${seconds.toFixed(seconds < 10 ? 1 : 0)} s` : `${(seconds / 60).toFixed(1)} min`
}

const decades = (lo: number, hi: number) => {
  const out: number[] = []
  for (let e = Math.ceil(Math.log10(lo) - 1e-9); e <= Math.floor(Math.log10(hi) + 1e-9); e++) out.push(10 ** e)
  return out
}

/**
 * Run time against one measure of the model's size, a line per tool, both axes logarithmic. Every point
 * answers to hover and focus with its region, bead count and time, and the table under the chart lists them all.
 */
export function LineChart({
  title,
  subtitle,
  xLabel,
  xTick,
  lines,
  better,
}: {
  title: string
  subtitle: string
  better?: Direction
  xLabel: string
  xTick: (x: number) => string
  lines: Line[]
}) {
  const [hot, setHot] = useState<{ line: Line; point: Point } | null>(null)
  const all = lines.flatMap((l) => l.points)
  const xs = all.map((p) => p.x)
  const ys = all.map((p) => p.seconds)
  const [x0, x1] = [Math.min(...xs) / 1.25, Math.max(...xs) * 1.25]
  const [y0, y1] = [Math.min(...ys) / 1.6, Math.max(...ys) * 1.6]
  const px = (x: number) => PAD.left + ((Math.log(x) - Math.log(x0)) / (Math.log(x1) - Math.log(x0))) * (W - PAD.left - PAD.right)
  const py = (y: number) => H - PAD.bottom - ((Math.log(y) - Math.log(y0)) / (Math.log(y1) - Math.log(y0))) * (H - PAD.top - PAD.bottom)
  // The names at the line ends, pushed apart where two lines end close together.
  const ends = lines
    .map((l) => ({ l, y: py(l.points[l.points.length - 1].seconds) }))
    .sort((a, b) => a.y - b.y)
  for (let i = 1; i < ends.length; i++) ends[i].y = Math.max(ends[i].y, ends[i - 1].y + 13)
  return (
    <figure className="bar-chart line-chart" aria-label={title}>
      <figcaption>
        <span className="chart-head">
          <span className="chart-title">{title}</span>
          {better && <Better dir={better} />}
        </span>
        <span className="chart-sub">{subtitle}</span>
      </figcaption>
      <ChartKey series={lines.map((l) => l.series)} label={`Key of ${title}`} />
      <div className="line-plot">
        <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`${title}, values in the table below`}>
          {TIME_TICKS.filter(([t]) => t >= y0 && t <= y1).map(([t, label]) => (
            <g key={t}>
              <line className="grid" x1={PAD.left} x2={W - PAD.right} y1={py(t)} y2={py(t)} />
              <text className="tick" x={PAD.left - 6} y={py(t) + 3.5} textAnchor="end">
                {label}
              </text>
            </g>
          ))}
          {decades(x0, x1).map((x) => (
            <text key={x} className="tick" x={px(x)} y={H - PAD.bottom + 14} textAnchor="middle">
              {xTick(x)}
            </text>
          ))}
          <line className="axis" x1={PAD.left} x2={W - PAD.right} y1={H - PAD.bottom} y2={H - PAD.bottom} />
          <text className="tick" x={(PAD.left + W - PAD.right) / 2} y={H - 4} textAnchor="middle">
            {xLabel}
          </text>
          {lines.map((l) => (
            <polyline key={l.series} className={`line ${l.series}`} points={l.points.map((p) => `${px(p.x)},${py(p.seconds)}`).join(' ')} />
          ))}
          {lines.map((l) =>
            l.points.map((p) => (
              <g key={`${l.series}-${p.x}`}>
                <circle className={`dot ${l.series}`} cx={px(p.x)} cy={py(p.seconds)} r={hot?.point === p ? 5 : 3.5} />
                <circle
                  className="hit"
                  cx={px(p.x)}
                  cy={py(p.seconds)}
                  r={11}
                  tabIndex={0}
                  aria-label={`${l.name}, ${p.region}, ${p.beads.toLocaleString('en-US')} beads, ${duration(p.seconds)}`}
                  onPointerEnter={() => setHot({ line: l, point: p })}
                  onPointerLeave={() => setHot(null)}
                  onFocus={() => setHot({ line: l, point: p })}
                  onBlur={() => setHot(null)}
                />
              </g>
            )),
          )}
          {ends.map(({ l, y }) => {
            // Names sit in one column right of the plot. A line that ends short of it gets a leader.
            const last = l.points[l.points.length - 1]
            const edge = W - PAD.right + 6
            return (
              <g key={l.series}>
                {px(last.x) < edge - 12 && <line className="leader" x1={px(last.x) + 6} y1={py(last.seconds)} x2={edge} y2={y} />}
                <text className="end-label" x={edge + 3} y={y + 3.5}>
                  {l.name}
                </text>
              </g>
            )
          })}
        </svg>
        {hot && (
          <div
            className="line-tip"
            role="status"
            style={{ left: `${(px(hot.point.x) / W) * 100}%`, top: `${(py(hot.point.seconds) / H) * 100}%` }}
          >
            <strong>{duration(hot.point.seconds)}</strong>
            <span>
              <i className={`swatch ${hot.line.series}`} />
              {hot.line.name}
            </span>
            <span>
              {hot.point.region}, {hot.point.beads.toLocaleString('en-US')} beads
            </span>
          </div>
        )}
      </div>
      <details className="line-table">
        <summary>Table</summary>
        <table>
          <thead>
            <tr>
              <th>Region</th>
              {lines.map((l) => (
                <th key={l.series}>{l.name}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {lines[0].points.map((p, i) => (
              <tr key={p.region}>
                <td>{p.region}</td>
                {lines.map((l) => (
                  <td key={l.series}>
                    {l.points[i] ? `${duration(l.points[i].seconds)}, ${l.points[i].beads.toLocaleString('en-US')} beads` : ''}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </details>
    </figure>
  )
}
