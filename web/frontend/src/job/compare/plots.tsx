import { type ReactNode } from 'react'

const PLOT = { w: 340, h: 286, left: 48, right: 12, top: 12, bottom: 56 }

/** The ticks of an axis from lo to hi at a round step, six at most. */
function ticks(lo: number, hi: number): number[] {
  const step = [0.5, 1, 2, 5, 10, 20, 50, 100].find((s) => (hi - lo) / s <= 5) ?? 100
  const out: number[] = []
  for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step) out.push(Math.round(v * 10) / 10)
  return out
}

type Scale = (v: number) => number

export type Key = [className: string, text: string]

/** The y axis of a plot with its ticks and its title, and the key of the marks below the plot. */
function Sides({ yr, yTitle, capped, fy, keys }: { yr: [number, number]; yTitle: string; capped?: boolean; fy: Scale; keys: Key[] }) {
  const { h, left, top, bottom } = PLOT
  const middle = (top + h - bottom) / 2
  // Each mark of the key follows the text of the one before it.
  const starts = keys.map((_, i) => 14 + keys.slice(0, i).reduce((sum, [, text]) => sum + 18 + text.length * 4.6, 0))
  return (
    <>
      <line className="axis" x1={left} x2={left} y1={top} y2={h - bottom} />
      {ticks(...yr).map((v) => (
        <g key={`y${v}`}>
          <line className="axis" x1={left - 4} x2={left} y1={fy(v)} y2={fy(v)} />
          <text className="tick" x={left - 7} y={fy(v) + 3} textAnchor="end">
            {capped && v === yr[1] ? `${v}+` : v}
          </text>
        </g>
      ))}
      <text className="axis-title" x={12} y={middle} textAnchor="middle" transform={`rotate(-90 12 ${middle})`}>
        {yTitle}
      </text>
      {keys.map(([className, text], i) => (
        <g key={text}>
          <circle className={className} cx={starts[i]} cy={h - 9} r={className === 'back' ? 1.6 : 3} />
          <text className="tick" x={starts[i] + 7} y={h - 6}>
            {text}
          </text>
        </g>
      ))}
    </>
  )
}

/** A plot with an axis on the left and one below, each with its ticks and its title, a dashed
 * line at zero and the key of its marks. */
export function Frame({
  label,
  xr,
  yr,
  xTitle,
  yTitle,
  capped = false,
  keys,
  children,
}: {
  label: string
  xr: [number, number]
  yr: [number, number]
  xTitle: string
  yTitle: string
  /** The last tick of the y axis stands for its value and every value above it. */
  capped?: boolean
  keys: Key[]
  children: (fx: Scale, fy: Scale) => ReactNode
}) {
  const { w, h, left, right, top, bottom } = PLOT
  const fx: Scale = (v) => left + ((v - xr[0]) / (xr[1] - xr[0])) * (w - left - right)
  const fy: Scale = (v) => h - bottom - ((v - yr[0]) / (yr[1] - yr[0])) * (h - top - bottom)
  return (
    <svg className="expression-plot" viewBox={`0 0 ${w} ${h}`} role="img" aria-label={label}>
      {xr[0] < 0 && xr[1] > 0 && <line className="zero" x1={fx(0)} x2={fx(0)} y1={top} y2={h - bottom} />}
      {yr[0] < 0 && yr[1] > 0 && <line className="zero" x1={left} x2={w - right} y1={fy(0)} y2={fy(0)} />}
      <line className="axis" x1={left} x2={w - right} y1={h - bottom} y2={h - bottom} />
      {ticks(...xr).map((v) => (
        <g key={`x${v}`}>
          <line className="axis" x1={fx(v)} x2={fx(v)} y1={h - bottom} y2={h - bottom + 4} />
          <text className="tick" x={fx(v)} y={h - bottom + 15} textAnchor="middle">
            {v}
          </text>
        </g>
      ))}
      <text className="axis-title" x={(left + w - right) / 2} y={h - bottom + 30} textAnchor="middle">
        {xTitle}
      </text>
      <Sides yr={yr} yTitle={yTitle} capped={capped} fy={fy} keys={keys} />
      {children(fx, fy)}
    </svg>
  )
}

/** The value a share of the way through sorted values, between two of them where it falls there. */
function quantile(sorted: number[], share: number): number {
  const at = (sorted.length - 1) * share
  const low = Math.floor(at)
  return sorted[low] + (sorted[Math.min(low + 1, sorted.length - 1)] - sorted[low]) * (at - low)
}

/** The distance of one gene in the structures of two jobs: a box from the first to the third
 * quartile with the median, whiskers to the least and the greatest value and a point for each structure. */
export function BoxPlot({ gene, names, a, b }: { gene: string; names: [string, string]; a: number[]; b: number[] }) {
  const { w, h, left, right, top, bottom } = PLOT
  const hi = Math.max(1, Math.ceil(Math.max(...a, ...b) * 1.05))
  const fy: Scale = (v) => h - bottom - (v / hi) * (h - top - bottom)
  const width = w - left - right
  const sides = [a, b].map((values, i) => {
    const sorted = [...values].sort((x, y) => x - y)
    return { values, sorted, x: left + width * (i === 0 ? 0.28 : 0.72), q1: quantile(sorted, 0.25), median: quantile(sorted, 0.5), q3: quantile(sorted, 0.75) }
  })
  const half = 34
  return (
    <svg
      className="expression-plot"
      viewBox={`0 0 ${w} ${h}`}
      role="img"
      aria-label={`Enhancer to promoter distance of ${gene} in the structures of ${names[0]} and of ${names[1]}`}
    >
      <line className="axis" x1={left} x2={w - right} y1={h - bottom} y2={h - bottom} />
      {sides.map((s, i) => (
        <g key={i} className={`box side-${i}`}>
          <line className="whisker" x1={s.x} x2={s.x} y1={fy(s.sorted[0])} y2={fy(s.sorted[s.sorted.length - 1])} />
          <rect className="quartiles" x={s.x - half} width={half * 2} y={fy(s.q3)} height={Math.max(1, fy(s.q1) - fy(s.q3))} />
          <line className="median" x1={s.x - half} x2={s.x + half} y1={fy(s.median)} y2={fy(s.median)} />
          {s.values.map((v, k) => (
            // The points of a side are spread across the box in the order of the structures.
            <circle key={k} className="structure" cx={s.x - half + 6 + ((half * 2 - 12) * (k + 0.5)) / s.values.length} cy={fy(v)} r={2}>
              <title>{`structure ${k + 1}, ${v.toFixed(1)} bead units`}</title>
            </circle>
          ))}
          <text className="tick" x={s.x} y={h - bottom + 15} textAnchor="middle">
            {names[i]}
          </text>
          <text className="tick" x={s.x} y={h - bottom + 27} textAnchor="middle">
            {`${s.values.length} structure${s.values.length === 1 ? '' : 's'}`}
          </text>
        </g>
      ))}
      <Sides yr={[0, hi]} yTitle="3D distance, bead units" fy={fy} keys={[['structure', 'a structure']]} />
    </svg>
  )
}
