import './barchart.css'

export type Series = 'original' | 'ctcf' | 'rnapii'
export type Bar = { series: Series; value: number | null; label: string }
export type Group = { name: string; bars: Bar[] }

const SERIES_NAME: Record<Series, string> = {
  original: 'original models, cudaMMC',
  ctcf: '3D-GNOMEᴺᴳ, CTCF loops',
  rnapii: '3D-GNOMEᴺᴳ, CTCF and RNAPII loops',
}

/** Horizontal bars in groups, each bar labelled with its value. A bar without a value is a placeholder. */
export function BarChart({ title, subtitle, groups }: { title: string; subtitle: string; groups: Group[] }) {
  const max = Math.max(...groups.flatMap((g) => g.bars.map((b) => b.value ?? 0)), 1e-9)
  return (
    <figure className="bar-chart" aria-label={title}>
      <figcaption>
        <span className="chart-title">{title}</span>
        <span className="chart-sub">{subtitle}</span>
      </figcaption>
      {groups.map((g) => (
        <div key={g.name} className="bar-group">
          {g.name && <span className="group-name">{g.name}</span>}
          {g.bars.map((b, i) => (
            <div key={i} className="bar-row" title={`${SERIES_NAME[b.series]}: ${b.label}`}>
              <span
                className={`bar ${b.series}${b.value === null ? ' placeholder' : ''}`}
                style={{ width: b.value === null ? '45%' : `${(b.value / max) * 82}%` }}
              />
              <span className={b.value === null ? 'bar-value muted' : 'bar-value'} data-testid="bar-value">
                {b.label}
              </span>
            </div>
          ))}
        </div>
      ))}
    </figure>
  )
}

/** The key of the three kinds of model. */
export function ChartKey() {
  return (
    <ul className="chart-key" aria-label="Key">
      {(Object.keys(SERIES_NAME) as Series[]).map((s) => (
        <li key={s}>
          <span className={`swatch ${s}`} />
          {SERIES_NAME[s]}
        </li>
      ))}
    </ul>
  )
}
