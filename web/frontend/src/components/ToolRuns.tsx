import './barchart.css'
import type { Series } from './BarChart'

/** One tool's run. A run without seconds did not finish and shows its note in place of a time. */
export type ToolRun = { series: Series; name: string; beads: number; seconds: number | null; note?: string }

type Measure = { key: string; label: string; value: (r: ToolRun) => number | null; text: (v: number) => string }

/**
 * The runs of several tools on one task as a table of bars. Each tool has three bars, the beads
 * in its model, its run time and its run time per 1,000 beads. Every column has its own scale.
 */
export function ToolRuns({ title, subtitle, unit, runs }: { title: string; subtitle: string; unit: 'seconds' | 'minutes'; runs: ToolRun[] }) {
  const measures: Measure[] = [
    { key: 'beads', label: 'beads in the model', value: (r) => r.beads, text: (v) => v.toLocaleString('en-US') },
    { key: 'bar', label: `run time, ${unit}`, value: (r) => (r.seconds === null ? null : unit === 'minutes' ? r.seconds / 60 : r.seconds), text: (v) => v.toFixed(1) },
    { key: 'rate', label: 'seconds per 1,000 beads', value: (r) => (r.seconds === null ? null : (r.seconds / r.beads) * 1000), text: (v) => v.toFixed(1) },
  ]
  const max = measures.map((m) => Math.max(...runs.map((r) => m.value(r) ?? 0), 1e-9))
  return (
    <figure className="bar-chart tool-runs" aria-label={title}>
      <figcaption>
        <span className="chart-title">{title}</span>
        <span className="chart-sub">{subtitle}</span>
      </figcaption>
      <table>
        <thead>
          <tr>
            <th scope="col">tool</th>
            {measures.map((m) => (
              <th key={m.key} scope="col">
                {m.label}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {runs.map((r) => (
            <tr key={r.name}>
              <th scope="row">{r.name}</th>
              {measures.map((m, i) => {
                const v = m.value(r)
                return (
                  <td key={m.key} data-label={m.label} title={`${r.name}, ${m.label}: ${v === null ? (r.note ?? '') : m.text(v)}`}>
                    <div className="bar-row">
                      {v !== null && <span className={`bar ${r.series}`} style={{ width: `${(v / max[i]) * 68}%` }} />}
                      <span className={v === null ? 'bar-value muted' : 'bar-value'} data-testid={`${m.key}-value`}>
                        {v === null ? (m.key === 'bar' ? (r.note ?? '') : '') : m.text(v)}
                      </span>
                    </div>
                  </td>
                )
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </figure>
  )
}
