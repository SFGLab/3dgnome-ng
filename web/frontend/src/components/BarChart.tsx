import { scoped } from '../lib/css'
import sheet from './barchart.module.css'

const cn = scoped(sheet)



export type Series = 'original' | 'ctcf' | 'rnapii' | 'cudammc' | 'multimm'
export type Bar = { series: Series; value: number | null; label: string }
export type Group = { name: string; bars: Bar[] }

const SERIES_NAME: Record<Series, string> = {
  original: 'original models, cudaMMC',
  ctcf: '3D-GNOMEᴺᴳ, CTCF loops',
  rnapii: '3D-GNOMEᴺᴳ, CTCF and RNAPII loops',
  cudammc: 'cudaMMC',
  multimm: 'MultiMM',
}

export type Direction = 'higher' | 'lower'

/** The direction that counts as better on a measure, as an arrow and a few words. */
export function Better({ dir, text }: { dir: Direction; text?: string }) {
  return (
    <span className={cn('better')} data-testid="better">
      <span aria-hidden="true">{dir === 'higher' ? '↑' : '↓'}</span> {text ?? `${dir} is better`}
    </span>
  )
}

/** Horizontal bars in groups, each bar labelled with its value. A bar without a value is a placeholder. */
export function BarChart({ title, subtitle, groups, better }: { title: string; subtitle: string; groups: Group[]; better?: Direction }) {
  const max = Math.max(...groups.flatMap((g) => g.bars.map((b) => b.value ?? 0)), 1e-9)
  return (
    <figure className="bar-chart" aria-label={title}>
      <figcaption>
        <span className={cn('chart-head')}>
          <span className="chart-title">{title}</span>
          {better && <Better dir={better} />}
        </span>
        <span className="chart-sub">{subtitle}</span>
      </figcaption>
      {groups.map((g) => (
        <div key={g.name} className={cn('bar-group')}>
          {g.name && <span className={cn('group-name')}>{g.name}</span>}
          {g.bars.map((b, i) => (
            <div key={i} className={cn('bar-row')} title={`${SERIES_NAME[b.series]}: ${b.label}`}>
              <span
                className={cn(`bar ${b.series}${b.value === null ? ' placeholder' : ''}`)}
                style={{ width: b.value === null ? '45%' : `${(b.value / max) * 82}%` }}
              />
              <span className={cn(b.value === null ? 'bar-value muted' : 'bar-value')} data-testid="bar-value">
                {b.label}
              </span>
            </div>
          ))}
        </div>
      ))}
    </figure>
  )
}

const MODEL_SERIES: Series[] = ['original', 'ctcf', 'rnapii']

/** The key of the three kinds of model, or of the series given. */
export function ChartKey({ series = MODEL_SERIES, label = 'Key' }: { series?: Series[]; label?: string }) {
  return (
    <ul className={cn('chart-key')} aria-label={label}>
      {series.map((s) => (
        <li key={s}>
          <span className={cn(`swatch ${s}`)} />
          {SERIES_NAME[s]}
        </li>
      ))}
    </ul>
  )
}
