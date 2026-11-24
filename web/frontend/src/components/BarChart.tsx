import { scoped } from '../lib/css'
import sheet from './barchart.module.css'

const cn = scoped(sheet)



export type Series = 'original' | 'ctcf' | 'rnapii' | 'anchors' | 'cudammc' | 'multimm'
/** A bar. `p` is the p value of the test of the bar against the reference bar of its group, as text, and the word reference on that bar. */
export type Bar = { series: Series; value: number | null; label: string; p?: string }
export type Group = { name: string; bars: Bar[] }

const SERIES_NAME: Record<Series, string> = {
  original: 'original models, cudaMMC',
  ctcf: '3D-GNOMEᴺᴳ, chain of one bead per kilobase, CTCF loops',
  rnapii: '3D-GNOMEᴺᴳ, chain of one bead per kilobase, CTCF and RNAPII loops',
  anchors: '3D-GNOMEᴺᴳ, chain of loop anchors, CTCF loops',
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

/** Horizontal bars in groups, each bar labelled with its value and its p value where it has one. A bar without a value is a placeholder. */
export function BarChart({ title, subtitle, groups, better }: { title: string; subtitle: string; groups: Group[]; better?: Direction }) {
  const max = Math.max(...groups.flatMap((g) => g.bars.map((b) => b.value ?? 0)), 1e-9)
  // The longest bar leaves room for its value and for a p value beside it.
  const widest = groups.some((g) => g.bars.some((b) => b.p)) ? 58 : 82
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
            <div key={i} className={cn('bar-row')} title={`${SERIES_NAME[b.series]}: ${b.label}${b.p ? `, ${b.p}` : ''}`}>
              <span
                className={cn(`bar ${b.series}${b.value === null ? ' placeholder' : ''}`)}
                style={{ width: b.value === null ? '45%' : `${(b.value / max) * widest}%` }}
              />
              <span className={cn(b.value === null ? 'bar-value muted' : 'bar-value')} data-testid="bar-value">
                {b.label}
              </span>
              {b.p && (
                <span className={cn('bar-p')} data-testid="bar-p">
                  {b.p}
                </span>
              )}
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
