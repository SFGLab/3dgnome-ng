import { ArrowDown, ArrowUp, Download, Search } from 'lucide-react'
import { useEffect, useMemo, useState } from 'react'
import { CsvButton } from '../components/CsvButton'
import { Pager } from '../components/Pager'
import { usePage } from '../components/usePage'
import type { GeneRow } from '../api'
import { bp, kb } from '../lib/format'
import { scoped } from '../lib/css'
import sheet from './job.module.css'

const cn = scoped(sheet)

type Key = 'gene' | 'enhancer' | 'linear_bp' | 'distance' | 'expression'
type Chip = 'all' | 'loop' | 'rnapii'

const median = (xs: number[]) => {
  if (!xs.length) return NaN
  const s = [...xs].sort((a, b) => a - b)
  const m = s.length >> 1
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2
}

type Props = {
  rows: GeneRow[]
  chrom: string
  structures: number
  /** The unit of the expression of the job, TPM for one. The column is shown when a row has a level. */
  expressionUnit?: string
  epUrl: string
  filter: string
  onSelect: (gene: string) => void
  selected?: string | null
}

/**
 * The Genes tab. It has every marked gene with its nearest enhancer, the 3D distance over the
 * ensemble and the loop joining them and a histogram of the distances.
 */
export function GenesTab({ rows, chrom, structures, expressionUnit = '', epUrl, filter, onSelect, selected }: Props) {
  const [text, setText] = useState(filter)
  const [chip, setChip] = useState<Chip>('all')
  const [sort, setSort] = useState<{ key: Key; up: boolean }>({ key: 'distance', up: true })
  useEffect(() => setText(filter), [filter])
  const shown = useMemo(() => {
    const t = text.trim().toUpperCase()
    const kept = rows.filter((r) => r.gene.toUpperCase().includes(t) && (chip === 'all' || (chip === 'loop' ? r.joined_by !== '' : r.joined_by === 'RNAPII')))
    const dir = sort.up ? 1 : -1
    // A gene without a level sorts as the lowest.
    const of = (r: GeneRow, key: Exclude<Key, 'gene'>) => (key === 'expression' ? (r.expression ?? -1) : r[key])
    return [...kept].sort((a, b) => (sort.key === 'gene' ? a.gene.localeCompare(b.gene) : of(a, sort.key) - of(b, sort.key)) * dir)
  }, [rows, text, chip, sort])
  const page = usePage(shown, `${text}|${chip}|${sort.key}|${sort.up}`)
  const max = Math.max(...rows.map((r) => r.max), 1)
  const hasExpression = rows.some((r) => r.expression != null)
  const header = (key: Key, label: string, num = false) => (
    <th className={cn(num ? 'num' : undefined)} aria-sort={sort.key === key ? (sort.up ? 'ascending' : 'descending') : 'none'} onClick={() => setSort((s) => ({ key, up: s.key === key ? !s.up : true }))}>
      {label}
      {sort.key === key && (sort.up ? <ArrowUp size={11} aria-hidden /> : <ArrowDown size={11} aria-hidden />)}
    </th>
  )
  const looped = rows.filter((r) => r.joined_by).map((r) => r.distance)
  const other = rows.filter((r) => !r.joined_by).map((r) => r.distance)
  const hist = useMemo(() => {
    const n = 13
    const counts = Array(n).fill(0)
    for (const r of rows) counts[Math.min(n - 1, Math.floor((r.distance / max) * n))]++
    return counts
  }, [rows, max])
  const top = Math.max(...hist, 1)
  return (
    <div className={cn('genes-tab')}>
      <div className={cn('genes-main')}>
        <div className={cn('genes-toolbar')}>
          <label className={cn('filter-genes')}>
            <Search size={14} aria-hidden />
            <input placeholder="Filter genes" value={text} onChange={(e) => setText(e.target.value)} />
          </label>
          {(
            [
              ['all', 'All genes'],
              ['loop', 'Joined by a loop'],
              ['rnapii', 'Joined by RNAPII only'],
            ] as [Chip, string][]
          ).map(([id, label]) => (
            <button key={id} type="button" className={cn(chip === id ? 'chip-button on' : 'chip-button')} aria-pressed={chip === id} onClick={() => setChip(id)}>
              {label}
            </button>
          ))}
          <a className="chip-button" href={epUrl}>
            <Download size={13} aria-hidden /> <span className="mono">ep.tsv</span>
          </a>
          <CsvButton
            file="genes"
            head={['gene', 'tss', 'nearest_enhancer', 'separation_bp', 'distance_mean', 'distance_sd', 'distance_min', 'distance_q1', 'distance_median', 'distance_q3', 'distance_max', 'structures', 'joined_by', ...(hasExpression ? [`expression${expressionUnit ? `_${expressionUnit}` : ''}`] : [])]}
            rows={() => shown.map((r) => [r.gene, r.tss, r.enhancer, r.linear_bp, r.distance, r.sd, r.min, r.q1, r.median, r.q3, r.max, r.n, r.joined_by, ...(hasExpression ? [r.expression] : [])])}
          />
        </div>
        <Pager page={page} noun="genes" />
        <table className={cn('genes-table')}>
          <thead>
            <tr>
              {header('gene', 'Gene')}
              {header('enhancer', 'Nearest enhancer')}
              {header('linear_bp', 'Separation', true)}
              {header('distance', '3D distance', true)}
              {hasExpression && header('expression', `Expression${expressionUnit ? `, ${expressionUnit}` : ''}`, true)}
              <th>{`Over ${structures} structures`}</th>
              <th>Loop between them</th>
            </tr>
          </thead>
          <tbody>
            {page.rows.map((r) => (
              <tr key={r.gene} className={cn(r.gene === selected ? 'selected' : undefined)} onClick={() => onSelect(r.gene)}>
                <td className={cn('gene')}>{r.gene}</td>
                <td className="mono" title={`${chrom}:${bp(r.enhancer)}`}>
                  {bp(r.enhancer)}
                </td>
                <td className={cn('num')}>{kb(r.linear_bp)}</td>
                <td className={cn('num strong')}>{r.distance.toFixed(1)}</td>
                {hasExpression && <td className={cn('num')}>{r.expression == null ? '' : r.expression >= 100 ? r.expression.toFixed(0) : r.expression.toFixed(1)}</td>}
                <td>
                  <span className={cn('range')} title={`${r.min.toFixed(1)} to ${r.max.toFixed(1)}, median ${r.median.toFixed(1)}`}>
                    <span className={cn('whisker')} style={{ left: `${(r.min / max) * 100}%`, width: `${((r.max - r.min) / max) * 100}%` }} />
                    <span className={cn('box')} style={{ left: `${(r.q1 / max) * 100}%`, width: `${((r.q3 - r.q1) / max) * 100}%` }} />
                    <span className={cn('med')} style={{ left: `${(r.median / max) * 100}%` }} />
                  </span>
                </td>
                <td>
                  {r.joined_by ? (
                    <span className={cn(`loop-badge ${r.joined_by === 'RNAPII' ? 'rnapii' : 'ctcf'}`)}>{`${r.joined_by} loop`}</span>
                  ) : (
                    <span className={cn('no-loop')}>{r.promoter_loops ? `none, the promoter anchors ${r.promoter_loops} loop${r.promoter_loops === 1 ? '' : 's'}` : 'none'}</span>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <section className={cn('side-card hist-card')} aria-label="3D distance, all genes">
        <p className={cn('card-label')}>3D distance, all genes</p>
        <div className={cn('hist')}>
          {hist.map((c, i) => (
            <span key={i} style={{ height: `${(c / top) * 100}%` }} />
          ))}
        </div>
        <p className={cn('hist-axis')}>
          <span>0</span>
          <span>bead units</span>
          <span>{Math.ceil(max)}</span>
        </p>
        <p className={cn('card-text')}>
          {`The median is ${median(rows.map((r) => r.distance)).toFixed(1)}.`}
          {looped.length > 0 &&
            other.length > 0 &&
            ` It is ${median(looped).toFixed(1)} for genes joined to their enhancer by a loop and ${median(other).toFixed(1)} for the other genes.`}
        </p>
      </section>
    </div>
  )
}
