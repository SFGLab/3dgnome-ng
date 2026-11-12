import { ArrowDown, ArrowUp, Download, Search } from 'lucide-react'
import { useEffect, useMemo, useState } from 'react'
import type { GeneRow } from './api'
import { bp, kb } from './format'

type Key = 'gene' | 'enhancer' | 'linear_bp' | 'distance'
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
  epUrl: string
  filter: string
  onSelect: (gene: string) => void
  selected?: string | null
}

/** The Genes tab: every marked gene with its nearest enhancer, the 3D distance over the
 * ensemble and the loop joining them, and a histogram of the distances. */
export function GenesTab({ rows, chrom, structures, epUrl, filter, onSelect, selected }: Props) {
  const [text, setText] = useState(filter)
  const [chip, setChip] = useState<Chip>('all')
  const [sort, setSort] = useState<{ key: Key; up: boolean }>({ key: 'distance', up: true })
  useEffect(() => setText(filter), [filter])
  const shown = useMemo(() => {
    const t = text.trim().toUpperCase()
    const kept = rows.filter(
      (r) => r.gene.toUpperCase().includes(t) && (chip === 'all' || (chip === 'loop' ? r.joined_by !== '' : r.joined_by === 'RNAPII')),
    )
    const dir = sort.up ? 1 : -1
    return [...kept].sort((a, b) => (sort.key === 'gene' ? a.gene.localeCompare(b.gene) : a[sort.key] - b[sort.key]) * dir)
  }, [rows, text, chip, sort])
  const max = Math.max(...rows.map((r) => r.max), 1)
  const header = (key: Key, label: string, num = false) => (
    <th className={num ? 'num' : undefined} onClick={() => setSort((s) => ({ key, up: s.key === key ? !s.up : true }))}>
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
    <div className="genes-tab">
      <div className="genes-main">
        <div className="genes-toolbar">
          <label className="filter-genes">
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
            <button key={id} type="button" className={chip === id ? 'chip-button on' : 'chip-button'} aria-pressed={chip === id} onClick={() => setChip(id)}>
              {label}
            </button>
          ))}
          <a className="chip-button" href={epUrl}>
            <Download size={13} aria-hidden /> <span className="mono">ep.tsv</span>
          </a>
        </div>
        <table className="genes-table">
          <thead>
            <tr>
              {header('gene', 'Gene')}
              {header('enhancer', 'Nearest enhancer')}
              {header('linear_bp', 'Separation', true)}
              {header('distance', '3D distance', true)}
              <th>{`Over ${structures} structures`}</th>
              <th>Joined by</th>
            </tr>
          </thead>
          <tbody>
            {shown.map((r) => (
              <tr key={r.gene} className={r.gene === selected ? 'selected' : undefined} onClick={() => onSelect(r.gene)}>
                <td className="gene">{r.gene}</td>
                <td className="mono" title={`${chrom}:${bp(r.enhancer)}`}>
                  {bp(r.enhancer)}
                </td>
                <td className="num">{kb(r.linear_bp)}</td>
                <td className="num strong">{r.distance.toFixed(1)}</td>
                <td>
                  <span className="range" title={`${r.min.toFixed(1)} to ${r.max.toFixed(1)}, median ${r.median.toFixed(1)}`}>
                    <span className="whisker" style={{ left: `${(r.min / max) * 100}%`, width: `${((r.max - r.min) / max) * 100}%` }} />
                    <span className="box" style={{ left: `${(r.q1 / max) * 100}%`, width: `${((r.q3 - r.q1) / max) * 100}%` }} />
                    <span className="med" style={{ left: `${(r.median / max) * 100}%` }} />
                  </span>
                </td>
                <td>
                  {r.joined_by ? (
                    <span className={`loop-badge ${r.joined_by === 'RNAPII' ? 'rnapii' : 'ctcf'}`}>{`${r.joined_by} loop`}</span>
                  ) : (
                    <span className="no-loop">no loop</span>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <section className="side-card hist-card" aria-label="3D distance, all genes">
        <p className="card-label">3D distance, all genes</p>
        <div className="hist">
          {hist.map((c, i) => (
            <span key={i} style={{ height: `${(c / top) * 100}%` }} />
          ))}
        </div>
        <p className="hist-axis">
          <span>0</span>
          <span>bead units</span>
          <span>{Math.ceil(max)}</span>
        </p>
        <p className="card-text">
          {`The median is ${median(rows.map((r) => r.distance)).toFixed(1)}.`}
          {looped.length > 0 && other.length > 0 && ` It is ${median(looped).toFixed(1)} for genes joined to their enhancer by a loop and ${median(other).toFixed(1)} for the other genes.`}
        </p>
      </section>
    </div>
  )
}
