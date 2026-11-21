import { useEffect } from 'react'
import { Pager } from '../../components/Pager'
import { usePage } from '../../components/usePage'
import { scoped } from '../../lib/css'
import sheet from '../job.module.css'

const cn = scoped(sheet)

/** A table of the comparison with every gene, a page at a time. The page with the selected gene is shown. */
export function Paged<T extends { gene: string }>({
  rows,
  gene,
  noun,
  label,
  head,
  cells,
  faint,
  onGene,
}: {
  rows: T[]
  gene: string | null
  noun: string
  label: string
  head: string[]
  cells: (r: T) => string[]
  /** A row whose change is not significant, drawn in grey. */
  faint: (r: T) => boolean
  onGene?: (gene: string | null) => void
}) {
  const page = usePage(rows, label)
  const at = rows.findIndex((r) => r.gene === gene)
  const { go, size } = page
  useEffect(() => {
    if (at >= 0) go(Math.floor(at / size))
    // The page follows the selection alone.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [gene, at, size])
  return (
    <>
      <div className={cn('compare-scroll')}>
        <table className={cn('genes-table compare-table')} aria-label={label}>
          <thead>
            <tr>
              {head.map((h) => (
                <th key={h}>{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {page.rows.map((r) => (
              <tr
                key={r.gene}
                className={cn([r.gene === gene ? 'selected' : '', faint(r) ? 'insignificant' : ''].join(' ').trim() || undefined)}
                aria-selected={r.gene === gene}
                onClick={() => onGene?.(r.gene === gene ? null : r.gene)}
              >
                {cells(r).map((c, i) => (
                  <td key={i}>{c}</td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <Pager page={page} noun={noun} />
    </>
  )
}
