import { useQuery } from '@tanstack/react-query'
import { useMemo } from 'react'
import { Pager } from '../components/Pager'
import { usePage } from '../components/usePage'
import type { JobApi, Loop } from './api'
import { bp, kb } from './format'

/** The Loops tab: every loop the engine read, with its factor, ends, span and PET count. A row
 * can be shown in the model, where its two anchors are marked. */
export function LoopsTab({ api, onShow }: { api: JobApi; onShow?: (loop: Loop) => void }) {
  const loops = useQuery({ queryKey: ['loops', api.base], queryFn: api.loops })
  const sorted = useMemo(() => [...(loops.data ?? [])].sort((a, b) => a.a[0] - b.a[0]), [loops.data])
  const page = usePage(sorted)
  if (!loops.data) return <div aria-busy="true" />
  return (
    <>
      <Pager page={page} noun="loops" />
      <table className="genes-table loops-table">
        <thead>
          <tr>
            <th>Factor</th>
            <th>Anchor A</th>
            <th>Anchor B</th>
            <th className="num">Span</th>
            <th className="num">PETs</th>
            {onShow && <th />}
          </tr>
        </thead>
        <tbody>
          {page.rows.map((l, i) => (
            <tr key={i}>
              <td>
                <span className={`loop-badge ${l.factor === 'RNAPII' ? 'rnapii' : 'ctcf'}`}>{l.factor}</span>
              </td>
              <td className="mono">{`${l.chrom}:${bp(l.a[0])}-${bp(l.a[1])}`}</td>
              <td className="mono">{`${l.chrom}:${bp(l.b[0])}-${bp(l.b[1])}`}</td>
              <td className="num">{kb(l.span)}</td>
              <td className="num">{l.pets}</td>
              {onShow && (
                <td>
                  <button type="button" className="row-action" onClick={() => onShow(l)}>
                    Show in the model
                  </button>
                </td>
              )}
            </tr>
          ))}
        </tbody>
      </table>
    </>
  )
}
