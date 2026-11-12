import { useQuery } from '@tanstack/react-query'
import type { JobApi } from './api'
import { bp, kb } from './format'

/** The Loops tab: every loop the engine read, with its factor, ends, span and PET count. */
export function LoopsTab({ api }: { api: JobApi }) {
  const loops = useQuery({ queryKey: ['loops', api.base], queryFn: api.loops })
  if (!loops.data) return <div aria-busy="true" />
  return (
    <table className="genes-table loops-table">
      <thead>
        <tr>
          <th>Factor</th>
          <th>Anchor A</th>
          <th>Anchor B</th>
          <th className="num">Span</th>
          <th className="num">PETs</th>
        </tr>
      </thead>
      <tbody>
        {[...loops.data]
          .sort((a, b) => a.a[0] - b.a[0])
          .map((l, i) => (
            <tr key={i}>
              <td>
                <span className={`loop-badge ${l.factor === 'RNAPII' ? 'rnapii' : 'ctcf'}`}>{l.factor}</span>
              </td>
              <td className="mono">{`${l.chrom}:${bp(l.a[0])}-${bp(l.a[1])}`}</td>
              <td className="mono">{`${l.chrom}:${bp(l.b[0])}-${bp(l.b[1])}`}</td>
              <td className="num">{kb(l.span)}</td>
              <td className="num">{l.pets}</td>
            </tr>
          ))}
      </tbody>
    </table>
  )
}
