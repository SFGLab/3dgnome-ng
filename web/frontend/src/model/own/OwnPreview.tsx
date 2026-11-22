import { asked, type Region } from '../../api'
import { useQueries } from '@tanstack/react-query'
import type { Entry } from './entry'
import { scoped } from '../../lib/css'
import sheet1 from '../model.module.css'
import sheet2 from './owndata.module.css'

const cn = scoped(sheet1, sheet2)

const W = 780
const H = 120
const AXIS = 60

/** The loops of the uploaded files in the region, the first factor above the axis and the rest below. */
export function OwnPreview({ cell, region, loops }: { cell: string; region: Region; loops: Entry[] }) {
  const span = `${region.chrom}:${region.start}-${region.end}`
  const ready = loops.filter((l) => l.result?.ok)
  const answers = useQueries({
    queries: ready.map((l) => asked.uploadPreview(l.result!.id, cell, span)),
  })
  const x = (p: number) => ((p - region.start) / (region.end - region.start)) * W
  return (
    <figure className={cn('own-preview')}>
      <figcaption>{`Loops in ${region.label.split(', ')[0]}`}</figcaption>
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`Uploaded loops in ${span}`}>
        {answers.flatMap((a, k) =>
          (a.data ?? []).map(([s, e], i) => {
            const above = loops.indexOf(ready[k]) === 0
            const r = Math.max((x(e) - x(s)) / 2, 1)
            return (
              <path
                key={`${k}-${i}`}
                className={cn(`arc f${Math.min(loops.indexOf(ready[k]), 2)}`)}
                d={`M ${x(s)} ${AXIS} A ${r} ${Math.min(r, AXIS - 6)} 0 0 ${above ? 1 : 0} ${x(e)} ${AXIS}`}
              />
            )
          }),
        )}
        <line className={cn('axis')} x1={0} x2={W} y1={AXIS} y2={AXIS} />
      </svg>
    </figure>
  )
}
