import { useQuery } from '@tanstack/react-query'
import { api, type Region } from '../api'
import type { Factors } from './form'

const W = 780
const H = 150
const AXIS = 75

const plural = (n: number, word: string) => `${n.toLocaleString('en-US')} ${word}${n === 1 ? '' : 's'}`

/** The loops of a region as arcs over an axis, CTCF above and RNAPII below, and the genes as ticks. */
export function LoopPreview({ cell, region, factors }: { cell: string; region: Region; factors: Factors }) {
  const span = `${region.chrom}:${region.start}-${region.end}`
  const preview = useQuery({ queryKey: ['preview', cell, span], queryFn: () => api.preview(cell, span) })
  if (!preview.data) return <div className="loop-preview" aria-busy="true" />
  const x = (p: number) => ((p - region.start) / (region.end - region.start)) * W
  const chosen = (f: string) => factors === 'CTCF+RNAPII' || factors === f
  const arcs = (factor: 'CTCF' | 'RNAPII', above: boolean) =>
    (preview.data.loops[factor] ?? []).map(([a, b, n], i) => {
      const x1 = x(a)
      const x2 = x(b)
      const r = Math.max((x2 - x1) / 2, 1)
      return (
        <path
          key={`${factor}${i}`}
          className={`arc ${factor.toLowerCase()}${chosen(factor) ? '' : ' faint'}`}
          d={`M ${x1} ${AXIS} A ${r} ${Math.min(r, AXIS - 6)} 0 0 ${above ? 1 : 0} ${x2} ${AXIS}`}
        >
          <title>{`${factor} loop, ${a.toLocaleString('en-US')} to ${b.toLocaleString('en-US')}, ${n} PETs`}</title>
        </path>
      )
    })
  const gene = region.gene ? preview.data.genes.find((g) => g.name === region.gene) : undefined
  return (
    <div className="loop-preview">
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`Loops in ${region.label || span}`}>
        <text className="side-label" x={6} y={14}>
          CTCF loops
        </text>
        <text className="side-label" x={6} y={H - 6}>
          RNAPII loops
        </text>
        {arcs('CTCF', true)}
        {arcs('RNAPII', false)}
        <line className="axis" x1={0} x2={W} y1={AXIS} y2={AXIS} />
        {preview.data.genes.map((g) => (
          <line key={g.name + g.tss} className={g === gene ? 'gene-tick named' : 'gene-tick'} x1={x(g.tss)} x2={x(g.tss)} y1={AXIS - 5} y2={AXIS + 5} />
        ))}
        {gene && (
          <text className="gene-label" x={x(gene.tss)} y={AXIS - 9} textAnchor="middle">
            {gene.name}
          </text>
        )}
      </svg>
      <p className="preview-key">
        <span className="key-item">
          <span className="key-line ctcf" />
          {plural(preview.data.counts.CTCF ?? 0, 'CTCF loop')}
        </span>
        <span className="key-item">
          <span className="key-line rnapii" />
          {plural(preview.data.counts.RNAPII ?? 0, 'RNAPII loop')}
        </span>
      </p>
    </div>
  )
}
