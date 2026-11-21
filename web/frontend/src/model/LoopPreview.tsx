import { keepPreviousData, useQuery } from '@tanstack/react-query'
import { useEffect, useRef, useState } from 'react'
import { api, type Region } from '../api'
import type { Factors } from './form'

const W = 780
const H = 150
const AXIS = 75
/** The shortest region a drag leaves, and the step positions are rounded to, in bp. */
const MIN_SPAN = 20_000
const STEP = 1_000

const plural = (n: number, word: string) => `${n.toLocaleString('en-US')} ${word}${n === 1 ? '' : 's'}`
const mb = (n: number) => `${(n / 1e6).toFixed(n >= 1e7 ? 1 : 2)} Mb`
const round = (p: number) => Math.round(p / STEP) * STEP

/**
 * The stretch drawn around a region: the region with its own length on either side, from 0. The
 * stretch is `cap` bp at most. A region that leaves less room gets less on either side, and a
 * region of `cap` or more is drawn from a little before its start.
 */
export function around(start: number, end: number, cap = Infinity): [number, number] {
  const span = end - start
  if (span >= cap) {
    const from = Math.max(0, start - Math.floor(cap / 20))
    return [from, from + cap]
  }
  const pad = Math.min(span, Math.floor((cap - span) / 2))
  const from = Math.max(0, start - pad)
  return [from, from === 0 ? Math.max(end + pad, span + 2 * pad) : end + pad]
}

/** Whether a region is drawn well in a view: inside it and between an eighth and seven tenths of it. */
const fits = (start: number, end: number, view: [number, number]) => {
  const share = (end - start) / (view[1] - view[0])
  return start >= view[0] && end <= view[1] && share >= 0.125 && share <= 0.7
}

type Drag = { kind: 'start' | 'end' | 'move'; at: number; from: [number, number] }

/**
 * The loops around a region as arcs over an axis, CTCF above and RNAPII below, the genes as ticks
 * and the region as a box. With `onRegion` the edges of the box are sliders and the box can be
 * moved. The new coordinates are reported when the drag ends.
 *
 * `maxSpan` is the longest region of a job in bp. A drag does not make the region longer than
 * that. A region that is longer already is drawn in the colour of an error and a drag makes it
 * shorter.
 */
export function LoopPreview({ cell, region, factors, maxSpan = Infinity, onRegion }: { cell: string; region: Region; factors: Factors; maxSpan?: number; onRegion?: (text: string) => void }) {
  // The server draws three times the longest region at most.
  const cap = 3 * maxSpan
  // The view moves only when the region no longer fits it, so the picture stays still while the
  // region is dragged inside it.
  const [view, setView] = useState<[number, number]>(() => around(region.start, region.end, cap))
  const [chrom, setChrom] = useState(region.chrom)
  useEffect(() => {
    if (region.chrom !== chrom || !fits(region.start, region.end, view)) {
      setView(around(region.start, region.end, cap))
      setChrom(region.chrom)
    }
  }, [region.chrom, region.start, region.end, cap]) // eslint-disable-line react-hooks/exhaustive-deps
  const span = `${region.chrom}:${view[0]}-${view[1]}`
  const preview = useQuery({ queryKey: ['preview', cell, span], queryFn: () => api.preview(cell, span), placeholderData: keepPreviousData })
  const svg = useRef<SVGSVGElement>(null)
  const drag = useRef<Drag | null>(null)
  const [moved, setMoved] = useState<[number, number] | null>(null)
  const latest = useRef<[number, number] | null>(null)
  if (!preview.data) return <div className="loop-preview" aria-busy="true" />
  const [start, end] = moved ?? [region.start, region.end]
  const width = view[1] - view[0]
  const x = (p: number) => ((p - view[0]) / width) * W
  const chosen = (f: string) => factors === 'CTCF+RNAPII' || factors === f
  // A loop counts for the region when both its anchors lie inside it, as the job reads it.
  const inside = (l: number[]) => (l[3] ?? l[0]) >= start && (l[4] ?? l[1]) <= end
  const arcs = (factor: 'CTCF' | 'RNAPII', above: boolean) =>
    (preview.data.loops[factor] ?? []).map((l, i) => {
      const [a, b, n] = l
      const x1 = x(a)
      const x2 = x(b)
      const r = Math.max((x2 - x1) / 2, 1)
      return (
        <path
          key={`${factor}${i}`}
          className={`arc ${factor.toLowerCase()}${chosen(factor) && inside(l) ? '' : ' faint'}`}
          d={`M ${x1} ${AXIS} A ${r} ${Math.min(r, AXIS - 6)} 0 0 ${above ? 1 : 0} ${x2} ${AXIS}`}
        >
          <title>{`${factor} loop, ${a.toLocaleString('en-US')} to ${b.toLocaleString('en-US')}, ${n} PETs`}</title>
        </path>
      )
    })
  const count = (factor: string) => (preview.data.loops[factor] ?? []).filter(inside).length
  const gene = region.gene ? preview.data.genes.find((g) => g.name === region.gene) : undefined
  const text = (a: number, b: number) => `${region.chrom}:${a.toLocaleString('en-US')}-${b.toLocaleString('en-US')}`
  const at = (clientX: number) => {
    const box = svg.current!.getBoundingClientRect()
    return view[0] + (box.width ? (clientX - box.left) / box.width : 0) * width
  }
  /**
   * The region a drag gives, kept inside the view, at least the shortest span long and at most
   * the longest. An edge of a region that is too long already moves inwards alone.
   */
  const dragged = (d: Drag, p: number): [number, number] => {
    const [a, b] = d.from
    if (d.kind === 'start') {
      const least = Math.max(view[0], Math.min(a, b - maxSpan))
      return [Math.max(round(Math.min(Math.max(a + p - d.at, least), b - MIN_SPAN)), least), b]
    }
    if (d.kind === 'end') {
      const most = Math.min(view[1], Math.max(b, a + maxSpan))
      return [a, Math.min(round(Math.max(Math.min(b + p - d.at, most), a + MIN_SPAN)), most)]
    }
    const shift = Math.min(Math.max(p - d.at, view[0] - a), view[1] - b)
    return [round(a + shift), round(a + shift) + (b - a)]
  }
  const grab = (kind: Drag['kind']) => (e: React.PointerEvent<SVGRectElement>) => {
    drag.current = { kind, at: at(e.clientX), from: [region.start, region.end] }
    latest.current = null
    try {
      e.currentTarget.setPointerCapture?.(e.pointerId)
    } catch {
      /* not an active pointer */
    }
  }
  const move = (e: React.PointerEvent<SVGRectElement>) => {
    if (!drag.current) return
    latest.current = dragged(drag.current, at(e.clientX))
    setMoved(latest.current)
  }
  const release = () => {
    const next = latest.current
    drag.current = null
    latest.current = null
    setMoved(null)
    if (next && onRegion && (next[0] !== region.start || next[1] !== region.end)) onRegion(text(next[0], next[1]))
  }
  /** The arrow keys move an edge by a hundredth of the view. */
  const nudge = (kind: 'start' | 'end') => (e: React.KeyboardEvent) => {
    if (e.key !== 'ArrowLeft' && e.key !== 'ArrowRight') return
    e.preventDefault()
    const by = (e.key === 'ArrowLeft' ? -1 : 1) * Math.max(STEP, width / 100)
    const next = dragged({ kind, at: 0, from: [region.start, region.end] }, by)
    if (next[0] !== region.start || next[1] !== region.end) onRegion?.(text(next[0], next[1]))
  }
  const handlers = { onPointerMove: move, onPointerUp: release, onPointerCancel: release }
  const tooLong = end - start > maxSpan
  const longest = `${maxSpan / 1e6} Mb`
  return (
    <div className={tooLong ? 'loop-preview too-long' : 'loop-preview'}>
      <svg ref={svg} viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`Loops in ${region.label || text(region.start, region.end)}`}>
        <rect className={onRegion ? 'region-box movable' : 'region-box'} x={x(start)} y={0} width={Math.max(x(end) - x(start), 2)} height={H} onPointerDown={onRegion ? grab('move') : undefined} {...(onRegion ? handlers : {})} />
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
        {onRegion &&
          (['start', 'end'] as const).map((kind) => {
            const p = kind === 'start' ? start : end
            return (
              <g key={kind}>
                <line className="region-edge" x1={x(p)} x2={x(p)} y1={0} y2={H} />
                <rect
                  className="region-handle"
                  x={x(p) - 7}
                  y={AXIS - 16}
                  width={14}
                  height={32}
                  rx={4}
                  role="slider"
                  tabIndex={0}
                  aria-label={kind === 'start' ? 'Start of the region' : 'End of the region'}
                  aria-valuemin={view[0]}
                  aria-valuemax={view[1]}
                  aria-valuenow={p}
                  aria-valuetext={`${region.chrom}:${p.toLocaleString('en-US')}`}
                  onPointerDown={grab(kind)}
                  onKeyDown={nudge(kind)}
                  {...handlers}
                />
              </g>
            )
          })}
      </svg>
      <p className="preview-key">
        <span className="key-item">
          <span className="key-line ctcf" />
          {plural(count('CTCF'), 'CTCF loop')}
        </span>
        <span className="key-item">
          <span className="key-line rnapii" />
          {plural(count('RNAPII'), 'RNAPII loop')}
        </span>
        <span className="preview-span">{`${text(start, end)}, ${mb(end - start)}${end - start === maxSpan ? ', the limit' : ''}`}</span>
      </p>
      {tooLong ? (
        <p className="preview-hint too-long" role="status">
          {`This region is ${mb(end - start)} long and a job takes ${longest} at most.${onRegion ? ' Drag an edge inwards to shorten it.' : ''}`}
          {onRegion && (
            <button type="button" className="link-button" onClick={() => onRegion(text(region.start, region.start + maxSpan))}>
              {`Keep the first ${longest}`}
            </button>
          )}
        </p>
      ) : (
        onRegion && <p className="preview-hint">{`Drag the box to move the region and its two edges to make it longer or shorter${Number.isFinite(maxSpan) ? `, up to ${longest}` : ''}. The loops outside the region are drawn faint.`}</p>
      )}
    </div>
  )
}
