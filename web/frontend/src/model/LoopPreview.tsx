import { asked, type Region } from '../api'
import { keepPreviousData, useQuery } from '@tanstack/react-query'
import { useEffect, useRef, useState } from 'react'
import type { Factors } from './form'
import { around } from './loopWindow'
import { FeatureTracks } from '../components/FeatureTracks'
import { labelRows } from '../components/packGenes'
import { scoped } from '../lib/css'
import sheet from './model.module.css'

const cn = scoped(sheet)

const W = 780
const H = 150
const AXIS = 75
/** The height of the band of variants under the loops, with two rows of labels. */
const BAND = 42
/** The shortest region a drag leaves and the step of a rounded position, in bp. */
const MIN_SPAN = 20_000
const STEP = 1_000

const plural = (n: number, word: string) => `${n.toLocaleString('en-US')} ${word}${n === 1 ? '' : 's'}`
const mb = (n: number) => `${(n / 1e6).toFixed(n >= 1e7 ? 1 : 2)} Mb`
const round = (p: number) => Math.round(p / STEP) * STEP

/** Returns true for a region that is drawn well in a view, inside it and between an eighth and seven tenths of it. */
const fits = (start: number, end: number, view: [number, number]) => {
  const share = (end - start) / (view[1] - view[0])
  return start >= view[0] && end <= view[1] && share >= 0.125 && share <= 0.7
}

type Drag = { kind: 'start' | 'end' | 'move'; at: number; from: [number, number] }

/**
 * The loops around a region as arcs over an axis, CTCF above and RNAPII below, the region as a box
 * and the genes as a track under them. With `onRegion` the edges of the box are sliders and the box can be
 * moved. The new coordinates are reported when the drag ends.
 *
 * `maxSpan` is the longest region of a job in bp. A drag does not make the region longer than
 * that. A region that is longer already is drawn in the colour of an error and a drag makes it
 * shorter.
 *
 * `variants` are structural variants of the region, drawn in a band under the loops with their type.
 */
export function LoopPreview({
  cell,
  region,
  factors,
  maxSpan = Infinity,
  onRegion,
  variants,
}: {
  cell: string
  region: Region
  factors: Factors
  maxSpan?: number
  onRegion?: (text: string) => void
  variants?: { type: string; pos: number; end: number }[]
}) {
  // The server draws three times the longest region at most.
  const cap = 3 * maxSpan
  // The view moves only when the region no longer fits it. The picture stays still while the region
  // is dragged inside it.
  const [view, setView] = useState<[number, number]>(() => around(region.start, region.end, cap))
  const [chrom, setChrom] = useState(region.chrom)
  useEffect(() => {
    if (region.chrom !== chrom || !fits(region.start, region.end, view)) {
      setView(around(region.start, region.end, cap))
      setChrom(region.chrom)
    }
  }, [region.chrom, region.start, region.end, cap]) // eslint-disable-line react-hooks/exhaustive-deps
  const span = `${region.chrom}:${view[0]}-${view[1]}`
  const preview = useQuery({ ...asked.preview(cell, span), placeholderData: keepPreviousData })
  const svg = useRef<SVGSVGElement>(null)
  const drag = useRef<Drag | null>(null)
  const [moved, setMoved] = useState<[number, number] | null>(null)
  const latest = useRef<[number, number] | null>(null)
  if (!preview.data) return <div className={cn('loop-preview')} aria-busy="true" />
  const height = H + (variants ? BAND : 0)
  const [start, end] = moved ?? [region.start, region.end]
  const width = view[1] - view[0]
  const x = (p: number) => ((p - view[0]) / width) * W
  const svRows = labelRows(
    (variants ?? []).map((v) => (x(v.pos) + Math.max(x(v.end), x(v.pos) + 4)) / 2),
    (variants ?? []).map((v) => v.type.length * 6.5),
    2,
  )
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
          className={cn(`arc ${factor.toLowerCase()}${chosen(factor) && inside(l) ? '' : ' faint'}`)}
          d={`M ${x1} ${AXIS} A ${r} ${Math.min(r, AXIS - 6)} 0 0 ${above ? 1 : 0} ${x2} ${AXIS}`}
        >
          <title>{`${factor} loop, ${a.toLocaleString('en-US')} to ${b.toLocaleString('en-US')}, ${n} PETs`}</title>
        </path>
      )
    })
  const count = (factor: string) => (preview.data.loops[factor] ?? []).filter(inside).length
  const text = (a: number, b: number) => `${region.chrom}:${a.toLocaleString('en-US')}-${b.toLocaleString('en-US')}`
  const at = (clientX: number) => {
    const box = svg.current!.getBoundingClientRect()
    return view[0] + (box.width ? (clientX - box.left) / box.width : 0) * width
  }
  /**
   * Returns the region a drag gives, kept inside the view, at least the shortest span long and at
   * most the longest. An edge of a region that is too long already moves inwards alone.
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
  /** Moves an edge by a hundredth of the view at an arrow key. */
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
    <div className={cn(tooLong ? 'loop-preview too-long' : 'loop-preview')}>
      <svg ref={svg} viewBox={`0 0 ${W} ${height}`} role="img" aria-label={`Loops in ${region.label || text(region.start, region.end)}`}>
        <rect className={cn(onRegion ? 'region-box movable' : 'region-box')} x={x(start)} y={0} width={Math.max(x(end) - x(start), 2)} height={height} onPointerDown={onRegion ? grab('move') : undefined} {...(onRegion ? handlers : {})} />
        <text className={cn('side-label')} x={6} y={14}>
          CTCF loops
        </text>
        <text className={cn('side-label')} x={6} y={H - 6}>
          RNAPII loops
        </text>
        {arcs('CTCF', true)}
        {arcs('RNAPII', false)}
        <line className={cn('axis')} x1={0} x2={W} y1={AXIS} y2={AXIS} />
        {variants?.map((v, i) => {
          const x1 = x(v.pos)
          const w = Math.max(x(v.end) - x1, 4)
          return (
            <g key={i} className={cn(`sv-mark sv-${v.type.toLowerCase()}`)}>
              <rect x={x1} y={H + 2} width={w} height={10} rx={1} />
              {/* Two variants close to each other have their types on two rows. A third has none. */}
              {svRows[i] !== null && (
                <text x={x1 + w / 2} y={H + 24 + svRows[i]! * 11} textAnchor="middle">
                  {v.type}
                </text>
              )}
            </g>
          )
        })}
        {onRegion &&
          (['start', 'end'] as const).map((kind) => {
            const p = kind === 'start' ? start : end
            return (
              <g key={kind}>
                <line className={cn('region-edge')} x1={x(p)} x2={x(p)} y1={0} y2={height} />
                <rect
                  className={cn('region-handle')}
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
      <FeatureTracks chrom={region.chrom} view={view} genes={preview.data.genes} first={region.gene} />
      <p className={cn('preview-key')}>
        <span className={cn('key-item')}>
          <span className={cn('key-line ctcf')} />
          {plural(count('CTCF'), 'CTCF loop')}
        </span>
        <span className={cn('key-item')}>
          <span className={cn('key-line rnapii')} />
          {plural(count('RNAPII'), 'RNAPII loop')}
        </span>
        <span className={cn('preview-span')}>{`${text(start, end)}, ${mb(end - start)}${end - start === maxSpan ? ', the limit' : ''}`}</span>
      </p>
      {tooLong ? (
        <p className={cn('preview-hint too-long')} role="status">
          {`This region is ${mb(end - start)} long and a job takes ${longest} at most.${onRegion ? ' Drag an edge inwards to shorten it.' : ''}`}
          {onRegion && (
            <button type="button" className="link-button" onClick={() => onRegion(text(region.start, region.start + maxSpan))}>
              {`Keep the first ${longest}`}
            </button>
          )}
        </p>
      ) : (
        onRegion && <p className={cn('preview-hint')}>{`Drag the box to move the region and its two edges to make it longer or shorter${Number.isFinite(maxSpan) ? `, up to ${longest}` : ''}. The loops outside the region are drawn faint.`}</p>
      )}
    </div>
  )
}
