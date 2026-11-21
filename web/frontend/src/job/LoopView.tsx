import { RefreshCw, X } from 'lucide-react'
import { useEffect, useMemo, useRef, useState } from 'react'
import type { Loop } from '../api'
import './loop-view.css'
import { wheelFactor } from './wheel'
import { zoomed, describeLoop, ticks } from './loops'

const W = 1000
const H = 70

/** A position in megabases with no more digits than the step between ticks needs. */
const mb = (v: number, step: number) => `${(v / 1e6).toFixed(step >= 1e6 ? 0 : step >= 1e5 ? 1 : step >= 1e4 ? 2 : 3)}`

type Props = {
  chrom: string
  /** The stretch of the genome the view spans. */
  region: [number, number]
  loops: Loop[]
  /** The loop marked as selected, an index into `loops`. */
  selected?: number | null
  onSelect?: (index: number | null) => void
  /** A part of the region another view shows, drawn as a box. */
  window?: [number, number]
  /** A drag along the view asks for that part, and so does the wheel, which scales the part in
   * use about the pointer. */
  onWindow?: (start: number, end: number) => void
  /** The stretch a zoom out may reach, the whole region where `region` is a part of it. */
  bounds?: [number, number]
  /** A position to mark, the bead under the pointer in the model. */
  cursor?: number | null
  /** Shown as a button while a part is in use, to go back to the whole region. */
  onReset?: () => void
  /** The line under the view while no loop is under the pointer and none is selected. */
  hint?: string
}

/**
 * The loops of a region in two dimensions: the genome as a line and every loop as an arc between
 * its anchors, taller for a longer loop.
 * An arc can be selected and a drag selects a part of the region. The view only reports both.
 */
export function LoopView({ chrom, region, loops, selected = null, onSelect, window, onWindow, bounds, cursor = null, onReset, hint = '' }: Props) {
  const [start, end] = region
  const width = Math.max(end - start, 1)
  const x = (p: number) => ((p - start) / width) * W
  const svg = useRef<SVGSVGElement>(null)
  const [over, setOver] = useState<number | null>(null)
  // The dragged part is drawn from the state and read from the ref on release, since a release
  // can come before the redraw that the last move asked for.
  const [brush, setBrush] = useState<[number, number] | null>(null)
  const dragged = useRef<[number, number] | null>(null)
  const drag = useRef<number | null>(null)
  // The loop the pointer went down on. The view takes the pointer for a drag, so a click never
  // reaches the arc and the release decides between a click on it and a drag.
  const pressed = useRef<number | null>(null)
  // The loops that reach into the region, longest first so a short arc is drawn over a long one
  // and stays easy to point at.
  const shown = useMemo(
    () =>
      loops
        .map((l, i) => ({ l, i, a: (l.a[0] + l.a[1]) / 2, b: (l.b[0] + l.b[1]) / 2 }))
        .filter((r) => r.b >= start && r.a <= end)
        .sort((p, q) => q.b - q.a - (p.b - p.a)),
    [loops, start, end],
  )
  const longest = Math.max(1, ...shown.map((r) => r.b - r.a))
  const arc = (a: number, b: number) => {
    const [xa, xb] = [x(a), x(b)]
    // The longest loop in view fills the height and the others grow with the root of their
    // span, so short loops are still seen beside long ones.
    const h = Math.max(6, Math.sqrt((b - a) / longest) * (H - 6))
    return `M ${xa} ${H} A ${(xb - xa) / 2} ${h} 0 0 1 ${xb} ${H}`
  }
  const frac = (clientX: number) => {
    const box = svg.current!.getBoundingClientRect()
    return box.width ? Math.min(1, Math.max(0, (clientX - box.left) / box.width)) : 0
  }
  // The wheel scales the part in use, the box where there is one and the view otherwise. A
  // native listener, since React's wheel handler is passive and cannot stop the page scrolling.
  const wheel = useRef<(e: WheelEvent) => void>(() => {})
  wheel.current = (e) => {
    if (!onWindow) return
    e.preventDefault()
    const next = zoomed(window ?? [start, end], bounds ?? [start, end], start + frac(e.clientX) * width, wheelFactor(e))
    onWindow(next[0], next[1])
  }
  useEffect(() => {
    const el = svg.current
    if (!el) return
    const on = (e: WheelEvent) => wheel.current(e)
    el.addEventListener('wheel', on, { passive: false })
    return () => el.removeEventListener('wheel', on)
  }, [])
  const pick = (i: number) => onSelect?.(i === selected ? null : i)
  const marks = ticks(start, end)
  const step = marks.length > 1 ? marks[1] - marks[0] : width
  const named = over ?? selected
  return (
    <div className="loop-view">
      {/* The line that says what the view does, or names the loop under the pointer or the selected
          one, with the two buttons at its end. */}
      <div className="loop-view-head">
        <p className="loop-view-line">
          {named !== null && loops[named] ? <span className="mono">{`${named === selected ? 'Selected: ' : ''}${describeLoop(loops[named])}`}</span> : hint}
        </p>
        <div className="loop-view-tools">
          {selected !== null && onSelect && (
            <button type="button" className="icon-tool small" aria-label="Clear the selected loop" title="Clear the selected loop" onClick={() => onSelect(null)}>
              <X size={13} aria-hidden />
            </button>
          )}
          {onReset && (
            <button type="button" className="icon-tool small" aria-label="Whole region" title="Whole region" onClick={onReset}>
              <RefreshCw size={13} aria-hidden />
            </button>
          )}
        </div>
      </div>
      <div className="loop-view-box">
      <svg
        ref={svg}
        className={onWindow ? 'loop-view-arcs brushable' : 'loop-view-arcs'}
        viewBox={`0 0 ${W} ${H}`}
        preserveAspectRatio="none"
        role="group"
        aria-label="Loops along the region"
        onPointerDown={(e) => {
          const hit = (e.target as Element).getAttribute?.('data-loop')
          pressed.current = hit === null || hit === undefined ? null : Number(hit)
          if (!onWindow) return
          drag.current = frac(e.clientX)
          dragged.current = null
          // The drag goes on when the pointer leaves the view. A pointer that cannot be captured
          // still drags inside it.
          try {
            e.currentTarget.setPointerCapture?.(e.pointerId)
          } catch {
            /* not an active pointer */
          }
        }}
        onPointerMove={(e) => {
          if (drag.current === null) return
          const f = frac(e.clientX)
          if (Math.abs(f - drag.current) <= 0.01) return
          dragged.current = [Math.min(drag.current, f), Math.max(drag.current, f)]
          setBrush(dragged.current)
        }}
        onPointerUp={() => {
          const part = dragged.current
          if (part && onWindow) onWindow(Math.round(start + part[0] * width), Math.round(start + part[1] * width))
          else if (pressed.current !== null) pick(pressed.current)
          pressed.current = null
          drag.current = null
          dragged.current = null
          setBrush(null)
        }}
      >
        {window && <rect className="loop-window" x={x(window[0])} y={0} width={Math.max(x(window[1]) - x(window[0]), 2)} height={H} />}
        {shown.map(({ l, i, a, b }) => (
          <path key={i} className={`loop-arc ${l.factor === 'RNAPII' ? 'rnapii' : 'ctcf'}${i === selected ? ' selected' : selected !== null ? ' dimmed' : ''}${i === over ? ' over' : ''}`} d={arc(a, b)} />
        ))}
        {/* A wide unseen stroke over each arc takes the pointer. A thin arc is hard to hit. */}
        {onSelect &&
          shown.map(({ l, i, a, b }) => (
            <path
              key={i}
              className="loop-hit"
              data-loop={i}
              d={arc(a, b)}
              role="button"
              tabIndex={0}
              aria-label={describeLoop(l)}
              aria-pressed={i === selected}
              onPointerEnter={() => setOver(i)}
              onPointerLeave={() => setOver(null)}
              onFocus={() => setOver(i)}
              onBlur={() => setOver(null)}
              onKeyDown={(e) => e.key === 'Enter' && pick(i)}
            />
          ))}
        {brush && <rect className="loop-brush" x={brush[0] * W} y={0} width={(brush[1] - brush[0]) * W} height={H} />}
        {cursor !== null && cursor >= start && cursor <= end && <line className="loop-cursor" x1={x(cursor)} x2={x(cursor)} y1={0} y2={H} />}
      </svg>
        {/* The two anchors of the selected loop, lettered as in the model. */}
        {selected !== null &&
          loops[selected] &&
          (['a', 'b'] as const).map((k) => {
            const at = (loops[selected][k][0] + loops[selected][k][1]) / 2
            return at >= start && at <= end ? (
              <span key={k} className="marker loop-anchor" style={{ left: `${((at - start) / width) * 100}%` }}>
                {k.toUpperCase()}
              </span>
            ) : null
          })}
      </div>
      <div className="loop-view-axis" aria-hidden>
        {/* A tick at either end would run into the edge or the unit. */}
        {marks
          .filter((v) => (v - start) / width > 0.02 && (v - start) / width < 0.9)
          .map((v) => (
            <span key={v} style={{ left: `${((v - start) / width) * 100}%` }}>
              {mb(v, step)}
            </span>
          ))}
        <span className="unit">{`${chrom}, Mb`}</span>
      </div>
    </div>
  )
}
