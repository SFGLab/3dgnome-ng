import { useQuery } from '@tanstack/react-query'
import { RefreshCw, ZoomIn, ZoomOut } from 'lucide-react'
import { useEffect, useRef, useState } from 'react'
import type { Contacts, JobApi, Summary } from './api'
import { bp } from './format'
import { colour, percentile, transform, type Show } from './heatmap'
import { parseRegion } from './StructureTab'

const RESOLUTIONS = [5_000, 10_000, 25_000, 50_000, 100_000, 250_000]
const MAX_BINS = 800
const SETTLE_MS = 250

/** A window cut to the job's region and at least ten bins wide, or the whole region when it
 * does not overlap it. */
function clamp(l: Locus, job: Locus, res: number): Locus {
  if (l.chrom !== job.chrom || l.end <= job.start || l.start >= job.end) return job
  const width = Math.min(Math.max(l.end - l.start, res * 10), job.end - job.start)
  const start = Math.min(Math.max(l.start, job.start), job.end - width)
  return { chrom: job.chrom, start: Math.round(start), end: Math.round(start + width) }
}

/** The finest resolution not finer than `res` that keeps a window under MAX_BINS bins. */
const fitting = (l: Locus, res: number) => RESOLUTIONS.find((r) => r >= res && (l.end - l.start) / r <= MAX_BINS) ?? RESOLUTIONS[RESOLUTIONS.length - 1]
const SHOWS: [Show, string][] = [
  ['observed', 'Observed'],
  ['expected', 'Expected'],
  ['oe', 'O/E'],
  ['pearson', 'Pearson'],
]

type Locus = { chrom: string; start: number; end: number }
const fmt = (l: Locus) => `${l.chrom}:${bp(l.start)}-${bp(l.end)}`
const parse = (t: string): Locus | null => {
  const m = t.replace(/,/g, '').match(/^(\w+):(\d+)-(\d+)$/)
  return m && Number(m[3]) > Number(m[2]) ? { chrom: m[1], start: Number(m[2]), end: Number(m[3]) } : null
}

function draw(canvas: HTMLCanvasElement, upper: number[][], lower: number[][], range: [number, number], log: boolean) {
  const ctx = canvas.getContext('2d')
  const n = upper.length
  if (!ctx || !n) return
  canvas.width = n
  canvas.height = n
  // Each half on its own scale: the lower half's values are brought to the upper half's range.
  const ratio = percentile(upper, 0.99) / (percentile(lower, 0.99) || 1)
  const img = ctx.createImageData(n, n)
  for (let i = 0; i < n; i++)
    for (let j = 0; j < n; j++) {
      const v = j >= i ? upper[i][j] : lower[i][j] * ratio
      const [r, g, b] = colour(v, range[0], range[1], log)
      const k = (i * n + j) * 4
      img.data.set([r, g, b, 255], k)
    }
  ctx.putImageData(img, 0, 0)
}

/** The Contact map tab: the job's Hi-C and its model on one grid, upper and lower half. */
export function ContactMapTab({ api, summary }: { api: JobApi; summary: Summary }) {
  const job = parseRegion(summary.region)!
  const [locus, setLocus] = useState<Locus>(job)
  const [text, setText] = useState(fmt(job))
  const [res, setRes] = useState(25_000)
  const [show, setShow] = useState<Show>('observed')
  const [norm, setNorm] = useState<'ice' | 'raw'>('ice')
  const [upperKind, setUpper] = useState('hic')
  const [lowerKind, setLower] = useState('model')
  const [log, setLog] = useState(true)
  const [range, setRange] = useState<[number, number] | null>(null)
  const [hover, setHover] = useState<string | null>(null)
  const [note, setNote] = useState('')
  const canvas = useRef<HTMLCanvasElement>(null)
  // The map follows the window once a burst of wheel steps settles, not at every step.
  const [settled, setSettled] = useState<Locus>(job)
  useEffect(() => {
    const t = setTimeout(() => setSettled(locus), SETTLE_MS)
    return () => clearTimeout(t)
  }, [locus])
  const region = `${settled.chrom}:${settled.start}-${settled.end}`
  const maps = useQuery<Contacts, Error>({
    queryKey: ['contacts', api.base, region, res, upperKind, lowerKind, norm],
    queryFn: () => api.contacts({ region, res, upper: upperKind, lower: lowerKind, norm }),
    retry: false,
  })
  const annotations = useQuery({ queryKey: ['annotations', api.base], queryFn: api.annotations })
  const upper = maps.data ? transform(maps.data.upper, show) : []
  const lower = maps.data ? transform(maps.data.lower, show) : []
  const top = maps.data ? percentile(upper, 0.99) : 1
  const shownRange: [number, number] = range ?? [0, Math.round(top * 100) / 100]
  useEffect(() => {
    if (canvas.current && maps.data) draw(canvas.current, upper, lower, shownRange, log)
  })
  const moveTo = (want: Locus) => {
    const next = clamp(want, job, res)
    const r = fitting(next, res)
    if (r !== res) setRes(r)
    setLocus(next)
    setText(fmt(next))
    return next
  }
  const zoom = (f: number, centre = (locus.start + locus.end) / 2, frac?: number) => {
    const width = (locus.end - locus.start) * f
    // Under the pointer the genomic position stays where it was.
    const anchor = frac === undefined ? centre : locus.start + frac * (locus.end - locus.start)
    const start = frac === undefined ? centre - width / 2 : anchor - frac * width
    moveTo({ chrom: locus.chrom, start, end: start + width })
  }
  const go = () => {
    const want = parse(text)
    if (!want) return
    const next = moveTo(want)
    setNote(next.start !== want.start || next.end !== want.end || next.chrom !== want.chrom ? `The job covers ${fmt(job)}. The map shows only that region.` : '')
  }
  // A native listener, since React's wheel handler is passive and cannot stop the page scrolling.
  const zoomRef = useRef(zoom)
  zoomRef.current = zoom
  useEffect(() => {
    const c = canvas.current
    if (!c) return
    const onWheel = (e: WheelEvent) => {
      e.preventDefault()
      const box = c.getBoundingClientRect()
      const frac = box.width ? (e.clientX - box.left) / box.width : 0.5
      zoomRef.current(e.deltaY > 0 ? 1.25 : 0.8, undefined, frac)
    }
    c.addEventListener('wheel', onWheel, { passive: false })
    return () => c.removeEventListener('wheel', onWheel)
  }, [])
  const at = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const box = e.currentTarget.getBoundingClientRect()
    const n = maps.data?.bins.length ?? 0
    return [Math.floor(((e.clientY - box.top) / box.height) * n), Math.floor(((e.clientX - box.left) / box.width) * n)]
  }
  const loops = (annotations.data?.loops ?? []).filter((l) => l.a[0] >= settled.start && l.b[1] <= settled.end)
  const x = (p: number) => ((p - settled.start) / (settled.end - settled.start)) * 600
  return (
    <div className="contacts-tab">
      <div className="contacts-main">
        <div className="map-toolbar">
          <label className="field">
            <span>Locus</span>
            <span className="locus-input">
              <input value={text} onChange={(e) => setText(e.target.value)} onKeyDown={(e) => e.key === 'Enter' && go()} />
              <button type="button" onClick={go}>
                Go
              </button>
            </span>
          </label>
          <div className="field">
            <span>Show</span>
            <div className="segmented" role="group" aria-label="Show">
              {SHOWS.map(([id, label]) => (
                <button key={id} type="button" aria-pressed={show === id} onClick={() => setShow(id)}>
                  {label}
                </button>
              ))}
            </div>
          </div>
          <label className="field">
            <span>Normalization</span>
            <select value={norm} onChange={(e) => setNorm(e.target.value as 'ice' | 'raw')}>
              <option value="ice">Balanced, ICE</option>
              <option value="raw">Raw counts</option>
            </select>
          </label>
          <label className="field">
            <span>Resolution</span>
            <select aria-label="Resolution" value={res} onChange={(e) => setRes(fitting(locus, Number(e.target.value)))}>
              {RESOLUTIONS.map((r) => (
                <option key={r} value={r}>{`${r / 1000} kb`}</option>
              ))}
            </select>
          </label>
          <div className="zoom-buttons">
            <button type="button" className="icon-tool" aria-label="Zoom out" onClick={() => zoom(2)}>
              <ZoomOut size={15} aria-hidden />
            </button>
            <button type="button" className="icon-tool" aria-label="Zoom in" onClick={() => zoom(0.5)}>
              <ZoomIn size={15} aria-hidden />
            </button>
            <button
              type="button"
              className="icon-tool"
              aria-label="Back to the job region"
              onClick={() => {
                moveTo(job)
                setNote('')
              }}
            >
              <RefreshCw size={15} aria-hidden />
            </button>
          </div>
        </div>
        <svg className="loop-track" viewBox="0 0 600 34" preserveAspectRatio="none" aria-label="Loops">
          <text x={4} y={10}>
            loops
          </text>
          {loops.map((l, i) => {
            const a = x((l.a[0] + l.a[1]) / 2)
            const b = x((l.b[0] + l.b[1]) / 2)
            return <path key={i} className={`arc ${l.factor.toLowerCase()}`} d={`M ${a} 34 A ${(b - a) / 2} 30 0 0 1 ${b} 34`} />
          })}
        </svg>
        {maps.data && (
          <p className="map-labels">
            <span>{maps.data.lower_label}</span>
            <span>{maps.data.upper_label}</span>
          </p>
        )}
        {note && <p className="map-note">{note}</p>}
        {maps.isError && <p className="field-error">{maps.error.message}</p>}
        <canvas
          ref={canvas}
          className="contact-canvas"
          onMouseMove={(e) => {
            if (!maps.data) return
            const [i, j] = at(e)
            const d = maps.data
            if (i < 0 || j < 0 || i >= d.bins.length || j >= d.bins.length) return
            const [r, c] = [Math.min(i, j), Math.max(i, j)]
            setHover(
              `x ${settled.chrom}:${bp(d.bins[j])}   y ${settled.chrom}:${bp(d.bins[i])}   ${d.upper_label.split(' ')[0]} ${d.upper[r][c].toFixed(1)}   model ${d.lower[c][r].toFixed(1)}   apart ${(Math.abs(d.bins[j] - d.bins[i]) / 1e6).toFixed(2)} Mb`,
            )
          }}
          onMouseLeave={() => setHover(null)}
          onDoubleClick={(e) => {
            if (!maps.data) return
            const [, j] = at(e)
            zoom(1, maps.data.bins[Math.max(0, Math.min(j, maps.data.bins.length - 1))])
          }}
        />
        <p className="hover-bar mono">{hover ?? ' '}</p>
      </div>
      <div className="side-cards">
        <section className="side-card" aria-label="Colour range">
          <p className="card-label">Colour range</p>
          <div className="range-inputs">
            <label>
              min <input type="number" value={shownRange[0]} onChange={(e) => setRange([Number(e.target.value), shownRange[1]])} />
            </label>
            <label>
              max <input type="number" value={shownRange[1]} onChange={(e) => setRange([shownRange[0], Number(e.target.value)])} />
            </label>
          </div>
          <label className="check">
            <input type="checkbox" checked={log} onChange={(e) => setLog(e.target.checked)} /> Log scale, separate for each half
          </label>
        </section>
        <label className="field">
          <span>Upper half</span>
          <select value={upperKind} onChange={(e) => setUpper(e.target.value)}>
            <option value="hic">{`Hi-C, ${summary.request.cell_type}`}</option>
            <option value="model">This job, sum of structures</option>
          </select>
        </label>
        <label className="field">
          <span>Lower half</span>
          <select value={lowerKind} onChange={(e) => setLower(e.target.value)}>
            <option value="model">{`This job, sum of ${summary.structures} structures`}</option>
            <option value="hic">{`Hi-C, ${summary.request.cell_type}`}</option>
          </select>
        </label>
        {maps.data && (
          <section className="side-card" aria-label="Agreement with Hi-C">
            <p className="card-label">Agreement with Hi-C</p>
            <p className="big">
              <span className="big-value">{maps.data.scc === null ? 'none' : maps.data.scc.toFixed(2)}</span> <span className="unit">stratum adjusted correlation</span>
            </p>
            <p className="card-text">{`Computed on ${res / 1000} kb bins up to 2 Mb apart in this view.`}</p>
          </section>
        )}
        <p className="card-text">Scroll to zoom and double click to centre. Grey bins contain no bead.</p>
      </div>
    </div>
  )
}
