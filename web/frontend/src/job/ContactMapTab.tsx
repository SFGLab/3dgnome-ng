import { keepPreviousData, useQuery } from '@tanstack/react-query'
import { RefreshCw, ZoomIn, ZoomOut } from 'lucide-react'
import { useEffect, useRef, useState } from 'react'
import type { Contacts, JobApi, Summary } from './api'
import { bp } from './format'
import { colour, diverging, emptyRows, halfRange, percentile, transform, type Show } from './heatmap'
import { parseRegion } from './StructureTab'

const RESOLUTIONS = [5_000, 10_000, 25_000, 50_000, 100_000, 250_000]
const MAX_BINS = 800
const NO_DATA: [number, number, number] = [226, 225, 221]
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
  ['oe', 'O/E'],
  ['pearson', 'Pearson'],
]

type Locus = { chrom: string; start: number; end: number }
const fmt = (l: Locus) => `${l.chrom}:${bp(l.start)}-${bp(l.end)}`
const parse = (t: string): Locus | null => {
  const m = t.replace(/,/g, '').match(/^(\w+):(\d+)-(\d+)$/)
  return m && Number(m[3]) > Number(m[2]) ? { chrom: m[1], start: Number(m[2]), end: Number(m[3]) } : null
}

type Half = { m: number[][]; range: [number, number]; loops: boolean }

/** Each half on its own colour range. A loop half has no empty bins to grey: a row without a loop
 * is a row of zeros, drawn white. */
function draw(canvas: HTMLCanvasElement, up: Half, low: Half, log: boolean, show: Show) {
  const [upper, lower] = [up.m, low.m]
  const ctx = canvas.getContext('2d')
  const n = upper.length
  if (!ctx || !n) return
  canvas.width = n
  canvas.height = n
  // Each half on its own scale: the lower half's values are brought to the upper half's range.
  const span = Math.max(Math.abs(up.range[0]), Math.abs(up.range[1]))
  const none = upper.map(() => false)
  const [emptyUp, emptyLow] = [up.loops ? none : emptyRows(upper), low.loops ? none : emptyRows(lower)]
  const img = ctx.createImageData(n, n)
  for (let i = 0; i < n; i++)
    for (let j = 0; j < n; j++) {
      const v = j >= i ? upper[i][j] : lower[i][j]
      const range = j >= i ? up.range : low.range
      // The maps hold no pair within one bin. The main diagonal is drawn as no data.
      const empty = j >= i ? emptyUp[i] || emptyUp[j] : emptyLow[i] || emptyLow[j]
      const [r, g, b] =
        i === j || empty ? NO_DATA : show === 'oe' ? diverging(v > 0 ? Math.log2(v) : -span, span) : show === 'pearson' ? diverging(v, span) : colour(v, range[0], range[1], log)
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
  const halves: [string, string][] = [
    ['hic', `Hi-C, ${summary.request.cell_type}`],
    ['model', 'This job, 1/(d+1)³ between bins'],
    ['model_count', 'This job, contacts within the median bond'],
    ...Object.keys(summary.loops).map((f): [string, string] => [`loops_${f.toLowerCase()}`, `${f} loops`]),
  ]
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
    // The map on screen stays, dimmed, while the next view is read.
    placeholderData: keepPreviousData,
  })
  const annotations = useQuery({ queryKey: ['annotations', api.base], queryFn: api.annotations })
  const upper = maps.data ? transform(maps.data.upper, show) : []
  const lower = maps.data ? transform(maps.data.lower, show) : []
  const round = (v: number) => Math.round(v * 100) / 100
  // Each half gets its own range; a range typed in applies to the upper half and the lower half
  // keeps the same share of its own spread.
  const oeSpan =
    maps.data && show === 'oe'
      ? Math.max(
          0.5,
          round(
            percentile(
              upper.map((r) => r.map((v) => (v > 0 ? Math.abs(Math.log2(v)) : 0))),
              0.95,
            ),
          ),
        )
      : 1
  const auto = (m: number[][]): [number, number] => (show === 'oe' ? [-oeSpan, oeSpan] : show === 'pearson' ? [-1, 1] : halfRange(m, log))
  const autoUp = maps.data ? auto(upper) : ([0, 1] as [number, number])
  const autoLow = maps.data ? auto(lower) : ([0, 1] as [number, number])
  const shownRange: [number, number] = range ?? [round(autoUp[0]), round(autoUp[1])]
  const scale = show === 'observed' && autoUp[1] > 0 ? autoLow[1] / autoUp[1] : 1
  const lowRange: [number, number] = range ? [range[0] * scale, range[1] * scale] : autoLow
  useEffect(() => {
    if (canvas.current && maps.data)
      draw(canvas.current, { m: upper, range: shownRange, loops: upperKind.startsWith('loops_') }, { m: lower, range: lowRange, loops: lowerKind.startsWith('loops_') }, log, show)
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
              {RESOLUTIONS.map((r) => {
                const tooFine = (locus.end - locus.start) / r > MAX_BINS
                return (
                  <option key={r} value={r} disabled={tooFine}>
                    {`${r / 1000} kb${tooFine ? ', zoom in to use' : ''}`}
                  </option>
                )
              })}
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
        <div className={`contact-frame${maps.data ? '' : ' empty'}${maps.isFetching ? ' loading' : ''}`}>
          {maps.isFetching && (
            <div className="map-loading" role="status">
              <div className="map-loading-card">
                <span className="map-spinner" aria-hidden />
                <strong>{`Reading the map at ${res / 1000} kb.`}</strong>
              </div>
            </div>
          )}
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
        </div>
        {maps.data && (
          <p className={`hover-bar mono${hover ? '' : ' idle'}`}>{hover ?? 'Move over the map to read a pair of positions. Double click to zoom in.'}</p>
        )}
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
          <select aria-label="Upper half" value={upperKind} onChange={(e) => setUpper(e.target.value)}>
            {halves.map(([id, label]) => (
              <option key={id} value={id}>
                {label}
              </option>
            ))}
          </select>
        </label>
        <label className="field">
          <span>Lower half</span>
          <select aria-label="Lower half" value={lowerKind} onChange={(e) => setLower(e.target.value)}>
            {[...halves.slice(1, 3), halves[0], ...halves.slice(3)].map(([id, label]) => (
              <option key={id} value={id}>
                {label}
              </option>
            ))}
          </select>
        </label>
        {maps.data && (
          <section className="side-card" aria-label="Agreement of the two halves">
            <p className="card-label">Agreement of the two halves</p>
            <p className="big">
              <span className="big-value">{typeof maps.data.rank === 'number' ? maps.data.rank.toFixed(2) : 'none'}</span> <span className="unit">rank agreement</span>
            </p>
            <p className="card-text">{`Rank correlation at ${res / 1000} kb of ${maps.data.rank_about ?? 'the two maps'}. It compares the pattern at each distance and not the level.`}</p>
            <p className="card-text">{`Stratum adjusted correlation ${maps.data.scc === null ? 'none' : maps.data.scc.toFixed(2)}, of ${maps.data.scc_about ?? 'the two maps'}. It also compares the level and a stretch without beads scores well where Hi-C is depleted.`}</p>
            {typeof maps.data.loop_enrichment === 'number' && (
              <p className="card-text">{`At the loops the other half is ${maps.data.loop_enrichment.toFixed(1)} times its expected contact at the same distance.`}</p>
            )}
          </section>
        )}
        <p className="card-text">Scroll to zoom and double click to centre. Grey bins contain no bead.</p>
        <p className="card-text">
          O/E divides each contact by the average contact at its distance, blue below and red above it. Pearson correlates the rows of O/E, red for rows that rise and fall
          together.
        </p>
      </div>
    </div>
  )
}
