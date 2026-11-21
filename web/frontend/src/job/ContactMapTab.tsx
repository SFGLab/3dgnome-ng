import { askedOf } from '../api'
import { keepPreviousData, useQuery } from '@tanstack/react-query'
import { Camera, RefreshCw, ZoomIn, ZoomOut } from 'lucide-react'
import { useEffect, useMemo, useRef, useState } from 'react'
import type { JobApi, Summary } from '../api'
import { bp } from '../lib/format'
import { LoopView } from './LoopView'
import { shownMaps, type Show } from './heatmap'
import { mapPicture } from './export/mapPicture'
import { saveDataUrl } from './export/savePng'
import { type Locus, parseRegion, readRegion, regionText } from '../lib/region'
import { wheelFactor } from './wheel'
import { draw } from './mapDraw'
import { RESOLUTIONS, MAX_BINS, clamp, binCount, fitting } from './mapWindow'
const SETTLE_MS = 250
const PREDICTED = 'hicdiffusion.bedpe'
// HiCDiffusion predicts 8,192 bp bins. This is the resolution of the tab nearest to them.
const PREDICTED_RES = 10_000
const SHOWS: [Show, string][] = [
  ['observed', 'Observed'],
  ['oe', 'O/E'],
  ['pearson', 'Pearson'],
]
/** The separation the rank correlation reached, as the end of a sentence. Nothing when unknown. */
const reach = (bp?: number) => (bp ? `, up to ${bp >= 1e6 ? `${bp / 1e6} Mb` : `${bp / 1000} kb`}` : '')

/** The Contact map tab: the job's Hi-C and its model on one grid, upper and lower half. */
export function ContactMapTab({ api, summary }: { api: JobApi; summary: Summary }) {
  const job = parseRegion(summary.region)!
  const [locus, setLocus] = useState<Locus>(job)
  const [text, setText] = useState(regionText(job))
  // The map opens at 25 kb, or at the finest resolution the region fits when it is too long for that.
  const [res, setRes] = useState(() => fitting(job, 25_000))
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
    // A job with a predicted background keeps the prediction as a file.
    ...(summary.files.some((f) => f.name === PREDICTED) ? [['hicdiffusion', 'HiCDiffusion, predicted from the sequence'] as [string, string]] : []),
    ...Object.keys(summary.loops).map((f): [string, string] => [`loops_${f.toLowerCase()}`, `${f} loops`]),
  ]
  // The map goes to the resolution of the prediction when the prediction is chosen for a half.
  const choose = (set: (kind: string) => void, kind: string) => {
    set(kind)
    if (kind === 'hicdiffusion') setRes(fitting(locus, PREDICTED_RES))
  }
  // The map follows the window once a burst of wheel steps settles, not at every step.
  const [settled, setSettled] = useState<Locus>(job)
  useEffect(() => {
    const t = setTimeout(() => setSettled(locus), SETTLE_MS)
    return () => clearTimeout(t)
  }, [locus])
  const region = `${settled.chrom}:${settled.start}-${settled.end}`
  const maps = useQuery({
    ...askedOf(api).contacts({ region, res, upper: upperKind, lower: lowerKind, norm }),
    retry: false,
    // The map on screen stays, dimmed, while the next view is read.
    placeholderData: keepPreviousData,
  })
  const annotations = useQuery(askedOf(api).annotations())
  // The halves as drawn and their ranges are worked out once for a map and its settings, not at
  // every move of the pointer over it.
  const view = useMemo(() => (maps.data ? shownMaps(maps.data.upper, maps.data.lower, show, log, range) : null), [maps.data, show, log, range])
  const shownRange: [number, number] = view?.upperRange ?? range ?? [0, 1]
  const [upperLoops, lowerLoops] = [upperKind.startsWith('loops_'), lowerKind.startsWith('loops_')]
  useEffect(() => {
    if (canvas.current && view) draw(canvas.current, { m: view.upper, range: view.upperRange, loops: upperLoops }, { m: view.lower, range: view.lowerRange, loops: lowerLoops }, log, show)
  }, [view, upperLoops, lowerLoops, log, show])
  // The window asked for as a part of the map on screen, its start and its width, while the map
  // on screen is still the wider one. It is the square the next map will fill.
  const bins = maps.data?.bins ?? []
  const shown = bins.length > 1 ? { start: bins[0], width: bins.length * (bins[1] - bins[0]) } : null
  const part = shown ? [(locus.start - shown.start) / shown.width, (locus.end - locus.start) / shown.width] : null
  const target = part && part[1] < 0.97 && part[0] > -0.02 && part[0] + part[1] < 1.02 ? [Math.max(0, part[0]), Math.min(part[1], 1 - Math.max(0, part[0]))] : null
  const moveTo = (want: Locus) => {
    const next = clamp(want, job, res)
    const r = fitting(next, res)
    if (r !== res) setRes(r)
    setLocus(next)
    setText(regionText(next))
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
    const want = readRegion(text)
    if (!want) return
    const next = moveTo(want)
    setNote(next.start !== want.start || next.end !== want.end || next.chrom !== want.chrom ? `The job covers ${regionText(job)}. The map shows only that region.` : '')
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
      zoomRef.current(wheelFactor(e), undefined, frac)
    }
    c.addEventListener('wheel', onWheel, { passive: false })
    return () => c.removeEventListener('wheel', onWheel)
  }, [])
  const at = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const box = e.currentTarget.getBoundingClientRect()
    const n = maps.data?.bins.length ?? 0
    return [Math.floor(((e.clientY - box.top) / box.height) * n), Math.floor(((e.clientX - box.left) / box.width) * n)]
  }
  return (
    <div className="contacts-tab">
      <div className="contacts-main">
        <div className="map-toolbar">
          {/* No label stands over a control, as on the other tabs. Each control says what it is by
              its own text and has its name for a screen reader and as a tooltip. */}
          <span className="locus-input">
            <input aria-label="Locus" title="Locus" value={text} onChange={(e) => setText(e.target.value)} onKeyDown={(e) => e.key === 'Enter' && go()} />
            <button type="button" onClick={go}>
              Go
            </button>
          </span>
          <div className="segmented" role="group" aria-label="Show">
            {SHOWS.map(([id, label]) => (
              <button key={id} type="button" aria-pressed={show === id} onClick={() => setShow(id)}>
                {label}
              </button>
            ))}
          </div>
          <select aria-label="Normalization" title="Normalization" value={norm} onChange={(e) => setNorm(e.target.value as 'ice' | 'raw')}>
            <option value="ice">Balanced, ICE</option>
            <option value="raw">Raw counts</option>
          </select>
          <select aria-label="Resolution" title="Resolution" value={res} onChange={(e) => setRes(fitting(locus, Number(e.target.value)))}>
            {RESOLUTIONS.map((r) => {
              const tooFine = binCount(locus, r) > MAX_BINS
              return (
                <option key={r} value={r} disabled={tooFine}>
                  {`${r / 1000} kb bins${tooFine ? ', zoom in to use' : ''}`}
                </option>
              )
            })}
          </select>
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
            <button
              type="button"
              className="icon-tool"
              aria-label="Save the map as a PNG"
              disabled={!maps.data}
              onClick={() => {
                const map = canvas.current
                if (!map || !maps.data) return
                const out = mapPicture(map, { chrom: settled.chrom, bins: maps.data.bins, res, upper: maps.data.upper_label, lower: maps.data.lower_label })
                if (!out) return
                saveDataUrl(out.toDataURL('image/png'), `contact_map_${settled.chrom}_${settled.start}_${settled.end}`)
              }}
            >
              <Camera size={15} aria-hidden />
            </button>
          </div>
        </div>
        {(annotations.data?.loops.length ?? 0) > 0 && (
          <div className="loop-overview">
            <LoopView
              chrom={job.chrom}
              region={[job.start, job.end]}
              loops={annotations.data!.loops}
              window={[locus.start, locus.end]}
              onWindow={(start, end) => {
                moveTo({ chrom: job.chrom, start, end })
                setNote('')
              }}
              onSelect={(i) => {
                if (i === null) return
                // The map goes to the loop with a quarter of its length either side.
                const l = annotations.data!.loops[i]
                const pad = Math.max((l.b[1] - l.a[0]) / 4, res * 5)
                moveTo({ chrom: job.chrom, start: l.a[0] - pad, end: l.b[1] + pad })
                setNote('')
              }}
              hint="The box is the part the map shows. Scroll or drag along the loops to choose a part, or click a loop to go to it."
            />
          </div>
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
          {target && <div className="zoom-box" aria-hidden style={{ left: `${target[0] * 100}%`, top: `${target[0] * 100}%`, width: `${target[1] * 100}%`, height: `${target[1] * 100}%` }} />}
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
              `x ${settled.chrom}:${bp(d.bins[j])}   y ${settled.chrom}:${bp(d.bins[i])}   ${d.upper_label.split(' ')[0].replace(/,$/, '')} ${d.upper[r][c].toFixed(1)}   model ${d.lower[c][r].toFixed(1)}   apart ${(Math.abs(d.bins[j] - d.bins[i]) / 1e6).toFixed(2)} Mb`,
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
          <select aria-label="Upper half" value={upperKind} onChange={(e) => choose(setUpper, e.target.value)}>
            {halves.map(([id, label]) => (
              <option key={id} value={id}>
                {label}
              </option>
            ))}
          </select>
        </label>
        <label className="field">
          <span>Lower half</span>
          <select aria-label="Lower half" value={lowerKind} onChange={(e) => choose(setLower, e.target.value)}>
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
            {/* The full account of each number is its tooltip. */}
            <p className="big" title={maps.data.rank_about}>
              <span className="big-value">{typeof maps.data.rank === 'number' ? maps.data.rank.toFixed(2) : 'none'}</span> <span className="unit">Spearman correlation</span>
            </p>
            <p className="card-text">{`Pairs of bins the same distance apart are compared${reach(maps.data.rank_reach)}. The fall of contacts with distance adds nothing to this number.`}</p>
            <p className="card-text" title={maps.data.scc_about}>{`SCC ${maps.data.scc === null ? 'none' : maps.data.scc.toFixed(2)}, the same comparison on the values and not their ranks.`}</p>
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
