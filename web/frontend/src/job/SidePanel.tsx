import { parseRegion, regionText } from '../lib/region'
import { Braces, ChevronDown, ChevronLeft, ChevronRight, ChevronUp, CopyPlus, FileText, Folder, Pencil } from 'lucide-react'
import { useEffect, useRef, useState } from 'react'
import { served } from '../base'
import { recentJobs, rememberJob } from '../myjobs/recent'
import { ROUTES } from '../routes'
import type { JobApi, Summary } from '../api'
import { variants } from './format'
import { bp, duration, FACTORS, size, shortRegion } from '../lib/format'
import { scoped } from '../lib/css'
import sheet from './job.module.css'

const cn = scoped(sheet)

type Props = {
  api: JobApi
  summary: Summary
  selected: number
  onSelect: (i: number) => void
  onPick: (which: 'medoid' | 'compact' | 'random') => void
}

function background(s: Summary): string {
  const b = s.request.background ?? (s.request.hicdiffusion ? 'hicdiffusion' : 'hic')
  if (b === 'hic' && s.request.own?.contacts) return 'your map'
  return { hic: 'Hi-C, 4DN', hicdiffusion: 'HiCDiffusion', none: 'none' }[b] ?? b
}

function ownLoops(s: Summary): string | null {
  const f = s.request.own?.loops ?? []
  if (!f.length) return null
  const words = f.length === 1 ? f[0] : `${f.slice(0, -1).join(', ')} and ${f.at(-1)}`
  return `${words}, your files`
}

function listedLoops(s: Summary): string {
  const set = FACTORS[s.request.factors] ?? s.request.factors
  return s.request.own?.loops_from ? `${set} of ${s.request.own.loops_from}` : set
}

/** The job's name, state, facts, structure selector and files. */
export function JobSidePanel({ api, summary, selected, onSelect, onPick }: Props) {
  const remembered = recentJobs().find((j) => j.url === api.url)
  const [name, setName] = useState(remembered?.name ?? '')
  const [editing, setEditing] = useState(false)
  const [unfolded, setUnfolded] = useState(false)
  // A waiting job has no region of its own yet. Its heading reads the coordinates of the request.
  const at = parseRegion(summary.region) ?? parseRegion(summary.request.region.replace(/,/g, ''))
  const region = at ? regionText(at) : summary.request.region
  // The heading has the region in megabases, which fits the panel. The line under it has the exact positions.
  const short = at ? shortRegion(at.chrom, at.start, at.end) : summary.request.region
  const n = summary.structures
  // The file of the structure in view is kept in sight in the list of all of them.
  const shown = useRef<HTMLLIElement>(null)
  useEffect(() => {
    // In braces: a browser may return a promise from the scroll and an effect must return nothing
    // but its cleanup.
    shown.current?.scrollIntoView?.({ block: 'nearest' })
  }, [selected])
  const save = (value: string) => {
    setName(value.trim())
    setEditing(false)
    rememberJob({ url: api.url, label: remembered?.label ?? `${summary.request.cell_type} ${short}`, name: value.trim(), at: remembered?.at ?? Date.now() })
  }
  const state =
    summary.state === 'done' && summary.started && summary.finished
      ? `Done in ${duration(summary.finished - summary.started)}`
      : summary.state === 'queued'
        ? summary.ahead
          ? `Queued, ${summary.ahead} job${summary.ahead === 1 ? '' : 's'} ahead`
          : 'Queued'
        : summary.state === 'running'
          ? `Running, ${summary.stage ?? 'starting'} stage`
          : summary.state === 'failed'
            ? 'Failed'
            : 'Done'
  const loops = Object.entries(summary.loops)
    .map(([f, c]) => `${bp(c)} ${f}`)
    .join(', ')
  return (
    <aside className={cn(unfolded ? 'job-side' : 'job-side folded')}>
      <p className={cn(`job-state ${summary.state}`)}>
        <span className={cn('state-dot')} /> {state}
      </p>
      {editing ? (
        <input
          className={cn('rename')}
          aria-label="Job name"
          defaultValue={name}
          autoFocus
          onKeyDown={(e) => e.key === 'Enter' && save((e.target as HTMLInputElement).value)}
          onBlur={(e) => save(e.target.value)}
        />
      ) : (
        <h1 className={cn('job-title')}>
          {name || (
            <>
              {summary.request.cell_type} <span className={cn('nowrap')}>{short}</span>
            </>
          )}
        </h1>
      )}
      <p className={cn('job-locus')}>
        {summary.request.cell_type}, {region}
      </p>
      <button type="button" className="link-button" onClick={() => setEditing(true)}>
        <Pencil size={12} aria-hidden /> Rename job
      </button>
      {/* The form of Model a region, filled with the request of this job. */}
      <a className="link-button" href={served(`${ROUTES.model}?from=${encodeURIComponent(api.url)}`)}>
        <CopyPlus size={12} aria-hidden /> New job from this request
      </a>
      {/* On a narrow screen the panel is above the tabs. Its facts and files fold away there, so
          the model is not a screen below. */}
      <button type="button" className={cn('side-fold-toggle')} aria-expanded={unfolded} onClick={() => setUnfolded((u) => !u)}>
        {unfolded ? <ChevronUp size={13} aria-hidden /> : <ChevronDown size={13} aria-hidden />} Details and files
      </button>
      <dl className={cn('facts')}>
        <dt>Loops</dt>
        <dd>{ownLoops(summary) ?? listedLoops(summary)}</dd>
        <dt>Loops in region</dt>
        <dd>{loops || 'none'}</dd>
        <dt>Structural variants</dt>
        <dd>{variants(summary)}</dd>
        <dt>Contact background</dt>
        <dd>{background(summary)}</dd>
        {summary.expression && (
          <>
            <dt>Expression</dt>
            <dd>{summary.expression.label}</dd>
          </>
        )}
        <dt>Structures</dt>
        <dd>{n || summary.request.n_models}</dd>
      </dl>
      {n > 0 && (
        <>
          <p className={cn('side-label')}>Structure</p>
          <div className={cn('structure-select')}>
            <button type="button" aria-label="Previous structure" disabled={selected <= 1} onClick={() => onSelect(selected - 1)}>
              <ChevronLeft size={14} aria-hidden />
            </button>
            <select aria-label="Structure" value={selected} onChange={(e) => onSelect(Number(e.target.value))}>
              {Array.from({ length: n }, (_, i) => (
                <option key={i + 1} value={i + 1}>
                  {i + 1} of {n}
                </option>
              ))}
            </select>
            <button type="button" aria-label="Next structure" disabled={selected >= n} onClick={() => onSelect(selected + 1)}>
              <ChevronRight size={14} aria-hidden />
            </button>
          </div>
          <div className={cn('picks')}>
            <button type="button" onClick={() => onPick('medoid')}>
              Medoid
            </button>
            <button type="button" onClick={() => onPick('compact')}>
              Most compact
            </button>
            <button type="button" onClick={() => onPick('random')}>
              Random
            </button>
          </div>
          <div className={cn('side-files')}>
          <p className={cn('side-label')}>Files</p>
          <ul className={cn('files')}>
            <li className="folder">
              <Folder size={14} aria-hidden /> structures <span className={cn('size')}>{n} files</span>
            </li>
            <li className={cn('models')}>
              <ul className={cn('files')} aria-label="Structure files">
                {summary.models.map((f, i) => (
                  <li key={f.name} className={cn(i === selected - 1 ? 'file selected' : 'file')} ref={i === selected - 1 ? shown : undefined}>
                    {/* The sizes of the structures are alike. The size is in the tooltip and the name has the row. */}
                    <a href={api.fileUrl(f.name)} title={`${f.name}, ${size(f.size)}`}>
                      <FileText size={14} aria-hidden /> {f.name}
                    </a>
                  </li>
                ))}
              </ul>
            </li>
            {summary.files.map((f) => (
              <li key={f.name} className="file">
                <a href={api.fileUrl(f.name)}>
                  {f.name.endsWith('.json') ? <Braces size={14} aria-hidden /> : <FileText size={14} aria-hidden />} {f.name}
                </a>
                <span className={cn('size')}>{size(f.size)}</span>
              </li>
            ))}
          </ul>
          </div>
        </>
      )}
    </aside>
  )
}
