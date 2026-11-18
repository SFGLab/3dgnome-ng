import { Archive, Braces, ChevronLeft, ChevronRight, FileText, Folder, Pencil } from 'lucide-react'
import { useState } from 'react'
import { recentJobs, rememberJob } from '../model/recent'
import type { JobApi, Summary } from './api'
import { bp, duration, FACTORS, size } from './format'

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

function variants(s: Summary): string {
  if (!s.n_svs) return 'none'
  const sample = s.sample_note ? `${s.request.sample} (${s.sample_note})` : s.request.sample
  const who = s.request.sample ? `${sample}, haplotype ${s.request.haplotype + 1}` : 'uploaded VCF'
  return `${s.n_svs} from ${who}`
}

/** The job's name, state, facts, structure selector and files. */
export function JobSidePanel({ api, summary, selected, onSelect, onPick }: Props) {
  const remembered = recentJobs().find((j) => j.url === api.url)
  const [name, setName] = useState(remembered?.name ?? '')
  const [editing, setEditing] = useState(false)
  const m = summary.region?.match(/^(\w+):(\d+)-(\d+)$/)
  const region = m ? `${m[1]}:${bp(Number(m[2]))}-${bp(Number(m[3]))}` : summary.request.region
  const title = name || `${summary.request.cell_type} ${region}`
  const n = summary.structures
  const model = summary.models[selected - 1]
  const save = (value: string) => {
    setName(value.trim())
    setEditing(false)
    rememberJob({ url: api.url, label: remembered?.label ?? `${summary.request.cell_type} ${region}`, name: value.trim(), at: remembered?.at ?? Date.now() })
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
    <aside className="job-side">
      <p className={`job-state ${summary.state}`}>
        <span className="state-dot" /> {state}
      </p>
      {editing ? (
        <input
          className="rename"
          aria-label="Job name"
          defaultValue={name}
          autoFocus
          onKeyDown={(e) => e.key === 'Enter' && save((e.target as HTMLInputElement).value)}
          onBlur={(e) => save(e.target.value)}
        />
      ) : (
        <h1 className="job-title">{title}</h1>
      )}
      <p className="job-locus">
        {summary.request.cell_type}, {region}
      </p>
      <button type="button" className="link-button" onClick={() => setEditing(true)}>
        <Pencil size={12} aria-hidden /> Rename job
      </button>
      <dl className="facts">
        <dt>Loops</dt>
        <dd>{ownLoops(summary) ?? listedLoops(summary)}</dd>
        <dt>Loops in region</dt>
        <dd>{loops || 'none'}</dd>
        <dt>Structural variants</dt>
        <dd>{variants(summary)}</dd>
        <dt>Contact background</dt>
        <dd>{background(summary)}</dd>
        <dt>Structures</dt>
        <dd>{n || summary.request.n_models}</dd>
      </dl>
      {n > 0 && (
        <>
          <p className="side-label">Structure</p>
          <div className="structure-select">
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
          <div className="picks">
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
          <p className="side-label">Files</p>
          <ul className="files">
            <li className="folder">
              <Folder size={14} aria-hidden /> structures <span className="size">{n} files</span>
            </li>
            {model && (
              <li className="file selected">
                <a href={api.fileUrl(model.name)}>
                  <FileText size={14} aria-hidden /> {model.name}
                </a>
                <span className="size">{size(model.size)}</span>
              </li>
            )}
            {n > 1 && <li className="more">{n - 1} more</li>}
            {summary.files
              .filter((f) => ['ep.tsv', 'marks.json', 'engine.log'].includes(f.name))
              .map((f) => (
                <li key={f.name} className="file">
                  <a href={api.fileUrl(f.name)}>
                    {f.name.endsWith('.json') ? <Braces size={14} aria-hidden /> : <FileText size={14} aria-hidden />} {f.name}
                  </a>
                  <span className="size">{size(f.size)}</span>
                </li>
              ))}
            <li className="file">
              <a href={api.zipUrl}>
                <Archive size={14} aria-hidden /> all files, zip
              </a>
            </li>
          </ul>
        </>
      )}
    </aside>
  )
}
