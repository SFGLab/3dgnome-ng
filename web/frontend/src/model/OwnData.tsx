import { useQueries } from '@tanstack/react-query'
import { CircleAlert, CircleCheck, FileText, LoaderCircle, Plus, Trash2, X } from 'lucide-react'
import { useEffect, useRef, useState } from 'react'
import type { Region } from '../api'
import { guessFactor, megabytes, NO_OWN, type OwnChoice, type UploadKind, type UploadResult, uploadFile, uploadPreview } from './upload'
import './owndata.css'

export type OwnTab = 'loops' | 'anchors' | 'contacts' | 'enhancers'

/** One uploaded file in the dialog: checking, checked, or refused by the server. */
type Entry = { key: number; factor: string; name: string; size: number; result?: UploadResult; failure?: string }

const fromResult = (key: number, factor: string, r: UploadResult): Entry => ({ key, factor, name: r.name, size: r.size, result: r })
const passed = (e: Entry | null) => !e || !!e.result?.ok
const pending = (e: Entry | null) => !!e && !e.result && !e.failure

const W = 780
const H = 120
const AXIS = 60

function CheckLine({ entry }: { entry: Entry }) {
  if (entry.failure) {
    return (
      <p className="check bad">
        <CircleAlert size={14} aria-hidden /> {entry.failure}
      </p>
    )
  }
  if (!entry.result) {
    return (
      <p className="check">
        <LoaderCircle size={14} aria-hidden className="spin" /> Checking the file.
      </p>
    )
  }
  const r = entry.result
  return r.ok ? (
    <p className="check good">
      <CircleCheck size={14} aria-hidden /> {r.summary}
    </p>
  ) : (
    <p className="check bad" role="alert">
      <CircleAlert size={14} aria-hidden /> {r.error?.found} {r.error?.fix}
    </p>
  )
}

function FileChip({ entry }: { entry: Entry }) {
  return (
    <span className="file-chip">
      <FileText size={14} aria-hidden />
      <span className="file-name">{entry.name}</span>
      <span className="file-size">{megabytes(entry.size)}</span>
    </span>
  )
}

/** The loops of the uploaded files in the region, the first factor above the axis and the rest below. */
function OwnPreview({ cell, region, loops }: { cell: string; region: Region; loops: Entry[] }) {
  const span = `${region.chrom}:${region.start}-${region.end}`
  const ready = loops.filter((l) => l.result?.ok)
  const answers = useQueries({
    queries: ready.map((l) => ({ queryKey: ['upload-preview', l.result!.id, span], queryFn: () => uploadPreview(l.result!.id, cell, span) })),
  })
  const x = (p: number) => ((p - region.start) / (region.end - region.start)) * W
  return (
    <figure className="own-preview">
      <figcaption>{`Loops in ${region.label.split(', ')[0]}`}</figcaption>
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`Uploaded loops in ${span}`}>
        {answers.flatMap((a, k) =>
          (a.data ?? []).map(([s, e], i) => {
            const above = loops.indexOf(ready[k]) === 0
            const r = Math.max((x(e) - x(s)) / 2, 1)
            return (
              <path
                key={`${k}-${i}`}
                className={`arc f${Math.min(loops.indexOf(ready[k]), 2)}`}
                d={`M ${x(s)} ${AXIS} A ${r} ${Math.min(r, AXIS - 6)} 0 0 ${above ? 1 : 0} ${x(e)} ${AXIS}`}
              />
            )
          }),
        )}
        <line className="axis" x1={0} x2={W} y1={AXIS} y2={AXIS} />
      </svg>
    </figure>
  )
}

/** A tab that takes one file: its rule, the file with its check, or a field to choose one. */
function OneFile({ kind, rule, entry, onFile, onRemove }: { kind: string; rule: string; entry: Entry | null; onFile: (f: File) => void; onRemove: () => void }) {
  return (
    <div className="own-tab">
      <p className="rule">{rule}</p>
      {entry ? (
        <div className={`factor-row${entry.failure || entry.result?.ok === false ? ' bad' : ''}`} role="group" aria-label={`${kind} file`}>
          <div className="row-head">
            <FileChip entry={entry} />
            <button type="button" className="icon-button" aria-label={`Remove ${entry.name}`} onClick={onRemove}>
              <Trash2 size={15} aria-hidden />
            </button>
          </div>
          <CheckLine entry={entry} />
        </div>
      ) : (
        <label className="add-file">
          <Plus size={15} aria-hidden /> {`Choose an ${kind} file`}
          <input type="file" className="visually-hidden" onChange={(e) => e.target.files?.[0] && onFile(e.target.files[0])} />
        </label>
      )}
    </div>
  )
}

/** The dialog for a job's own data: loops per factor, anchors, contacts and enhancers, each
 * checked by the server on arrival. "Use these loops" hands the checked uploads to the form. */
export function OwnData({
  cell,
  region,
  initial,
  tab = 'loops',
  onClose,
  onUse,
}: {
  cell: string
  region: Region | null
  initial: OwnChoice | null
  tab?: OwnTab
  onClose: () => void
  onUse: (own: OwnChoice) => void
}) {
  const start = initial ?? NO_OWN
  const next = useRef(100)
  const [active, setActive] = useState<OwnTab>(tab)
  const [loops, setLoops] = useState<Entry[]>(start.loops.map((l, i) => fromResult(i, l.factor, l.upload)))
  const [single, setSingle] = useState<Record<'anchors' | 'contacts' | 'enhancers', Entry | null>>({
    anchors: start.anchors && fromResult(90, '', start.anchors),
    contacts: start.contacts && fromResult(91, '', start.contacts),
    enhancers: start.enhancers && fromResult(92, '', start.enhancers),
  })
  const where = { cell, region: region ? `${region.chrom}:${region.start}-${region.end}` : '' }

  useEffect(() => {
    const esc = (e: KeyboardEvent) => e.key === 'Escape' && onClose()
    window.addEventListener('keydown', esc)
    return () => window.removeEventListener('keydown', esc)
  }, [onClose])

  const send = (kind: UploadKind, file: File, factor: string, put: (patch: Partial<Entry>) => void) =>
    uploadFile(kind, file, { ...where, factor })
      .then((result) => put({ result }))
      .catch((e: Error) => put({ failure: e.message }))

  const addLoops = (files: FileList | null) => {
    Array.from(files ?? []).forEach((file, i) => {
      const key = next.current++
      const factor = guessFactor(file.name, loops.length + i)
      setLoops((ls) => [...ls, { key, factor, name: file.name, size: file.size }])
      void send('loops', file, factor, (patch) => setLoops((now) => now.map((l) => (l.key === key ? { ...l, ...patch } : l))))
    })
  }
  const addSingle = (kind: 'anchors' | 'contacts' | 'enhancers', file: File) => {
    const key = next.current++
    setSingle((s) => ({ ...s, [kind]: { key, factor: '', name: file.name, size: file.size } }))
    void send(kind, file, '', (patch) => setSingle((s) => (s[kind]?.key === key ? { ...s, [kind]: { ...s[kind]!, ...patch } } : s)))
  }

  const blocking = [...loops.map((l) => [l, `${l.factor} file`] as const), ...(['anchors', 'contacts', 'enhancers'] as const).map((k) => [single[k], `${k} file`] as const)].find(
    ([e]) => !passed(e) && !pending(e),
  )
  const checking = loops.some(pending) || Object.values(single).some(pending)
  const anything = loops.length > 0 || Object.values(single).some(Boolean)
  const use = () =>
    onUse({
      loops: loops.map((l) => ({ factor: l.factor.trim() || 'loops', upload: l.result! })),
      anchors: single.anchors?.result ?? null,
      contacts: single.contacts?.result ?? null,
      enhancers: single.enhancers?.result ?? null,
    })

  const count = (n: number) => `${n} file${n === 1 ? '' : 's'}`
  const tabs: [OwnTab, string, string][] = [
    ['loops', 'Loops', count(loops.length)],
    ['anchors', 'Anchors', single.anchors ? '1 file' : 'loop ends'],
    ['contacts', 'Contacts', single.contacts ? '1 file' : '.hic, .mcool or BEDPE'],
    ['enhancers', 'Enhancers', single.enhancers ? '1 file' : 'cell type'],
  ]

  return (
    <div className="dialog-backdrop">
      <div className="dialog own-data" role="dialog" aria-modal="true" aria-labelledby="own-title">
        <header className="dialog-head">
          <div>
            <h2 id="own-title">Your own data</h2>
            <p className="dialog-sub">Files stay private to this job and are deleted after 14 days.</p>
          </div>
          <button type="button" className="icon-button" aria-label="Close" onClick={onClose}>
            <X size={18} aria-hidden />
          </button>
        </header>
        <div className="dialog-tabs" role="tablist">
          {tabs.map(([id, title, meta]) => (
            <button key={id} type="button" role="tab" aria-selected={active === id} onClick={() => setActive(id)}>
              {title} <span className="tab-meta">{meta}</span>
            </button>
          ))}
        </div>
        <div className="dialog-body" role="tabpanel">
          {active === 'loops' && (
            <div className="own-tab">
              <p className="rule">
                One BEDPE file per factor, columns chr1 start1 end1 chr2 start2 end2 count. The first factor is read as CTCF and its loops get the motif orientation term.
              </p>
              {loops.map((l, i) => (
                <div key={l.key} className={`factor-row${l.failure || l.result?.ok === false ? ' bad' : ''}`} role="group" aria-label={`Loops of factor ${i + 1}`}>
                  <div className="row-head">
                    <span className={`factor-name f${Math.min(i, 2)}`}>
                      <span className="key-line" aria-hidden />
                      <input aria-label="Factor" value={l.factor} onChange={(e) => setLoops((ls) => ls.map((x) => (x.key === l.key ? { ...x, factor: e.target.value } : x)))} />
                    </span>
                    <FileChip entry={l} />
                    <button type="button" className="icon-button" aria-label={`Remove ${l.name}`} onClick={() => setLoops((ls) => ls.filter((x) => x.key !== l.key))}>
                      <Trash2 size={15} aria-hidden />
                    </button>
                  </div>
                  <CheckLine entry={l} />
                </div>
              ))}
              <label
                className="add-file"
                onDragOver={(e) => e.preventDefault()}
                onDrop={(e) => {
                  e.preventDefault()
                  addLoops(e.dataTransfer.files)
                }}
              >
                <Plus size={15} aria-hidden /> Add a factor, or drop a BEDPE file here
                <input type="file" multiple className="visually-hidden" onChange={(e) => addLoops(e.target.files)} />
              </label>
              {region && loops.some((l) => l.result?.ok) && <OwnPreview cell={cell} region={region} loops={loops} />}
            </div>
          )}
          {active === 'anchors' && (
            <OneFile
              kind="anchors"
              rule="One BED file, columns chr start end and an optional fourth column with the motif orientation L, R or N. Without a file the anchors are the loop ends, merged where they overlap."
              entry={single.anchors}
              onFile={(f) => addSingle('anchors', f)}
              onRemove={() => setSingle((s) => ({ ...s, anchors: null }))}
            />
          )}
          {active === 'contacts' && (
            <OneFile
              kind="contacts"
              rule="Contact pairs of the region in BEDPE, columns chr1 start1 end1 chr2 start2 end2 count. A job on your own cell type needs contacts."
              entry={single.contacts}
              onFile={(f) => addSingle('contacts', f)}
              onRemove={() => setSingle((s) => ({ ...s, contacts: null }))}
            />
          )}
          {active === 'enhancers' && (
            <OneFile
              kind="enhancers"
              rule="One BED file, columns chr start end. Without a file the job marks the enhancers of the cell type, and none on your own cell type."
              entry={single.enhancers}
              onFile={(f) => addSingle('enhancers', f)}
              onRemove={() => setSingle((s) => ({ ...s, enhancers: null }))}
            />
          )}
        </div>
        <footer className="dialog-foot">
          <p className="foot-note">{blocking ? `Fix or remove the ${blocking[1]} to continue.` : checking ? 'Checking the files.' : ''}</p>
          <button type="button" className="secondary" onClick={onClose}>
            Cancel
          </button>
          <button type="button" className="primary" disabled={!!blocking || checking || !anything} onClick={use}>
            Use these loops
          </button>
        </footer>
      </div>
    </div>
  )
}
