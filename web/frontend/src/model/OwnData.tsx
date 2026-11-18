import { useQueries } from '@tanstack/react-query'
import { CircleAlert, CircleCheck, FileText, HardDrive, Link2, LoaderCircle, SquareTerminal, Trash2, Upload, X } from 'lucide-react'
import { useEffect, useRef, useState } from 'react'
import type { Region } from '../api'
import { readHicWindow } from './hicLocal'
import { guessFactor, LinkRefused, megabytes, NO_OWN, type OwnChoice, type UploadKind, type UploadResult, uploadFile, uploadLink, uploadPreview } from './upload'
import './owndata.css'

export type OwnTab = 'loops' | 'anchors' | 'contacts' | 'enhancers'

/** One uploaded file in the dialog: checking, checked, or refused by the server. */
type Entry = { key: number; factor: string; name: string; size: number; result?: UploadResult; failure?: string; fix?: string }

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

/** What a step takes: a sentence or two and the columns of the file on a line of their own. */
function Rule({ text, columns }: { text: string; columns?: string }) {
  return (
    <div className="rule">
      <p>{text}</p>
      {columns && (
        <p className="columns">
          <span>Columns</span>
          <code>{columns}</code>
        </p>
      )}
    </div>
  )
}

/** A tab that takes one file: its rule, the file with its check, or a field to choose one. */
function OneFile({ kind, rule, columns, entry, onFile, onRemove }: { kind: string; rule: string; columns: string; entry: Entry | null; onFile: (f: File) => void; onRemove: () => void }) {
  return (
    <div className="own-tab">
      <Rule text={rule} columns={columns} />
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
          <Upload size={14} aria-hidden /> {`Choose an ${kind} file`}
          <input type="file" className="visually-hidden" onChange={(e) => e.target.files?.[0] && onFile(e.target.files[0])} />
        </label>
      )}
    </div>
  )
}

type EnhSource = 'upload' | 'listed' | 'none'
const ENH_SOURCES: [EnhSource, string, string][] = [
  ['upload', 'Upload a BED file', 'your enhancers'],
  ['listed', 'A listed cell type', 'its hosted enhancers'],
  ['none', 'None', 'promoters only'],
]

const CONTACT_RES = 25_000
const STEPS: OwnTab[] = ['loops', 'anchors', 'contacts', 'enhancers']
const STEP_TITLE: Record<OwnTab, string> = { loops: 'Loops', anchors: 'Anchors', contacts: 'Contacts', enhancers: 'Enhancers' }
type Route = 'link' | 'local' | 'bedpe'
const ROUTES: [Route, string, string, typeof Link2][] = [
  ['link', 'Link to a file', '.hic or .mcool', Link2],
  ['local', 'Local .hic file', 'read in your browser', HardDrive],
  ['bedpe', 'BEDPE of the region', 'upload', Upload],
]

/** A contact map with its check: name and size over the summary, or the refusal and its fix. */
function ContactCard({ entry, onRemove }: { entry: Entry; onRemove: () => void }) {
  const r = entry.result
  const bad = !!entry.failure || r?.ok === false
  const title = entry.failure ?? (r ? `${r.name}, ${megabytes(r.size)}` : entry.name)
  const body = entry.failure ? entry.fix : r ? (r.ok ? r.summary : `${r.error?.found} ${r.error?.fix}`) : 'Reading the region.'
  const Icon = bad ? CircleAlert : r ? CircleCheck : LoaderCircle
  return (
    <div className={`contact-card${bad ? ' bad' : r ? ' good' : ''}`} role="group" aria-label="Contact map">
      <Icon size={15} aria-hidden className={r || bad ? undefined : 'spin'} />
      <div className="contact-text">
        <p className="contact-title">{title}</p>
        {body && <p className="contact-body">{body}</p>}
      </div>
      <button type="button" className="icon-button" aria-label="Remove the contact map" onClick={onRemove}>
        <Trash2 size={15} aria-hidden />
      </button>
    </div>
  )
}

/** The contacts tab: a map by link, a local .hic read in the browser, or a BEDPE of the region. */
function ContactsTab({
  region,
  maxMb,
  entry,
  onLink,
  onLocal,
  onFile,
  onRemove,
}: {
  region: Region | null
  maxMb: number
  entry: Entry | null
  onLink: (url: string) => void
  onLocal: (f: File) => void
  onFile: (f: File) => void
  onRemove: () => void
}) {
  const [route, setRoute] = useState<Route>('link')
  const [url, setUrl] = useState('')
  const span = region ? `${region.chrom}:${region.start}-${region.end}` : 'chr1:11500000-12600000'
  return (
    <div className="own-tab">
      <Rule text={`A contact map at 25 kb is used for the contacts between loops. The server reads only the bins of your region, up to ${maxMb} Mb and thins a deep map to the density the cell type maps use.`} />
      <div className="routes" role="radiogroup" aria-label="Contact map from">
        {ROUTES.map(([id, title, sub, Icon]) => (
          <button key={id} type="button" role="radio" aria-checked={route === id} className="route" onClick={() => setRoute(id)}>
            <Icon size={16} aria-hidden />
            <span>
              <span className="route-title">{title}</span>
              <span className="route-sub">{sub}</span>
            </span>
          </button>
        ))}
      </div>
      {route === 'link' && (
        <form
          className="link-field"
          onSubmit={(e) => {
            e.preventDefault()
            if (url.trim()) onLink(url.trim())
          }}
        >
          <label htmlFor="contact-link">Link to a .hic or .mcool file</label>
          <div className="link-row">
            <span className="link-input">
              <Link2 size={14} aria-hidden />
              <input
                id="contact-link"
                type="url"
                value={url}
                placeholder="https://4dn-open-data-public.s3.amazonaws.com/…/4DNFI1UEG1HD.hic"
                onChange={(e) => setUrl(e.target.value)}
              />
            </span>
            <button type="submit" className="read-button">
              Read
            </button>
          </div>
          <p className="hint">
            The server must answer range requests. 4DN, ENCODE, GEO and S3 links do. The file is not copied and the server keeps only the bins of your region.
          </p>
        </form>
      )}
      {route === 'local' && (
        <label className="add-file">
          <Upload size={14} aria-hidden /> Choose a .hic file. The page reads only your region and uploads it.
          <input type="file" accept=".hic" className="visually-hidden" onChange={(e) => e.target.files?.[0] && onLocal(e.target.files[0])} />
        </label>
      )}
      {route === 'bedpe' && (
        <p className="columns">
          <span>Columns</span>
          <code>chr1 start1 end1 chr2 start2 end2 count</code>
        </p>
      )}
      {route === 'bedpe' && (
        <label className="add-file">
          <Upload size={14} aria-hidden /> Choose a BEDPE file of the region
          <input type="file" className="visually-hidden" onChange={(e) => e.target.files?.[0] && onFile(e.target.files[0])} />
        </label>
      )}
      {entry && <ContactCard entry={entry} onRemove={onRemove} />}
      <div className="contact-card hint-card">
        <SquareTerminal size={15} aria-hidden />
        <div className="contact-text">
          <p className="contact-title">A local .mcool file</p>
          <p className="contact-body">Extract the region on your computer with this command and upload region.bedpe as a BEDPE of the region.</p>
          <code className="command">{`gnome3d-ng extract --mcool sample.mcool --region ${span} --out region.bedpe`}</code>
        </div>
      </div>
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
  maxMb = 20,
  ownCell = false,
  cells = [],
  onClose,
  onUse,
}: {
  cell: string
  region: Region | null
  initial: OwnChoice | null
  tab?: OwnTab
  maxMb?: number
  ownCell?: boolean
  cells?: string[]
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
  const borrowed = start.enhancersFrom && start.enhancersFrom !== 'none' ? start.enhancersFrom : ''
  const [enhSource, setEnhSource] = useState<EnhSource>(start.enhancers || !ownCell ? 'upload' : borrowed ? 'listed' : 'none')
  const [enhFrom, setEnhFrom] = useState(borrowed || cells[0] || '')

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

  const putContacts = (key: number) => (patch: Partial<Entry>) => setSingle((s) => (s.contacts?.key === key ? { ...s, contacts: { ...s.contacts, ...patch } } : s))
  const addLink = (url: string) => {
    const key = next.current++
    const name = url.split('?')[0].split('/').pop() || url
    setSingle((s) => ({ ...s, contacts: { key, factor: '', name, size: 0 } }))
    const put = putContacts(key)
    uploadLink(url, where)
      .then((result) => put({ result }))
      .catch((e: Error) => put({ failure: e.message, fix: e instanceof LinkRefused ? e.fix : '' }))
  }
  const addLocal = (file: File) => {
    const key = next.current++
    setSingle((s) => ({ ...s, contacts: { key, factor: '', name: file.name, size: file.size } }))
    const put = putContacts(key)
    if (!region) {
      put({ failure: 'The region is not set', fix: 'Enter the region in step 1 first. The page reads only the bins of the region from the file.' })
      return
    }
    const win = { chrom: region.chrom, start: region.start, end: region.end }
    readHicWindow(file, win, CONTACT_RES)
      .then((text) => uploadFile('contacts', new File([text], `${file.name.replace(/\.hic$/i, '')}.${region.chrom}_${region.start}_${region.end}.bedpe`), where))
      .then((result) => put({ result }))
      .catch((e: Error) => put({ failure: `${file.name} could not be read`, fix: e.message }))
  }

  const kinds = enhSource === 'upload' ? (['anchors', 'contacts', 'enhancers'] as const) : (['anchors', 'contacts'] as const)
  const blocking = [...loops.map((l) => [l, `${l.factor} file`] as const), ...kinds.map((k) => [single[k], `${k} file`] as const)].find(([e]) => !passed(e) && !pending(e))
  const checking = loops.some(pending) || Object.values(single).some(pending)
  const anything = loops.length > 0 || Object.values(single).some(Boolean) || (ownCell && enhSource === 'listed')
  const use = () =>
    onUse({
      loops: loops.map((l) => ({ factor: l.factor.trim() || 'loops', upload: l.result! })),
      anchors: single.anchors?.result ?? null,
      contacts: single.contacts?.result ?? null,
      enhancers: enhSource === 'upload' ? (single.enhancers?.result ?? null) : null,
      loopsFrom: start.loopsFrom,
      enhancersFrom: ownCell ? (enhSource === 'listed' ? enhFrom : enhSource === 'none' ? 'none' : undefined) : undefined,
    })

  // Each step's subtitle names what the job will use from it, in one form for all four.
  const named = (e: Entry | null) => (e ? (e.result?.name ?? e.name) : null)
  const uses: Record<OwnTab, string> = {
    loops: loops.length ? `${loops.length} file${loops.length === 1 ? '' : 's'}` : 'the listed sets',
    anchors: named(single.anchors) ?? 'the loop ends',
    contacts: named(single.contacts) ?? (ownCell ? 'needed' : "the cell type's Hi-C"),
    enhancers: !ownCell ? (named(single.enhancers) ?? "the cell type's") : enhSource === 'upload' ? (named(single.enhancers) ?? 'none') : enhSource === 'listed' ? enhFrom : 'none',
  }
  const at = STEPS.indexOf(active)

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
        <div className="wizard">
          <div className="wizard-steps" role="tablist" aria-orientation="vertical" aria-label="Steps">
            {STEPS.map((id, i) => (
              <button key={id} type="button" role="tab" aria-selected={active === id} onClick={() => setActive(id)}>
                <span className="step-number" aria-hidden>
                  {i + 1}
                </span>
                <span className="step-text">
                  <span className="step-title">{STEP_TITLE[id]}</span>
                  <span className="step-uses">{uses[id]}</span>
                </span>
              </button>
            ))}
          </div>
          <div className="dialog-body" role="tabpanel">
            {active === 'loops' && (
              <div className="own-tab">
                <Rule text="One BEDPE file per factor. The first factor is read as CTCF and its loops get the motif orientation term." columns="chr1 start1 end1 chr2 start2 end2 count" />
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
                  <Upload size={14} aria-hidden /> Add a factor, or drop a BEDPE file here
                  <input type="file" multiple className="visually-hidden" onChange={(e) => addLoops(e.target.files)} />
                </label>
                {region && loops.some((l) => l.result?.ok) && <OwnPreview cell={cell} region={region} loops={loops} />}
              </div>
            )}
            {active === 'anchors' && (
              <OneFile
                kind="anchors"
                rule="One BED file. The fourth column is optional and gives the motif orientation, L, R or N. Without a file the anchors are the loop ends, merged where they overlap."
                columns="chr start end orientation"
                entry={single.anchors}
                onFile={(f) => addSingle('anchors', f)}
                onRemove={() => setSingle((s) => ({ ...s, anchors: null }))}
              />
            )}
            {active === 'contacts' && (
              <ContactsTab
                region={region}
                maxMb={maxMb}
                entry={single.contacts}
                onLink={addLink}
                onLocal={addLocal}
                onFile={(f) => addSingle('contacts', f)}
                onRemove={() => setSingle((s) => ({ ...s, contacts: null }))}
              />
            )}
            {active === 'enhancers' && ownCell && (
              <div className="own-tab">
                <div className="routes" role="radiogroup" aria-label="Enhancers from">
                  {ENH_SOURCES.map(([id, title, sub]) => (
                    <button key={id} type="button" role="radio" aria-checked={enhSource === id} className="route" onClick={() => setEnhSource(id)}>
                      <span>
                        <span className="route-title">{title}</span>
                        <span className="route-sub">{sub}</span>
                      </span>
                    </button>
                  ))}
                </div>
                {enhSource === 'listed' && (
                  <label className="own-select">
                    <span>Enhancers of</span>
                    <select value={enhFrom} onChange={(e) => setEnhFrom(e.target.value)}>
                      {cells.map((c) => (
                        <option key={c} value={c}>
                          {c}
                        </option>
                      ))}
                    </select>
                  </label>
                )}
                {enhSource === 'none' && <Rule text="The job marks promoters only." />}
              </div>
            )}
            {active === 'enhancers' && (!ownCell || enhSource === 'upload') && (
              <OneFile
                kind="enhancers"
                rule={ownCell ? 'One BED file.' : 'One BED file. Without a file the job marks the enhancers of the cell type.'}
                columns="chr start end"
                entry={single.enhancers}
                onFile={(f) => addSingle('enhancers', f)}
                onRemove={() => setSingle((s) => ({ ...s, enhancers: null }))}
              />
            )}
          </div>
        </div>
        <footer className="dialog-foot">
          <p className={`foot-note${blocking ? ' bad' : ''}`}>
            {blocking
              ? `Fix or remove the ${blocking[1]} to continue.`
              : checking
                ? 'Checking the files.'
                : active === 'contacts' && !ownCell
                  ? "The map replaces the cell type's map for this job."
                  : ''}
          </p>
          <button type="button" className="secondary" onClick={onClose}>
            Cancel
          </button>
          <button type="button" className="secondary" disabled={at === 0} onClick={() => setActive(STEPS[at - 1])}>
            Back
          </button>
          <button type="button" className="secondary" disabled={at === STEPS.length - 1} onClick={() => setActive(STEPS[at + 1])}>
            Next
          </button>
          <button type="button" className="primary" disabled={!!blocking || checking || !anything} onClick={use}>
            Use this data
          </button>
        </footer>
      </div>
    </div>
  )
}
