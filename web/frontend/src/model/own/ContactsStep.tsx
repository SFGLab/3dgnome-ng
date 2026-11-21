import { CircleAlert, CircleCheck, HardDrive, Link2, LoaderCircle, SquareTerminal, Trash2, Upload } from 'lucide-react'
import { useState } from 'react'
import type { Region } from '../../api'
import { megabytes } from '../../lib/format'
import type { Entry } from './entry'
import { Rule } from './parts'

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

/** The contacts step: a map by link, a local .hic read in the browser, or a BEDPE of the region. */
export function ContactsStep({
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
          <code className="command">{`gnome3d extract --mcool sample.mcool --region ${span} --out region.bedpe`}</code>
        </div>
      </div>
    </div>
  )
}
