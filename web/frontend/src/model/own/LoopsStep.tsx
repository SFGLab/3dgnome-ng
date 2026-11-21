import { Trash2, Upload } from 'lucide-react'
import type { Dispatch, SetStateAction } from 'react'
import type { Region } from '../../api'
import type { Entry } from './entry'
import { OwnPreview } from './OwnPreview'
import { CheckLine, FileChip, Rule } from './parts'
import { scoped } from '../../lib/css'
import sheet1 from '../model.module.css'
import sheet2 from './owndata.module.css'

const cn = scoped(sheet1, sheet2)

/** The loops step: one file per factor with its name and its check, and the loops of the files in the region. */
export function LoopsStep({
  cell,
  region,
  loops,
  setLoops,
  addLoops,
}: {
  cell: string
  region: Region | null
  loops: Entry[]
  setLoops: Dispatch<SetStateAction<Entry[]>>
  addLoops: (files: FileList | null) => void
}) {
  return (
    <div className={cn('own-tab')}>
      <Rule text="One BEDPE file per factor. The first factor is read as CTCF and its loops get the motif orientation term." columns="chr1 start1 end1 chr2 start2 end2 count" />
      {loops.map((l, i) => (
        <div key={l.key} className={cn(`factor-row${l.failure || l.result?.ok === false ? ' bad' : ''}`)} role="group" aria-label={`Loops of factor ${i + 1}`}>
          <div className={cn('row-head')}>
            <span className={cn(`factor-name f${Math.min(i, 2)}`)}>
              <span className={cn('key-line')} aria-hidden />
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
        className={cn('add-file')}
        onDragOver={(e) => e.preventDefault()}
        onDrop={(e) => {
          e.preventDefault()
          addLoops(e.dataTransfer.files)
        }}
      >
        <Upload size={14} aria-hidden /> Add a factor, or drop a BEDPE file here
        <input type="file" multiple className={cn('visually-hidden')} onChange={(e) => addLoops(e.target.files)} />
      </label>
      {region && loops.some((l) => l.result?.ok) && <OwnPreview cell={cell} region={region} loops={loops} />}
    </div>
  )
}
