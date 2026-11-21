/** The pieces the steps of the own data dialog share. */
import { CircleAlert, CircleCheck, FileText, LoaderCircle, Trash2, Upload } from 'lucide-react'
import type { ReactNode } from 'react'
import { megabytes } from '../../lib/format'
import type { Entry } from './entry'
import { scoped } from '../../lib/css'
import sheet from './owndata.module.css'

const cn = scoped(sheet)

export function CheckLine({ entry }: { entry: Entry }) {
  if (entry.failure) {
    return (
      <p className={cn('check bad')}>
        <CircleAlert size={14} aria-hidden /> {entry.failure}
      </p>
    )
  }
  if (!entry.result) {
    return (
      <p className={cn('check')}>
        <LoaderCircle size={14} aria-hidden className={cn('spin')} /> Checking the file.
      </p>
    )
  }
  const r = entry.result
  return r.ok ? (
    <p className={cn('check good')}>
      <CircleCheck size={14} aria-hidden /> {r.summary}
    </p>
  ) : (
    <p className={cn('check bad')} role="alert">
      <CircleAlert size={14} aria-hidden /> {r.error?.found} {r.error?.fix}
    </p>
  )
}

export function FileChip({ entry }: { entry: Entry }) {
  return (
    <span className={cn('file-chip')}>
      <FileText size={14} aria-hidden />
      <span className={cn('file-name')}>{entry.name}</span>
      <span className={cn('file-size')}>{megabytes(entry.size)}</span>
    </span>
  )
}

/** The rule of a step, a sentence or two and the columns of the file on a line of their own. */
export function Rule({ text, columns }: { text: string; columns?: string }) {
  return (
    <div className={cn('rule')}>
      <p>{text}</p>
      {columns && (
        <p className={cn('columns')}>
          <span>Columns</span>
          <code>{columns}</code>
        </p>
      )}
    </div>
  )
}

/** A step that takes one file. It has its rule, the file with its check or a field to choose one, then what the step adds. */
export function OneFile({
  kind,
  rule,
  columns,
  entry,
  onFile,
  onRemove,
  children,
}: {
  kind: string
  rule: string
  columns: string
  entry: Entry | null
  onFile: (f: File) => void
  onRemove: () => void
  children?: ReactNode
}) {
  return (
    <div className={cn('own-tab')}>
      <Rule text={rule} columns={columns} />
      {entry ? (
        <div className={cn(`factor-row${entry.failure || entry.result?.ok === false ? ' bad' : ''}`)} role="group" aria-label={`${kind} file`}>
          <div className={cn('row-head')}>
            <FileChip entry={entry} />
            <button type="button" className="icon-button" aria-label={`Remove ${entry.name}`} onClick={onRemove}>
              <Trash2 size={15} aria-hidden />
            </button>
          </div>
          <CheckLine entry={entry} />
        </div>
      ) : (
        <label className={cn('add-file')}>
          <Upload size={14} aria-hidden /> {`Choose an ${kind} file`}
          <input type="file" className={cn('visually-hidden')} onChange={(e) => e.target.files?.[0] && onFile(e.target.files[0])} />
        </label>
      )}
      {children}
    </div>
  )
}
