import type { Entry } from './entry'
import { OneFile, Rule } from './parts'
import { scoped } from '../../lib/css'
import sheet from './owndata.module.css'

const cn = scoped(sheet)

export type EnhSource = 'upload' | 'listed' | 'none'
const ENH_SOURCES: [EnhSource, string, string][] = [
  ['upload', 'Upload a BED file', 'your enhancers'],
  ['listed', 'A listed cell type', 'its hosted enhancers'],
  ['none', 'None', 'promoters only'],
]

/** The enhancers step. A listed cell type takes a file in place of its enhancers. Your own cell type
 * takes a file, the enhancers of a listed cell type or none. */
export function EnhancersStep({
  ownCell,
  cells,
  enhSource,
  setEnhSource,
  enhFrom,
  setEnhFrom,
  entry,
  onFile,
  onRemove,
}: {
  ownCell: boolean
  cells: string[]
  enhSource: EnhSource
  setEnhSource: (s: EnhSource) => void
  enhFrom: string
  setEnhFrom: (cell: string) => void
  entry: Entry | null
  onFile: (f: File) => void
  onRemove: () => void
}) {
  return (
    <>
      {ownCell && (
        <div className={cn('own-tab')}>
          <div className={cn('routes')} role="radiogroup" aria-label="Enhancers from">
            {ENH_SOURCES.map(([id, title, sub]) => (
              <button key={id} type="button" role="radio" aria-checked={enhSource === id} className={cn('route')} onClick={() => setEnhSource(id)}>
                <span>
                  <span className={cn('route-title')}>{title}</span>
                  <span className={cn('route-sub')}>{sub}</span>
                </span>
              </button>
            ))}
          </div>
          {enhSource === 'listed' && (
            <label className={cn('own-select')}>
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
      {(!ownCell || enhSource === 'upload') && (
        <OneFile
          kind="enhancers"
          rule={ownCell ? 'One BED file.' : 'One BED file. Without a file the job marks the enhancers of the cell type.'}
          columns="chr start end"
          entry={entry}
          onFile={onFile}
          onRemove={onRemove}
        />
      )}
    </>
  )
}
