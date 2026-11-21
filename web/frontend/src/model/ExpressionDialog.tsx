import { Trash2, Upload, X } from 'lucide-react'
import { useEffect, useState } from 'react'
import { CheckLine, type Entry, FileChip, Rule } from './OwnData'
import { type OwnExpression, uploadFile } from './upload'
import '../styles/dialog.css'
import './owndata.css'

/**
 * The dialog for an expression table of the user: the file, checked by the server on arrival,
 * the samples the job takes from it and what its values are. "Use this table" hands the
 * checked upload to the form.
 */
export function ExpressionDialog({ initial, onClose, onUse }: { initial: OwnExpression | null; onClose: () => void; onUse: (own: OwnExpression) => void }) {
  const [entry, setEntry] = useState<Entry | null>(initial ? { key: 0, factor: '', name: initial.upload.name, size: initial.upload.size, result: initial.upload } : null)
  const [columns, setColumns] = useState<string[]>(initial?.columns ?? [])
  const [kind, setKind] = useState<OwnExpression['kind']>(initial?.kind ?? 'counts')

  useEffect(() => {
    const esc = (e: KeyboardEvent) => e.key === 'Escape' && onClose()
    window.addEventListener('keydown', esc)
    return () => window.removeEventListener('keydown', esc)
  }, [onClose])

  const add = (file: File) => {
    const key = Date.now()
    setEntry({ key, factor: '', name: file.name, size: file.size })
    const put = (patch: Partial<Entry>) => setEntry((e) => (e?.key === key ? { ...e, ...patch } : e))
    uploadFile('expression', file, { cell: '', region: '' })
      .then((result) => {
        put({ result })
        if (result.ok) {
          setColumns(result.samples)
          setKind(result.counts ? 'counts' : 'values')
        }
      })
      .catch((e: Error) => put({ failure: e.message }))
  }
  const upload = entry?.result?.ok ? entry.result : null
  const bad = !!entry && (!!entry.failure || entry.result?.ok === false)
  const toggle = (sample: string) => setColumns((now) => (upload?.samples ?? []).filter((s) => (s === sample ? !now.includes(s) : now.includes(s))))

  return (
    <div className="dialog-backdrop">
      <div className="dialog expression-dialog" role="dialog" aria-modal="true" aria-labelledby="expression-title">
        <header className="dialog-head">
          <div>
            <h2 id="expression-title">Your own expression table</h2>
            <p className="dialog-sub">The file stays private to this job and is deleted after 14 days.</p>
          </div>
          <button type="button" className="icon-button" aria-label="Close" onClick={onClose}>
            <X size={18} aria-hidden />
          </button>
        </header>
        <div className="dialog-body">
          <div className="own-tab">
            <Rule
              text="One table of genes by samples, separated by tabs or commas. The first column holds the gene as an Ensembl id or a gene name. Every other column of numbers is a sample. A column named length is optional and gives the gene length in bp. The output of featureCounts is read as it is."
              columns="gene length sample_1 sample_2"
            />
            {entry ? (
              <div className={`factor-row${bad ? ' bad' : ''}`} role="group" aria-label="Expression table">
                <div className="row-head">
                  <FileChip entry={entry} />
                  <button
                    type="button"
                    className="icon-button"
                    aria-label={`Remove ${entry.name}`}
                    onClick={() => {
                      setEntry(null)
                      setColumns([])
                    }}
                  >
                    <Trash2 size={15} aria-hidden />
                  </button>
                </div>
                <CheckLine entry={entry} />
              </div>
            ) : (
              <label
                className="add-file"
                onDragOver={(e) => e.preventDefault()}
                onDrop={(e) => {
                  e.preventDefault()
                  if (e.dataTransfer.files[0]) add(e.dataTransfer.files[0])
                }}
              >
                <Upload size={14} aria-hidden /> Choose a table, or drop it here
                <input type="file" accept=".tsv,.txt,.csv,.gz" className="visually-hidden" onChange={(e) => e.target.files?.[0] && add(e.target.files[0])} />
              </label>
            )}
            {upload && (
              <>
                <fieldset className="own-samples">
                  <legend>Samples of this job</legend>
                  {upload.samples.map((s) => (
                    <label key={s} className="own-sample">
                      <input type="checkbox" checked={columns.includes(s)} onChange={() => toggle(s)} />
                      {s}
                    </label>
                  ))}
                </fieldset>
                <label className="own-select">
                  <span>The values are</span>
                  <select value={kind} onChange={(e) => setKind(e.target.value === 'values' ? 'values' : 'counts')}>
                    <option value="counts">read counts</option>
                    <option value="values">normalised, such as TPM or TMM</option>
                  </select>
                </label>
                <p className="hint">
                  {kind === 'counts'
                    ? 'Two jobs with two samples or more each are compared with DESeq2. With one sample on a side the comparison is a ratio with no test.'
                    : 'Two jobs with normalised values are compared by the ratio of their values, with no test.'}
                </p>
              </>
            )}
          </div>
        </div>
        <footer className="dialog-foot">
          <p className={`foot-note${bad ? ' bad' : ''}`}>
            {bad ? 'Fix or remove the table to continue.' : entry && !entry.result ? 'Checking the file.' : upload && columns.length === 0 ? 'Choose one sample or more.' : ''}
          </p>
          <button type="button" className="secondary" onClick={onClose}>
            Cancel
          </button>
          <button type="button" className="primary" disabled={!upload || columns.length === 0} onClick={() => upload && onUse({ upload, columns, kind })}>
            Use this table
          </button>
        </footer>
      </div>
    </div>
  )
}
