import { CircleAlert, FileSpreadsheet, Info, LoaderCircle, Upload, X } from 'lucide-react'
import { useEffect, useState } from 'react'
import type { ExpressionOption } from '../api'
import { megabytes, type OwnExpression, uploadFile } from './upload'
import './owndata.css'

type Props = {
  /** The hosted expression of the cell type and of the sample of the form. */
  options: ExpressionOption[]
  /** The id of the chosen source, own for an uploaded table, empty for none. */
  value: string
  /** The cell type of the form, as its card names it. */
  cell: string
  onChange: (source: string) => void
  /** The uploaded table and the samples the job takes from it. */
  own: OwnExpression | null
  onOwn: (own: OwnExpression | null) => void
}

const about = (o: ExpressionOption) =>
  o.kind === 'counts'
    ? 'Read counts. Two jobs with two samples or more each are compared with DESeq2.'
    : `Normalised values${o.unit ? `, ${o.unit}` : ''}. Two jobs of this source are compared by the ratio of their values.`

/** The upload of a table of genes by samples, the samples the job takes and what the values are. */
function OwnTable({ own, onOwn }: Pick<Props, 'own' | 'onOwn'>) {
  const [file, setFile] = useState<File | null>(null)
  const [failure, setFailure] = useState('')

  useEffect(() => {
    if (!file) return
    setFailure('')
    let live = true
    uploadFile('expression', file, { cell: '', region: '' })
      .then((u) => {
        if (!live) return
        if (u.ok) onOwn({ upload: u, columns: u.samples, kind: u.counts ? 'counts' : 'values' })
        else setFailure(`${u.error?.found} ${u.error?.fix}`)
      })
      .catch((e: Error) => live && setFailure(e.message))
    return () => {
      live = false
    }
    // The upload belongs to the file alone.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [file])

  const remove = () => {
    setFile(null)
    setFailure('')
    onOwn(null)
  }
  const toggle = (sample: string) => {
    if (!own) return
    const has = own.columns.includes(sample)
    const columns = own.upload.samples.filter((s) => (s === sample ? !has : own.columns.includes(s)))
    if (columns.length) onOwn({ ...own, columns })
  }

  return (
    <div className="own-expression">
      {file || own ? (
        <span className="file-chip">
          <FileSpreadsheet size={14} aria-hidden />
          <span className="file-name">{file?.name ?? own?.upload.name}</span>
          <span className="file-size">{megabytes(file?.size ?? own?.upload.size ?? 0)}</span>
          <button type="button" className="icon-button" aria-label="Remove the table" onClick={remove}>
            <X size={14} aria-hidden />
          </button>
        </span>
      ) : (
        <>
          <label className="add-file">
            <Upload size={14} aria-hidden /> Choose a table of genes by samples
            <input type="file" accept=".tsv,.txt,.csv,.gz" className="visually-hidden" onChange={(e) => e.target.files?.[0] && setFile(e.target.files[0])} />
          </label>
          <p className="catalog-note">
            The table is tab or comma separated. Its first column holds the gene, as an Ensembl id or a gene name, and every other column of numbers is a sample. The output of featureCounts is read as it is. The
            table stays private to this job.
          </p>
        </>
      )}
      {file && !own && !failure && (
        <p className="check">
          <LoaderCircle size={14} aria-hidden className="spin" /> Reading the samples of the table.
        </p>
      )}
      {failure && (
        <p className="check bad" role="alert">
          <CircleAlert size={14} aria-hidden /> {failure}
        </p>
      )}
      {own && (
        <>
          <p className="catalog-note">{own.upload.summary}</p>
          <fieldset className="own-samples">
            <legend>Samples of this job</legend>
            {own.upload.samples.map((s) => (
              <label key={s} className="own-sample">
                <input type="checkbox" checked={own.columns.includes(s)} onChange={() => toggle(s)} />
                {s}
              </label>
            ))}
          </fieldset>
          <label className="field own-kind">
            <span>The values are</span>
            <select value={own.kind} onChange={(e) => onOwn({ ...own, kind: e.target.value === 'values' ? 'values' : 'counts' })}>
              <option value="counts">read counts</option>
              <option value="values">normalised, such as TPM or TMM</option>
            </select>
          </label>
          <p className="catalog-note">
            {own.kind === 'counts'
              ? 'Two jobs with two samples or more each are compared with DESeq2. With one sample a side the comparison is a ratio with no test.'
              : 'Two jobs with normalised values are compared by the ratio of their values, with no test.'}
          </p>
        </>
      )}
    </div>
  )
}

/**
 * The step of the form for gene expression, off unless a source is chosen. The job then shows
 * the expression of each gene, and the comparison of two jobs shows the change of expression
 * beside the change of the 3D distance.
 */
export function Expression({ options, value, cell, onChange, own, onOwn }: Props) {
  const chosen = options.find((o) => o.id === value)
  return (
    <>
      <div className="options two" role="radiogroup" aria-label="Gene expression">
        <button type="button" role="radio" aria-checked={value === ''} className="option card" onClick={() => onChange('')}>
          <span className="option-text">
            <span className="option-title">None</span>
            <span className="option-desc">The job has no expression data.</span>
          </span>
        </button>
        {options.map((o) => (
          <button key={o.id} type="button" role="radio" aria-checked={value === o.id} className="option card" onClick={() => onChange(o.id)}>
            <span className="option-text">
              <span className="option-title">{o.label}</span>
              <span className="option-desc">{about(o)}</span>
            </span>
          </button>
        ))}
        <button type="button" role="radio" aria-checked={value === 'own'} className="option card" onClick={() => onChange('own')}>
          <span className="option-text">
            <span className="option-title">Your own table</span>
            <span className="option-desc">Read counts or normalised values of your samples. The job is private.</span>
          </span>
        </button>
      </div>
      {options.length === 0 && <p className="catalog-note">The server holds no RNA-seq for this cell type.</p>}
      {value === 'own' && <OwnTable own={own} onOwn={onOwn} />}
      {chosen?.other_cell_type && (
        <p className="notice">
          <Info size={14} aria-hidden /> {`This RNA-seq is of a lymphoblastoid cell line. ${cell} is another cell type and its expression differs.`}
        </p>
      )}
    </>
  )
}
