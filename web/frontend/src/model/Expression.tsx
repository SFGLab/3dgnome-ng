import { Info } from 'lucide-react'
import type { ExpressionOption } from '../api'

type Props = {
  /** The hosted expression of the cell type and of the sample of the form. */
  options: ExpressionOption[]
  /** The id of the chosen source, empty for none. */
  value: string
  /** The cell type of the form, as its card names it. */
  cell: string
  onChange: (source: string) => void
}

const about = (o: ExpressionOption) =>
  o.kind === 'counts'
    ? 'Read counts. Two jobs with two samples or more each are compared with DESeq2.'
    : `Normalised values${o.unit ? `, ${o.unit}` : ''}. Two jobs of this source are compared by the ratio of their values.`

/**
 * The step of the form for gene expression, off unless a source is chosen. The job then shows
 * the expression of each gene, and the comparison of two jobs shows the change of expression
 * beside the change of the 3D distance.
 */
export function Expression({ options, value, cell, onChange }: Props) {
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
      </div>
      {options.length === 0 && <p className="catalog-note">The server holds no RNA-seq for this cell type.</p>}
      {chosen?.other_cell_type && (
        <p className="notice">
          <Info size={14} aria-hidden /> {`This RNA-seq is of a lymphoblastoid cell line. ${cell} is another cell type and its expression differs.`}
        </p>
      )}
    </>
  )
}
