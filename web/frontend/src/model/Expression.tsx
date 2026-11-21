import { FileSpreadsheet, Info, X } from 'lucide-react'
import type { ExpressionOption } from '../api'
import type { OwnExpression } from './upload'
import { scoped } from '../lib/css'
import sheet from './model.module.css'

const cn = scoped(sheet)

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
  /** Opens the dialog that takes a table of the user. */
  onOpen: () => void
  onRemove: () => void
}

const about = (o: ExpressionOption) =>
  o.kind === 'counts'
    ? 'Read counts. Two jobs with two samples or more each are compared with DESeq2.'
    : `Normalised values${o.unit ? `, ${o.unit}` : ''}. Two jobs of this source are compared by the ratio of their values.`

const samples = (n: number) => (n === 1 ? '1 sample' : `${n} samples`)

/**
 * The step of the form for gene expression, off unless a source is chosen. The job then shows
 * the expression of each gene, and the comparison of two jobs shows the change of expression
 * beside the change of the 3D distance. A table of the user comes through the dialog for own data.
 */
export function Expression({ options, value, cell, onChange, own, onOpen, onRemove }: Props) {
  const chosen = options.find((o) => o.id === value)
  return (
    <>
      <div className={cn('options two')} role="radiogroup" aria-label="Gene expression">
        <button type="button" role="radio" aria-checked={value === ''} className={cn('option card')} onClick={() => onChange('')}>
          <span className={cn('option-text')}>
            <span className={cn('option-title')}>None</span>
            <span className={cn('option-desc')}>The job has no expression data.</span>
          </span>
        </button>
        {options.map((o) => (
          <button key={o.id} type="button" role="radio" aria-checked={value === o.id} className={cn('option card')} onClick={() => onChange(o.id)}>
            <span className={cn('option-text')}>
              <span className={cn('option-title')}>{o.label}</span>
              <span className={cn('option-desc')}>{about(o)}</span>
            </span>
          </button>
        ))}
        <button type="button" role="radio" aria-checked={value === 'own'} className={cn('option card')} onClick={() => (own ? onChange('own') : onOpen())}>
          <span className={cn('option-text')}>
            <span className={cn('option-title')}>Your own table</span>
            <span className={cn('option-desc')}>Read counts or normalised values of your samples. The job is private.</span>
          </span>
        </button>
      </div>
      {options.length === 0 && <p className={cn('catalog-note')}>The server holds no RNA-seq for this cell type.</p>}
      {value === 'own' && own && (
        <div className={cn('upload chosen own-table')}>
          <FileSpreadsheet size={14} aria-hidden />
          <span>{`${own.upload.name}, ${samples(own.columns.length)}, ${own.kind === 'counts' ? 'read counts' : 'normalised values'}`}</span>
          <button type="button" className="link-button" onClick={onOpen}>
            Change
          </button>
          <button type="button" className="icon-button" aria-label="Remove the table" onClick={onRemove}>
            <X size={14} aria-hidden />
          </button>
        </div>
      )}
      {chosen?.other_cell_type && (
        <p className={cn('notice')}>
          <Info size={14} aria-hidden /> {`This RNA-seq is of a lymphoblastoid cell line. ${cell} is another cell type and its expression differs.`}
        </p>
      )}
    </>
  )
}
