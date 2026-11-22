import { Upload } from 'lucide-react'
import type { Catalog } from '../api'
import { ownData, type OwnChoice } from './upload'
import { scoped } from '../lib/css'
import sheet from './model.module.css'

const cn = scoped(sheet)

/**
 * The cell type cards of step 1, a radio group. The chips of a card are the data the cell type
 * has. The own data card opens the uploads and its chips are the data given through them.
 */
export function CellTypes({ catalog, value, own, onChange, onOwn }: { catalog: Catalog; value: string; own: OwnChoice; onChange: (cell: string) => void; onOwn: () => void }) {
  const given = ownData(own)
  return (
    <div className={cn('cell-types')} role="radiogroup" aria-label="Cell type">
      {Object.entries(catalog.cell_types).map(([name, c]) => (
        <button key={name} type="button" role="radio" aria-checked={value === name} className={cn('cell-card')} onClick={() => onChange(name)}>
          <span className={cn('cell-name')}>{name}</span>
          <span className={cn('cell-desc')}>{c.description}</span>
          <span className={cn('chips')}>
            {c.data.map((d) => (
              <span key={d} className={cn('chip')}>
                {d}
              </span>
            ))}
          </span>
        </button>
      ))}
      <button type="button" role="radio" aria-checked={value === 'own'} className={cn('cell-card own')} onClick={onOwn}>
        <span className={cn('cell-name')}>
          <Upload size={14} aria-hidden /> Your own data
        </span>
        <span className={cn('cell-desc')}>Loops and anchors in BEDPE and BED, contacts as .hic, .mcool or BEDPE, enhancers in BED</span>
        {given.length > 0 && (
          <span className={cn('chips')}>
            {given.map((d) => (
              <span key={d} className={cn('chip')}>
                {d}
              </span>
            ))}
          </span>
        )}
      </button>
      <p className={cn('catalog-note')}>For a cell type that is not listed, upload its loops and contacts as your own data.</p>
    </div>
  )
}
