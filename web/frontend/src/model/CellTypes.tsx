import { Upload } from 'lucide-react'
import type { Catalog } from '../api'

/** The cell type cards of step 1, a radio group, with the own data card disabled. */
export function CellTypes({ catalog, value, onChange }: { catalog: Catalog; value: string; onChange: (cell: string) => void }) {
  return (
    <div className="cell-types" role="radiogroup" aria-label="Cell type">
      {Object.entries(catalog.cell_types).map(([name, c]) => (
        <button key={name} type="button" role="radio" aria-checked={value === name} className="cell-card" onClick={() => onChange(name)}>
          <span className="cell-name">{name}</span>
          <span className="cell-desc">{c.description}</span>
          <span className="chips">
            {c.data.map((d) => (
              <span key={d} className="chip">
                {d}
              </span>
            ))}
          </span>
        </button>
      ))}
      <button type="button" role="radio" aria-checked={false} className="cell-card own" disabled title="Uploads open in a later version.">
        <span className="cell-name">
          <Upload size={14} aria-hidden /> Your own data
        </span>
        <span className="cell-desc">Loops and anchors in BEDPE and BED, contacts as .hic, .mcool or BEDPE, enhancers in BED</span>
        <span className="chips">
          <span className="chip">private</span>
          <span className="chip">kept 14 days</span>
        </span>
      </button>
      <p className="catalog-note">For a cell type that is not listed, upload its loops and contacts as your own data.</p>
    </div>
  )
}
