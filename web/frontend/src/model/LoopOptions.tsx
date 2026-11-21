import type { Factors } from './form'
import { scoped } from '../lib/css'
import sheet from './model.module.css'

const cn = scoped(sheet)

const OPTIONS: [Factors | 'own', string, string][] = [
  ['CTCF', 'CTCF', 'CTCF ChIA-PET loops, the loop set of the original 3D-GNOME.'],
  ['RNAPII', 'RNAPII', 'RNAPII ChIA-PET loops only.'],
  ['CTCF+RNAPII', 'CTCF and RNAPII', 'Both loop sets. Enhancers are closer to their genes.'],
  ['own', 'Your own loops', 'Any loop set in BEDPE, for example ChIA-PET, HiChIP, Micro-C or Hi-C loop calls.'],
]

/** Step 2, the loop set of the job. Own loops open the uploads. On your own cell type a listed
 * set comes from the cell type named under "Loops of". */
export function LoopOptions({
  value,
  ownCell,
  cells,
  loopsFrom,
  onLoopsFrom,
  onChange,
  onOwn,
}: {
  value: Factors | 'own'
  ownCell: boolean
  cells: string[]
  loopsFrom: string
  onLoopsFrom: (cell: string) => void
  onChange: (f: Factors) => void
  onOwn: () => void
}) {
  return (
    <>
      <div className={cn('options two')} role="radiogroup" aria-label="Loops">
        {OPTIONS.map(([id, title, desc]) => (
          <button key={id} type="button" role="radio" aria-checked={value === id} className={cn('option')} onClick={() => (id === 'own' ? onOwn() : onChange(id))}>
            <span className={cn('radio-dot')} aria-hidden />
            <span className={cn('option-text')}>
              <span className={cn('option-title')}>{title}</span>
              <span className={cn('option-desc')}>{desc}</span>
            </span>
          </button>
        ))}
      </div>
      {ownCell && value !== 'own' && (
        <label className={cn('loops-of')}>
          <span>Loops of</span>
          <select value={loopsFrom} onChange={(e) => onLoopsFrom(e.target.value)}>
            {cells.map((c) => (
              <option key={c} value={c}>
                {c}
              </option>
            ))}
          </select>
        </label>
      )}
    </>
  )
}
