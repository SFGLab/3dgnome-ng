import type { Chain as ChainId } from './form'
import { scoped } from '../lib/css'
import sheet from './model.module.css'

const cn = scoped(sheet)

const OPTIONS: [ChainId, string, string][] = [
  [
    'uniform',
    'One bead per kilobase',
    'Every kilobase of the region is one bead and a loop ends on the bead it falls in. Under the Hi-C background the contacts between the loops are read from the map of the cell type.',
  ],
  [
    'anchors',
    'Loop anchors and segments',
    'The anchors of the loops are placed first and the chromatin between two anchors is filled in after, the chain of the original 3D-GNOME.',
  ],
]

/** The chain of the model at the top of step 6, two cards. */
export function Chain({ value, onChange }: { value: ChainId; onChange: (c: ChainId) => void }) {
  return (
    <div className={cn('options two')} role="radiogroup" aria-label="Chain">
      {OPTIONS.map(([id, title, desc]) => (
        <button key={id} type="button" role="radio" aria-checked={value === id} className={cn('option')} onClick={() => onChange(id)}>
          <span className={cn('radio-dot')} aria-hidden />
          <span className={cn('option-text')}>
            <span className={cn('option-title')}>{title}</span>
            <span className={cn('option-desc')}>{desc}</span>
          </span>
        </button>
      ))}
    </div>
  )
}
