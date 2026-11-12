import type { Factors } from './form'

const OPTIONS: [Factors | 'own', string, string][] = [
  ['CTCF', 'CTCF', 'CTCF ChIA-PET loops, the loop set of the original 3D-GNOME.'],
  ['RNAPII', 'RNAPII', 'RNAPII ChIA-PET loops only.'],
  ['CTCF+RNAPII', 'CTCF and RNAPII', 'Both loop sets. Enhancers are closer to their genes.'],
  ['own', 'Your own loops', 'Any loop set in BEDPE, for example ChIA-PET, HiChIP, Micro-C or Hi-C loop calls.'],
]

/** Step 2, the loop set of the job. Own loops open the uploads. On your own cell type only own loops apply. */
export function LoopOptions({
  value,
  ownCell,
  onChange,
  onOwn,
}: {
  value: Factors | 'own'
  ownCell: boolean
  onChange: (f: Factors) => void
  onOwn: () => void
}) {
  return (
    <div className="options two" role="radiogroup" aria-label="Loops">
      {OPTIONS.map(([id, title, desc]) => (
        <button
          key={id}
          type="button"
          role="radio"
          aria-checked={value === id}
          className="option"
          disabled={ownCell && id !== 'own'}
          onClick={() => (id === 'own' ? onOwn() : onChange(id))}
        >
          <span className="radio-dot" aria-hidden />
          <span className="option-text">
            <span className="option-title">{title}</span>
            <span className="option-desc">{desc}</span>
          </span>
        </button>
      ))}
    </div>
  )
}
