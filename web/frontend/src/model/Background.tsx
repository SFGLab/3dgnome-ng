import { Info } from 'lucide-react'
import type { Region } from '../api'
import type { Background as BackgroundId } from './form'

const HICDIFFUSION_SPAN = 2_000_000

type Props = {
  value: BackgroundId
  region: Region | undefined
  available: boolean
  lymphoblastoid: boolean
  onChange: (b: BackgroundId) => void
}

/** Step 4, the contact background: the cell type's Hi-C, a HiCDiffusion prediction or none. */
export function Background({ value, region, available, lymphoblastoid, onChange }: Props) {
  const tooLong = region !== undefined && region.end - region.start > HICDIFFUSION_SPAN
  const reason = !available ? 'HiCDiffusion is not set up on this server.' : tooLong ? 'HiCDiffusion predicts regions up to 2 Mb.' : ''
  const options: [BackgroundId, string, string][] = [
    ['hic', 'Hi-C of this cell type', 'The Hi-C experiment on this cell line, from 4DN. The default.'],
    ['hicdiffusion', 'Predicted by HiCDiffusion', 'A contact map predicted from the DNA sequence, for regions up to 2 Mb. Structural variants change the prediction.'],
    ['none', 'None', 'No contact map. Two points that no loop joins are placed by their distance along the genome alone.'],
  ]
  return (
    <>
      <div className="options three" role="radiogroup" aria-label="Contact background">
        {options.map(([id, title, desc]) => {
          const disabled = id === 'hicdiffusion' && reason !== ''
          return (
            <button key={id} type="button" role="radio" aria-checked={value === id} className="option card" disabled={disabled} onClick={() => onChange(id)}>
              <span className="option-text">
                <span className="option-title">{title}</span>
                <span className="option-desc">{desc}</span>
                {disabled && <span className="option-reason">{reason}</span>}
              </span>
            </button>
          )
        })}
      </div>
      {!lymphoblastoid && (
        <p className="notice">
          <Info size={14} aria-hidden /> HiCDiffusion was trained on GM12878. Its predictions for H1ESC and HFFc6 are outside its training data and the job page shows a warning.
        </p>
      )}
    </>
  )
}
