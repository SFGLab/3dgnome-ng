import { Info } from 'lucide-react'
import type { Region } from '../api'
import type { Background as BackgroundId } from './form'
import { scoped } from '../lib/css'
import sheet from './model.module.css'

const cn = scoped(sheet)

type Props = {
  value: BackgroundId
  region: Region | undefined
  available: boolean
  /** The longest region HiCDiffusion takes on this server, in Mb. */
  limitMb?: number
  lymphoblastoid: boolean
  onChange: (b: BackgroundId) => void
}

/** Step 4, the contact background. It is the Hi-C of the cell type, a HiCDiffusion prediction or none. */
export function Background({ value, region, available, limitMb = 2, lymphoblastoid, onChange }: Props) {
  const tooLong = region !== undefined && region.end - region.start > limitMb * 1e6
  const reason = !available ? 'HiCDiffusion is not set up on this server.' : tooLong ? `HiCDiffusion predicts regions up to ${limitMb} Mb.` : ''
  const options: [BackgroundId, string, string][] = [
    ['hic', 'Hi-C of this cell type', 'The Hi-C experiment on this cell line, from 4DN.'],
    ['hicdiffusion', 'Predicted by HiCDiffusion', `A contact map predicted from the DNA sequence, for regions up to ${limitMb} Mb. Structural variants change the prediction.`],
    ['none', 'None', 'Turns off the targets from the Hi-C map. No pair of anchors without a loop gets a Hi-C target and the contacts of a loop do not shorten its target. The exponent of the distance law is still fitted on the contact data.'],
  ]
  return (
    <>
      <div className={cn('options three')} role="radiogroup" aria-label="Contact background">
        {options.map(([id, title, desc]) => {
          const disabled = id === 'hicdiffusion' && reason !== ''
          return (
            <button key={id} type="button" role="radio" aria-checked={value === id} className={cn('option card')} disabled={disabled} onClick={() => onChange(id)}>
              <span className={cn('option-text')}>
                <span className={cn('option-title')}>{title}</span>
                <span className={cn('option-desc')}>{desc}</span>
                {disabled && <span className={cn('option-reason')}>{reason}</span>}
              </span>
            </button>
          )
        })}
      </div>
      {value === 'hicdiffusion' && !lymphoblastoid && (
        <p className={cn('notice')}>
          <Info size={14} aria-hidden /> HiCDiffusion was trained on GM12878. Its predictions for H1ESC and HFFc6 are outside its training data.
        </p>
      )}
    </>
  )
}
