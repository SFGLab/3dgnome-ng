import { scoped } from '../lib/css'
import sheet from './layout.module.css'

const cn = scoped(sheet)

/** The site name, 3D-GNOME with a superscript NG. */
export function Wordmark() {
  return (
    <span className={cn('wordmark')} data-testid="wordmark">
      3D-GNOME<sup className="ng">NG</sup>
    </span>
  )
}
