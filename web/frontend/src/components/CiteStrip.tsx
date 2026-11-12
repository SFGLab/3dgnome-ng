import { ArrowRight, Quote } from 'lucide-react'
import { Link } from 'react-router-dom'
import { ROUTES } from '../routes'

/** The strip at the foot of every page that asks users to cite the work. */
export function CiteStrip() {
  return (
    <footer className="cite-strip">
      <Quote size={14} aria-hidden />
      <p>If you use these models, please cite 3D-GNOMEᴺᴳ and the method and data papers on the Cite page.</p>
      <Link className="cite-button" to={ROUTES.cite}>
        All citations <ArrowRight size={13} aria-hidden />
      </Link>
    </footer>
  )
}
