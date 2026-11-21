import { ArrowRight, Code, Quote } from 'lucide-react'
import { Link } from 'react-router-dom'
import { ROUTES } from '../routes'

/** The strip at the foot of every page that asks users to cite the work, with the source code,
 * its licence and the privacy notice. */
export function CiteStrip() {
  return (
    <footer className="cite-strip">
      <Quote size={14} aria-hidden />
      <p>If you use any structures, figures or data produced on this website, please cite 3D-GNOMEᴺᴳ and the method and data papers on the Cite page.</p>
      <a className="cite-button" href="https://github.com/SFGLab/3dgnome-ng">
        <Code size={13} aria-hidden /> Source code
      </a>
      <Link className="cite-button" to={ROUTES.cite}>
        All citations <ArrowRight size={13} aria-hidden />
      </Link>
      <a className="cite-link" href="https://www.gnu.org/licenses/gpl-3.0.html">
        GNU GPL v3
      </a>
      <Link className="cite-link" to={ROUTES.privacy}>
        Privacy
      </Link>
    </footer>
  )
}
