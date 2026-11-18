import { ArrowRight, Database, ExternalLink, History, Play } from 'lucide-react'
import { Link } from 'react-router-dom'
import contacts from '../assets/contacts.png'
import structure from '../assets/structure.png'
import structureCtcf from '../assets/structure_ctcf.png'
import { EXAMPLES as ALL } from '../model/examples'
import { ROUTES } from '../routes'
import { useOpenExample } from '../model/useOpenExample'
import './title.css'

const EXAMPLES = [
  { name: 'TNFRSF8', title: 'TNFRSF8, CTCF and RNAPII loops', detail: 'GM12878, compared with CTCF loops alone', image: structure, fit: 'contain' },
  { name: 'TNFRSF8-CTCF', title: 'The same locus, CTCF loops alone', detail: 'GM12878, chr1:11.5-12.6 Mb', image: structureCtcf, fit: 'contain' },
  { name: 'contacts', title: 'Contacts against Hi-C', detail: 'GM12878, chr1:20-25 Mb', image: contacts, fit: 'cover' },
] as const

/** The landing page: what the site computes, the examples and the previous services. */
export function Title() {
  const { open, busy, error } = useOpenExample()
  return (
    <div className="title-page">
      <section className="hero">
        <div className="hero-text">
          <h1 className="hero-name">
            3D-GNOME<sup>NG</sup>
          </h1>
          <p className="lead">
            3D-GNOMEᴺᴳ computes 3D structures of genomic regions up to 20 Mb from CTCF and RNAPII loops, Hi-C contacts and structural variants. Each job returns an ensemble of
            structures with promoters and enhancers marked and the 3D distances between them.
          </p>
          <div className="buttons">
            <Link className="button primary" to={ROUTES.model}>
              Model a region <ArrowRight size={16} aria-hidden />
            </Link>
            <Link className="button secondary" to={`${ROUTES.model}?example=TNFRSF8`}>
              <Play size={14} aria-hidden /> Open the TNFRSF8 example
            </Link>
          </div>
          <p className="note">Jobs with uploaded data stay private.</p>
        </div>
        <figure className="hero-image">
          <img src={structure} alt="A 3D structure of the TNFRSF8 locus with its promoter and enhancer marked" />
          <figcaption>
            <span className="key">
              <span className="dot promoter" /> TNFRSF8 promoter
            </span>
            <span className="key">
              <span className="dot enhancer" /> its enhancer, 385 kb away
            </span>
          </figcaption>
        </figure>
      </section>

      <section className="previous">
        <div className="service">
          <History size={16} aria-hidden />
          <p>3D-GNOME 3.0, the previous server with its 1000 Genomes structures, is still available at 3dgnome.mini.pw.edu.pl/old.</p>
          <a href="https://3dgnome.mini.pw.edu.pl/old">
            Open 3D-GNOME 3.0 <ExternalLink size={13} aria-hidden />
          </a>
        </div>
        <div className="service">
          <Database size={16} aria-hidden />
          <p>
            enhancer3D, the 3D chromatin structures and enhancer to promoter distance profiles of archaic and modern human genomes, is available at
            3dgnome.mini.pw.edu.pl/old/enhancer3D.
          </p>
          <a href="https://3dgnome.mini.pw.edu.pl/old/enhancer3D/">
            Open enhancer3D <ExternalLink size={13} aria-hidden />
          </a>
        </div>
      </section>

      <section className="examples">
        <div className="examples-head">
          <h2>Examples</h2>
          <Link to={ROUTES.examples}>All examples</Link>
        </div>
        {error && (
          <p className="field-error" role="alert">
            {error}
          </p>
        )}
        <div className="example-row">
          {EXAMPLES.map((e) => (
            <button key={e.name} type="button" className="example" disabled={!!busy} onClick={() => open(ALL.find((x) => x.id === e.name)!)}>
              <img src={e.image} alt="" style={{ objectFit: e.fit }} />
              <span className="example-title">{e.title}</span>
              <span className="example-detail">{busy === e.name ? 'Opening the job' : e.detail}</span>
            </button>
          ))}
        </div>
      </section>

      <p className="lab">
        The Laboratory of Functional and Structural Genomics at Warsaw University of Technology and the University of Warsaw develops 3D-GNOMEᴺᴳ, the successor of 3D-GNOME 1.0, 2.0
        and 3.0.
      </p>
    </div>
  )
}
