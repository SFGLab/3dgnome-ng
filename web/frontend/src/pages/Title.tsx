import { ArrowRight, Database, ExternalLink, History, Play } from 'lucide-react'
import { Link } from 'react-router-dom'
import contacts from '../assets/contacts.png'
import previews from '../assets/previews.json'
import structure from '../assets/structure.png'
import { StructurePreview, type PreviewData } from '../components/StructurePreview'
import { EXAMPLES as ALL } from '../model/examples'
import { ROUTES } from '../routes'
import { useOpenExample } from '../model/useOpenExample'
import './title.css'

/** The drawings of the TNFRSF8 locus with and without RNAPII loops. A is the promoter and B the
 * enhancer 385 kb away. web/scripts/make_previews.py writes them from the two example jobs. */
const WITH = previews.rnapii as PreviewData
const WITHOUT = previews.ctcf as PreviewData
const spread = (d: PreviewData) => `${d.mean.toFixed(1)} ± ${d.sd.toFixed(1)}`
/** The hero's picture in its own pixels: a point on the promoter and one on the enhancer, the
 * letter beside each where the picture is empty and the middle of the distance's box. */
const HERO = { a: [354, 397], b: [705, 196], letterA: [350, 452], letterB: [668, 150], box: [525, 232] } as const

/** The part of each drawing the example shows. One view for both, so the two lines can be compared. */
const STORY_VIEW: [number, number, number, number] = [-170, -372, 770, 764]

/** The two drawings of the first example, the tag over each and the words a screen reader gets. */
const STORY = [
  { tag: 'CTCF and RNAPII loops', data: WITH },
  { tag: 'CTCF loops alone', data: WITHOUT },
]

/** The landing page: what the site computes, the examples and the previous services. */
export function Title() {
  const { open, busy, error } = useOpenExample()
  return (
    <div className="title-page">
      <section className="hero">
        <div className="hero-text">
          <h1 className="hero-name">
            3D-GNOME<sup className="ng">NG</sup>
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
        </div>
        <figure className="hero-image">
          {/* The rendered structure with its marks drawn over it. The picture is 1238 by 768 and
              A and B are at the middle of its promoter and its enhancer. */}
          <svg className="hero-art" viewBox="0 0 1238 768" role="img" aria-label="A 3D structure of the TNFRSF8 locus with its promoter, A, its enhancer, B and the distance between them">
            <image href={structure} width="1238" height="768" />
            <line className="hero-ab" x1={HERO.a[0]} y1={HERO.a[1]} x2={HERO.b[0]} y2={HERO.b[1]} />
            <rect className="hero-box" x={HERO.box[0] - 62} y={HERO.box[1] - 23} width="124" height="46" rx="9" />
            <text className="hero-distance" x={HERO.box[0]} y={HERO.box[1] + 9}>
              {spread(WITH)}
            </text>
            {(
              [
                ['A', HERO.a, HERO.letterA],
                ['B', HERO.b, HERO.letterB],
              ] as const
            ).map(([name, at, letter]) => (
              <g key={name}>
                <circle className="hero-point" cx={at[0]} cy={at[1]} r="8" />
                <circle className="hero-end" cx={letter[0]} cy={letter[1]} r="21" />
                <text className="hero-end-text" x={letter[0]} y={letter[1] + 9}>
                  {name}
                </text>
              </g>
            ))}
          </svg>
          <figcaption>
            <span className="key">
              <span className="dot promoter" /> A, the TNFRSF8 promoter
            </span>
            <span className="key">
              <span className="dot enhancer" /> B, its enhancer 385 kb away
            </span>
            <span className="key">{`A to B ${spread(WITH)} bead units`}</span>
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
          <button type="button" className="example story" disabled={!!busy} onClick={() => open(ALL.find((x) => x.id === 'TNFRSF8')!)}>
            <span className="story-panels">
              {STORY.map((p) => (
                <span key={p.tag} className="story-panel">
                  <StructurePreview data={p.data} view={STORY_VIEW} label={`A structure of the TNFRSF8 locus, ${p.tag}, with the promoter A and the enhancer B ${spread(p.data)} bead units apart`} />
                  <span className="story-tag">{p.tag}</span>
                </span>
              ))}
            </span>
            <span className="example-title">TNFRSF8 with and without its RNAPII loop</span>
            <span className="example-detail">
              {busy === 'TNFRSF8'
                ? 'Opening the job'
                : `GM12878. From the promoter A to the enhancer B ${spread(WITH)} bead units with the loop and ${spread(WITHOUT)} with CTCF loops alone`}
            </span>
          </button>
          <button type="button" className="example" disabled={!!busy} onClick={() => open(ALL.find((x) => x.id === 'contacts')!)}>
            <img src={contacts} alt="" />
            <span className="example-title">Contacts against Hi-C</span>
            <span className="example-detail">{busy === 'contacts' ? 'Opening the job' : 'GM12878, chr1:20-25 Mb'}</span>
          </button>
        </div>
      </section>

      <p className="lab">
        The Laboratory of Functional and Structural Genomics at Warsaw University of Technology and the University of Warsaw develops 3D-GNOMEᴺᴳ, the successor of 3D-GNOME 1.0, 2.0
        and 3.0.
      </p>
    </div>
  )
}
