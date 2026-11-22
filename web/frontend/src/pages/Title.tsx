import { asked } from '../api'
import { useQuery } from '@tanstack/react-query'
import { ArrowRight, Database, ExternalLink, History, Play } from 'lucide-react'
import { Link } from 'react-router-dom'
import contacts from '../assets/contacts.png'
import previews from '../assets/previews.json'
import deletion from '../assets/previews_deletion.json'
import expression from '../assets/previews_expression.json'
import structure from '../assets/structure.png'
import { StructurePreview, type PreviewData } from '../components/StructurePreview'
import { EXAMPLES as ALL } from '../model/examples'
import { ROUTES } from '../routes'
import { useOpenExample } from '../model/useOpenExample'
import { scoped } from '../lib/css'
import sheet from './title.module.css'

const cn = scoped(sheet)

/** The drawings of the TNFRSF8 locus with and without RNAPII loops. A is the promoter and B the
 * enhancer 385 kb away. web/scripts/make_previews.py writes them from the two example jobs. */
const WITH = previews.rnapii as PreviewData
const WITHOUT = previews.ctcf as PreviewData
const spread = (d: PreviewData) => `${d.mean.toFixed(1)} ± ${d.sd.toFixed(1)}`
/**
 * The picture of the hero in its own pixels, with a point on the promoter and one on the enhancer,
 * the letter beside each on an empty part of the picture and the middle of the box of the distance.
 */
const HERO = { a: [354, 397], b: [705, 196], letterA: [350, 452], letterB: [668, 150], box: [525, 232] } as const

/** The part of each drawing the example shows. Both have one view and the two lines can be compared. */
const STORY_VIEW: [number, number, number, number] = [-170, -372, 770, 764]

/** The two drawings of the first example, the tag over each and the words a screen reader gets. */
const STORY = [
  { tag: 'CTCF and RNAPII loops', data: WITH },
  { tag: 'CTCF loops alone', data: WITHOUT },
]

/**
 * The drawing of the deletion example, GM12878 with the variants of HG00512. A and B are the promoters of GSTM2
 * and GSTM5, the genes either side of the deleted GSTM1.
 */
const DELETION = deletion.deletion as PreviewData
const DELETION_VIEW: [number, number, number, number] = [-240, -362, 600, 704]

/**
 * The drawings of the expression example, VPS13D in two cell types, each with the enhancer nearest to it in
 * that cell type, 20 kb away in GM12878 and 849 bp away in H1ESC.
 */
const CELLS = [
  { tag: 'GM12878', data: expression.gm12878 as PreviewData },
  { tag: 'H1ESC', data: expression.h1esc as PreviewData },
]
/** A view of one size about the middle of each drawing. The two are on one scale and each is centred in its panel. */
const CELLS_SIZE = [
  Math.max(...CELLS.map((c) => Math.max(...c.data.points.map((p) => p[0])) - Math.min(...c.data.points.map((p) => p[0])))) + 60,
  Math.max(...CELLS.map((c) => Math.max(...c.data.points.map((p) => p[1])) - Math.min(...c.data.points.map((p) => p[1])))) + 60,
]
const centred = (d: PreviewData): [number, number, number, number] => {
  const xs = d.points.map((p) => p[0])
  const ys = d.points.map((p) => p[1])
  return [(Math.min(...xs) + Math.max(...xs) - CELLS_SIZE[0]) / 2, (Math.min(...ys) + Math.max(...ys) - CELLS_SIZE[1]) / 2, CELLS_SIZE[0], CELLS_SIZE[1]]
}
/** The expression of VPS13D in H1ESC over GM12878, from DESeq2 on the ENCODE counts of the site, read on 2026-11-21. */
const VPS13D_FOLD = 1.7

/** The front page, with the output of the site, the examples and the previous services. */
export function Title() {
  const { open, busy, error } = useOpenExample()
  // The server gives the longest region of a job. 20 Mb is shown until the server answers.
  const catalog = useQuery(asked.catalog())
  const maxMb = catalog.data?.max_region_mb ?? 20
  return (
    <div className={cn('title-page')}>
      <section className={cn('hero')}>
        <div className={cn('hero-text')}>
          <h1 className={cn('hero-name')}>
            3D-GNOME<sup className="ng">NG</sup>
          </h1>
          <p className="lead">
            {`3D-GNOMEᴺᴳ computes 3D structures of genomic regions up to ${maxMb} Mb from CTCF`} and RNAPII loops, Hi-C contacts and structural variants. Each job returns an ensemble of
            structures with promoters and enhancers marked and the 3D distances between them.
          </p>
          <p className={cn('hero-terms')}>
            The server is free to use for all, your data stays private. 3D-GNOMEᴺᴳ is free software under the{' '}
            <a href="https://www.gnu.org/licenses/gpl-3.0.html">GNU General Public License, version 3</a>.
          </p>
          <div className={cn('buttons')}>
            <Link className="button primary" to={ROUTES.model}>
              Model a region <ArrowRight size={16} aria-hidden />
            </Link>
            <Link className="button secondary" to={`${ROUTES.model}?example=TNFRSF8`}>
              <Play size={14} aria-hidden /> Open the TNFRSF8 example
            </Link>
          </div>
        </div>
        <figure className={cn('hero-image')}>
          {/* The rendered structure with its marks drawn over it. The picture is 1238 by 768 and
              A and B are at the middle of its promoter and its enhancer. */}
          <svg className={cn('hero-art')} viewBox="0 0 1238 768" role="img" aria-label="A 3D structure of the TNFRSF8 locus with its promoter, A, its enhancer, B and the distance between them">
            <image href={structure} width="1238" height="768" />
            <line className={cn('hero-ab')} x1={HERO.a[0]} y1={HERO.a[1]} x2={HERO.b[0]} y2={HERO.b[1]} />
            <rect className={cn('hero-box')} x={HERO.box[0] - 62} y={HERO.box[1] - 23} width="124" height="46" rx="9" />
            <text className={cn('hero-distance')} x={HERO.box[0]} y={HERO.box[1] + 9}>
              {spread(WITH)}
            </text>
            {(
              [
                ['A', HERO.a, HERO.letterA],
                ['B', HERO.b, HERO.letterB],
              ] as const
            ).map(([name, at, letter]) => (
              <g key={name}>
                <circle className={cn('hero-point')} cx={at[0]} cy={at[1]} r="8" />
                <circle className={cn('hero-end')} cx={letter[0]} cy={letter[1]} r="21" />
                <text className={cn('hero-end-text')} x={letter[0]} y={letter[1] + 9}>
                  {name}
                </text>
              </g>
            ))}
          </svg>
          <figcaption>
            <span className={cn('key')}>
              <span className={cn('dot promoter')} /> A, the TNFRSF8 promoter
            </span>
            <span className={cn('key')}>
              <span className={cn('dot enhancer')} /> B, its enhancer 385 kb away
            </span>
            <span className={cn('key')}>{`A to B ${spread(WITH)} bead units`}</span>
          </figcaption>
        </figure>
      </section>

      <section className={cn('previous')}>
        <div className={cn('service')}>
          <History size={16} aria-hidden />
          <p>3D-GNOME 3.0, the previous server with its 1000 Genomes structures, is still available at 3dgnome.mini.pw.edu.pl/old.</p>
          <a href="https://3dgnome.mini.pw.edu.pl/old">
            Open 3D-GNOME 3.0 <ExternalLink size={13} aria-hidden />
          </a>
        </div>
        <div className={cn('service')}>
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

      <section className={cn('examples')}>
        <div className={cn('examples-head')}>
          <h2>Examples</h2>
          <Link to={ROUTES.examples}>All examples</Link>
        </div>
        {error && (
          <p className="field-error" role="alert">
            {error}
          </p>
        )}
        <div className={cn('example-row')}>
          <button type="button" className={cn('example story')} disabled={!!busy} onClick={() => open(ALL.find((x) => x.id === 'TNFRSF8')!)}>
            <span className={cn('story-panels')}>
              {STORY.map((p) => (
                <span key={p.tag} className={cn('story-panel')}>
                  <StructurePreview data={p.data} view={STORY_VIEW} label={`A structure of the TNFRSF8 locus, ${p.tag}, with the promoter A and the enhancer B ${spread(p.data)} bead units apart`} />
                  <span className={cn('story-tag')}>{p.tag}</span>
                </span>
              ))}
            </span>
            <span className={cn('example-title')}>TNFRSF8 with and without its RNAPII loop</span>
            <span className={cn('example-detail')}>
              {busy === 'TNFRSF8'
                ? 'Opening the job'
                : `GM12878. From the promoter A to the enhancer B ${spread(WITH)} bead units with the loop and ${spread(WITHOUT)} with CTCF loops alone`}
            </span>
          </button>
          <button type="button" className={cn('example')} disabled={!!busy} onClick={() => open(ALL.find((x) => x.id === 'contacts')!)}>
            <img src={contacts} alt="" />
            <span className={cn('example-title')}>Contacts against Hi-C</span>
            <span className={cn('example-detail')}>{busy === 'contacts' ? 'Opening the job' : 'GM12878, chr1:20-25 Mb'}</span>
          </button>
        </div>
        <div className={cn('example-row')}>
          <button type="button" className={cn('example')} disabled={!!busy} onClick={() => open(ALL.find((x) => x.id === 'GSTM1')!)}>
            <span className={cn('story-panels one')}>
              <span className={cn('story-panel')}>
                <StructurePreview data={DELETION} view={DELETION_VIEW} label={`A structure of the GSTM locus with the deletion of GSTM1, with the promoters of GSTM2, A and of GSTM5, B ${spread(DELETION)} bead units apart`} />
                <span className={cn('story-tag')}>HG00512</span>
              </span>
            </span>
            <span className={cn('example-title')}>Deletion at GSTM1</span>
            <span className={cn('example-detail')}>
              {busy === 'GSTM1' ? 'Opening the job' : 'GM12878, the 18 kb deletion of GSTM1 in HG00512'}
            </span>
          </button>
          <button type="button" className={cn('example story')} disabled={!!busy} onClick={() => open(ALL.find((x) => x.id === 'VPS13D')!)}>
            <span className={cn('story-panels')}>
              {CELLS.map((p) => (
                <span key={p.tag} className={cn('story-panel')}>
                  <StructurePreview data={p.data} view={centred(p.data)} label={`A structure of the VPS13D locus in ${p.tag}, with the promoter A and its nearest enhancer B ${spread(p.data)} bead units apart`} />
                  <span className={cn('story-tag')}>{p.tag}</span>
                </span>
              ))}
            </span>
            <span className={cn('example-title')}>VPS13D in GM12878 and H1ESC, distance beside expression</span>
            <span className={cn('example-detail')}>
              {busy === 'VPS13D'
                ? 'Opening the jobs'
                : `From the promoter A to the nearest enhancer B of each cell type ${spread(CELLS[0].data)} and ${spread(CELLS[1].data)} bead units, expression ${VPS13D_FOLD} times higher in H1ESC`}
            </span>
          </button>
        </div>
      </section>

      <p className={cn('lab')}>
        The Laboratory of Functional and Structural Genomics at Warsaw University of Technology and the University of Warsaw develops 3D-GNOMEᴺᴳ, the successor of 3D-GNOME 1.0, 2.0
        and 3.0.
      </p>
    </div>
  )
}
