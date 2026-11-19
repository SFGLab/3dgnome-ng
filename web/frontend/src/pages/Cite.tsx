import { Copy, ExternalLink, Github, Type } from 'lucide-react'
import { ReferenceEntry } from '../components/ReferenceEntry'
import { DATA_SOURCES, REFERENCES, referenceLink } from '../content/references'
import { hidden } from '../flags'
import '../styles/doc.css'
import './cite.css'

const PRIMARY =
  'Authors and title to be set. 3D-GNOMEᴺᴳ: 3D chromatin models from CTCF and RNAPII loops, contacts and structural variants on the GPU. Manuscript in preparation.'
const PRIMARY_BIBTEX = `@unpublished{gnome3d_ng,
  title  = {3D-GNOME\\textsuperscript{NG}: 3D chromatin models from CTCF and RNAPII loops, contacts and structural variants on the GPU},
  author = {Authors to be set},
  note   = {Manuscript in preparation}
}`

const SECTIONS = [
  ['cite-gnome3d', 'Cite 3D-GNOMEᴺᴳ'],
  ['methods', '3D-GNOME and its methods'],
  ['algorithms', 'Algorithms and measures'],
  ['data', 'Data on this server'],
  ['related', 'Related tools'],
  ['contact', 'Contact'],
] as const

function copy(text: string) {
  void navigator.clipboard?.writeText(text)
}

/** Every citation for models made on this server. */
export function Cite() {
  const group = (g: string) => REFERENCES.filter((r) => r.group === g)
  return (
    <div className="doc-page">
      <aside className="side-panel">
        <p className="eyebrow">On this page</p>
        {SECTIONS.map(([id, label]) => (
          <a key={id} href={`#${id}`}>
            {label}
          </a>
        ))}
      </aside>
      <div className="reading">
        <header className="page-head">
          <h1>Cite</h1>
          <p className="lead">
            This page lists every citation for models made on this server. Cite 3D-GNOMEᴺᴳ, the earlier 3D-GNOME papers and the data that your models used.
          </p>
        </header>

        <section id="cite-gnome3d">
          <h2>
            Cite 3D-GNOME<sup className="ng">NG</sup>
          </h2>
          <div className="primary-citation">
            <p>{PRIMARY}</p>
            <div className="buttons">
              <button type="button" onClick={() => copy(PRIMARY_BIBTEX)}>
                <Copy size={13} aria-hidden /> Copy BibTeX
              </button>
              <button type="button" onClick={() => copy(PRIMARY)}>
                <Type size={13} aria-hidden /> Copy text
              </button>
              <a className="button-link" href="https://github.com/SFGLab/3dgnome-ng">
                <Github size={13} aria-hidden /> Source code
              </a>
            </div>
          </div>
        </section>

        <section id="methods">
          <h2>3D-GNOME and its methods</h2>
          <p className="intro">
            3D-GNOMEᴺᴳ reimplements the 3D-GNOME multiscale Monte Carlo method and its GPU version, cudaMMC. Its loop rewiring at structural variants extends the rules of Sadowski and colleagues. enhancer3D introduced the enhancer to promoter distance profiles on the job page.
          </p>
          {group('gnome').map((r) => (
            <ReferenceEntry key={r.key} r={r} />
          ))}
        </section>

        <section id="algorithms">
          <h2>Algorithms and measures</h2>
          <p className="intro">{`${hidden('method') ? 'The method builds on these' : 'The Method page cites these'} for the multiscale approach, the distance law, the optimisers and the agreement measure.`}</p>
          {group('algorithms').map((r) => (
            <ReferenceEntry key={r.key} r={r} />
          ))}
        </section>

        <section id="data">
          <h2>Data on this server</h2>
          <p className="intro">
            {`Every data set behind the models${hidden('results') ? '' : ' and the Results page'}, with the accession of its archive.`} A row lists the experiment or the set first and its files after it. An entry that says to confirm is not recorded yet.
          </p>
          <table className="data-table">
            <thead>
              <tr>
                <th>Data</th>
                <th>Accession</th>
                <th>Used for</th>
                <th>Cite</th>
              </tr>
            </thead>
            <tbody>
              {DATA_SOURCES.map((d) => {
                const r = REFERENCES.find((x) => x.key === d.ref)
                return (
                  <tr key={d.data}>
                    <td className="data-name">
                      {d.data}
                      {d.note && <span className="data-note">{d.note}</span>}
                    </td>
                    <td className="accessions">
                      {d.accessions.map((a) => (
                        <a key={a.id} href={a.url}>
                          {a.id}
                        </a>
                      ))}
                      {d.pending && <span className="unconfirmed">{d.pending}</span>}
                      {d.accessions.length === 0 && !d.pending && <span className="no-accession">none</span>}
                    </td>
                    <td>{d.usedFor}</td>
                    <td>{r && !r.unconfirmed ? <a href={referenceLink(r)}>{d.label}</a> : <span className="unconfirmed">{d.label}</span>}</td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </section>

        <section id="related">
          <h2>Related tools</h2>
          {group('related').map((r) => (
            <ReferenceEntry key={r.key} r={r} />
          ))}
        </section>

        <section id="contact">
          <h2>Contact</h2>
          <p className="intro">
            Laboratory of Functional and Structural Genomics, led by Dariusz Plewczynski, at the Faculty of Mathematics and Information Science, Warsaw University of Technology and the Centre of New Technologies, University of Warsaw.
          </p>
          <a className="lab-link" href="https://sfglab.mini.pw.edu.pl">
            <ExternalLink size={13} aria-hidden /> sfglab.mini.pw.edu.pl
          </a>
        </section>
      </div>
    </div>
  )
}
