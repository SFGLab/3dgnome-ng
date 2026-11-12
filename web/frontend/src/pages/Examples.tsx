import { ArrowRight } from 'lucide-react'
import { EXAMPLES } from '../model/examples'
import { EXAMPLE_STRUCTURES } from '../model/openExample'
import { useOpenExample } from '../model/useOpenExample'
import '../styles/doc.css'
import './examples.css'

/** Every example, each opening its finished job. */
export function Examples() {
  const { open, busy, error } = useOpenExample()
  return (
    <div className="doc-page examples-page">
      <div className="reading">
        <header className="page-head">
          <h1>Examples</h1>
          <p className="lead">{`Each example opens its finished job of ${EXAMPLE_STRUCTURES} structures. The first visitor waits about a minute while it runs, and everyone after opens it at once.`}</p>
        </header>
        {error && (
          <p className="field-error" role="alert">
            {error}
          </p>
        )}
        <div className="example-list">
          {EXAMPLES.map((e) => (
            <button key={e.id} type="button" className="example-item" disabled={!!busy} aria-label={`Open ${e.title}`} onClick={() => open(e)}>
              <span className="example-top">
                <span className="example-gene">{e.title}</span>
                <span className="example-cell">{e.cellLabel}</span>
              </span>
              <span className="example-desc">{e.description}</span>
              <span className="example-open">
                {busy === e.id ? 'Opening the job' : 'Open the job'} <ArrowRight size={14} aria-hidden />
              </span>
            </button>
          ))}
        </div>
      </div>
    </div>
  )
}
