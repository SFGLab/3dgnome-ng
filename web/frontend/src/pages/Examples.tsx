import { ArrowRight } from 'lucide-react'
import { EXAMPLES } from '../model/examples'
import { EXAMPLE_STRUCTURES } from '../model/openExample'
import { useOpenExample } from '../model/useOpenExample'
import '../styles/doc.css'
import { scoped } from '../lib/css'
import sheet from './examples.module.css'

const cn = scoped(sheet)

/** Every example, each opening its finished job. */
export function Examples() {
  const { open, busy, error } = useOpenExample()
  return (
    <div className={cn('doc-page examples-page')}>
      <div className="reading">
        <header className="page-head">
          <h1>Examples</h1>
          <p className="lead">{`Each example opens its finished job of ${EXAMPLE_STRUCTURES} structures. If nobody has opened an example before, its job runs first and takes about a minute.`}</p>
        </header>
        {error && (
          <p className="field-error" role="alert">
            {error}
          </p>
        )}
        <div className={cn('example-list')}>
          {EXAMPLES.map((e) => (
            <button key={e.id} type="button" className={cn('example-item')} disabled={!!busy} aria-label={`Open ${e.title}`} onClick={() => open(e)}>
              <span className={cn('example-top')}>
                <span className={cn('example-gene')}>{e.title}</span>
                <span className={cn('example-cell')}>{e.cellLabel}</span>
              </span>
              <span className={cn('example-desc')}>{e.description}</span>
              <span className={cn('example-open')}>
                {busy === e.id ? 'Opening the job' : 'Open the job'} <ArrowRight size={14} aria-hidden />
              </span>
            </button>
          ))}
        </div>
      </div>
    </div>
  )
}
