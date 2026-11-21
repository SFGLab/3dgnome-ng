import { Link } from 'react-router-dom'
import { EXAMPLES, type Example } from './examples'
import { useJobStates } from '../myjobs/useJobStates'
import { useRecentJobs } from '../myjobs/useRecentJobs'
import { scoped } from '../lib/css'
import sheet from './model.module.css'

const cn = scoped(sheet)

/** The examples and the recent jobs of this browser. */
export function SidePanel({ onExample }: { onExample: (e: Example) => void }) {
  const jobs = useRecentJobs()
  const states = useJobStates(jobs)
  return (
    <aside className={cn('side-panel model-side')}>
      <p className="eyebrow">Start from an example</p>
      {EXAMPLES.map((e) => (
        <button key={e.id} type="button" className={cn('example-card')} onClick={() => onExample(e)}>
          <span className={cn('example-top')}>
            <span className={cn('example-gene')}>{e.title}</span>
            <span className={cn('example-cell')}>{e.cellLabel}</span>
          </span>
          <span className={cn('example-desc')}>{e.description}</span>
        </button>
      ))}
      {jobs.length > 0 && (
        <>
          <p className="eyebrow">Your recent jobs</p>
          {jobs.map((j, i) => (
            <Link key={j.url} className="recent-job" to={j.url} reloadDocument>
              <span className={cn('recent-region')}>{j.name || j.label}</span>
              <span className={cn('recent-state')}>{states[i]?.text}</span>
            </Link>
          ))}
        </>
      )}
      <p className={cn('retention')}>Jobs are kept for 14 days. A job with an uploaded VCF is private and reachable only through its link.</p>
    </aside>
  )
}
