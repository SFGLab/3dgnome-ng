import { useQueries } from '@tanstack/react-query'
import { Link } from 'react-router-dom'
import { api, type JobStatus } from '../api'
import { EXAMPLES, type Example } from './examples'
import { recentJobs } from './recent'

function stateText(s: JobStatus | undefined): string {
  if (!s) return ''
  if (s.state === 'running' && s.stage) return `running, ${s.stage} stage`
  return s.state
}

/** The examples and the recent jobs of this browser. */
export function SidePanel({ onExample }: { onExample: (e: Example) => void }) {
  const jobs = recentJobs()
  const statuses = useQueries({
    queries: jobs.map((j) => ({ queryKey: ['status', j.url], queryFn: () => api.status(j.url), refetchInterval: 15000, retry: false })),
  })
  return (
    <aside className="side-panel model-side">
      <p className="eyebrow">Start from an example</p>
      {EXAMPLES.map((e) => (
        <button key={e.id} type="button" className="example-card" onClick={() => onExample(e)}>
          <span className="example-top">
            <span className="example-gene">{e.title}</span>
            <span className="example-cell">{e.cellLabel}</span>
          </span>
          <span className="example-desc">{e.description}</span>
        </button>
      ))}
      {jobs.length > 0 && (
        <>
          <p className="eyebrow">Your recent jobs</p>
          {jobs.map((j, i) => (
            <Link key={j.url} className="recent-job" to={j.url} reloadDocument>
              <span className="recent-region">{j.name || j.label}</span>
              <span className="recent-state">{stateText(statuses[i]?.data)}</span>
            </Link>
          ))}
        </>
      )}
      <p className="retention">Jobs are kept for 14 days. A job with an uploaded VCF is private and reachable only through its link.</p>
    </aside>
  )
}
