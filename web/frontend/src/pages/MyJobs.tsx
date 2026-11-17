import { useQueries } from '@tanstack/react-query'
import { ArrowRight } from 'lucide-react'
import { Link, useSearchParams } from 'react-router-dom'
import { api, type JobStatus } from '../api'
import { useRecentJobs } from '../model/useRecentJobs'
import { ROUTES } from '../routes'
import { SyncPanel } from '../sync/SyncPanel'
import { syncState, useSyncStatus } from '../sync/sync'
import '../styles/doc.css'
import './myjobs.css'

function stateText(s: JobStatus | undefined): string {
  if (!s) return ''
  if (s.state === 'running' && s.stage) return `running, ${s.stage} stage`
  return s.state
}

const when = new Intl.DateTimeFormat('en-GB', { dateStyle: 'medium', timeStyle: 'short' })

/** The jobs of this browser, newest first, with their state, and the sync of the list. */
export function MyJobs() {
  const jobs = useRecentJobs()
  const [params, setParams] = useSearchParams()
  useSyncStatus()
  const sync = syncState()
  const statuses = useQueries({
    queries: jobs.map((j) => ({ queryKey: ['status', j.url], queryFn: () => api.status(j.url), refetchInterval: 15000, retry: false })),
  })
  return (
    <div className="doc-page my-jobs">
      <div className="reading">
        <header className="page-head">
          <h1>My jobs</h1>
          <p className="lead">
            {sync
              ? 'Your jobs, newest first. The list is shared with every computer that has its sync code.'
              : 'The jobs started in this browser, newest first. The list is stored in this browser only. A sync code shares it with your other computers.'}
          </p>
        </header>
        <SyncPanel initialCode={params.get('sync') ?? ''} onCodeUsed={() => setParams({}, { replace: true })} />
        {jobs.length === 0 ? (
          <div className="no-jobs">
            <p>You have no jobs in this browser yet.</p>
            <Link className="button primary" to={ROUTES.model}>
              Model a region <ArrowRight size={16} aria-hidden />
            </Link>
          </div>
        ) : (
          <table className="jobs-table">
            <thead>
              <tr>
                <th>Job</th>
                <th>State</th>
                <th>Started</th>
              </tr>
            </thead>
            <tbody>
              {jobs.map((j, i) => (
                <tr key={j.url}>
                  <td>
                    <Link to={j.url} reloadDocument>
                      {j.name || j.label}
                    </Link>
                    {j.name && <span className="job-label">{j.label}</span>}
                    {sync && !sync.private && j.url.startsWith('/p/') && <span className="job-label">private, on this computer only</span>}
                  </td>
                  <td>{statuses[i]?.isError ? 'removed or not found' : stateText(statuses[i]?.data)}</td>
                  <td>{when.format(j.at)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  )
}
