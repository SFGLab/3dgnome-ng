import { ArrowRight } from 'lucide-react'
import { Link, useSearchParams } from 'react-router-dom'
import { useJobStates } from '../myjobs/useJobStates'
import { useRecentJobs } from '../myjobs/useRecentJobs'
import { ROUTES } from '../routes'
import { SyncPanel } from '../myjobs/SyncPanel'
import { syncState, useSyncStatus } from '../myjobs/sync'
import '../styles/doc.css'
import { scoped } from '../lib/css'
import sheet from './myjobs.module.css'

const cn = scoped(sheet)

const when = new Intl.DateTimeFormat('en-GB', { dateStyle: 'medium', timeStyle: 'short' })

/** The jobs of this browser, newest first, with their state and the sync of the list. */
export function MyJobs() {
  const jobs = useRecentJobs()
  const [params, setParams] = useSearchParams()
  useSyncStatus()
  const sync = syncState()
  const states = useJobStates(jobs)
  return (
    <div className={cn('doc-page my-jobs')}>
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
          <div className={cn('no-jobs')}>
            <p>You have no jobs in this browser yet.</p>
            <Link className="button primary" to={ROUTES.model}>
              Model a region <ArrowRight size={16} aria-hidden />
            </Link>
          </div>
        ) : (
          <table className={cn('jobs-table')}>
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
                    {j.name && <span className={cn('job-label')}>{j.label}</span>}
                    {sync && !sync.private && j.url.startsWith('/p/') && <span className={cn('job-label')}>private, on this computer only</span>}
                  </td>
                  <td>{states[i]?.missing ? 'removed or not found' : states[i]?.text}</td>
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
