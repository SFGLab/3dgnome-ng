/** The state of each job of a list, asked of the server in one request. */
import { asked, type JobState } from '../api'
import { useQuery } from '@tanstack/react-query'
import type { RecentJob } from './recent'

/** The time between two requests while a job of the list waits or runs. */
const EVERY_MS = 15000

/** Returns the state of a job in words, with the stage of a running job. Empty before the first answer. */
export function stateText(s: JobState | null | undefined): string {
  if (!s) return ''
  if (s.state === 'running' && s.stage) return `running, ${s.stage} stage`
  return s.state
}

/**
 * Returns for each job its state in words and whether the server has the job. The list is asked
 * again every 15 seconds while one of its jobs waits or runs. A list of finished jobs is asked once.
 */
export function useJobStates(jobs: RecentJob[]): { text: string; missing: boolean }[] {
  const urls = jobs.map((j) => j.url)
  const states = useQuery({
    ...asked.states(urls),
    enabled: urls.length > 0,
    retry: false,
    refetchInterval: (q) => (Object.values(q.state.data ?? {}).some((s) => s && (s.state === 'queued' || s.state === 'running')) ? EVERY_MS : false),
  })
  return jobs.map((j) => {
    const s = states.data?.[j.url]
    return { text: stateText(s), missing: states.data !== undefined && !s }
  })
}
