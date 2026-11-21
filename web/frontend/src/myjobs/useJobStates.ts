/** The state of each job of a list, asked of the server every 15 seconds. */
import { asked, type JobStatus } from '../api'
import { useQueries } from '@tanstack/react-query'
import type { RecentJob } from './recent'

/** Returns the state of a job in words, with the stage of a running job. Empty before the first answer. */
export function stateText(s: JobStatus | undefined): string {
  if (!s) return ''
  if (s.state === 'running' && s.stage) return `running, ${s.stage} stage`
  return s.state
}

/** Returns for each job its state in words and whether the server has the job. */
export function useJobStates(jobs: RecentJob[]): { text: string; missing: boolean }[] {
  const statuses = useQueries({
    queries: jobs.map((j) => ({ ...asked.status(j.url), refetchInterval: 15000, retry: false })),
  })
  return statuses.map((s) => ({ text: stateText(s.data), missing: s.isError }))
}
