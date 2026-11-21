import { askedOf } from '../api'
import { useQuery } from '@tanstack/react-query'
import type { JobApi } from '../api'
import { Unanswered } from '../components/Unanswered'
import { scoped } from '../lib/css'
import sheet from './job.module.css'

const cn = scoped(sheet)

/** The Log tab: the engine's log of the job. */
export function LogTab({ api }: { api: JobApi }) {
  const log = useQuery(askedOf(api).log())
  if (log.isError) return <Unanswered of={log} />
  return <pre className={cn('log-tail full')}>{log.data ?? ''}</pre>
}
