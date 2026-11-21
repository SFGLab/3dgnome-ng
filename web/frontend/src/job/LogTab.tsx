import { askedOf } from '../api'
import { useQuery } from '@tanstack/react-query'
import type { JobApi } from '../api'

/** The Log tab: the engine's log of the job. */
export function LogTab({ api }: { api: JobApi }) {
  const log = useQuery(askedOf(api).log())
  return <pre className="log-tail full">{log.data ?? ''}</pre>
}
