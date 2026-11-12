import { useQuery } from '@tanstack/react-query'
import { useMemo, useState } from 'react'
import { useParams, useSearchParams } from 'react-router-dom'
import { jobApi } from '../job/api'
import { JobSidePanel } from '../job/SidePanel'
import { ContactMapTab } from '../job/ContactMapTab'
import { GenesPanel } from '../job/GenesPanel'
import { LogTab } from '../job/LogTab'
import { LoopsTab } from '../job/LoopsTab'
import { StructureTab } from '../job/StructureTab'
import { JobTabs, type Tab } from '../job/Tabs'
import './job.css'

/** The explorer of one job: its side panel, its tabs, and its state while it is not done. */
export function Job({ kind, pollMs = 5000 }: { kind: 'jobs' | 'p'; pollMs?: number }) {
  const { key = '' } = useParams()
  const api = useMemo(() => jobApi(kind, key), [kind, key])
  const [params, setParams] = useSearchParams()
  const tab = (params.get('tab') as Tab) ?? 'structure'
  const [selected, setSelected] = useState(1)
  const [comparing, setComparing] = useState(false)
  const [gene, setGene] = useState<string | null>(null)
  const [geneFilter, setGeneFilter] = useState('')
  const summary = useQuery({
    queryKey: ['summary', api.base],
    queryFn: api.summary,
    refetchInterval: (q) => (q.state.data && ['done', 'failed'].includes(q.state.data.state) ? false : pollMs),
  })
  if (summary.isError) return <div className="job-page"><p className="job-message">{summary.error.message}</p></div>
  if (!summary.data) return <div className="job-page" aria-busy="true" />
  const s = summary.data
  const pick = async (which: 'medoid' | 'compact' | 'random') => setSelected((await api.picks())[which])
  return (
    <div className="job-page">
      <JobSidePanel api={api} summary={s} selected={selected} onSelect={setSelected} onPick={(w) => void pick(w)} />
      <div className="job-main">
        {s.state === 'done' ? (
          <>
            <JobTabs
              tab={tab}
              counts={{ genes: s.genes, loops: s.loop_count }}
              comparing={comparing}
              zipUrl={api.zipUrl}
              onTab={(t) => setParams({ tab: t }, { replace: true })}
              onCompare={() => setComparing((c) => !c)}
            />
            <div className="tab-body" role="tabpanel">
              {tab === 'structure' && (
                <StructureTab
                  api={api}
                  summary={s}
                  selected={selected}
                  gene={gene}
                  onGene={setGene}
                  onOpenGenes={(g) => {
                    setGeneFilter(g)
                    setParams({ tab: 'genes' }, { replace: true })
                  }}
                />
              )}
              {tab === 'genes' && (
                <GenesPanel
                  api={api}
                  summary={s}
                  filter={geneFilter}
                  selected={gene}
                  onSelect={(g) => {
                    setGene(g)
                    setParams({ tab: 'structure' }, { replace: true })
                  }}
                />
              )}
              {tab === 'contacts' && <ContactMapTab api={api} summary={s} />}
              {tab === 'loops' && <LoopsTab api={api} />}
              {tab === 'log' && <LogTab api={api} />}
            </div>
          </>
        ) : s.state === 'failed' ? (
          <div className="job-message">
            <p className="field-error">{s.error}</p>
            {s.log_tail && <pre className="log-tail">{s.log_tail}</pre>}
          </div>
        ) : (
          <div className="job-message">
            <p>The job page refreshes itself and opens the structures when the job is done.</p>
          </div>
        )}
      </div>
    </div>
  )
}
