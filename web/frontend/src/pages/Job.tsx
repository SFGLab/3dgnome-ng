import { useQuery } from '@tanstack/react-query'
import { useMemo, useState } from 'react'
import { useParams, useSearchParams } from 'react-router-dom'
import { jobApi, type Loop } from '../job/api'
import { JobSidePanel } from '../job/SidePanel'
import { Compare } from '../job/Compare'
import { Progress } from '../job/Progress'
import { ContactMapTab } from '../job/ContactMapTab'
import { GenesPanel } from '../job/GenesPanel'
import { LogTab } from '../job/LogTab'
import { LoopsTab } from '../job/LoopsTab'
import { StructureTab } from '../job/StructureTab'
import { JobTabs, type Tab } from '../job/Tabs'
import './job.css'

/** The explorer of one job: its side panel, its tabs and its state while it is not done. */
export function Job({ kind, pollMs = 5000 }: { kind: 'jobs' | 'p'; pollMs?: number }) {
  const { key = '' } = useParams()
  const api = useMemo(() => jobApi(kind, key), [kind, key])
  const [params, setParams] = useSearchParams()
  const tab = (params.get('tab') as Tab) ?? 'structure'
  const [selected, setSelected] = useState(1)
  // The tab and the comparison are history entries in the URL. Back returns to the previous tab
  // or closes the comparison.
  const compareTo = params.get('compare')
  const comparing = compareTo !== null
  const go = (next: Record<string, string>) => setParams(next)
  const [gene, setGene] = useState<string | null>(params.get('gene'))
  const [geneFilter, setGeneFilter] = useState('')
  const [loop, setLoop] = useState<Loop | null>(null)
  const summary = useQuery({
    queryKey: ['summary', api.base],
    queryFn: api.summary,
    refetchInterval: (q) => (q.state.data && ['done', 'failed'].includes(q.state.data.state) ? false : pollMs),
  })
  if (summary.isError)
    return (
      <div className="job-page">
        <p className="job-message">{summary.error.message}</p>
      </div>
    )
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
              onTab={(t) => go({ tab: t })}
              onCompare={() => go(comparing ? { tab } : { tab, compare: '' })}
            />
            <div className="tab-body" role="tabpanel">
              {comparing && (
                <Compare
                  key={compareTo}
                  api={api}
                  summary={s}
                  selected={selected}
                  gene={gene}
                  onGene={setGene}
                  initial={compareTo || null}
                  onOther={(url) => go({ tab, compare: url ?? '' })}
                  onClose={() => go({ tab })}
                />
              )}
              {!comparing && tab === 'structure' && (
                <StructureTab
                  api={api}
                  summary={s}
                  selected={selected}
                  gene={gene}
                  onGene={setGene}
                  loop={loop}
                  onLoop={setLoop}
                  onOpenGenes={(g) => {
                    setGeneFilter(g)
                    go({ tab: 'genes' })
                  }}
                />
              )}
              {!comparing && tab === 'genes' && (
                <GenesPanel
                  api={api}
                  summary={s}
                  filter={geneFilter}
                  selected={gene}
                  onSelect={(g) => {
                    setLoop(null)
                    setGene(g)
                    go({ tab: 'structure' })
                  }}
                />
              )}
              {!comparing && tab === 'contacts' && <ContactMapTab api={api} summary={s} />}
              {!comparing && tab === 'loops' && (
                <LoopsTab
                  api={api}
                  onShow={(l) => {
                    setGene(null)
                    setLoop(l)
                    go({ tab: 'structure' })
                  }}
                />
              )}
              {!comparing && tab === 'log' && <LogTab api={api} />}
            </div>
          </>
        ) : s.state === 'failed' ? (
          <div className="job-message">
            <p className="field-error">{s.error}</p>
            {s.log_tail && <pre className="log-tail">{s.log_tail}</pre>}
          </div>
        ) : (
          <div className="job-message">
            <Progress summary={s} />
          </div>
        )}
      </div>
    </div>
  )
}
