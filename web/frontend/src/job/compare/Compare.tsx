import { Unanswered } from '../../components/Unanswered'
import { useQuery } from '@tanstack/react-query'
import { useMemo, useState } from 'react'
import { recentJobs } from '../../myjobs/recent'
import { compareExpression, compareJobs, jobApi, type JobApi, type Summary } from '../../api'
import { CompareResults } from './CompareResults'
import { sideNames } from './names'
import { makeCameraLink } from '../cameraLink'
import { parseJobUrl } from '../../routes'
import { PanelHead, Panel } from './Panel'
import { scoped } from '../../lib/css'
import sheet from '../job.module.css'

const cn = scoped(sheet)

/**
 * The compare view. This job and another are side by side, with the changes between them below,
 * `CompareResults`. The other side starts empty with its list of jobs open.
 */
export function Compare({
  api,
  summary,
  selected,
  gene,
  initial = null,
  onOther,
  onGene,
  onClose,
}: {
  api: JobApi
  summary: Summary
  selected: number
  gene: string | null
  initial?: string | null
  onOther?: (url: string | null) => void
  /** Called with the gene of a clicked row of the table. Both sides then mark its pair. A click
   * on the selected row, or on the button of the toolbar, calls it with null. */
  onGene?: (gene: string | null) => void
  onClose: () => void
}) {
  const [other, setOtherState] = useState<{ kind: 'jobs' | 'p'; key: string } | null>(initial ? parseJobUrl(initial) : null)
  const setOther = (o: { kind: 'jobs' | 'p'; key: string } | null) => {
    setOtherState(o)
    onOther?.(o ? `/${o.kind}/${o.key}` : null)
  }
  const [linked, setLinked] = useState(true)
  const link = useMemo(makeCameraLink, [])
  const otherApi = useMemo(() => (other ? jobApi(other.kind, other.key) : null), [other])
  const otherSummary = useQuery({ queryKey: ['summary', otherApi?.base], queryFn: () => otherApi!.summary(), enabled: !!otherApi })
  // Two cell types have two sets of enhancers. The comparison can keep to the enhancers both have. Positions are
  // compared as they are, which holds for two jobs without structural variants.
  const [sharedOnly, setSharedOnly] = useState(false)
  const o = otherSummary.data?.request
  const canShare = !!o && o.cell_type !== summary.request.cell_type && o.cell_type !== 'own' && summary.request.cell_type !== 'own' && !o.sample && !summary.request.sample
  const shared = sharedOnly && canShare
  const cmp = useQuery({ queryKey: ['compare', api.url, otherApi?.url, shared], queryFn: () => compareJobs(api.url, otherApi!.url, shared), enabled: !!otherApi })
  // The expression of the two jobs is compared when both have it. The test of two tables takes seconds the first time.
  const bothExpressed = !!summary.expression && !!otherSummary.data?.expression
  const expressed = useQuery({
    queryKey: ['compare-expression', api.url, otherApi?.url, shared],
    queryFn: () => compareExpression(api.url, otherApi!.url, shared),
    enabled: !!otherApi && bothExpressed,
    staleTime: Infinity,
  })
  const recent = recentJobs().filter((j) => j.url !== api.url)
  const choices = recent.filter((j) => j.url !== otherApi?.url).map((j) => ({ url: j.url, name: j.name || j.label }))
  const pick = (url: string) => setOther(parseJobUrl(url))
  const otherJob = otherApi ? recentJobs().find((j) => j.url === otherApi.url) : undefined
  // The row of the selected gene. Each side marks the enhancer the comparison measures.
  const chosen = cmp.data?.rows.find((r) => r.gene === gene)
  // With the enhancers of both jobs alone, a gene outside the comparison has no enhancer to mark.
  const absent = shared && cmp.data && gene ? null : undefined
  return (
    <div className="compare">
      <div className={cn('compare-toolbar')}>
        <label className={cn('check')}>
          <input type="checkbox" checked={linked} onChange={(e) => setLinked(e.target.checked)} /> Link rotation and zoom
        </label>
        {otherApi && (
          <label className={cn('check')}>
            <input type="checkbox" checked={cmp.data?.same_region ?? false} readOnly disabled /> Same region
          </label>
        )}
        {canShare && (
          <label className={cn('check')}>
            <input type="checkbox" checked={sharedOnly} onChange={(e) => setSharedOnly(e.target.checked)} /> Enhancers of both jobs only
          </label>
        )}
        {gene && onGene && (
          <button type="button" className="action" onClick={() => onGene(null)}>
            {`Clear the selection of ${gene}`}
          </button>
        )}
        <button type="button" className="action" onClick={onClose}>
          Close the comparison
        </button>
      </div>
      <div className={cn('compare-panels')}>
        <Panel api={api} label="this job" structure={selected} gene={gene} link={linked ? link : undefined} linkId="a" pair={chosen ? { bead: chosen.enhancer_bead_a, ...chosen.a } : absent} />
        {otherApi ? (
          <Panel
            key={otherApi.url}
            api={otherApi}
            label={otherJob?.name || otherJob?.label || 'the other job'}
            structure={1}
            gene={gene}
            link={linked ? link : undefined}
            linkId="b"
            pair={chosen ? { bead: chosen.enhancer_bead_b, ...chosen.b } : absent}
            choices={choices}
            onPick={pick}
          />
        ) : (
          <div className={cn('compare-panel')}>
            <PanelHead title="Choose a job to compare" sub="a job of this browser or a job link" choices={choices} onPick={pick} startOpen />
            <div className={cn('view-box compare-box empty-side')}>
              <p>The other job appears here.</p>
            </div>
          </div>
        )}
      </div>
      {otherApi && (cmp.isError || otherSummary.isError) && <Unanswered of={[otherSummary, cmp]} />}
      {cmp.data && cmp.data.rows.length > 0 && otherApi && otherSummary.data && (
        <CompareResults
          urls={[api.url, otherApi.url]}
          names={sideNames(summary, otherSummary.data)}
          cmp={cmp.data}
          shared={shared}
          expression={expressed.data}
          expressionState={bothExpressed ? (expressed.isPending ? 'wait' : expressed.isError ? (expressed.error as Error).message : '') : summary.expression || otherSummary.data.expression ? 'one' : ''}
          gene={gene}
          onGene={onGene}
        />
      )}
    </div>
  )
}
