import { useQuery } from '@tanstack/react-query'
import { Camera, ChevronDown } from 'lucide-react'
import { useEffect, useMemo, useRef, useState } from 'react'
import { recentJobs } from '../model/recent'
import { compareExpression, compareJobs, jobApi, type JobApi, type Summary } from '../api'
import { CompareResults, sideNames } from './CompareResults'
import { readPalette } from './colors'
import { FACTORS } from './format'
import { makeCameraLink, type CameraLink } from './cameraLink'
import { Scene, type SceneHandle } from './Scene'
import { saveDataUrl } from './savePng'
import { chainColors, focusRanges, parseRegion } from './structure'

const factorColor = (f: string) => (f === 'CTCF' ? 'var(--ctcf)' : 'var(--rnapii)')

/** A job url as its kind and key, or null. */
function parseJobUrl(text: string): { kind: 'jobs' | 'p'; key: string } | null {
  const m = text.trim().match(/\/(jobs|p)\/([A-Za-z0-9_-]{8,64})/)
  return m ? { kind: m[1] as 'jobs' | 'p', key: m[2] } : null
}

type Choice = { url: string; name: string }

/** The head of one side. With `onPick` it is a button that opens, under itself, the jobs that
 * side can show and a field for a job link. Without it the head is plain. */
function PanelHead({ line, title, sub, choices = [], onPick, startOpen = false }: { line?: string; title: string; sub: string; choices?: Choice[]; onPick?: (url: string) => void; startOpen?: boolean }) {
  const [open, setOpen] = useState(startOpen)
  const [pasted, setPasted] = useState('')
  const box = useRef<HTMLDivElement>(null)
  // A press outside the head and its list closes the list.
  useEffect(() => {
    if (!open) return
    const away = (e: PointerEvent) => {
      if (!box.current?.contains(e.target as Node)) setOpen(false)
    }
    document.addEventListener('pointerdown', away)
    return () => document.removeEventListener('pointerdown', away)
  }, [open])
  const pick = (url: string) => {
    setOpen(false)
    setPasted('')
    onPick?.(url)
  }
  const text = (
    <>
      {line && <span className="factor-line" style={{ background: line }} />}
      <span className="panel-text">
        <span className="panel-title">{title}</span>
        <span className="panel-sub">{sub}</span>
      </span>
    </>
  )
  if (!onPick) return <div className="panel-head fixed">{text}</div>
  return (
    <div className="panel-pick" ref={box} onKeyDown={(e) => e.key === 'Escape' && setOpen(false)}>
      <button type="button" className="panel-head" aria-haspopup="true" aria-expanded={open} onClick={() => setOpen(!open)}>
        {text}
        <ChevronDown size={14} aria-hidden />
      </button>
      {open && (
        <div className="panel-menu" role="group" aria-label="Jobs to compare with">
          {choices.map((j) => (
            <button key={j.url} type="button" className="menu-item" onClick={() => pick(j.url)}>
              {j.name}
            </button>
          ))}
          <div className="paste">
            <input placeholder="Paste a job link" value={pasted} onChange={(e) => setPasted(e.target.value)} onKeyDown={(e) => e.key === 'Enter' && parseJobUrl(pasted) && pick(pasted)} />
            <button type="button" className="action" disabled={!parseJobUrl(pasted)} onClick={() => pick(pasted)}>
              Compare
            </button>
          </div>
        </div>
      )}
    </div>
  )
}

function Panel({
  api,
  label,
  structure: index,
  gene,
  link,
  linkId,
  pair,
  choices,
  onPick,
}: {
  api: JobApi
  label: string
  structure: number
  gene: string | null
  link?: CameraLink
  linkId: string
  /** The enhancer the comparison measures the selected gene to in this job, and its distance. Null for a gene
   * the comparison does not hold, which is then not marked. */
  pair?: { bead?: number; mean: number; sd: number } | null
  choices?: Choice[]
  onPick?: (url: string) => void
}) {
  const summary = useQuery({ queryKey: ['summary', api.base], queryFn: api.summary })
  const structure = useQuery({ queryKey: ['structure', api.base, index], queryFn: () => api.structure(index) })
  const annotations = useQuery({ queryKey: ['annotations', api.base], queryFn: api.annotations })
  const genes = useQuery({ queryKey: ['genes', api.base], queryFn: api.genes })
  const palette = useMemo(readPalette, [])
  const scene = useRef<SceneHandle>(null)
  const s = summary.data
  const region = parseRegion(s?.region ?? null)
  const colors = useMemo(
    () => (structure.data && annotations.data && region ? chainColors(structure.data, annotations.data, [region.start, region.end], 'elements', palette) : []),
    [structure.data, annotations.data, region?.start, region?.end, palette], // eslint-disable-line react-hooks/exhaustive-deps
  )
  const row = pair === null ? undefined : genes.data?.find((r) => r.gene === gene)
  const enhancerBead = pair?.bead ?? row?.enhancer_bead
  // The selected gene's promoter and enhancer, marked A and B as on the Structure tab.
  const markers =
    row && structure.data
      ? [row.promoter_bead, enhancerBead!]
          .filter((k) => structure.data!.beads[k])
          .map((k, i) => ({ label: 'AB'[i], at: [structure.data!.beads[k][2], structure.data!.beads[k][3], structure.data!.beads[k][4]] as [number, number, number] }))
      : []
  // The chain at the selected gene's promoter and enhancer is solid and the rest see through.
  const focus = useMemo(
    () => (row && structure.data ? focusRanges(structure.data, [row.promoter_bead, enhancerBead!]) : undefined),
    [structure.data, row?.promoter_bead, enhancerBead], // eslint-disable-line react-hooks/exhaustive-deps
  )
  return (
    <div className="compare-panel">
      <PanelHead
        line={factorColor(s?.request.factors === 'CTCF' ? 'CTCF' : 'RNAPII')}
        title={s ? `${FACTORS[s.request.factors] ?? s.request.factors} loops` : ' '}
        sub={s ? `${label}, ${s.request.cell_type}, ${s.structures} structures` : ''}
        choices={choices}
        onPick={onPick}
      />
      <div className="view-box compare-box">
        {structure.data && (
          <Scene ref={scene} points={structure.data.chain.map(([x, y, z]) => [x, y, z] as [number, number, number])} colors={colors} markers={markers} focus={focus} link={link} linkId={linkId} />
        )}
        {structure.data && (
          <button
            type="button"
            className="icon-tool panel-save"
            aria-label={`Save the structure of ${label} as a PNG`}
            onClick={() => {
              const url = scene.current?.png(row ? `A to B ${(pair?.mean ?? row.distance).toFixed(1)} ± ${(pair?.sd ?? row.sd).toFixed(1)}` : undefined)
              if (url) saveDataUrl(url, `structure_${linkId === 'a' ? 'this_job' : 'other_job'}_${index}`)
            }}
          >
            <Camera size={15} aria-hidden />
          </button>
        )}
        {row && (
          <span className="ab-label">
            A to B <strong>{`${(pair?.mean ?? row.distance).toFixed(1)} ± ${(pair?.sd ?? row.sd).toFixed(1)}`}</strong>
          </span>
        )}
      </div>
    </div>
  )
}

/** The compare view: this job and another side by side, with the changes between them below,
 * `CompareResults`. The other side starts empty with its list of jobs open. */
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
      <div className="compare-toolbar">
        <label className="check">
          <input type="checkbox" checked={linked} onChange={(e) => setLinked(e.target.checked)} /> Link rotation and zoom
        </label>
        {otherApi && (
          <label className="check">
            <input type="checkbox" checked={cmp.data?.same_region ?? false} readOnly disabled /> Same region
          </label>
        )}
        {canShare && (
          <label className="check">
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
      <div className="compare-panels">
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
          <div className="compare-panel">
            <PanelHead title="Choose a job to compare" sub="a job of this browser or a job link" choices={choices} onPick={pick} startOpen />
            <div className="view-box compare-box empty-side">
              <p>The other job appears here.</p>
            </div>
          </div>
        )}
      </div>
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
