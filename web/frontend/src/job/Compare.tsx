import { useQuery } from '@tanstack/react-query'
import { ChevronDown, Plus } from 'lucide-react'
import { useMemo, useState } from 'react'
import { recentJobs } from '../model/recent'
import { compareJobs, jobApi, type JobApi, type Summary } from './api'
import { readPalette } from './colors'
import { FACTORS, kb } from './format'
import { makeCameraLink, type CameraLink } from './cameraLink'
import { Scene } from './Scene'
import { chainColors } from './StructureView'
import { parseRegion } from './StructureTab'

const factorColor = (f: string) => (f === 'CTCF' ? 'var(--ctcf)' : 'var(--rnapii)')

/** A job url as its kind and key, or null. */
function parseJobUrl(text: string): { kind: 'jobs' | 'p'; key: string } | null {
  const m = text.trim().match(/\/(jobs|p)\/([A-Za-z0-9_-]{8,64})/)
  return m ? { kind: m[1] as 'jobs' | 'p', key: m[2] } : null
}

function Panel({
  api,
  label,
  structure: index,
  gene,
  link,
  linkId,
  onChange,
}: {
  api: JobApi
  label: string
  structure: number
  gene: string | null
  link?: CameraLink
  linkId: string
  onChange: () => void
}) {
  const summary = useQuery({ queryKey: ['summary', api.base], queryFn: api.summary })
  const structure = useQuery({ queryKey: ['structure', api.base, index], queryFn: () => api.structure(index) })
  const annotations = useQuery({ queryKey: ['annotations', api.base], queryFn: api.annotations })
  const genes = useQuery({ queryKey: ['genes', api.base], queryFn: api.genes })
  const palette = useMemo(readPalette, [])
  const s = summary.data
  const region = parseRegion(s?.region ?? null)
  const colors = useMemo(
    () => (structure.data && annotations.data && region ? chainColors(structure.data, annotations.data, [region.start, region.end], 'elements', palette) : []),
    [structure.data, annotations.data, region?.start, region?.end, palette], // eslint-disable-line react-hooks/exhaustive-deps
  )
  const row = genes.data?.find((r) => r.gene === gene)
  // The selected gene's promoter and enhancer, marked A and B as on the Structure tab.
  const markers =
    row && structure.data
      ? [row.promoter_bead, row.enhancer_bead]
          .filter((k) => structure.data!.beads[k])
          .map((k, i) => ({ label: 'AB'[i], at: [structure.data!.beads[k][2], structure.data!.beads[k][3], structure.data!.beads[k][4]] as [number, number, number] }))
      : []
  return (
    <div className="compare-panel">
      <button type="button" className="panel-head" onClick={onChange}>
        <span className="factor-line" style={{ background: factorColor(s?.request.factors === 'CTCF' ? 'CTCF' : 'RNAPII') }} />
        <span className="panel-text">
          <span className="panel-title">{s ? `${FACTORS[s.request.factors] ?? s.request.factors} loops` : ' '}</span>
          <span className="panel-sub">{s ? `${label}, ${s.request.cell_type}, ${s.structures} structures` : ''}</span>
        </span>
        <ChevronDown size={14} aria-hidden />
      </button>
      <div className="view-box compare-box">
        {structure.data && (
          <Scene points={structure.data.chain.map(([x, y, z]) => [x, y, z] as [number, number, number])} colors={colors} markers={markers} link={link} linkId={linkId} />
        )}
        {row && (
          <span className="ab-label">
            A to B <strong>{`${row.distance.toFixed(1)} ± ${row.sd.toFixed(1)}`}</strong>
          </span>
        )}
      </div>
    </div>
  )
}

/** The compare view: this job and another side by side, with the largest changes in enhancer to
 * promoter distance between them. */
export function Compare({
  api,
  summary,
  selected,
  gene,
  initial = null,
  onOther,
  onClose,
}: {
  api: JobApi
  summary: Summary
  selected: number
  gene: string | null
  initial?: string | null
  onOther?: (url: string | null) => void
  onClose: () => void
}) {
  const [other, setOtherState] = useState<{ kind: 'jobs' | 'p'; key: string } | null>(initial ? parseJobUrl(initial) : null)
  const setOther = (o: { kind: 'jobs' | 'p'; key: string } | null) => {
    setOtherState(o)
    onOther?.(o ? `/${o.kind}/${o.key}` : null)
  }
  const [pasted, setPasted] = useState('')
  const [linked, setLinked] = useState(true)
  const link = useMemo(makeCameraLink, [])
  const otherApi = useMemo(() => (other ? jobApi(other.kind, other.key) : null), [other])
  const cmp = useQuery({ queryKey: ['compare', api.url, otherApi?.url], queryFn: () => compareJobs(api.url, otherApi!.url), enabled: !!otherApi })
  const otherSummary = useQuery({ queryKey: ['summary', otherApi?.base], queryFn: () => otherApi!.summary(), enabled: !!otherApi })
  const recent = recentJobs().filter((j) => j.url !== api.url)
  if (!otherApi) {
    return (
      <section className="side-card compare-choose" aria-label="Compare with another job">
        <p className="card-label">Compare with another job</p>
        {recent.map((j) => (
          <button key={j.url} type="button" className="recent-pick" onClick={() => setOther(parseJobUrl(j.url))}>
            <Plus size={13} aria-hidden /> {j.name || j.label}
          </button>
        ))}
        <div className="paste">
          <input placeholder="Paste a job link" value={pasted} onChange={(e) => setPasted(e.target.value)} />
          <button type="button" className="action" disabled={!parseJobUrl(pasted)} onClick={() => setOther(parseJobUrl(pasted))}>
            Compare
          </button>
        </div>
        <button type="button" className="link-button" onClick={onClose}>
          Close
        </button>
      </section>
    )
  }
  const thisLabel = FACTORS[summary.request.factors] ?? summary.request.factors
  const otherName = recentJobs().find((j) => j.url === otherApi.url)?.name
  return (
    <div className="compare">
      <div className="compare-toolbar">
        <label className="check">
          <input type="checkbox" checked={linked} onChange={(e) => setLinked(e.target.checked)} /> Link rotation and zoom
        </label>
        <label className="check">
          <input type="checkbox" checked={cmp.data?.same_region ?? false} readOnly disabled /> Same region
        </label>
        <button type="button" className="action" onClick={() => setOther(null)}>
          <Plus size={14} aria-hidden /> Add an ensemble
        </button>
      </div>
      <div className="compare-panels">
        <Panel api={api} label="this job" structure={selected} gene={gene} link={linked ? link : undefined} linkId="a" onChange={() => setOther(null)} />
        <Panel
          api={otherApi}
          label={otherName || recentJobs().find((j) => j.url === otherApi.url)?.label || 'the other job'}
          structure={1}
          gene={gene}
          link={linked ? link : undefined}
          linkId="b"
          onChange={() => setOther(null)}
        />
      </div>
      {cmp.data && cmp.data.rows.length > 0 && (
        <table className="genes-table compare-table" aria-label="Largest changes in enhancer to promoter distance">
          <thead>
            <tr>
              <th>Largest changes in enhancer to promoter distance</th>
              <th>{thisLabel}</th>
              <th>{otherSummary.data ? (FACTORS[otherSummary.data.request.factors] ?? otherSummary.data.request.factors) : ''}</th>
              <th>{otherSummary.data ? `${FACTORS[otherSummary.data.request.factors] ?? otherSummary.data.request.factors} against ${thisLabel}` : 'Change'}</th>
            </tr>
          </thead>
          <tbody>
            {[...cmp.data.rows.filter((r) => r.gene === gene), ...cmp.data.rows.filter((r) => r.gene !== gene).slice(0, 10)].map((r) => (
              <tr key={r.gene} className={r.gene === gene ? 'selected' : undefined}>
                <td>{`${r.gene}, enhancer ${kb(r.linear_bp)} away`}</td>
                <td>{`${r.a.mean.toFixed(1)} ± ${r.a.sd.toFixed(1)}`}</td>
                <td>{`${r.b.mean.toFixed(1)} ± ${r.b.sd.toFixed(1)}`}</td>
                {/* The ratio is the other job's distance over this job's. Above 1 the other is farther. */}
                <td>{r.ratio >= 1 ? `${r.ratio.toFixed(1)} times farther` : `${(1 / r.ratio).toFixed(1)} times closer`}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  )
}
