import { useQuery } from '@tanstack/react-query'
import { ChevronDown } from 'lucide-react'
import { useEffect, useMemo, useRef, useState } from 'react'
import { recentJobs } from '../model/recent'
import { compareExpression, compareJobs, jobApi, type ExpressionCompare, type JobApi, type Summary } from './api'
import { readPalette } from './colors'
import { FACTORS, kb } from './format'
import { makeCameraLink, type CameraLink } from './cameraLink'
import { Scene } from './Scene'
import { chainColors, focusRanges } from './StructureView'
import { parseRegion } from './StructureTab'

const factorColor = (f: string) => (f === 'CTCF' ? 'var(--ctcf)' : 'var(--rnapii)')

/** A job url as its kind and key, or null. */
function parseJobUrl(text: string): { kind: 'jobs' | 'p'; key: string } | null {
  const m = text.trim().match(/\/(jobs|p)\/([A-Za-z0-9_-]{8,64})/)
  return m ? { kind: m[1] as 'jobs' | 'p', key: m[2] } : null
}

type Choice = { url: string; name: string }

/** The ten largest changes in their order. A selected gene outside them is put first. The order
 * of the ten does not change with the selection, so a clicked row stays under the pointer. */
export function shownRows<T extends { gene: string }>(rows: T[], gene: string | null): T[] {
  const top = rows.slice(0, 10)
  const chosen = rows.find((r) => r.gene === gene)
  return chosen && !top.includes(chosen) ? [chosen, ...top] : top
}

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
  choices,
  onPick,
}: {
  api: JobApi
  label: string
  structure: number
  gene: string | null
  link?: CameraLink
  linkId: string
  choices?: Choice[]
  onPick?: (url: string) => void
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
  // The chain at the selected gene's promoter and enhancer is solid and the rest see through.
  const focus = useMemo(
    () => (row && structure.data ? focusRanges(structure.data, [row.promoter_bead, row.enhancer_bead]) : undefined),
    [structure.data, row?.promoter_bead, row?.enhancer_bead], // eslint-disable-line react-hooks/exhaustive-deps
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
          <Scene points={structure.data.chain.map(([x, y, z]) => [x, y, z] as [number, number, number])} colors={colors} markers={markers} focus={focus} link={link} linkId={linkId} />
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

const SCATTER = { w: 320, h: 220, pad: 28 }

/** How many times a level or a distance changed, from its log2 change, with the word for its direction. */
const times = (log2: number, up: string, down: string) => `${(2 ** Math.abs(log2)).toFixed(1)} times ${log2 >= 0 ? up : down}`

/**
 * The change of expression of the genes two jobs share, the other job against this one, beside
 * the change of their enhancer to promoter distance: what was computed, a plot of one change
 * against the other and the genes whose expression changed most.
 */
function ExpressionChange({ data, gene, onGene }: { data: ExpressionCompare; gene: string | null; onGene?: (gene: string | null) => void }) {
  if (data.mode === 'none') return <p className="compare-note">{data.reason}</p>
  const rows = data.rows
  const { w, h, pad } = SCATTER
  const reach = (get: (r: ExpressionCompare['rows'][number]) => number) => Math.max(1, ...rows.map((r) => Math.abs(get(r))))
  const [rx, ry] = [reach((r) => r.distance), reach((r) => r.log2fc)]
  const x = (v: number) => w / 2 + (v / rx) * (w / 2 - pad)
  const y = (v: number) => h / 2 - (v / ry) * (h / 2 - pad)
  return (
    <>
      <p className="compare-note">
        {`${data.label_b} against ${data.label_a}. `}
        {data.mode === 'deseq2' ? 'The fold change and its adjusted p are those of DESeq2.' : data.reason}
        {data.rho !== null && ` Over ${rows.length} genes the change of expression and the change of distance have a rank correlation of ${data.rho.toFixed(2)}.`}
      </p>
      {rows.length === 0 ? (
        <p className="compare-note">The two jobs share no gene with expression in both.</p>
      ) : (
        <div className="expression-change">
          <svg className="expression-scatter" viewBox={`0 0 ${w} ${h}`} role="img" aria-label={`Change of expression against change of distance for ${rows.length} genes`}>
            <line className="axis" x1={pad} x2={w - pad} y1={h / 2} y2={h / 2} />
            <line className="axis" x1={w / 2} x2={w / 2} y1={pad} y2={h - pad} />
            <text className="axis-label" x={w - pad} y={h / 2 - 6} textAnchor="end">
              farther
            </text>
            <text className="axis-label" x={pad} y={h / 2 - 6}>
              closer
            </text>
            <text className="axis-label" x={w / 2 + 6} y={pad + 4}>
              higher expression
            </text>
            <text className="axis-label" x={w / 2 + 6} y={h - pad}>
              lower
            </text>
            {rows.map((r) => (
              <circle key={r.gene} className={r.gene === gene ? 'point selected' : r.padj !== null && r.padj < 0.05 ? 'point sure' : 'point'} cx={x(r.distance)} cy={y(r.log2fc)} r={r.gene === gene ? 5 : 3}>
                <title>{`${r.gene}, distance ${times(r.distance, 'farther', 'closer')}, expression ${times(r.log2fc, 'higher', 'lower')}`}</title>
              </circle>
            ))}
          </svg>
          <table className="genes-table compare-table" aria-label="Largest changes in expression">
            <thead>
              <tr>
                <th>Largest changes in expression</th>
                <th>Expression</th>
                <th>Adjusted p</th>
                <th>3D distance</th>
              </tr>
            </thead>
            <tbody>
              {shownRows(rows, gene).map((r) => (
                <tr key={r.gene} className={r.gene === gene ? 'selected' : undefined} aria-selected={r.gene === gene} onClick={() => onGene?.(r.gene === gene ? null : r.gene)}>
                  <td>{r.gene}</td>
                  <td>{times(r.log2fc, 'higher', 'lower')}</td>
                  <td>{r.padj === null ? 'no test' : r.padj < 0.001 ? r.padj.toExponential(1) : r.padj.toFixed(3)}</td>
                  <td>{times(r.distance, 'farther', 'closer')}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </>
  )
}

/** The compare view: this job and another side by side, with the largest changes in enhancer to
 * promoter distance between them. The other side starts empty with its list of jobs open. */
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
  const cmp = useQuery({ queryKey: ['compare', api.url, otherApi?.url], queryFn: () => compareJobs(api.url, otherApi!.url), enabled: !!otherApi })
  const otherSummary = useQuery({ queryKey: ['summary', otherApi?.base], queryFn: () => otherApi!.summary(), enabled: !!otherApi })
  // The expression of the two jobs is compared when both have it. The test of two tables takes seconds the first time.
  const bothExpressed = !!summary.expression && !!otherSummary.data?.expression
  const expressed = useQuery({
    queryKey: ['compare-expression', api.url, otherApi?.url],
    queryFn: () => compareExpression(api.url, otherApi!.url),
    enabled: !!otherApi && bothExpressed,
    staleTime: Infinity,
  })
  const recent = recentJobs().filter((j) => j.url !== api.url)
  const choices = recent.filter((j) => j.url !== otherApi?.url).map((j) => ({ url: j.url, name: j.name || j.label }))
  const pick = (url: string) => setOther(parseJobUrl(url))
  const thisLabel = FACTORS[summary.request.factors] ?? summary.request.factors
  const otherJob = otherApi ? recentJobs().find((j) => j.url === otherApi.url) : undefined
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
        <Panel api={api} label="this job" structure={selected} gene={gene} link={linked ? link : undefined} linkId="a" />
        {otherApi ? (
          <Panel
            key={otherApi.url}
            api={otherApi}
            label={otherJob?.name || otherJob?.label || 'the other job'}
            structure={1}
            gene={gene}
            link={linked ? link : undefined}
            linkId="b"
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
            {shownRows(cmp.data.rows, gene).map((r) => (
              <tr key={r.gene} className={r.gene === gene ? 'selected' : undefined} aria-selected={r.gene === gene} onClick={() => onGene?.(r.gene === gene ? null : r.gene)}>
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
      {otherApi && bothExpressed && (
        <section className="compare-expression" aria-label="Expression against 3D distance">
          <h3>Expression against 3D distance</h3>
          {expressed.isPending && (
            <p className="compare-note" aria-busy="true">
              Comparing the expression of the two jobs.
            </p>
          )}
          {expressed.isError && <p className="compare-note">{(expressed.error as Error).message}</p>}
          {expressed.data && <ExpressionChange data={expressed.data} gene={gene} onGene={onGene} />}
        </section>
      )}
      {otherApi && otherSummary.data && !bothExpressed && (summary.expression || otherSummary.data.expression) && (
        <p className="compare-note">One of the two jobs has no expression data. Both need it for a comparison of expression.</p>
      )}
    </div>
  )
}
