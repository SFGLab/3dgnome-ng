import { useQuery } from '@tanstack/react-query'
import { ChevronRight, Download } from 'lucide-react'
import { type ReactNode, useEffect, useRef, useState } from 'react'
import { CsvButton } from '../components/CsvButton'
import { Pager } from '../components/Pager'
import { usePage } from '../components/usePage'
import { compareGene, type Compare, type ExpressionCompare, type Summary } from './api'
import { FACTORS, kb } from './format'
import { savePng } from './savePng'

const BACKGROUNDS: Record<string, string> = { hic: 'Hi-C', hicdiffusion: 'HiCDiffusion', none: 'no contact map' }

/**
 * A short name for each of two jobs, by the first thing they differ in: the cell type, the
 * sample of the variants, the loops or the contact background. Two jobs alike in all of them
 * are this job and the other job.
 */
export function sideNames(a: Summary, b: Summary): [string, string] {
  const tries: ((s: Summary) => string)[] = [
    (s) => (s.request.cell_type === 'own' ? 'own data' : s.request.cell_type),
    (s) => s.request.sample || 'no variants',
    (s) => `${FACTORS[s.request.factors] ?? s.request.factors} loops`,
    (s) => BACKGROUNDS[s.request.hicdiffusion ? 'hicdiffusion' : (s.request.background ?? 'hic')] ?? 'Hi-C',
  ]
  for (const name of tries) if (name(a) !== name(b)) return [name(a), name(b)]
  return ['this job', 'the other job']
}

/** How many times a level or a distance changed, from its log2 change, with the word for its direction. */
const times = (log2: number, up: string, down: string) => `${(2 ** Math.abs(log2)).toFixed(1)} times ${log2 >= 0 ? up : down}`

/** The change of an expression level. A gene with next to no reads on one side has a fold change
 * in the hundreds that says no more than over 100 times. */
const fold = (log2: number) => (2 ** Math.abs(log2) >= 100 ? `over 100 times ${log2 >= 0 ? 'higher' : 'lower'}` : times(log2, 'higher', 'lower'))

const pText = (p: number | null) => (p === null ? 'no test' : p < 0.001 ? p.toExponential(1) : p.toFixed(3))
const level = (v: number) => (v >= 100 ? v.toFixed(0) : v.toFixed(1))

const PLOT = { w: 340, h: 286, left: 48, right: 12, top: 12, bottom: 56 }

/** The ticks of an axis from lo to hi at a round step, six at most. */
function ticks(lo: number, hi: number): number[] {
  const step = [0.5, 1, 2, 5, 10, 20, 50, 100].find((s) => (hi - lo) / s <= 5) ?? 100
  const out: number[] = []
  for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step) out.push(Math.round(v * 10) / 10)
  return out
}

type Scale = (v: number) => number
type Key = [className: string, text: string]

/** The y axis of a plot with its ticks and its title, and the key of the marks below the plot. */
function Sides({ yr, yTitle, capped, fy, keys }: { yr: [number, number]; yTitle: string; capped?: boolean; fy: Scale; keys: Key[] }) {
  const { h, left, top, bottom } = PLOT
  const middle = (top + h - bottom) / 2
  // Each mark of the key follows the text of the one before it.
  const starts = keys.map((_, i) => 14 + keys.slice(0, i).reduce((sum, [, text]) => sum + 18 + text.length * 4.6, 0))
  return (
    <>
      <line className="axis" x1={left} x2={left} y1={top} y2={h - bottom} />
      {ticks(...yr).map((v) => (
        <g key={`y${v}`}>
          <line className="axis" x1={left - 4} x2={left} y1={fy(v)} y2={fy(v)} />
          <text className="tick" x={left - 7} y={fy(v) + 3} textAnchor="end">
            {capped && v === yr[1] ? `${v}+` : v}
          </text>
        </g>
      ))}
      <text className="axis-title" x={12} y={middle} textAnchor="middle" transform={`rotate(-90 12 ${middle})`}>
        {yTitle}
      </text>
      {keys.map(([className, text], i) => (
        <g key={text}>
          <circle className={className} cx={starts[i]} cy={h - 9} r={className === 'back' ? 1.6 : 3} />
          <text className="tick" x={starts[i] + 7} y={h - 6}>
            {text}
          </text>
        </g>
      ))}
    </>
  )
}

/** A plot with an axis on the left and one below, each with its ticks and its title, a dashed
 * line at zero and the key of its marks. */
function Frame({
  label,
  xr,
  yr,
  xTitle,
  yTitle,
  capped = false,
  keys,
  children,
}: {
  label: string
  xr: [number, number]
  yr: [number, number]
  xTitle: string
  yTitle: string
  /** The last tick of the y axis stands for its value and every value above it. */
  capped?: boolean
  keys: Key[]
  children: (fx: Scale, fy: Scale) => ReactNode
}) {
  const { w, h, left, right, top, bottom } = PLOT
  const fx: Scale = (v) => left + ((v - xr[0]) / (xr[1] - xr[0])) * (w - left - right)
  const fy: Scale = (v) => h - bottom - ((v - yr[0]) / (yr[1] - yr[0])) * (h - top - bottom)
  return (
    <svg className="expression-plot" viewBox={`0 0 ${w} ${h}`} role="img" aria-label={label}>
      {xr[0] < 0 && xr[1] > 0 && <line className="zero" x1={fx(0)} x2={fx(0)} y1={top} y2={h - bottom} />}
      {yr[0] < 0 && yr[1] > 0 && <line className="zero" x1={left} x2={w - right} y1={fy(0)} y2={fy(0)} />}
      <line className="axis" x1={left} x2={w - right} y1={h - bottom} y2={h - bottom} />
      {ticks(...xr).map((v) => (
        <g key={`x${v}`}>
          <line className="axis" x1={fx(v)} x2={fx(v)} y1={h - bottom} y2={h - bottom + 4} />
          <text className="tick" x={fx(v)} y={h - bottom + 15} textAnchor="middle">
            {v}
          </text>
        </g>
      ))}
      <text className="axis-title" x={(left + w - right) / 2} y={h - bottom + 30} textAnchor="middle">
        {xTitle}
      </text>
      <Sides yr={yr} yTitle={yTitle} capped={capped} fy={fy} keys={keys} />
      {children(fx, fy)}
    </svg>
  )
}

/** The whole number an axis reaches to, one at least. */
const reach = (values: number[]) => Math.max(1, Math.ceil(Math.max(0, ...values.map(Math.abs))))

/** The value a share of the way through sorted values, between two of them where it falls there. */
function quantile(sorted: number[], share: number): number {
  const at = (sorted.length - 1) * share
  const low = Math.floor(at)
  return sorted[low] + (sorted[Math.min(low + 1, sorted.length - 1)] - sorted[low]) * (at - low)
}

/** The distance of one gene in the structures of two jobs: a box from the first to the third
 * quartile with the median, whiskers to the least and the greatest value and a point for each structure. */
function BoxPlot({ gene, names, a, b }: { gene: string; names: [string, string]; a: number[]; b: number[] }) {
  const { w, h, left, right, top, bottom } = PLOT
  const hi = Math.max(1, Math.ceil(Math.max(...a, ...b) * 1.05))
  const fy: Scale = (v) => h - bottom - (v / hi) * (h - top - bottom)
  const width = w - left - right
  const sides = [a, b].map((values, i) => {
    const sorted = [...values].sort((x, y) => x - y)
    return { values, sorted, x: left + width * (i === 0 ? 0.28 : 0.72), q1: quantile(sorted, 0.25), median: quantile(sorted, 0.5), q3: quantile(sorted, 0.75) }
  })
  const half = 34
  return (
    <svg
      className="expression-plot"
      viewBox={`0 0 ${w} ${h}`}
      role="img"
      aria-label={`Enhancer to promoter distance of ${gene} in the structures of ${names[0]} and of ${names[1]}`}
    >
      <line className="axis" x1={left} x2={w - right} y1={h - bottom} y2={h - bottom} />
      {sides.map((s, i) => (
        <g key={i} className={`box side-${i}`}>
          <line className="whisker" x1={s.x} x2={s.x} y1={fy(s.sorted[0])} y2={fy(s.sorted[s.sorted.length - 1])} />
          <rect className="quartiles" x={s.x - half} width={half * 2} y={fy(s.q3)} height={Math.max(1, fy(s.q1) - fy(s.q3))} />
          <line className="median" x1={s.x - half} x2={s.x + half} y1={fy(s.median)} y2={fy(s.median)} />
          {s.values.map((v, k) => (
            // The points of a side are spread across the box in the order of the structures.
            <circle key={k} className="structure" cx={s.x - half + 6 + ((half * 2 - 12) * (k + 0.5)) / s.values.length} cy={fy(v)} r={2}>
              <title>{`structure ${k + 1}, ${v.toFixed(1)} bead units`}</title>
            </circle>
          ))}
          <text className="tick" x={s.x} y={h - bottom + 15} textAnchor="middle">
            {names[i]}
          </text>
          <text className="tick" x={s.x} y={h - bottom + 27} textAnchor="middle">
            {`${s.values.length} structure${s.values.length === 1 ? '' : 's'}`}
          </text>
        </g>
      ))}
      <Sides yr={[0, hi]} yTitle="3D distance, bead units" fy={fy} keys={[['structure', 'a structure']]} />
    </svg>
  )
}

/** A figure of the comparison as one part of an accordion: its title opens and closes it, and
 * an open figure has a button that saves it as a PNG. */
function Figure({ title, file, open, onToggle, children }: { title: string; file: string; open: boolean; onToggle: () => void; children: ReactNode }) {
  const box = useRef<HTMLElement>(null)
  const save = () => {
    // The plot of the figure, and not an icon of its head.
    const svg = box.current?.querySelector<SVGSVGElement>('svg.expression-plot')
    if (svg) void savePng(svg, title, file)
  }
  return (
    <figure className={`compare-figure${open ? ' open' : ''}`} ref={box}>
      <figcaption>
        <button type="button" className="figure-head" aria-expanded={open} onClick={onToggle}>
          <ChevronRight size={14} aria-hidden /> {title}
        </button>
        {open && (
          <button type="button" className="figure-save" aria-label={`Save the figure ${title} as a PNG`} onClick={save}>
            <Download size={12} aria-hidden /> PNG
          </button>
        )}
      </figcaption>
      {open && children}
    </figure>
  )
}

type Row = Compare['rows'][number]
type ExpressionRow = ExpressionCompare['rows'][number]
type Tab = 'distance' | 'expression'
type Fig = 'box' | 'scatter' | 'volcano'

/** A table of the comparison with every gene, a page at a time. The page with the selected gene is shown. */
function Paged<T extends { gene: string }>({
  rows,
  gene,
  noun,
  label,
  head,
  cells,
  onGene,
}: {
  rows: T[]
  gene: string | null
  noun: string
  label: string
  head: string[]
  cells: (r: T) => string[]
  onGene?: (gene: string | null) => void
}) {
  const page = usePage(rows, label)
  const at = rows.findIndex((r) => r.gene === gene)
  const { go, size } = page
  useEffect(() => {
    if (at >= 0) go(Math.floor(at / size))
    // The page follows the selection alone.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [gene, at, size])
  return (
    <>
      <div className="compare-scroll">
        <table className="genes-table compare-table" aria-label={label}>
          <thead>
            <tr>
              {head.map((h) => (
                <th key={h}>{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {page.rows.map((r) => (
              <tr key={r.gene} className={r.gene === gene ? 'selected' : undefined} aria-selected={r.gene === gene} onClick={() => onGene?.(r.gene === gene ? null : r.gene)}>
                {cells(r).map((c, i) => (
                  <td key={i}>{c}</td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <Pager page={page} noun={noun} />
    </>
  )
}

/**
 * What changed between two jobs, under their structures: the figures in a column, each saved as
 * a PNG by its button, and beside them the main values and the tables of every gene, one tab for
 * the 3D distance and one for the expression. The other job is compared against this one.
 */
export function CompareResults({
  urls,
  names,
  cmp,
  expression,
  expressionState,
  gene,
  onGene,
}: {
  /** The addresses of this job and of the other job. */
  urls: [string, string]
  names: [string, string]
  cmp: Compare
  expression?: ExpressionCompare
  /** `wait` while the expression is compared, `one` when one job alone has expression, the message of a failure, or empty. */
  expressionState: string
  gene: string | null
  onGene?: (gene: string | null) => void
}) {
  const [tab, setTab] = useState<Tab>('distance')
  // One figure is open at a time. Undefined is the figure the view starts with.
  const [opened, setOpened] = useState<Fig | null | undefined>(undefined)
  const [A, B] = names
  const against = `${B} against ${A}`
  const pick = (name: string) => onGene?.(name === gene ? null : name)
  const row = cmp.rows.find((r) => r.gene === gene)
  const expressed = expression && expression.mode !== 'none' && expression.rows.length > 0 ? expression : null
  const eRow = expressed?.rows.find((r) => r.gene === gene)
  const one = useQuery({ queryKey: ['compare-gene', urls[0], urls[1], gene], queryFn: () => compareGene(urls[0], urls[1], gene!), enabled: !!row, staleTime: Infinity })
  const volcano = expressed?.volcano ?? null
  const kind = (r: ExpressionRow) => (r.gene === gene ? 'point selected' : r.padj !== null && r.padj < 0.05 ? 'point sure' : 'point')
  const keys: Key[] = [
    ['point sure', 'adjusted p under 0.05'],
    ['point', '0.05 or more'],
  ]
  const tested =
    volcano && expressed
      ? expressed.rows
          .filter((r) => r.padj !== null)
          .map((r) => ({ r, x: Math.max(-volcano.fold, Math.min(volcano.fold, r.log2fc)), y: r.padj! <= 0 ? volcano.p : Math.min(-Math.log10(r.padj!), volcano.p) }))
      : []
  const vx = volcano ? reach([...volcano.points.map(([x]) => x), ...tested.map((t) => t.x)]) : 1
  const top = volcano ? Math.max(2, Math.ceil(Math.max(...volcano.points.map(([, y]) => y), ...tested.map((t) => t.y)))) : 2
  const line = -Math.log10(0.05)
  const shown: Tab = tab === 'expression' && expressed ? 'expression' : 'distance'
  const figure = opened === undefined ? (expressed ? 'scatter' : 'box') : opened
  const toggle = (f: Fig) => setOpened(figure === f ? null : f)
  return (
    <section className="compare-results" aria-label="Changes between the two jobs">
      <div className="compare-tables">
        <div className="compare-cards">
          {row && (
            <section className="side-card" aria-label={`Change of ${row.gene}`}>
              <p className="card-label">{`${row.gene}, ${against}`}</p>
              <p className="big">
                <span className="big-value">{(row.ratio >= 1 ? row.ratio : 1 / row.ratio).toFixed(1)}</span>{' '}
                <span className="unit">{`times ${row.ratio >= 1 ? 'farther' : 'closer'} in ${B}`}</span>
              </p>
              <p className="card-text">{`${row.a.mean.toFixed(1)} ± ${row.a.sd.toFixed(1)} bead units in ${A}, ${row.b.mean.toFixed(1)} ± ${row.b.sd.toFixed(1)} in ${B}. Mann-Whitney p ${pText(row.p)}.`}</p>
              {eRow && (
                <>
                  <p className="big">
                    <span className="big-value">{2 ** Math.abs(eRow.log2fc) >= 100 ? 'over 100' : (2 ** Math.abs(eRow.log2fc)).toFixed(1)}</span>{' '}
                    <span className="unit">{`times ${eRow.log2fc >= 0 ? 'higher' : 'lower'} expression in ${B}`}</span>
                  </p>
                  <p className="card-text">{`Level ${level(eRow.a)} in ${A}, ${level(eRow.b)} in ${B}. Adjusted p ${pText(eRow.padj)}.`}</p>
                </>
              )}
            </section>
          )}
          {expressed && (
            <section className="side-card" aria-label="Expression against 3D distance">
              <p className="card-label">Expression against 3D distance</p>
              {expressed.rho !== null && (
                <p className="big">
                  <span className="big-value">{expressed.rho.toFixed(2)}</span> <span className="unit">rank correlation</span>
                </p>
              )}
              <p className="card-text">
                {expressed.rho !== null
                  ? `Of the change of expression with the change of distance over ${expressed.rows.length} genes, ${against}.`
                  : `${expressed.rows.length} genes have expression in both jobs. Five or more give a rank correlation.`}{' '}
                {expressed.mode === 'deseq2' ? 'Tested with DESeq2.' : expressed.reason}
              </p>
            </section>
          )}
          {!expressed && (expressionState || expression) && (
            <p className="compare-note" aria-busy={expressionState === 'wait'}>
              {expressionState === 'wait'
                ? 'Comparing the expression of the two jobs.'
                : expressionState === 'one'
                  ? 'One of the two jobs has no expression data. Both need it for a comparison of expression.'
                  : expressionState || (expression?.mode === 'none' ? expression.reason : 'The two jobs share no gene with expression in both.')}
            </p>
          )}
        </div>
        <div className="compare-tabs-row">
          <div className="compare-tabs" role="tablist" aria-label="Tables of the comparison">
            <button type="button" role="tab" aria-selected={shown === 'distance'} className={`chip-button${shown === 'distance' ? ' on' : ''}`} onClick={() => setTab('distance')}>
              3D distance <span className="count">{cmp.rows.length}</span>
            </button>
            {expressed && (
              <button
                type="button"
                role="tab"
                aria-selected={shown === 'expression'}
                className={`chip-button${shown === 'expression' ? ' on' : ''}`}
                onClick={() => setTab('expression')}
              >
                Expression <span className="count">{expressed.rows.length}</span>
              </button>
            )}
          </div>
          {shown === 'distance' ? (
            <CsvButton
              file="distance_changes"
              head={[
                'gene',
                'enhancer_separation_bp',
                `distance_mean_${A}`,
                `distance_sd_${A}`,
                `distance_mean_${B}`,
                `distance_sd_${B}`,
                `distance_ratio_${B}_over_${A}`,
                'mann_whitney_p',
              ]}
              rows={() => cmp.rows.map((r) => [r.gene, r.linear_bp, r.a.mean, r.a.sd, r.b.mean, r.b.sd, r.ratio, r.p])}
            />
          ) : (
            <CsvButton
              file="expression_changes"
              head={['gene', `expression_${A}`, `expression_${B}`, `expression_log2_${B}_over_${A}`, 'adjusted_p', `distance_log2_${B}_over_${A}`]}
              rows={() => expressed!.rows.map((r) => [r.gene, r.a, r.b, r.log2fc, r.padj, r.distance])}
            />
          )}
        </div>
        {shown === 'distance' ? (
          <Paged<Row>
            rows={cmp.rows}
            gene={gene}
            noun="genes"
            label="Changes in enhancer to promoter distance"
            head={['Gene', A, B, against, 'Mann-Whitney p']}
            cells={(r) => [
              `${r.gene}, enhancer ${kb(r.linear_bp)} away`,
              `${r.a.mean.toFixed(1)} ± ${r.a.sd.toFixed(1)}`,
              `${r.b.mean.toFixed(1)} ± ${r.b.sd.toFixed(1)}`,
              // The ratio is the other job's distance over this job's. Above 1 the other is farther.
              r.ratio >= 1 ? `${r.ratio.toFixed(1)} times farther` : `${(1 / r.ratio).toFixed(1)} times closer`,
              pText(r.p),
            ]}
            onGene={onGene}
          />
        ) : (
          <Paged<ExpressionRow>
            rows={expressed!.rows}
            gene={gene}
            noun="genes"
            label="Changes in expression"
            head={['Gene', A, B, `Expression, ${against}`, 'Adjusted p', `3D distance, ${against}`]}
            cells={(r) => [r.gene, level(r.a), level(r.b), fold(r.log2fc), pText(r.padj), times(r.distance, 'farther', 'closer')]}
            onGene={onGene}
          />
        )}
      </div>
      <div className="compare-figures">
        <Figure
          title={row ? `Enhancer to promoter distance of ${row.gene}` : 'Enhancer to promoter distance of a gene'}
          file={`distance_${row?.gene ?? 'gene'}`}
          open={figure === 'box'}
          onToggle={() => toggle('box')}
        >
          {row && one.data ? (
            <BoxPlot gene={row.gene} names={names} a={one.data.a} b={one.data.b} />
          ) : (
            <p className="figure-empty">
              {row ? 'Reading the distances.' : 'Select a gene in a table or a plot for the box plot of its distance in the structures of the two jobs.'}
            </p>
          )}
        </Figure>
        {expressed && (
          <Figure title="Change of expression against change of distance" file="expression_against_distance" open={figure === 'scatter'} onToggle={() => toggle('scatter')}>
            <Frame
              label={`Change of expression against change of distance for ${expressed.rows.length} genes`}
              xr={[-reach(expressed.rows.map((r) => r.distance)), reach(expressed.rows.map((r) => r.distance))]}
              yr={[-reach(expressed.rows.map((r) => r.log2fc)), reach(expressed.rows.map((r) => r.log2fc))]}
              xTitle={`3D distance, log2 of ${B} over ${A}`}
              yTitle={`Expression, log2 of ${B} over ${A}`}
              keys={keys}
            >
              {(fx, fy) =>
                expressed.rows.map((r) => (
                  <circle key={r.gene} className={kind(r)} cx={fx(r.distance)} cy={fy(r.log2fc)} r={r.gene === gene ? 5 : 3} onClick={() => pick(r.gene)}>
                    <title>{`${r.gene}, distance ${times(r.distance, 'farther', 'closer')}, expression ${fold(r.log2fc)} in ${B}`}</title>
                  </circle>
                ))
              }
            </Frame>
          </Figure>
        )}
        {expressed && volcano && (
          <Figure title="Volcano plot of the change of expression" file="expression_volcano" open={figure === 'volcano'} onToggle={() => toggle('volcano')}>
            <Frame
              label={`Volcano plot of ${tested.length} genes of the region among the genes of the test`}
              xr={[-vx, vx]}
              yr={[0, top]}
              xTitle={`Expression, log2 of ${B} over ${A}`}
              yTitle="-log10 of the adjusted p"
              capped={top >= volcano.p}
              keys={[...keys, ['back', 'genes outside the region']]}
            >
              {(fx, fy) => (
                <>
                  {volcano.points.map(([x, y]) => (
                    <circle key={`${x} ${y}`} className="back" cx={fx(x)} cy={fy(y)} r={1.3} />
                  ))}
                  {line < top && <line className="threshold" x1={fx(-vx)} x2={fx(vx)} y1={fy(line)} y2={fy(line)} />}
                  {line < top && (
                    <text className="tick" x={fx(vx) - 2} y={fy(line) - 4} textAnchor="end">
                      adjusted p 0.05
                    </text>
                  )}
                  {tested.map(({ r, x, y }) => (
                    <circle key={r.gene} className={kind(r)} cx={fx(x)} cy={fy(y)} r={r.gene === gene ? 5 : 3} onClick={() => pick(r.gene)}>
                      <title>{`${r.gene}, expression ${fold(r.log2fc)} in ${B}, adjusted p ${pText(r.padj)}`}</title>
                    </circle>
                  ))}
                </>
              )}
            </Frame>
          </Figure>
        )}
      </div>
    </section>
  )
}
