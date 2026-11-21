import { asked, type Compare, type ExpressionCompare } from '../../api'
import { useQuery } from '@tanstack/react-query'
import { useState } from 'react'
import { CsvButton } from '../../components/CsvButton'
import { kb } from '../../lib/format'
import { times, fold, twoEnhancers, pText, level } from './names'
import { type Key, Frame, BoxPlot } from './plots'
import { Figure } from './Figure'
import { Paged } from './PagedTable'

/** The whole number an axis reaches to, one at least. */
const reach = (values: number[]) => Math.max(1, Math.ceil(Math.max(0, ...values.map(Math.abs))))

type Row = Compare['rows'][number]
type ExpressionRow = ExpressionCompare['rows'][number]
type Tab = 'distance' | 'expression'
type Fig = 'box' | 'scatter' | 'volcano'

/**
 * What changed between two jobs, under their structures: the tables of every gene, one tab for
 * the 3D distance and one for the expression, and beside them the figures, each saved as a PNG by
 * its button, with the main values under them. The other job is compared against this one.
 */
export function CompareResults({
  urls,
  names,
  cmp,
  shared = false,
  expression,
  expressionState,
  gene,
  onGene,
}: {
  /** The addresses of this job and of the other job. */
  urls: [string, string]
  names: [string, string]
  cmp: Compare
  /** The comparison keeps to the enhancers both jobs have. */
  shared?: boolean
  expression?: ExpressionCompare
  /** `wait` while the expression is compared, `one` when one job alone has expression, the message of a failure, or empty. */
  expressionState: string
  gene: string | null
  onGene?: (gene: string | null) => void
}) {
  const [tab, setTab] = useState<Tab>('distance')
  // The open figures. Undefined is the figure the view starts with.
  const [opened, setOpened] = useState<Fig[] | undefined>(undefined)
  const [A, B] = names
  const against = `${B} against ${A}`
  const pick = (name: string) => onGene?.(name === gene ? null : name)
  const row = cmp.rows.find((r) => r.gene === gene)
  const expressed = expression && expression.mode !== 'none' && expression.rows.length > 0 ? expression : null
  const eRow = expressed?.rows.find((r) => r.gene === gene)
  const one = useQuery({
    ...asked.compareGene(urls[0], urls[1], gene ?? '', shared),
    enabled: !!row,
    staleTime: Infinity,
  })
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
  const open: Fig[] = opened ?? [expressed ? 'scatter' : 'box']
  const toggle = (f: Fig) => setOpened(open.includes(f) ? open.filter((x) => x !== f) : [...open, f])
  return (
    <section className="compare-results" aria-label="Changes between the two jobs">
      <div className="compare-tables">
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
                `enhancer_separation_bp_${A}`,
                `enhancer_separation_bp_${B}`,
                `distance_mean_${A}`,
                `distance_sd_${A}`,
                `distance_mean_${B}`,
                `distance_sd_${B}`,
                `distance_ratio_${B}_over_${A}`,
                'mann_whitney_p',
              ]}
              rows={() => cmp.rows.map((r) => [r.gene, r.linear_bp, r.linear_bp_b ?? r.linear_bp, r.a.mean, r.a.sd, r.b.mean, r.b.sd, r.ratio, r.p])}
            />
          ) : (
            <CsvButton
              file="expression_changes"
              head={['gene', `expression_${A}`, `expression_${B}`, `expression_log2_${B}_over_${A}`, 'adjusted_p', `distance_log2_${B}_over_${A}`]}
              rows={() => expressed!.rows.map((r) => [r.gene, r.a, r.b, r.log2fc, r.padj, r.distance])}
            />
          )}
        </div>
        {cmp.shared && gene && !row && (
          <p className="compare-note">{`${gene} has no enhancer that both jobs have. It is not in this comparison and is not marked in the structures.`}</p>
        )}
        {(cmp.shared || (cmp.differ ?? 0) > 0) && (
          <p className="compare-note">
            {cmp.shared
              ? `Each gene is measured to its nearest enhancer among the ${cmp.shared_enhancers} that both jobs have.`
              : `Each job measures a gene to its own nearest enhancer. ${cmp.differ} of ${cmp.rows.length} genes have another enhancer in ${B} than in ${A}.`}
          </p>
        )}
        {shown === 'distance' ? (
          <Paged<Row>
            rows={cmp.rows}
            gene={gene}
            noun="genes"
            label="Changes in enhancer to promoter distance"
            head={['Gene', A, B, against, 'Mann-Whitney p']}
            cells={(r) => [
              // A job measures a gene to its own nearest enhancer. Two enhancers are both stated.
              twoEnhancers(r) ? `${r.gene}, enhancer ${kb(r.linear_bp)} away in ${A}, ${kb(r.linear_bp_b!)} in ${B}` : `${r.gene}, enhancer ${kb(r.linear_bp)} away`,
              `${r.a.mean.toFixed(1)} ± ${r.a.sd.toFixed(1)}`,
              `${r.b.mean.toFixed(1)} ± ${r.b.sd.toFixed(1)}`,
              // The ratio is the other job's distance over this job's. Above 1 the other is farther.
              r.ratio >= 1 ? `${r.ratio.toFixed(1)} times farther` : `${(1 / r.ratio).toFixed(1)} times closer`,
              pText(r.p),
            ]}
            faint={(r) => r.p !== null && r.p >= 0.05}
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
            faint={(r) => r.padj !== null && r.padj >= 0.05}
            onGene={onGene}
          />
        )}
      </div>
      <div className="compare-side">
        <div className="compare-figures">
          <Figure
            title={row ? `Enhancer to promoter distance of ${row.gene}` : 'Enhancer to promoter distance of a gene'}
            file={`distance_${row?.gene ?? 'gene'}`}
            open={open.includes('box')}
            onToggle={() => toggle('box')}
          >
            {row && one.data ? (
              <BoxPlot gene={row.gene} names={names} a={one.data.a} b={one.data.b} />
            ) : (
              <p className="figure-empty">
                {row ? (one.isError ? one.error.message : 'Reading the distances.') : 'Select a gene in a table or a plot for the box plot of its distance in the structures of the two jobs.'}
              </p>
            )}
          </Figure>
          {expressed && (
            <Figure title="Change of expression against change of distance" file="expression_against_distance" open={open.includes('scatter')} onToggle={() => toggle('scatter')}>
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
            <Figure title="Volcano plot of the change of expression" file="expression_volcano" open={open.includes('volcano')} onToggle={() => toggle('volcano')}>
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
      </div>
    </section>
  )
}
