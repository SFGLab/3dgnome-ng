import { ArrowRight } from 'lucide-react'
import type { GeneRow } from '../api'
import { bp, kb } from '../lib/format'
import { scoped } from '../lib/css'
import sheet from './job.module.css'

const cn = scoped(sheet)

/** The Selected gene card: the gene's nearest enhancer by mean 3D distance over the ensemble. */
export function GeneCard({ row, chrom, onOpenGenes }: { row: GeneRow; chrom: string; onOpenGenes: (gene: string) => void }) {
  return (
    <section className={cn('side-card')} aria-label="Selected gene">
      <p className={cn('card-label')}>Selected gene</p>
      <p className={cn('gene-name')}>{row.gene}</p>
      <p className={cn('mono small')}>{`TSS ${chrom}:${bp(row.tss)}`}</p>
      <p className={cn('big-value')}>{row.distance.toFixed(1)}</p>
      <p className={cn('card-text')}>{`bead units to the nearest enhancer, mean over ${row.n} structure${row.n === 1 ? '' : 's'}`}</p>
      <dl className={cn('gene-facts')}>
        <dt>Nearest enhancer</dt>
        <dd className="mono">{`${chrom}:${bp(row.enhancer)}`}</dd>
        <dt>Separation</dt>
        <dd>{kb(row.linear_bp)}</dd>
        <dt>Range over structures</dt>
        <dd>{`${row.min.toFixed(1)} to ${row.max.toFixed(1)}`}</dd>
        <dt>Loop between them</dt>
        <dd>{row.joined_by ? `${row.joined_by} loop` : row.promoter_loops ? `none, the promoter anchors ${row.promoter_loops} loops` : 'none'}</dd>
      </dl>
      <button type="button" className="link-button" onClick={() => onOpenGenes(row.gene)}>
        All enhancers of this gene <ArrowRight size={13} aria-hidden />
      </button>
    </section>
  )
}
