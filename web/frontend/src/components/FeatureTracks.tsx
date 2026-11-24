import { useEffect, useRef, useState } from 'react'
import { MIN_BAR, packGenes, type TrackGene, unnamed } from './packGenes'
import { scoped } from '../lib/css'
import sheet from './feature-tracks.module.css'

const cn = scoped(sheet)

const ROW = 15
const MAX_ROWS = 4
/** The width the rows are packed for before the track is measured. */
const NOMINAL = 900

const place = (p: number) => p.toLocaleString('en-US')

/**
 * The genes of a view under its loops, as a genome browser draws them, with its enhancers. A gene is
 * a bar over its span with an arrowhead at the end it is read towards and its name beside it. The
 * genes are packed into rows. An enhancer is a box. The tracks follow the view they are given.
 * @param chrom the chromosome, for the words of a gene under the pointer
 * @param view the start and the end of the view in bp
 * @param genes the genes with their spans
 * @param enhancers the enhancers as start and end. Without them there is no enhancer track.
 * @param first the name of the gene the view is about, which is placed first and drawn stronger
 * @param onGene takes the name of a gene that is clicked, or null for a click on the first gene. With it a gene is a button.
 * @param pickable the names of the genes that can be clicked, every gene without it
 */
export function FeatureTracks({
  chrom,
  view,
  genes,
  enhancers,
  first,
  onGene,
  pickable,
}: {
  chrom: string
  view: [number, number]
  genes: TrackGene[]
  enhancers?: [number, number][]
  first?: string
  onGene?: (name: string | null) => void
  pickable?: Set<string>
}) {
  const box = useRef<HTMLDivElement>(null)
  const [width, setWidth] = useState(NOMINAL)
  useEffect(() => {
    const el = box.current
    if (!el || typeof ResizeObserver === 'undefined') return
    const seen = new ResizeObserver(() => el.clientWidth > 0 && setWidth(el.clientWidth))
    seen.observe(el)
    return () => seen.disconnect()
  }, [])
  const span = Math.max(view[1] - view[0], 1)
  const pct = (p: number) => `${((Math.min(Math.max(p, view[0]), view[1]) - view[0]) / span) * 100}%`
  const { placed, left } = packGenes(genes, view, width, MAX_ROWS, first)
  const rows = Math.max(1, ...placed.map((g) => g.row + 1))
  const boxes = (enhancers ?? []).filter(([s, e]) => e > view[0] && s < view[1])
  return (
    <div ref={box} className={cn('feature-tracks')}>
      <div className={cn('track genes')} style={{ height: rows * ROW }} role="group" aria-label="Genes">
        <span className={cn('track-name')} title={left ? `${left.toLocaleString('en-US')} more genes of this stretch have no room in the rows` : undefined}>
          {left ? `genes, ${left.toLocaleString('en-US')} more` : 'genes'}
        </span>
        {placed.map((g) => {
          const from = (Math.max(g.start, view[0]) - view[0]) / span
          const to = (Math.min(g.end, view[1]) - view[0]) / span
          const picked = onGene && (!pickable || pickable.has(g.name))
          const shared = {
            className: cn(`gene ${g.strand === '-' ? 'minus' : 'plus'}${g.name === first ? ' first' : ''}${unnamed(g) ? ' unnamed' : ''}${picked ? ' pick' : ''}`),
            style: { left: `${from * 100}%`, top: g.row * ROW, width: `max(${MIN_BAR}px, ${(to - from) * 100}%)` },
            title: `${g.name}, ${chrom}:${place(g.start)}-${place(g.end)}, ${g.strand === '-' ? 'minus' : 'plus'} strand`,
            'data-testid': 'gene',
          }
          const drawn = (
            <>
              <span className={cn('gene-bar')} />
              {g.named && <span className={cn('gene-name')}>{g.name}</span>}
            </>
          )
          // A gene that can be picked is a button. A second click on the gene in use lets it go.
          return picked ? (
            <button key={`${g.name}${g.start}`} type="button" {...shared} aria-label={g.name} aria-pressed={g.name === first} onClick={() => onGene(g.name === first ? null : g.name)}>
              {drawn}
            </button>
          ) : (
            <span key={`${g.name}${g.start}`} {...shared}>
              {drawn}
            </span>
          )
        })}
      </div>
      {enhancers && (
        <div className={cn('track enhancers')} role="group" aria-label="Enhancers">
          <span className={cn('track-name')}>enhancers</span>
          {boxes.map(([s, e]) => (
            <span key={`${s}-${e}`} className={cn('enhancer')} style={{ left: pct(s), width: `max(2px, calc(${pct(e)} - ${pct(s)}))` }} title={`Enhancer, ${chrom}:${place(s)}-${place(e)}`} data-testid="enhancer" />
          ))}
        </div>
      )}
    </div>
  )
}
