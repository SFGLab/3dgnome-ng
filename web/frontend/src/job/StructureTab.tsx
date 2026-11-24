import { Unanswered } from '../components/Unanswered'
import { askedOf } from '../api'
import { useQuery } from '@tanstack/react-query'
import { Ruler } from 'lucide-react'
import { useState } from 'react'
import type { Annotations, JobApi, Loop, Structure, Summary } from '../api'
import type { Mode } from './colors'
import { FindGene } from './FindGene'
import { bp, kb } from '../lib/format'
import { GeneCard } from './GeneCard'
import { LoopView } from './LoopView'
import { sameLoop } from './loops'
import { DistanceCard } from './Measure'
import { StructureView } from './StructureView'
import { beadAt, solidSpan } from './structure'
import { parseRegion } from '../lib/region'
import { scoped } from '../lib/css'
import sheet from './job.module.css'

const cn = scoped(sheet)

/** Returns the kind of a measured position, the TSS of a gene, an enhancer or a plain coordinate. */
function describe(pos: number, chrom: string, ann: Annotations): string {
  const gene = ann.genes.find((g) => Math.abs(g.tss - pos) <= 5000)
  if (gene) return `${gene.name} TSS, ${chrom}:${bp(pos)}`
  if (ann.enhancers.some(([a, b]) => pos >= a - 5000 && pos <= b + 5000)) return `enhancer, ${chrom}:${bp(pos)}`
  return `${chrom}:${bp(pos)}`
}

const mid = (s: Structure, k: number) => Math.round((s.beads[k][0] + s.beads[k][1]) / 2)

/** The card of a selected loop, with its factor, its two anchors, its span and its PET count. */
function LoopCard({ loop }: { loop: Loop }) {
  return (
    <section className={cn('side-card')} aria-label="Selected loop">
      <p className={cn('card-label')}>Selected loop</p>
      <p>
        <span className={cn(`loop-badge ${loop.factor === 'RNAPII' ? 'rnapii' : 'ctcf'}`)}>{loop.factor}</span>
      </p>
      <p className={cn('point')}>
        <span className={cn('marker')}>A</span> <span className="mono">{`${loop.chrom}:${bp(loop.a[0])}-${bp(loop.a[1])}`}</span>
      </p>
      <p className={cn('point')}>
        <span className={cn('marker')}>B</span> <span className="mono">{`${loop.chrom}:${bp(loop.b[0])}-${bp(loop.b[1])}`}</span>
      </p>
      <p className={cn('card-text')}>{`The anchors are ${kb(loop.span)} apart on the genome and the loop has ${loop.pets} PET${loop.pets === 1 ? '' : 's'}.`}</p>
    </section>
  )
}

type Props = {
  api: JobApi
  summary: Summary
  selected: number
  gene?: string | null
  /** A position inside the enhancer to mark for the selected gene, in place of its nearest enhancer. */
  enhancer?: number | null
  onGene?: (gene: string | null) => void
  onOpenGenes?: (gene: string) => void
  /** The loop marked in the model and in the loop view. */
  loop?: Loop | null
  onLoop?: (loop: Loop | null) => void
}

/**
 * The Structure tab. It has the selected structure in 3D, its loops in two dimensions under it,
 * measurement between two beads over the ensemble and the card of a selected gene or loop.
 */
export function StructureTab({ api, summary, selected, gene: geneProp, enhancer = null, onGene, onOpenGenes = () => {}, loop: loopProp, onLoop }: Props) {
  const [mode, setMode] = useState<Mode>('elements')
  const [measure, setMeasure] = useState(false)
  const [picks, setPicks] = useState<number[]>([])
  const [ownGene, setOwnGene] = useState<string | null>(null)
  const [ownLoop, setOwnLoop] = useState<Loop | null>(null)
  const [cursor, setCursor] = useState<number | null>(null)
  // A part of the region chosen on the loop view. The view spans it and the model keeps it solid.
  const [part, setPart] = useState<[number, number] | null>(null)
  const loop = loopProp !== undefined ? loopProp : ownLoop
  const setLoop = (l: Loop | null) => (onLoop ? onLoop(l) : setOwnLoop(l))
  const gene = geneProp !== undefined ? geneProp : ownGene
  const setGene = (g: string | null) => (onGene ? onGene(g) : setOwnGene(g))
  const structure = useQuery(askedOf(api).structure(selected))
  const annotations = useQuery(askedOf(api).annotations())
  const genes = useQuery(askedOf(api).genes())
  const row = genes.data?.find((r) => r.gene === gene)
  const s = structure.data
  // A selected loop is its two anchors, the beads that hold the middles of its two ends.
  const loopBeads = s && loop ? [beadAt(s.beads, (loop.a[0] + loop.a[1]) / 2), beadAt(s.beads, (loop.b[0] + loop.b[1]) / 2)] : null
  const named = !!row && !!s && enhancer !== null
  const pair = row ? [row.promoter_bead, named ? beadAt(s!.beads, enhancer!) : row.enhancer_bead] : (loopBeads ?? picks)
  const distance = useQuery({ ...askedOf(api).distance(s && pair.length === 2 ? mid(s, pair[0]) : 0, s && pair.length === 2 ? mid(s, pair[1]) : 0), enabled: !!s && pair.length === 2 })
  const region = parseRegion(summary.region)
  if (!s || !annotations.data || !region) return <Unanswered of={[structure, annotations]} className={cn('view-box')} />
  const markers = pair.map((k, i) => ({ label: 'AB'[i], at: [s.beads[k][2], s.beads[k][3], s.beads[k][4]] as [number, number, number] }))
  const loops = annotations.data.loops
  const loopIndex = loop ? loops.findIndex((l) => sameLoop(l, loop)) : -1
  const pick = (k: number) => {
    if (!measure) return
    setGene(null)
    setLoop(null)
    setPicks((p) => (p.length >= 2 ? [k] : [...p, k]))
  }
  return (
    <div className={cn('structure-tab')}>
      <StructureView
        structure={s}
        annotations={annotations.data}
        region={[region.start, region.end]}
        chrom={region.chrom}
        index={selected}
        total={summary.structures}
        mode={mode}
        onMode={setMode}
        markers={markers}
        focusBeads={pair}
        onPickBead={pick}
        onHoverPosition={setCursor}
        focusSpan={solidSpan(loop, part)}
        above={
          loops.length > 0 && (
            <LoopView
              chrom={region.chrom}
              region={part ?? [region.start, region.end]}
              bounds={[region.start, region.end]}
              loops={loops}
              genes={annotations.data.genes}
              enhancers={annotations.data.enhancers}
              gene={gene ?? undefined}
              pickableGenes={new Set((genes.data ?? []).map((r) => r.gene))}
              onGene={(g) => {
                setLoop(null)
                setPicks([])
                setGene(g)
              }}
              onWindow={(a, b) => setPart(a <= region.start && b >= region.end ? null : [a, b])}
              onReset={part ? () => setPart(null) : undefined}
              selected={loopIndex < 0 ? null : loopIndex}
              onSelect={(i) => {
                setGene(null)
                setPicks([])
                setLoop(i === null ? null : loops[i])
              }}
              cursor={cursor}
              hint={part ? 'This part of the region stays solid in the model. Click a loop to mark its two anchors and the chain between them, or a gene to mark it and its nearest enhancer. Scroll to zoom.' : 'Click a loop to mark its two anchors and the chain between them in the model, or a gene to mark it and its nearest enhancer. Scroll or drag along the loops to look at a part of the region.'}
            />
          )
        }
        toolbarStart={<FindGene names={(genes.data ?? []).map((r) => r.gene)} onFind={(g) => { setLoop(null); setGene(g) }} />}
        toolbarEnd={
          <button type="button" className={cn(measure ? 'measure on' : 'measure')} aria-pressed={measure} onClick={() => setMeasure((m) => !m)}>
            <Ruler size={14} aria-hidden /> Measure
          </button>
        }
      />
      <div className={cn('side-cards')}>
        {distance.data && pair.length === 2 && (
          <DistanceCard
            d={distance.data}
            current={selected}
            labels={
              row
                ? [`${row.gene} TSS, ${region.chrom}:${bp(distance.data.a.bp)}`, `enhancer, ${region.chrom}:${bp(distance.data.b.bp)}`]
                : loop
                  ? [`${loop.factor} anchor, ${region.chrom}:${bp(loop.a[0])}-${bp(loop.a[1])}`, `${loop.factor} anchor, ${region.chrom}:${bp(loop.b[0])}-${bp(loop.b[1])}`]
                  : [describe(distance.data.a.bp, region.chrom, annotations.data), describe(distance.data.b.bp, region.chrom, annotations.data)]
            }
          />
        )}
        {row && <GeneCard row={row} chrom={region.chrom} onOpenGenes={onOpenGenes} marked={named && pair[1] !== row.enhancer_bead ? enhancer! : undefined} />}
        {!row && loop && <LoopCard loop={loop} />}
      </div>
    </div>
  )
}
