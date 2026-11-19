import { useQuery } from '@tanstack/react-query'
import { Ruler } from 'lucide-react'
import { useState } from 'react'
import type { Annotations, JobApi, Loop, Structure, Summary } from './api'
import type { Mode } from './colors'
import { FindGene } from './FindGene'
import { bp, kb } from './format'
import { GeneCard } from './GeneCard'
import { LoopView, sameLoop } from './LoopView'
import { DistanceCard } from './Measure'
import { beadAt, StructureView } from './StructureView'

export function parseRegion(region: string | null): { chrom: string; start: number; end: number } | null {
  const m = region?.match(/^(\w+):(\d+)-(\d+)$/)
  return m ? { chrom: m[1], start: Number(m[2]), end: Number(m[3]) } : null
}

/** What a measured position is: a gene's TSS, an enhancer or a plain coordinate. */
function describe(pos: number, chrom: string, ann: Annotations): string {
  const gene = ann.genes.find((g) => Math.abs(g.tss - pos) <= 5000)
  if (gene) return `${gene.name} TSS, ${chrom}:${bp(pos)}`
  if (ann.enhancers.some(([a, b]) => pos >= a - 5000 && pos <= b + 5000)) return `enhancer, ${chrom}:${bp(pos)}`
  return `${chrom}:${bp(pos)}`
}

const mid = (s: Structure, k: number) => Math.round((s.beads[k][0] + s.beads[k][1]) / 2)

/** The card of a selected loop: its factor, its two anchors, its span and its PET count. */
function LoopCard({ loop }: { loop: Loop }) {
  return (
    <section className="side-card" aria-label="Selected loop">
      <p className="card-label">Selected loop</p>
      <p>
        <span className={`loop-badge ${loop.factor === 'RNAPII' ? 'rnapii' : 'ctcf'}`}>{loop.factor}</span>
      </p>
      <p className="point">
        <span className="marker">A</span> <span className="mono">{`${loop.chrom}:${bp(loop.a[0])}-${bp(loop.a[1])}`}</span>
      </p>
      <p className="point">
        <span className="marker">B</span> <span className="mono">{`${loop.chrom}:${bp(loop.b[0])}-${bp(loop.b[1])}`}</span>
      </p>
      <p className="card-text">{`The anchors are ${kb(loop.span)} apart on the genome and the loop has ${loop.pets} PET${loop.pets === 1 ? '' : 's'}.`}</p>
    </section>
  )
}

type Props = {
  api: JobApi
  summary: Summary
  selected: number
  gene?: string | null
  onGene?: (gene: string | null) => void
  onOpenGenes?: (gene: string) => void
  /** The loop marked in the model and in the loop view. */
  loop?: Loop | null
  onLoop?: (loop: Loop | null) => void
}

/** The Structure tab: the selected structure in 3D, its loops in two dimensions under it,
 * measurement between two beads over the ensemble and the card of a selected gene or loop. */
export function StructureTab({ api, summary, selected, gene: geneProp, onGene, onOpenGenes = () => {}, loop: loopProp, onLoop }: Props) {
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
  const structure = useQuery({ queryKey: ['structure', api.base, selected], queryFn: () => api.structure(selected) })
  const annotations = useQuery({ queryKey: ['annotations', api.base], queryFn: api.annotations })
  const genes = useQuery({ queryKey: ['genes', api.base], queryFn: api.genes })
  const row = genes.data?.find((r) => r.gene === gene)
  const s = structure.data
  // A selected loop is its two anchors: the beads that hold the middles of its two ends.
  const loopBeads = s && loop ? [beadAt(s.beads, (loop.a[0] + loop.a[1]) / 2), beadAt(s.beads, (loop.b[0] + loop.b[1]) / 2)] : null
  const pair = row ? [row.promoter_bead, row.enhancer_bead] : (loopBeads ?? picks)
  const distance = useQuery({
    queryKey: ['distance', api.base, pair[0], pair[1]],
    queryFn: () => api.distance(mid(s!, pair[0]), mid(s!, pair[1])),
    enabled: !!s && pair.length === 2,
  })
  const region = parseRegion(summary.region)
  if (!s || !annotations.data || !region) return <div className="view-box" aria-busy="true" />
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
    <div className="structure-tab">
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
        focusSpan={part}
        below={
          loops.length > 0 && (
            <LoopView
              chrom={region.chrom}
              region={part ?? [region.start, region.end]}
              loops={loops}
              onWindow={(a, b) => setPart([a, b])}
              onReset={part ? () => setPart(null) : undefined}
              selected={loopIndex < 0 ? null : loopIndex}
              onSelect={(i) => {
                setGene(null)
                setPicks([])
                setLoop(i === null ? null : loops[i])
              }}
              cursor={cursor}
              hint={part ? 'This part of the region stays solid in the model. Click a loop to mark its two anchors.' : 'Click a loop to mark its two anchors in the model. Drag along the loops to look at a part of the region.'}
            />
          )
        }
        toolbarStart={<FindGene names={(genes.data ?? []).map((r) => r.gene)} onFind={(g) => { setLoop(null); setGene(g) }} />}
        toolbarEnd={
          <button type="button" className={measure ? 'measure on' : 'measure'} aria-pressed={measure} onClick={() => setMeasure((m) => !m)}>
            <Ruler size={14} aria-hidden /> Measure
          </button>
        }
      />
      <div className="side-cards">
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
        {row && <GeneCard row={row} chrom={region.chrom} onOpenGenes={onOpenGenes} />}
        {!row && loop && <LoopCard loop={loop} />}
      </div>
    </div>
  )
}
