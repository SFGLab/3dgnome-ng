import { useQuery } from '@tanstack/react-query'
import { Ruler } from 'lucide-react'
import { useState } from 'react'
import type { Annotations, JobApi, Structure, Summary } from './api'
import type { Mode } from './colors'
import { FindGene } from './FindGene'
import { bp } from './format'
import { GeneCard } from './GeneCard'
import { DistanceCard } from './Measure'
import { StructureView } from './StructureView'

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

type Props = {
  api: JobApi
  summary: Summary
  selected: number
  gene?: string | null
  onGene?: (gene: string | null) => void
  onOpenGenes?: (gene: string) => void
}

/** The Structure tab: the selected structure in 3D, measurement between two beads over the
 * ensemble, and the card of a selected gene. */
export function StructureTab({ api, summary, selected, gene: geneProp, onGene, onOpenGenes = () => {} }: Props) {
  const [mode, setMode] = useState<Mode>('elements')
  const [measure, setMeasure] = useState(false)
  const [picks, setPicks] = useState<number[]>([])
  const [ownGene, setOwnGene] = useState<string | null>(null)
  const gene = geneProp !== undefined ? geneProp : ownGene
  const setGene = (g: string | null) => (onGene ? onGene(g) : setOwnGene(g))
  const structure = useQuery({ queryKey: ['structure', api.base, selected], queryFn: () => api.structure(selected) })
  const annotations = useQuery({ queryKey: ['annotations', api.base], queryFn: api.annotations })
  const genes = useQuery({ queryKey: ['genes', api.base], queryFn: api.genes })
  const row = genes.data?.find((r) => r.gene === gene)
  const pair = row ? [row.promoter_bead, row.enhancer_bead] : picks
  const s = structure.data
  const distance = useQuery({
    queryKey: ['distance', api.base, pair[0], pair[1]],
    queryFn: () => api.distance(mid(s!, pair[0]), mid(s!, pair[1])),
    enabled: !!s && pair.length === 2,
  })
  const region = parseRegion(summary.region)
  if (!s || !annotations.data || !region) return <div className="view-box" aria-busy="true" />
  const markers = pair.map((k, i) => ({ label: 'AB'[i], at: [s.beads[k][2], s.beads[k][3], s.beads[k][4]] as [number, number, number] }))
  const pick = (k: number) => {
    if (!measure) return
    setGene(null)
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
        onPickBead={pick}
        toolbarStart={<FindGene names={(genes.data ?? []).map((r) => r.gene)} onFind={(g) => setGene(g)} />}
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
            labels={[describe(distance.data.a.bp, region.chrom, annotations.data), describe(distance.data.b.bp, region.chrom, annotations.data)]}
          />
        )}
        {row && <GeneCard row={row} chrom={region.chrom} onOpenGenes={onOpenGenes} />}
      </div>
    </div>
  )
}
